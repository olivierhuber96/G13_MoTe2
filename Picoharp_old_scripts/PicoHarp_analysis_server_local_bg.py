import os, time, threading
import Pyro4       
import numpy as np     
import pickle
import glob
from time import gmtime, strftime
import time
from collections import deque

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


from multiprocessing import cpu_count
import math

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}

def curtime():
    return strftime("UTC %Y-%m-%d %H:%M:%S", gmtime())

from numba import njit, prange

# ---------------- constants ----------------
T2WRAPAROUND = 210698240  # PicoHarp T2 rollover (28-bit)
BASE_RES = 4e-12          # seconds per raw tick


# ---------------- low-level decoders ----------------
def extract_records_from_memmap(memmap, start_idx, stop_idx):
    """Return views (words) from memmap without copying (slice returns a view)."""
    return memmap[start_idx:stop_idx]

def decode_words(words):
    """Decode raw uint32 words -> channels (uint8), raw_timetags (uint32), special flags."""
    channels = (words >> 28) & 0xF
    timetags = words & 0x0FFFFFFF
    is_special = channels == 0xF
    # special types: overflow when low4 == 0, marker when low4 >0
    low4 = timetags & 0xF
    is_overflow = is_special & (low4 == 0)
    is_marker = is_special & (low4 != 0)
    return channels.astype(np.uint8, copy=False), timetags.astype(np.uint32, copy=False), is_overflow, is_marker


@njit(nogil=True, fastmath=True)
def correlate_numba(ch0_times, ch1_times, tau_min, tau_max, bin_width, hist):
    """
    Count pairs with tau_min < (t1 - t0) <= tau_max, binning (t1 - t0)//bin_width.
    Inputs must be sorted ascending and uint64.
    """
    nbins = hist.shape[0]
    n1 = ch1_times.size

    j_ge = 0  # first index with ch1 >= t0 (slides monotonically forward)

    for i in range(ch0_times.size):
        t0 = ch0_times[i]

        # advance to first ch1 >= t0 + tau_min (exclude delays below tau_min)
        while j_ge < n1 and ch1_times[j_ge] < t0 + tau_min:
            j_ge += 1

        # scan forward until t1 > t0 + tau_max
        tmax = t0 + tau_max
        k = j_ge  # reset inner pointer for this t0
        while k < n1:
            t1 = ch1_times[k]
            if t1 > tmax:
                break
            dt = t1 - t0 - tau_min # uint64, so dt >= 0
            if dt != 0:  
                b = dt // bin_width
                if b < nbins:
                    hist[b] += 1
            k += 1


def worker_process_chunk_streaming(args):
    """
    args:
        filepath, total_records, start_idx, stop_idx, overlap_stop_idx,
        tau_min, tau_max, bin_width, nbins, mode, overflow_positions
    """
    (filepath, total_records, start_idx, stop_idx, overlap_stop_idx,
        tau_min, tau_max, bin_width, nbins, mode, overflow_positions) = args

    memmap = np.memmap(filepath, dtype=np.uint32, mode='r', shape=(total_records,))
    words  = memmap[start_idx:overlap_stop_idx]
    if words.size == 0:
        return (start_idx, stop_idx, np.zeros(nbins, dtype=np.uint64), 0, 0)

    channels_raw, timetags_raw, is_overflow_local, _ = decode_words(words)

    # overflows before start_idx (global)
    if overflow_positions.size == 0:
        base_overflows_before_start = 0
    else:
        base_overflows_before_start = int(np.searchsorted(overflow_positions, start_idx, side='left'))

    # local cumulative wraps
    if is_overflow_local.any():
        local_wraps = np.cumsum(is_overflow_local.astype(np.uint8), dtype=np.uint64)
    else:
        local_wraps = np.zeros(words.shape[0], dtype=np.uint64)

    total_wraps = local_wraps + np.uint64(base_overflows_before_start)
    unwrapped   = timetags_raw.astype(np.uint64) + total_wraps * np.uint64(T2WRAPAROUND)

    # primary part of chunk
    rel_stop = max(0, min(stop_idx - start_idx, unwrapped.size))

    # choose events per mode
    if mode == "X01":
        mask_ch0 = (channels_raw[:rel_stop] == 0)
        ch0_times = unwrapped[:rel_stop][mask_ch0]
        ch1_times = unwrapped[channels_raw == 1]
    elif mode == "X10":
        mask_ch0 = (channels_raw[:rel_stop] == 1)
        ch0_times = unwrapped[:rel_stop][mask_ch0]
        ch1_times = unwrapped[channels_raw == 0]
    elif mode == "A0":
        mask_ch0 = (channels_raw[:rel_stop] == 0)
        ch0_times = unwrapped[:rel_stop][mask_ch0]
        ch1_times = unwrapped[channels_raw == 0]
    elif mode == "A1":
        mask_ch0 = (channels_raw[:rel_stop] == 1)
        ch0_times = unwrapped[:rel_stop][mask_ch0]
        ch1_times = unwrapped[channels_raw == 1]
    elif mode == "all":
        # ignore channels entirely
        ch0_times = unwrapped[:rel_stop]  # triggers from primary region only
        ch1_times = unwrapped             # matches can come from primary+overlap
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ch0_times = ch0_times.astype(np.uint64, copy=False)
    ch1_times = ch1_times.astype(np.uint64, copy=False)

    hist_local = np.zeros(nbins, dtype=np.uint32)
    if ch0_times.size and ch1_times.size:
        correlate_numba(ch0_times, ch1_times,
                        np.uint64(tau_min), np.uint64(tau_max), np.uint64(bin_width),
                        hist_local)
    return (start_idx, stop_idx, hist_local, ch0_times.size, ch1_times.size)

# ------------ streaming chunk builder (no full unwrapped array) ------------
def build_chunk_ranges_streaming(filepath, total_records, chunk_ticks, tau_max_ticks,
                                    overflow_positions, block_records=10_000_000):
    """
    Single-pass time-based chunking without materializing the full unwrapped timeline.
    Returns list of (start_idx, stop_idx, overlap_stop_idx).
    """
    mm = np.memmap(filepath, dtype=np.uint32, mode='r', shape=(total_records,))
    ranges = []

    start_idx = 0
    cur_idx   = 0

    # state across blocks
    wraps_before = 0
    have_t0 = False
    t0 = np.uint64(0)
    target_end_time = np.uint64(0)

    while cur_idx < total_records:
        end_idx = min(cur_idx + block_records, total_records)
        words   = mm[cur_idx:end_idx]
        # decode fast
        ch, tt, is_wrap, _ = decode_words(words)

        # local wraps and unwrapped times
        if is_wrap.any():
            local_wraps = np.cumsum(is_wrap.astype(np.uint8), dtype=np.uint64)
        else:
            local_wraps = np.zeros(words.size, dtype=np.uint64)
        total_wraps = local_wraps + np.uint64(wraps_before)
        unwrapped   = tt.astype(np.uint64) + total_wraps * np.uint64(T2WRAPAROUND)

        # initialize t0 and target if first block or new chunk
        if not have_t0:
            t0 = unwrapped[0]
            target_end_time = t0 + np.uint64(chunk_ticks)
            have_t0 = True

        # process all chunk boundaries that fall inside this block
        pos = 0  # relative pointer in this block
        while True:
            # find first index in this block where unwrapped >= target_end_time
            rel = np.searchsorted(unwrapped[pos:], target_end_time, side='left')
            if rel == (unwrapped.size - pos):
                # boundary not in this block
                break
            stop_abs = cur_idx + pos + rel  # absolute stop index
            # primary stop time is previous event’s time (in this block array)
            stop_time = unwrapped[pos + rel - 1] if (pos + rel - 1) >= 0 else target_end_time

            # find overlap_stop >= stop_time + tau_max_ticks
            tgt = stop_time + np.uint64(tau_max_ticks)

            # search within current block
            rel2 = np.searchsorted(unwrapped[pos + rel:], tgt, side='right')
            if (pos + rel + rel2) < unwrapped.size:
                overlap_abs = cur_idx + pos + rel + rel2
            else:
                # need to look ahead across blocks; scan forward minimally
                # we’ll walk subsequent blocks until we pass tgt
                overlap_abs = end_idx
                look_idx = end_idx
                add_wraps_before = int(total_wraps[-1])
                while look_idx < total_records:
                    look_end = min(look_idx + block_records, total_records)
                    w2 = mm[look_idx:look_end]
                    _, tt2, is_wrap2, _ = decode_words(w2)
                    if is_wrap2.any():
                        lw2 = np.cumsum(is_wrap2.astype(np.uint8), dtype=np.uint64) + np.uint64(add_wraps_before)
                    else:
                        lw2 = np.zeros(w2.size, dtype=np.uint64) + np.uint64(add_wraps_before)
                    u2 = tt2.astype(np.uint64) + lw2 * np.uint64(T2WRAPAROUND)
                    r2 = np.searchsorted(u2, tgt, side='right')
                    if r2 < u2.size:
                        overlap_abs = look_idx + r2
                        break
                    # advance
                    add_wraps_before = int(lw2[-1])
                    look_idx = look_end
                # cap at total_records
                overlap_abs = min(overlap_abs, total_records)

            ranges.append((start_idx, stop_abs, overlap_abs))

            # next chunk starts at stop_abs; set next target
            start_idx = stop_abs
            if start_idx < total_records:
                # define new t0 as this stop’s time; if it lies in this block, we can
                # read it from unwrapped; otherwise it will be initialized next block
                # when we re-enter the loop
                if (cur_idx <= start_idx < end_idx):
                    t0 = unwrapped[start_idx - cur_idx]
                    target_end_time = t0 + np.uint64(chunk_ticks)
                    have_t0 = True
                    # advance pos at least to here to keep search monotonic
                    pos = max(pos, (start_idx - cur_idx))
                else:
                    have_t0 = False  # force reinit in next block
            else:
                have_t0 = False
                break  # done

        # update block/global state and advance
        wraps_before = int(total_wraps[-1])
        cur_idx = end_idx

    # handle possible trailing tail (if nothing closed the last chunk)
    if start_idx < total_records:
        # find overlap to end by scanning forward once
        # we can re-use the while-loop logic, but do a simple conservative overlap
        overlap_abs = total_records
        ranges.append((start_idx, total_records, overlap_abs))

    return ranges






# ---------------- per-chunk worker ----------------
def worker_process_chunk(args):
    """
    args:
    filepath, total_records, start_idx, stop_idx, overlap_stop_idx,tau_min,
    tau_max, bin_width, nbins, mode, overflow_positions
    Worker:
    - memmap slice file
    - decode words for indices [start_idx : overlap_stop_idx) (includes overlap)
    - compute per-chunk unwrapped timestamps using overflow_positions to get base correction
    - select events where t0 in [start_idx, stop_idx) (i.e. main chunk only), and ch0 is according to the corr. mode
    - correlate with ch1 events anywhere in the chunk+overlap (ch1 == ... according to corr. mode)
    - return (start_idx, stop_idx, hist_local (uint64 array), count_ch0, count_ch1)
    """
    (filepath, total_records, start_idx, stop_idx, overlap_stop_idx,tau_min,
    tau_max, bin_width, nbins, mode, overflow_positions) = args

    # memmap the whole file but we only slice
    memmap = np.memmap(filepath, dtype=np.uint32, mode='r', shape=(total_records,))

    # slice from start_idx to overlap_stop_idx
    words = memmap[start_idx:overlap_stop_idx]  
    if words.size == 0:
        return (start_idx, stop_idx, np.zeros(nbins, dtype=np.uint64), 0, 0)

    channels_raw, timetags_raw, is_overflow_local, is_marker_local = decode_words(words)

    # compute cumulative overflow count before this chunk
    # overflow_positions is an array of absolute indices in the file
    # find how many overflows happened before start_idx
    if overflow_positions.size == 0:
        base_overflows_before_start = 0
    else:
        base_overflows_before_start = int(np.searchsorted(overflow_positions, start_idx, side='left'))

    # build local overflow counts within the chunk (vectorized)
    # positions of overflow within this slice (relative indices)
    # Instead of building a full array, compute cumulative count by using boolean cumulative sum:
    local_overflow_flags = np.empty(words.shape[0], dtype=np.uint8)
    # local_overflow_flags = ( (channels_raw == 0xF) & ((timetags_raw & 0xF) == 0) ).astype(np.uint8)
    # above is equivalent to is_overflow_local
    local_overflow_flags[:] = is_overflow_local.astype(np.uint8)

    # create overflow_count_per_record by cumsum of local flags and add base
    if local_overflow_flags.sum() == 0:
        overflow_counts = np.zeros(words.shape[0], dtype=np.uint64)
    else:
        overflow_counts = np.cumsum(local_overflow_flags, dtype=np.uint64)
    # total_overflow_count_at_pos = base_overflows_before_start + overflow_counts
    total_overflow_counts = overflow_counts + np.uint64(base_overflows_before_start)

    # now build unwrapped timestamps for entire slice:
    timetags64 = timetags_raw.astype(np.uint64)
    unwrapped = timetags64 + total_overflow_counts * np.uint64(T2WRAPAROUND)

    # select ch0 (reference) events that lie within [start_idx, stop_idx) - i.e., within the primary chunk region
    # Note: we used words built from start_idx so index 0 corresponds to start_idx
    rel_stop = stop_idx - start_idx  # number of primary records in slice
    # ensure bounds
    rel_stop = max(0, min(rel_stop, unwrapped.size))


    ############## Here we choose the different modes

    if mode == "X01":
        mask_ch0 = (channels_raw[:rel_stop] == 0)
        ch0_times = unwrapped[:rel_stop][mask_ch0].astype(np.uint64, copy=False)

        # ch1 pool: from start_idx to overlap_stop_idx (entire slice)
        mask_ch1 = (channels_raw == 1)
        ch1_times = unwrapped[mask_ch1].astype(np.uint64, copy=False)
    if mode == "X10":
        mask_ch0 = (channels_raw[:rel_stop] == 1)
        ch0_times = unwrapped[:rel_stop][mask_ch0].astype(np.uint64, copy=False)

        # ch1 pool: from start_idx to overlap_stop_idx (entire slice)
        mask_ch1 = (channels_raw == 0)
        ch1_times = unwrapped[mask_ch1].astype(np.uint64, copy=False)

    if mode == "A1":
        mask_ch0 = (channels_raw[:rel_stop] == 1)
        ch0_times = unwrapped[:rel_stop][mask_ch0].astype(np.uint64, copy=False)

        # ch1 pool: from start_idx to overlap_stop_idx (entire slice)
        mask_ch1 = (channels_raw == 1)
        ch1_times = unwrapped[mask_ch1].astype(np.uint64, copy=False)

    if mode == "A0":
        mask_ch0 = (channels_raw[:rel_stop] == 0)
        ch0_times = unwrapped[:rel_stop][mask_ch0].astype(np.uint64, copy=False)

        # ch1 pool: from start_idx to overlap_stop_idx (entire slice)
        mask_ch1 = (channels_raw == 0)
        ch1_times = unwrapped[mask_ch1].astype(np.uint64, copy=False)

    if mode == "all":
        ch0_times = unwrapped[:rel_stop].astype(np.uint64, copy=False)
        ch1_times = unwrapped.astype(np.uint64, copy=False)

    # Prepare histogram buffer
    hist_local = np.zeros(nbins, dtype=np.uint32)

    if ch0_times.size == 0 or ch1_times.size == 0:
        return (start_idx, stop_idx, hist_local, ch0_times.size, ch1_times.size)

    # correlate: only count t0 from main region (we already selected that),
    # and matches t1 can be anywhere in the slice (including overlap)
    correlate_numba(ch0_times, ch1_times,np.uint64(tau_min), np.uint64(tau_max), np.uint64(bin_width), hist_local)

    return (start_idx, stop_idx, hist_local, ch0_times.size, ch1_times.size)


@Pyro4.expose
class AnalysisService(object):
        
    def __init__(self):
        self._log = deque(maxlen=10000)
        self._cv = threading.Condition()

    def _append_log(self, msg):
        with self._cv:
            self._log.append(msg)
            self._cv.notify_all()

    def clear_log(self):
        with self._cv:
            self._log.clear()
            self._log.append(f"[log cleared @ {time.strftime('%Y-%m-%d %H:%M:%S')}]")
            self._cv.notify_all()

    def get_logs(self, last_idx=0, timeout=5.0):
        """Client polls for new logs; waits up to `timeout` seconds."""
        end = time.time() + timeout
        with self._cv:
            while last_idx >= len(self._log):
                remaining = end - time.time()
                if remaining <= 0:
                    return last_idx, []
                self._cv.wait(remaining)
            new = list(self._log)[last_idx:]
            return len(self._log), new
                    
    def ping(self):
        self._append_log("Analysis server reached.")

    def unwrap_chunks(self,filepath, show_progress, total_records):
        
        # memory-map entire file once (for quick vectorized scans)
        memmap = np.memmap(filepath, dtype=np.uint32, mode='r', shape=(total_records,))


        # Quick vectorized scan to find overflow positions (indices)
        # Warning: this will allocate intermediate arrays for the operations but not a full copy of memmap
        # We only compute where (channel==0xF) & (low4==0)
        low4 = memmap & 0xF
        channels_full = (memmap >> 28) & 0xF
        overflow_mask = (channels_full == 0xF) & (low4 == 0)
        overflow_positions = np.nonzero(overflow_mask)[0].astype(np.int64, copy=True)  # copy small array
        if show_progress:
            # msg = f"Found {overflow_positions.size} overflow markers in file."
            #print(msg)
            self._append_log(msg)

        # Now build time-based chunk boundaries.
        # We'll create chunks that try to contain approximately chunk_ms milliseconds of (unwrapped) time.
        # To do this cheaply: we compute unwrapped timestamps for the whole file incrementally via vectorized approach
        # but we avoid full extra copy by computing a compact array of 'unwrapped' times at coarse step positions.
        # Simpler pragmatic approach: compute unwrapped timestamps for the entire file (one array).
        # This uses extra memory ~8 bytes * total_records (uint64). If file is huge this may be large.
        # If memory is tight, reduce chunk_ms so chunks are smaller and choose chunk indices by record count instead.

        # Attempt to compute full unwrapped timestamps (fast & vectorized):
        if show_progress:
            msg =f"Computing full unwrapped timestamps for {total_records} total records (may use significant RAM for very large files)..."
            #print(msg)
            self._append_log(msg)
            start_timestamps = time.time()
        timetags_raw = memmap & 0x0FFFFFFF
        # Compute overflow cumulative counts per index by using searchsorted on overflow_positions for vector of indices
        # This builds an array of size total_records (uint64). Memory usage = 8 * total_records bytes.
        # If this is too large for our system, set use_record_based_chunking = True below.
        try:
            words = memmap
            # flags: overflow when ch==0xF and low nibble == 0
            is_wrap = (((words >> 28) & 0xF) == 0xF) & ((words & 0xF) == 0)
            cum_wraps = np.cumsum(is_wrap, dtype=np.uint64)
            cum_wraps -= is_wrap.astype(np.uint64)       # prefix (wraps BEFORE index i)

            # build unwrapped times with a SINGLE extra array
            timetags32 = (words & 0x0FFFFFFF)            # view (uint32) — no copy
            unwrapped_all = timetags32.astype(np.uint64) # one allocation of size N
            unwrapped_all += cum_wraps * np.uint64(T2WRAPAROUND)
            del cum_wraps, is_wrap, timetags32
            memory_ok = True
        except MemoryError:
            memory_ok = False
        if not memory_ok:
            # fallback: create chunks based on fixed record counts (no full unwrapped array)
            if show_progress:
                msg = "MemoryError when building unwrapped_all. Falling back to record-count-based chunking."
                #print(msg)
                self._append_log(msg)
            use_record_based_chunking = True
        else:
            if show_progress:
                msg = f"Finished computing full unwrapped timestamps in {int(time.time()-start_timestamps):d} s"
                #print(msg)
                self._append_log(msg)
            use_record_based_chunking = False

        return unwrapped_all, use_record_based_chunking, overflow_positions

    def build_chunk_ranges(self, unwrapped_all, chunk_ms, tau_max_ticks, total_records, use_record_based_chunking, show_progress):
        # Build chunk ranges (start_idx, stop_idx). pick chunk boundaries by time (unwrapped timestamps).
        chunk_ranges = []
        start_chunks = time.time()
        if not use_record_based_chunking:
                 
            chunk_ticks = np.uint64(round((chunk_ms * 1e-3) / BASE_RES))
            if chunk_ticks == 0:
                chunk_ticks = np.uint64(1)

            t0   = unwrapped_all[0]
            tend = unwrapped_all[-1]
        
            ntargets = int((tend - t0) // chunk_ticks) + 1
            targets  = t0 + chunk_ticks * np.arange(1, ntargets, dtype=np.uint64)

            stops = np.searchsorted(unwrapped_all, targets, side='left').astype(np.int64)
            stops = stops[(stops > 0) & (stops < unwrapped_all.size)]
            stops = np.unique(stops)  # remove duplicates

            # Timetag ranges for chunks
            starts = np.empty(stops.size + 1, dtype=np.int64)
            starts[0] = 0
            starts[1:] = stops
            final_stops = np.empty_like(starts)
            final_stops[:-1] = stops
            final_stops[-1]  = unwrapped_all.size

            stop_times   = unwrapped_all[final_stops[:-1] - 1]
            over_targets = stop_times + np.uint64(tau_max_ticks)
            over_stops   = np.searchsorted(unwrapped_all, over_targets, side='right').astype(np.int64)
            over_stops   = np.minimum(over_stops, total_records)

            chunk_ranges = list(zip(starts, final_stops, np.r_[over_stops, total_records]))


            if show_progress:
                msg = f"Built {len(chunk_ranges)} time-based chunks (chunk_ticks ~ {chunk_ticks}) in {time.time() - start_chunks:.1f} s."
                #print(msg)
                self._append_log(msg)
        else:
            # record based ch unking: choose records_per_chunk proportional to chunk_ms and approximate average rate
            approx_rate = max(1, total_records // max(1, int((unwrapped_all[-1] - unwrapped_all[0]) if 'unwrapped_all' in locals() else 1)))
            records_per_chunk = int(max(10_000, (approx_rate * (chunk_ms*1e-3))))  # very rough
            if show_progress:
                msg = f"Using records_per_chunk = {records_per_chunk}"
                #print(msg)
                self._append_log(msg)
            start = 0
            while start < total_records:
                stop = min(total_records, start + records_per_chunk)
                chunk_ranges.append((start, stop))
                start = stop
        return chunk_ranges

    def do_correlation(self,chunk_ranges,filepath, tau_min_ticks, tau_max_ticks, bin_width_ticks, nbins, total_records,overflow_positions, mode, n_workers, use_record_based_chunking, show_progress):
        
        # For each chunk, determine overlap_stop_idx = stop index extended to include any records within tau_max ticks after stop
        # We will extend up to the first index whose unwrapped time >= stop_time + tau_max_ticks
        if not use_record_based_chunking:
            # chunk_ranges already has (start, stop, overlap_stop)
            overlap_ranges = chunk_ranges
        else:
            overlap_ranges = []
            for (start, stop) in chunk_ranges:  # here chunk_ranges are pairs
                overlap_margin_records = int(max(1000, tau_max_ticks*2))
                overlap_stop = min(total_records, stop + overlap_margin_records)
                overlap_ranges.append((start, stop, overlap_stop))
        
        # Prepare tasks
        tasks = []
        for (start, stop, overlap_stop) in overlap_ranges:
            # skip empty
            if stop <= start:
                continue
            tasks.append((filepath, total_records, start, stop, overlap_stop,tau_min_ticks,
                        tau_max_ticks, bin_width_ticks, nbins, mode, overflow_positions))

        if show_progress:
            self._append_log(f"Submitting {len(tasks)} tasks to {n_workers} workers...")

        # run tasks in parallel

        hist_accum = np.zeros(nbins, dtype=np.uint32)
        total_ch0 = 0
        total_ch1 = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(worker_process_chunk, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                start_idx, stop_idx, hist_local, cnt0, cnt1 = res
                hist_accum += hist_local
                total_ch0 += cnt0
                total_ch1 += cnt1
                if show_progress and (i%1000 == 0):
                    elapsed = time.time() - t0
                    frac = i / len(tasks)
                    eta = elapsed / frac - elapsed if frac > 0 else 0.0
                    msg = f"\rTasks {100.0*i/len(tasks):.2f} % done, elapsed {elapsed:.1f}s, ETA {eta:.1f}s"
                    #print(msg, end='\r')
                    self._append_log(msg)
        corr_duration = time.time()-t0
        return hist_accum, total_ch0, total_ch1, corr_duration


    def convert_to_ticks(self, tau_min_us, tau_max_us, tau_bin_ps):
        # Convert units to raw PicoHarp ticks:
        bin_width_ticks = np.uint64(max(1,round((tau_bin_ps * 1e-12 / BASE_RES))))
        tau_max_ticks = np.uint64(round((tau_max_us * 1e-6) / BASE_RES))
        tau_min_ticks = np.uint64(round((tau_min_us * 1e-6) / BASE_RES))

        if bin_width_ticks <= 0:
            bin_width_ticks = 1
        nbins = int((tau_max_ticks - tau_min_ticks + bin_width_ticks - 1) // bin_width_ticks)
        idx   = np.arange(nbins, dtype=np.uint64) # don't need this and the ticks, saves memory
        ticks = tau_min_ticks + idx * bin_width_ticks
        tau   = ticks.astype(np.float64) * np.float64(BASE_RES)  # seconds
        del ticks
        del idx 
        return tau, tau_min_ticks, tau_max_ticks, bin_width_ticks, nbins

    #@staticmethod
    # ---------------- driver function ----------------
    def correlate_b_file_highperf(self, filepath, mode = "X01", chunk_ms=10.0,tau_min_us = 0.0,
                                tau_max_us=10.0, tau_bin_ps=16.0, n_workers=None, only_n_events = False, show_progress=True):
        """
        High-performance correlator for PicoHarp .b file.
        Parameters
        ----------
        filepath : str
            Path to .b file
        mode: str
            mode to correlate: "X01" is the typical crosscorrelation, "X10", "A0", "A1" and "all" are also permitted
        chunk_ms : float
            chunk length in milliseconds (ms) for splitting file into parallel tasks
        tau_min_us : float
            minimum correlation delay in microseconds (us)
        tau_max_us : float
            maximum correlation delay in microseconds (us)
        tau_bin_ps : float
            correlation bin width in picoseconds
        n_workers : int
            number of parallel workers (defaults to cpu_count()-1)
        only_n_events : Bool, returns # events only if True
        """
        
        ####### INITIALIZE ANALYSIS
        filesize = os.path.getsize(filepath)
        total_records = filesize // 4
        if total_records == 0:
            raise ValueError("Empty file")
        if only_n_events:
            return (),(),total_records
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)
        
        analysis_params = {"filepath":filepath, "mode":mode,
                           "chunk_ms":chunk_ms, "tau_max_us":tau_max_us,"tau_bin_ps":tau_bin_ps}

        # SAVE ANALYSIS PARAMETERS TO FOLDER
        with open(os.path.join(os.path.dirname(filepath),f'analysis_params{os.path.basename(filepath)[:-2]}_{mode}.pkl'), 'wb') as f:
            pickle.dump(analysis_params, f)
        print("Tried to save analysis params to "+os.path.join(os.path.dirname(filepath),f'analysis_params_{mode}.pkl'))
    

        ########## CONVERT TO TICKS
        tau, tau_min_ticks, tau_max_ticks, bin_width_ticks, nbins = self.convert_to_ticks(tau_min_us, tau_max_us, tau_bin_ps)
        if show_progress:
            msg= (f"[ANALYSIS - {curtime()}] File {filepath}: {total_records} records" + "\n"
                   + f"Using {n_workers} workers, chunk_ms={chunk_ms} ms, tau_min = {tau_min_us}, tau_max={tau_max_us} us -> {tau_max_ticks} ticks, bin_width={bin_width_ticks} ticks ({tau_bin_ps} ps)"
                   + "\n"+f"nbins = {nbins}")
            self._append_log(msg)


        ########### UNWRAP TIMESTAMPS
        unwrapped_all, use_record_based_chunking, overflow_positions = self.unwrap_chunks(filepath, show_progress, total_records)

        ###################### For proper pair correlation
        ########### Build chunk ranges
        chunk_ranges = self.build_chunk_ranges(unwrapped_all, chunk_ms, tau_max_ticks, total_records, use_record_based_chunking, show_progress)
        ########### Do proper pair correlation
        hist_accum, total_ch0, total_ch1, corr_duration = self.do_correlation(chunk_ranges,filepath, tau_min_ticks, tau_max_ticks, bin_width_ticks, nbins, total_records,overflow_positions, mode, n_workers, use_record_based_chunking, show_progress)
        if show_progress:
            print()
            msg = (f"Correlation finished in {corr_duration:.1f}s" +"\n"
                   + f"Total ch0 events counted: {total_ch0}, total ch1 events available: {total_ch1}"
                   +"\n"+f"Final histogram bins: {nbins}")
            self._append_log(msg)       
        


        ########### Save the correlated data to the folder
        np.save(os.path.join(os.path.dirname(filepath),f"histogram{os.path.basename(filepath)[:-2]}_{mode}.npy"), hist_accum)
        np.save(os.path.join(os.path.dirname(filepath),f"tau_bins{os.path.basename(filepath)[:-2]}_{mode}.npy"), tau + tau_min_us*1e-6)
        print("Analysis done.")
        return tau+tau_min_us*1e-6, hist_accum, total_records


    def correlate_b_file_streaming(self, filepath, mode="X01", chunk_ms=10.0,
                                   tau_min_us=0.0, tau_max_us=10.0, tau_bin_ps=16.0,
                                   n_workers=None, show_progress=True, block_records=10_000_000,only_do_first_n_chunks = None):

        
        filesize = os.path.getsize(filepath)
        total_records = filesize // 4
        if total_records == 0:
            raise ValueError("Empty file")

        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)

        # ---- unit conversions (unchanged) ----
        bin_width_ticks = np.uint64(max(1, round((tau_bin_ps * 1e-12) / BASE_RES)))
        tau_min_ticks   = np.uint64(round((tau_min_us * 1e-6) / BASE_RES))
        tau_max_ticks   = np.uint64(round((tau_max_us * 1e-6) / BASE_RES))

        nbins = int(((tau_max_ticks - tau_min_ticks) + bin_width_ticks - np.uint64(1)) // bin_width_ticks)
        idx   = np.arange(nbins, dtype=np.uint64)
        ticks = idx * bin_width_ticks
        tau   = ticks.astype(np.float64) * np.float64(BASE_RES)   # seconds

        if show_progress:
            msg = (f"File {filepath}: {total_records} records" + "\n" +
                   f"mode={mode}, chunk_ms={chunk_ms}, tau_min={tau_min_us}us, tau_max={tau_max_us}us, bin={tau_bin_ps}ps" + "\n" +
                   f"ticks: min={tau_min_ticks}, max={tau_max_ticks}, bin={bin_width_ticks}, nbins={nbins}")
            self._append_log(msg)

        # ----------------------------------------------------------------------
        # STREAMED overflow marker scan (drop-in replacement for memmap block)
        # ----------------------------------------------------------------------
        overflow_pos_chunks = []
        record_offset = 0
        # Cap read size to avoid large resident sets; keep using your block_records as a guide.
        block_records_scan = max(1, min(block_records, 25_000_000))  # ~100 MB per read

        with open(filepath, 'rb') as fh:
            while True:
                buf = np.fromfile(fh, dtype=np.uint32, count=block_records_scan)
                if buf.size == 0:
                    break
                low4 = buf & np.uint32(0xF)
                ch_full = (buf >> np.uint32(28)) & np.uint32(0xF)
                mask = (ch_full == 0xF) & (low4 == 0)
                if mask.any():
                    rel = np.nonzero(mask)[0].astype(np.int64, copy=False)
                    overflow_pos_chunks.append(rel + record_offset)
                record_offset += buf.size

        overflow_positions = (np.concatenate(overflow_pos_chunks).astype(np.int64, copy=False)
                              if overflow_pos_chunks else np.empty(0, dtype=np.int64))
        # ----------------------------------------------------------------------

        if show_progress:
            msg = f"Found {overflow_positions.size} overflow markers."
            self._append_log(msg)

        # build chunk ranges by time with a single streamed pass (unchanged)
        chunk_ticks = np.uint64(round((chunk_ms * 1e-3) / BASE_RES))
        if chunk_ticks <= 0:
            chunk_ticks = 1

        t0 = time.time()
        overlap_ranges = build_chunk_ranges_streaming(filepath, total_records,
                                                      chunk_ticks, tau_max_ticks,
                                                      overflow_positions,
                                                      block_records=block_records)
        if show_progress:
            msg = f"Built {len(overlap_ranges)} chunks of length {float(chunk_ms/1e3):.2f}s, this took {time.time()-t0:.1f}s."
            self._append_log(msg)

        # Optional: Only analyze the first n chunks 
        if only_do_first_n_chunks is not None:
            overlap_ranges = overlap_ranges[:only_do_first_n_chunks]
            if show_progress:
                msg = f"Only correlating the first {len(overlap_ranges)} chunks."
                self._append_log(msg)

        analysis_params = {"filepath": filepath, "mode": mode,
                           "chunk_ms": chunk_ms, "tau_max_us": tau_max_us,
                           "tau_min_us": tau_min_us, "tau_bin_ps": tau_bin_ps,"n_chunks":len(overlap_ranges)}

        # SAVE ANALYSIS PARAMETERS TO FOLDER
        with open(os.path.join(os.path.dirname(filepath),
                               f'analysis_params{os.path.basename(filepath)[:-2]}_{mode}.pkl'), 'wb') as f:
            pickle.dump(analysis_params, f)
        print("Tried to save analysis params to "
              + os.path.join(os.path.dirname(filepath), f'analysis_params_{mode}.pkl'))


        # correlate per chunk in parallel (unchanged)
        hist_accum = np.zeros(nbins, dtype=np.uint32)
        total_ch0 = total_ch1 = 0
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            tasks = [(filepath, total_records, s, e, o,
                      tau_min_ticks, tau_max_ticks, bin_width_ticks, nbins, mode, overflow_positions)
                     for (s, e, o) in overlap_ranges if e > s]
            futures = [ex.submit(worker_process_chunk_streaming, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                start_idx, stop_idx, hist_local, cnt0, cnt1 = fut.result()
                hist_accum += hist_local
                total_ch0  += cnt0
                total_ch1  += cnt1
                if show_progress and (i % 5 == 0 or i == len(futures)):
                    elapsed = time.time() - t0
                    frac = i / len(futures)
                    eta = (elapsed/frac - elapsed) if frac > 0 else 0.0
                    msg = f"\rTasks{100.0*frac:5.1f} % done, elapsed {elapsed:.1f}s  ETA {eta:.1f}s"
                    self._append_log(msg)

        if show_progress:
            msg = f"Correlation done in {time.time()-t0:.1f}s; ch0={total_ch0}, ch1={total_ch1}"
            self._append_log(msg)

        # Save (unchanged filenames & contents)
        np.save(os.path.join(os.path.dirname(filepath),
                             f"histogram{os.path.basename(filepath)[:-2]}_{mode}.npy"), hist_accum)
        np.save(os.path.join(os.path.dirname(filepath),
                             f"tau_bins{os.path.basename(filepath)[:-2]}_{mode}.npy"),
                tau + tau_min_us*1e-6)

        print("Correlation done.")
        self._append_log("Correlation done.")
        return tau + tau_min_us*1e-6, hist_accum, total_records


    
    def correlate_b_file_classical(self,
                                filepath,
                                tau_bin_ms=0.01,
                                chunk_minutes=30,
                                block_records=10_000_000,
                                include_channels=None,   # None = all photon channels
                                show_progress=True):
        """
        Minimal streaming binning: PicoHarp .b -> binned counts per chunk.

        - Unwraps timestamps on the fly (handles overflows).
        - Bins *photon* events (non-special) into fixed bins of width tau_bin_ms.
        - Splits into chunks of length `chunk_minutes`, resets time origin per chunk.
        - Writes each chunk to disk as NPZ with arrays:
            counts (uint32), and scalars dt_s, chunk_minutes, chunk_index,
            t0_abs_ticks (first tick of the chunk), wrap_ticks.

        Returns:
            out_dir (str), list_of_npz_files (list[str])
        """
        
        # Convert bin width and chunk length to ticks
        dt_s = tau_bin_ms * 1e-3
        dt_ticks = int(round(dt_s / BASE_RES))
        if dt_ticks <= 0:
            dt_ticks = 1

        chunk_s = float(chunk_minutes) * 60.0
        chunk_ticks = np.uint64(round(chunk_s / BASE_RES))
        if chunk_ticks == 0:
            chunk_ticks = np.uint64(1)

        # Bins per chunk (fixed)
        n_bins_chunk = int(np.ceil(chunk_s / dt_s))
        if n_bins_chunk <= 0:
            n_bins_chunk = 1

        # ---------- input and output ----------
        filesize = os.path.getsize(filepath)
        total_records = filesize // 4
        if total_records == 0:
            raise ValueError("Empty file")

        # Output dir next to the file
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_dir = os.path.join(os.path.dirname(filepath), f"{base}_classical_bins")
        os.makedirs(out_dir, exist_ok=True)

        # Save minimal analysis params
        analysis_params = {
            "filepath": filepath,
            "tau_bin_ms": float(tau_bin_ms),
            "chunk_minutes": float(chunk_minutes),
            "block_records": int(block_records),
            "BASE_RES": float(BASE_RES),
            "include_channels":include_channels
        }
        with open(os.path.join(out_dir, "analysis_params_classical.pkl"), "wb") as f:
            pickle.dump(analysis_params, f)

        if show_progress:
            self._append_log(f"[classical] File {filepath} ({total_records} records)")
            self._append_log(f"[classical] dt={tau_bin_ms} ms (dt_ticks={dt_ticks}), "
                            f"chunk={chunk_minutes} min (chunk_ticks={int(chunk_ticks)}) "
                            f"-> {n_bins_chunk} bins/chunk including channels (none = all): {include_channels}.")

        # ---------- helpers ----------
        def save_chunk(chunk_idx, counts, t0_abs_ticks,channel_info = None):
            # counts: uint32 length n_bins_chunk
            if channel_info is not None:
                channel_name = "_ch_"+str(channel_info)
            else:
                channel_name = ""
            fn = os.path.join(out_dir,
                            f"{base}_bins_dt{tau_bin_ms}ms_chunk{chunk_idx:05d}"+channel_name+".npz")
            np.savez_compressed(
                fn,
                counts=counts.astype(np.uint32, copy=False),
                dt_s=np.array(dt_s, dtype=float),
                chunk_minutes=np.array(chunk_minutes, dtype=float),
                chunk_index=np.array(chunk_idx, dtype=np.int64),
                t0_abs_ticks=np.array(np.uint64(t0_abs_ticks), dtype=np.uint64),
                wrap_ticks=np.array(T2WRAPAROUND, dtype=np.uint64),
            )
            return fn

        def decode_words_local(words_u32):
            """Return channels_full, timetags_u32, is_special, is_overflow."""
            ch_full = (words_u32 >> 28) & 0xF
            timetags = (words_u32 & 0x0FFFFFFF).astype(np.uint32, copy=False)
            is_special = (ch_full == 0xF)
            low4 = (words_u32 & 0xF)
            is_overflow = is_special & (low4 == 0)
            return ch_full, timetags, is_special, is_overflow

        # ---------- streaming state ----------
        memmap = np.memmap(filepath, dtype=np.uint32, mode='r', shape=(total_records,))
        overflow_total = np.uint64(0)

        # chunk state
        chunk_idx = 0
        have_open_chunk = False
        t0_chunk = np.uint64(0)
        t1_chunk = np.uint64(0)
        counts = None
        out_files = []

        t_start = time.time()

        # ---------- stream in record blocks ----------
        for rec_start in range(0, total_records, block_records):
            rec_stop = min(total_records, rec_start + block_records)
            words = memmap[rec_start:rec_stop]

            # decode
            ch_full, tt32, is_special, is_overflow = decode_words_local(words)

            # block-local prefix wrap count
            is_wrap = is_overflow.astype(np.uint8)
            if is_wrap.size:
                cum_wrap = np.cumsum(is_wrap, dtype=np.uint64)
                prefix_wraps = cum_wrap - is_wrap.astype(np.uint64)
                block_wraps = cum_wrap[-1]
            else:
                prefix_wraps = np.zeros(0, dtype=np.uint64)
                block_wraps = np.uint64(0)

            # unwrapped (all records)
            unwrapped = tt32.astype(np.uint64) + (overflow_total + prefix_wraps) * T2WRAPAROUND
            overflow_total += block_wraps  # carry to next block

            # select photon events (non-special)
            photon_mask = ~is_special
            if include_channels is not None:
                # include only those channels (on PicoHarp, non-special channels are 0..14)
                in_mask = np.isin(ch_full, np.array(include_channels, dtype=np.uint8))
                photon_mask &= in_mask

            if not photon_mask.any():
                if show_progress:
                    # lightweight progress ping
                    self._append_log(f"[classical] processed records {rec_stop}/{total_records}")
                continue

            times = unwrapped[photon_mask]
            # Times are monotonic because the file is in acquisition order

            # Open the very first chunk at the time of the first photon
            if not have_open_chunk:
                t0_chunk = np.uint64(times[0])
                t1_chunk = t0_chunk + chunk_ticks
                counts = np.zeros(n_bins_chunk, dtype=np.uint32)
                have_open_chunk = True

            # Process this block's events across chunk boundaries
            p = 0
            N = times.size
            while p < N:
                # Finalize chunks until current event fits
                while times[p] >= t1_chunk:
                    # save current chunk
                    out_files.append(save_chunk(chunk_idx, counts, t0_chunk, channel_info = include_channels))
                    chunk_idx += 1
                    # start next chunk (advance by fixed stride)
                    t0_chunk = t1_chunk
                    t1_chunk = t0_chunk + chunk_ticks
                    counts[:] = 0  # reset

                # all events for current chunk
                q = int(np.searchsorted(times, t1_chunk, side='left'))
                # Slice belongs to [t0_chunk, t1_chunk)
                if q > p:
                    dticks = times[p:q] - t0_chunk
                    # integer bin indices
                    idx = (dticks // dt_ticks).astype(np.int64, copy=False)
                    # Guard against any numerical edge case
                    idx = idx[(idx >= 0) & (idx < n_bins_chunk)]
                    if idx.size:
                        bc = np.bincount(idx, minlength=n_bins_chunk)
                        # bincount returns int64; add into uint32 (safe for normal count ranges)
                        counts += bc.astype(np.uint32, copy=False)
                p = q

            if show_progress:
                done = 100.0 * rec_stop / total_records
                self._append_log(f"\r [classical autocorrelation] processed {done:5.1f}% ({rec_stop}/{total_records}) records")

        # End of file: flush last open chunk (if any)
        if have_open_chunk:
            out_files.append(save_chunk(chunk_idx, counts, t0_chunk, channel_info = include_channels))

        if show_progress:
            dt = time.time() - t_start
            self._append_log(f"[classical] done in {dt:.1f} s. Wrote {len(out_files)} chunks to: {out_dir}")

        return out_dir, out_files

    def read_existing_classical(self, folder, chunk_range=None, tau_bin_ms=None, tol=1e-12, include_channel = None):
        """
        Scan `folder` for files like ..._bins_dt<tau_bin_ms>ms_chunk<idx>.npz, sort by idx,
        optionally filter to indices in `chunk_range` (inclusive) and/or to a specific
        tau_bin_ms (float).

        """
        
        if include_channel is not None:
            channel_name = "_ch_" + str(include_channel)
        else:
            channel_name = ""
        pattern = os.path.join(folder, f"*_bins_dt*ms_chunk*{channel_name}.npz")
        candidates = glob.glob(pattern)
        items = []
        for path in candidates:
            base = os.path.basename(path)
            stem = base[:-4]  # drop ".npz"


            # --- extract chunk index ---

            stem_main = stem.split("_ch_")[0]            
            pos_chunk = stem_main.rfind("chunk")
            if pos_chunk == -1:
                continue
            idx_str = stem_main[pos_chunk + len("chunk"):]
            if not idx_str.isdigit():
                continue
            idx = int(idx_str)
        
            # --- optional: match tau_bin_ms from filename ---
            if tau_bin_ms is not None:
                pos_dt = stem.rfind("_bins_dt")
                pos_ms = stem.rfind("ms_chunk")
                if pos_dt == -1 or pos_ms == -1 or pos_ms <= pos_dt:
                    continue
                dt_str = stem[pos_dt + len("_bins_dt"):pos_ms]
                try:
                    dt_val = float(dt_str)
                except ValueError:
                    continue
                if abs(dt_val - float(tau_bin_ms)) > tol:
                    continue

            items.append((idx, path))

        if not items:
            return [], [], [], [], [], [], None, None  # keep your return arity

        # sort by chunk index
        items.sort(key=lambda t: t[0])

        # optional filter by chunk_range
        if chunk_range is not None:
            lo, hi = chunk_range
            items = [(i, p) for (i, p) in items if lo <= i <= hi]
            if not items:
                return [], [], [], [], [], [], None, None

        # load & unpack
        counts_list, dt_s_list, chunk_minutes_list = [], [], []
        chunk_index_list, t0_abs_ticks_list, wrap_ticks_list = [], [], []
        for i, path in items:
            with np.load(path, allow_pickle=False) as z:
                counts_list.append(z["counts"])             # (n_bins_chunk,) uint32
                dt_s_list.append(float(z["dt_s"]))          # scalar
                chunk_minutes_list.append(int(z["chunk_minutes"]))
                chunk_index_list.append(int(z["chunk_index"]))
                t0_abs_ticks_list.append(z["t0_abs_ticks"]) # scalar uint64
                wrap_ticks_list.append(z["wrap_ticks"])     # scalar uint64

        idxs = [i for (i, _) in items]
        return (counts_list, dt_s_list, chunk_minutes_list, chunk_index_list,
                t0_abs_ticks_list, wrap_ticks_list, min(idxs), max(idxs))

    def read_existing(self,folder,name, mode = "X01"):
        def _name_part_from_hist(base, mode):
            # base like: "histogram..._{mode}.npy" -> returns "..."
            return base[len("histogram"):-len(f"_{mode}.npy")]

        def _strip_filepath(d):
            d = dict(d)
            d.pop('filepath', None)
            return d

        if name == "all":
            patt = os.path.join(folder, f"histogram*_{mode}.npy")
            h_files = glob.glob(patt)
            if not h_files:
                raise FileNotFoundError(f"No files matching {patt}")

            items = []
            for hpath in h_files:
                base = os.path.basename(hpath)
                if not (base.startswith("histogram") and base.endswith(f"_{mode}.npy")):
                    continue
                name_part = _name_part_from_hist(base, mode)
                tpath = os.path.join(folder, f"tau_bins{name_part}_{mode}.npy")
                ppath = os.path.join(folder, f"analysis_params{name_part}_{mode}.pkl")
                if not os.path.exists(tpath) or not os.path.exists(ppath):
                    continue
                items.append((name_part, hpath, tpath, ppath))

            if not items:
                raise FileNotFoundError(f"No complete (hist, tau, params) triples found in {folder} for mode {mode}")

            # Load first set as reference
            name0, h0p, t0p, p0p = items[0]
            h0 = np.load(h0p)
            tau0 = np.load(t0p)
            with open(p0p, 'rb') as f:
                p0 = pickle.load(f)
            p0_norm = _strip_filepath(p0)

            # Sum all histograms, validating compatibility
            h_sum = np.zeros_like(h0, dtype=np.uint64)
            for (nm, hp, tp, pp) in items:
                h = np.load(hp)
                tau = np.load(tp)
                with open(pp, 'rb') as f:
                    p = pickle.load(f)

                # shape check
                if h.shape != h0.shape:
                    raise ValueError(f"Incompatible histogram shapes for {nm}: {h.shape} vs {h0.shape}")
                # tau check
                if not np.array_equal(tau, tau0):
                    raise ValueError(f"Incompatible tau bins for {nm}")
                # params check (ignore filepath)
                if _strip_filepath(p) != p0_norm:
                    raise ValueError(f"Analysis parameters not identical (except filepath) for {nm}")

                h_sum += h.astype(np.uint64, copy=False)

            self._append_log(f"[ANALYSIS - {curtime()}] Combined {len(items)} chunks in mode {mode}")
            return h_sum, tau0, p0

        if isinstance(name, str):
            self._append_log(f"[ANALYSIS - {curtime()}] Reading file with name {name} in mode {mode} from folder {folder}")
            h_dat = np.load(os.path.join(folder,f"histogram{name}_{mode}.npy"))
            tau_bins = np.load(os.path.join(folder,f"tau_bins{name}_{mode}.npy"))
            with open(os.path.join(folder,f'analysis_params{name}_{mode}.pkl'),'rb') as f:
                loaded_analysis_dict = pickle.load(f)
            return h_dat, tau_bins, loaded_analysis_dict
        
        elif isinstance(name, list) and all(isinstance(n, str) for n in name):
            self._append_log(f"[ANALYSIS - {curtime()}] Reading files with names {name} in mode {mode} from folder {folder}")
            h_dat = []
            tau_bins = []
            loaded_analysis_dicts = []
            for n in name:
                h_dat.append(np.load(os.path.join(folder,f"histogram{n}_{mode}.npy")))
                tau_bins.append(np.load(os.path.join(folder,f"tau_bins{n}_{mode}.npy")))

                with open(os.path.join(folder,f'analysis_params{n}_{mode}.pkl'),'rb') as f:
                    loaded_analysis_dicts.append(pickle.load(f))

            return h_dat, tau_bins, loaded_analysis_dicts
        
        else:
            raise ValueError("Value Error in read_existing: name argument neither a string nor a list of strings!")
        

if __name__ == "__main__":
    svc = AnalysisService()
    daemon = Pyro4.Daemon(host="127.0.0.1", port=9097)
    uri = daemon.register(svc, objectId="Analysis_g13")
    print("Analysis server URI:", uri)
    daemon.requestLoop()
