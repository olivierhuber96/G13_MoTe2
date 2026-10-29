# pico_corr_fast.py
import os
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import math

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# ---------------- constants ----------------
T2WRAPAROUND = 210698240  # PicoHarp T2 rollover (28-bit)
BASE_RES = 4e-12          # seconds per raw tick (adjust if different)

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

# ---------------- numba-accelerated correlator ----------------
if NUMBA_AVAILABLE:
    @njit(cache=True)
    def correlate_numba(ch0, ch1, tau_max, bin_width, hist):
        """
        Count delays dt = t1 - t0 where 0 <= dt < tau_max.
        ch0 and ch1 must be uint64 sorted arrays.
        hist: preallocated uint64 histogram array (nbins).
        This is single-sided (t1 > t0).
        """
        n0 = ch0.shape[0]
        n1 = ch1.shape[0]
        nbins = hist.shape[0]
        for i in range(n0):
            t0 = ch0[i]
            # binary search start index in ch1
            lo = 0
            hi = n1
            # find leftmost index >= t0
            while lo < hi:
                mid = (lo + hi) // 2
                if ch1[mid] < t0:
                    lo = mid + 1
                else:
                    hi = mid
            j = lo
            # iterate until dt >= tau_max
            while j < n1:
                dt = ch1[j] - t0
                if dt >= tau_max:
                    break
                binidx = dt // bin_width
                if 0 <= binidx < nbins:
                    hist[binidx] += 1
                j += 1
        return hist
else:
    def correlate_numba(ch0, ch1, tau_max, bin_width, hist):
        # fallback pure-python/numpy version (slower)
        n1 = ch1.shape[0]
        for t0 in ch0:
            j = np.searchsorted(ch1, t0, 'left')
            if j >= n1:
                continue
            dt = ch1[j:] - t0
            mask = (dt >= 0) & (dt < tau_max)
            if not mask.any():
                continue
            bins = (dt[mask] // bin_width).astype(np.int64)
            for b in bins:
                hist[b] += 1
        return hist

# ---------------- per-chunk worker ----------------
def _worker_process_chunk(args):
    """
    args:
      filepath, total_records, start_idx, stop_idx, overlap_stop_idx,
      tau_max, bin_width, nbins, channel_a, channel_b, overflow_positions
    Worker:
      - memmap slice file
      - decode words for indices [start_idx : overlap_stop_idx) (includes overlap)
      - compute per-chunk unwrapped timestamps using overflow_positions to get base correction
      - select events where t0 in [start_idx, stop_idx) (i.e. main chunk only), and ch0 == channel_a
      - correlate with ch1 events anywhere in the chunk+overlap (ch1 == channel_b)
      - return (start_idx, stop_idx, hist_local (uint64 array), count_ch0, count_ch1)
    """
    (filepath, total_records, start_idx, stop_idx, overlap_stop_idx,
     tau_max, bin_width, nbins, channel_a, channel_b, overflow_positions) = args

    # memmap the whole file but we only slice
    memmap = np.memmap(filepath, dtype=np.uint32, mode='r', shape=(total_records,))

    # slice from start_idx to overlap_stop_idx
    words = memmap[start_idx:overlap_stop_idx]  # view -> cheap
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

    # build masks (no copy of unwrapped arrays besides boolean mask)
    mask_ch0 = (channels_raw[:rel_stop] == channel_a)
    ch0_times = unwrapped[:rel_stop][mask_ch0].astype(np.uint64, copy=False)

    # ch1 pool: from start_idx to overlap_stop_idx (entire slice)
    mask_ch1 = (channels_raw == channel_b)
    ch1_times = unwrapped[mask_ch1].astype(np.uint64, copy=False)

    # Prepare histogram buffer
    hist_local = np.zeros(nbins, dtype=np.uint64)

    if ch0_times.size == 0 or ch1_times.size == 0:
        return (start_idx, stop_idx, hist_local, ch0_times.size, ch1_times.size)

    # correlate: only count t0 from main region (we already selected that),
    # and matches t1 can be anywhere in the slice (including overlap)
    correlate_numba(ch0_times, ch1_times, np.uint64(tau_max), np.uint64(bin_width), hist_local)

    return (start_idx, stop_idx, hist_local, ch0_times.size, ch1_times.size)

# ---------------- driver function ----------------
def correlate_b_file_highperf(filepath,
                              channel_a=0,
                              channel_b=1,
                              chunk_ms=10.0,
                              tau_max_us=10.0,
                              tau_bin_ps=16.0,
                              n_workers=None,
                              show_progress=True):
    """
    High-performance correlator for PicoHarp .b file.

    Parameters
    ----------
    filepath : str
        Path to .b file
    channel_a, channel_b : int
        Channels to correlate (0/1 typical)
    chunk_ms : float
        chunk length in milliseconds (ms) for splitting file into parallel tasks
    tau_max_us : float
        maximum correlation delay in microseconds (us)
    tau_bin_ps : float
        correlation bin width in picoseconds
    n_workers : int
        number of parallel workers (defaults to cpu_count()-1)
    """
    filesize = os.path.getsize(filepath)
    total_records = filesize // 4
    if total_records == 0:
        raise ValueError("Empty file")

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    # Convert units to raw PicoHarp ticks:
    bin_width_ticks = int(round((tau_bin_ps * 1e-12 / BASE_RES)))
    tau_max_ticks = int(round((tau_max_us * 1e-6) / BASE_RES))
    if bin_width_ticks <= 0:
        bin_width_ticks = 1
    nbins = int((tau_max_ticks + bin_width_ticks - 1) // bin_width_ticks)

    if show_progress:
        print(f"File {filepath}: {total_records} records")
        print(f"Using {n_workers} workers, chunk_ms={chunk_ms} ms, tau_max={tau_max_us} us -> {tau_max_ticks} ticks, bin_width={bin_width_ticks} ticks ({tau_bin_ps} ps)")
        print(f"nbins = {nbins}")

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
        print(f"Found {overflow_positions.size} overflow markers in file.")

    # Now build time-based chunk boundaries.
    # We'll create chunks that try to contain approximately chunk_ms milliseconds of (unwrapped) time.
    # To do this cheaply: we compute unwrapped timestamps for the whole file incrementally via vectorized approach
    # but we avoid full extra copy by computing a compact array of 'unwrapped' times at coarse step positions.
    # Simpler pragmatic approach: compute unwrapped timestamps for the entire file (one array).
    # This uses extra memory ~8 bytes * total_records (uint64). If file is huge this may be large.
    # If memory is tight, reduce chunk_ms so chunks are smaller and choose chunk indices by record count instead.

    # Attempt to compute full unwrapped timestamps (fast & vectorized):
    if show_progress:
        print("Computing full unwrapped timestamps (may use significant RAM for very large files)...")
    timetags_raw = memmap & 0x0FFFFFFF
    # Compute overflow cumulative counts per index by using searchsorted on overflow_positions for vector of indices
    # This builds an array of size total_records (uint64). Memory usage = 8 * total_records bytes.
    # If this is too large for your system, set use_record_based_chunking = True below.
    try:
        idxs = np.arange(total_records, dtype=np.int64)
        # For each index i, number of overflows before i = searchsorted(overflow_positions, i, 'left')
        overflow_counts_prefix = np.searchsorted(overflow_positions, idxs, side='left').astype(np.uint64)
        unwrapped_all = timetags_raw.astype(np.uint64) + overflow_counts_prefix * np.uint64(T2WRAPAROUND)
        memory_ok = True
    except MemoryError:
        memory_ok = False

    if not memory_ok:
        # fallback: create chunks based on fixed record counts (no full unwrapped array)
        if show_progress:
            print("MemoryError when building unwrapped_all. Falling back to record-count-based chunking.")
        use_record_based_chunking = True
    else:
        use_record_based_chunking = False

    # Build chunk ranges (start_idx, stop_idx). We'll pick chunk boundaries by time (unwrapped timestamps).
    chunk_ranges = []
    if not use_record_based_chunking:
        # desired chunk length in ticks
        chunk_ticks = int(round((chunk_ms * 1e-3) / BASE_RES))
        if chunk_ticks <= 0:
            chunk_ticks = int(1e6)  # fall-back
        t0 = unwrapped_all[0]
        start = 0
        while start < total_records:
            # find index where unwrapped_all >= t0 + chunk_ticks
            target = t0 + chunk_ticks
            # searchsorted to find stop index
            stop = int(np.searchsorted(unwrapped_all, target, side='left'))
            if stop <= start:
                # ensure progress
                stop = min(start + 1_000_00, total_records)
            chunk_ranges.append((start, stop))
            start = stop
            if start < total_records:
                t0 = unwrapped_all[start]
        if show_progress:
            print(f"Built {len(chunk_ranges)} time-based chunks (chunk_ticks ~ {chunk_ticks}).")
    else:
        # record based chunking: choose records_per_chunk proportional to chunk_ms and approximate average rate
        approx_rate = max(1, total_records // max(1, int((unwrapped_all[-1] - unwrapped_all[0]) if 'unwrapped_all' in locals() else 1)))
        records_per_chunk = int(max(10_000, (approx_rate * (chunk_ms*1e-3))))  # very rough
        if show_progress:
            print(f"Using records_per_chunk = {records_per_chunk}")
        start = 0
        while start < total_records:
            stop = min(total_records, start + records_per_chunk)
            chunk_ranges.append((start, stop))
            start = stop

    # For each chunk, determine overlap_stop_idx = stop index extended to include any records within tau_max ticks after stop
    # We will extend up to the first index whose unwrapped time >= stop_time + tau_max_ticks
    overlap_ranges = []
    for (start, stop) in chunk_ranges:
        if not use_record_based_chunking:
            stop_time = unwrapped_all[stop-1] if stop-1 < unwrapped_all.size else unwrapped_all[-1]
            # find first index >= stop_time + tau_max_ticks
            tgt = stop_time + tau_max_ticks
            overlap_stop = int(np.searchsorted(unwrapped_all, tgt, side='right'))
            overlap_stop = min(overlap_stop, total_records)
        else:
            # conservative fixed overlap: add an extra margin of records
            overlap_margin_records = int(max(1000, tau_max_ticks*2))
            overlap_stop = min(total_records, stop + overlap_margin_records)
        overlap_ranges.append((start, stop, overlap_stop))

    # Prepare tasks
    tasks = []
    for (start, stop, overlap_stop) in overlap_ranges:
        # skip empty
        if stop <= start:
            continue
        tasks.append((filepath, total_records, start, stop, overlap_stop,
                      tau_max_ticks, bin_width_ticks, nbins, channel_a, channel_b, overflow_positions))

    if show_progress:
        print(f"Submitting {len(tasks)} tasks to {n_workers} workers...")

    # run tasks in parallel
    hist_accum = np.zeros(nbins, dtype=np.uint64)
    total_ch0 = 0
    total_ch1 = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_worker_process_chunk, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            start_idx, stop_idx, hist_local, cnt0, cnt1 = res
            hist_accum += hist_local
            total_ch0 += cnt0
            total_ch1 += cnt1
            if show_progress:
                elapsed = time.time() - t0
                frac = i / len(tasks)
                eta = elapsed / frac - elapsed if frac > 0 else 0.0
                print(f"Tasks {i}/{len(tasks)} done, elapsed {elapsed:.1f}s, ETA {eta:.1f}s", end='\r')

    if show_progress:
        print()
        print(f"Correlation finished in {time.time()-t0:.1f}s")
        print(f"Total ch0 events counted: {total_ch0}, total ch1 events available: {total_ch1}")
        print(f"Final histogram bins: {nbins}")

    # convert ticks to physical time
    tau = (np.arange(nbins) * bin_width_ticks) * BASE_RES  # seconds

    return tau, hist_accum

# ---------------- convenience plot ----------------
def plot_corr(tau_seconds, hist, title="g2 histogram"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.step(tau_seconds*1e6, hist, where='mid')  # x in microseconds
    plt.xlabel("Delay τ [µs]")
    plt.ylabel("Counts")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------- example usage ----------------
if __name__ == "__main__":
    filepath = r"C:\Users\SSPDs\Documents\Offline SSPD Data\20251116\Toptica_AOM_50kHz_300s\tests\_chunk0.b"
    tau, hist = correlate_b_file_highperf(
        filepath,
        channel_a=0,
        channel_b=1,
        chunk_ms=5e3,        # chunk size in ms
        tau_max_us=400.0,    # tau max in microseconds
        tau_bin_ps=4000.0,    # bin width in ps 
        n_workers=max(1, cpu_count()-1),
        show_progress=True
    )
    plot_corr(tau, hist, title=os.path.basename(filepath))
