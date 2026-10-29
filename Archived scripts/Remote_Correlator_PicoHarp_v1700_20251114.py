import numpy as np
from time import gmtime, strftime
import time
import numba

def curtime():
    return strftime("UTC %Y-%m-%d %H:%M:%S", gmtime())
import datetime
import os
import struct
import matplotlib as mpl
#mpl.use("Agg") # non-interactive backend to be able to save plots in a thread that is not the main thread
import matplotlib.pyplot as plt
import pickle
#from scipy.signal import find_peaks
from multiprocessing import Process
import sys, multiprocessing as mp
mp.set_start_method("spawn", force=True)
mp.set_executable(sys.executable)
#print(f"[analysis server] sys.executable = {sys.executable}", flush=True)

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import glob
import threading
import hashlib

#Picoquant Tag Types
tyEmpty8 = struct.unpack(">i", bytes.fromhex("FFFF0008"))[0]
tyBool8 = struct.unpack(">i", bytes.fromhex("00000008"))[0]
tyInt8 = struct.unpack(">i", bytes.fromhex("10000008"))[0]
tyBitSet64 = struct.unpack(">i", bytes.fromhex("11000008"))[0]
tyColor8 = struct.unpack(">i", bytes.fromhex("12000008"))[0]
tyFloat8 = struct.unpack(">i", bytes.fromhex("20000008"))[0]
tyTDateTime = struct.unpack(">i", bytes.fromhex("21000008"))[0]
tyFloat8Array = struct.unpack(">i", bytes.fromhex("2001FFFF"))[0]
tyAnsiString = struct.unpack(">i", bytes.fromhex("4001FFFF"))[0]
tyWideString = struct.unpack(">i", bytes.fromhex("4002FFFF"))[0]
tyBinaryBlob = struct.unpack(">i", bytes.fromhex("FFFFFFFF"))[0]

#Picoquant Record types
rtPicoHarpT3 = struct.unpack(">i", bytes.fromhex('00010303'))[0]
rtPicoHarpT2 = struct.unpack(">i", bytes.fromhex('00010203'))[0]
rtHydraHarpT3 = struct.unpack(">i", bytes.fromhex('00010304'))[0]
rtHydraHarpT2 = struct.unpack(">i", bytes.fromhex('00010204'))[0]
rtHydraHarp2T3 = struct.unpack(">i", bytes.fromhex('01010304'))[0]
rtHydraHarp2T2 = struct.unpack(">i", bytes.fromhex('01010204'))[0]
rtTimeHarp260NT3 = struct.unpack(">i", bytes.fromhex('00010305'))[0]
rtTimeHarp260NT2 = struct.unpack(">i", bytes.fromhex('00010205'))[0]
rtTimeHarp260PT3 = struct.unpack(">i", bytes.fromhex('00010306'))[0]
rtTimeHarp260PT2 = struct.unpack(">i", bytes.fromhex('00010206'))[0]
rtMultiHarpNT3 = struct.unpack(">i", bytes.fromhex('00010307'))[0]
rtMultiHarpNT2 = struct.unpack(">i", bytes.fromhex('00010207'))[0]

def read_ptu_header(input_filepath):
    """Read PTU header and return tag dictionary plus data-offset."""
    inputfile = open(input_filepath, "rb")
    magic = inputfile.read(8).decode("utf-8").strip('\0')
    if magic != "PQTTTR":
        print("ERROR: Magic invalid, this is not a PTU file.")
        inputfile.close()
        raise AssertionError
    version = inputfile.read(8).decode("utf-8").strip('\0')
    tagDataList = []
    while True:
        tagIdent = inputfile.read(32).decode("utf-8").strip('\0')
        tagIdx = struct.unpack("<i", inputfile.read(4))[0]
        tagTyp = struct.unpack("<i", inputfile.read(4))[0]
        if tagIdx > -1:
            evalName = tagIdent + '(' + str(tagIdx) + ')'
        else:
            evalName = tagIdent
        if tagTyp == tyEmpty8:
            inputfile.read(8)
            tagDataList.append((evalName, "<empty Tag>"))
        elif tagTyp == tyBool8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            if tagInt == 0:
                tagDataList.append((evalName, "False"))
            else:
                tagDataList.append((evalName, "True"))
        elif tagTyp == tyInt8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyBitSet64:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyColor8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyFloat8:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            tagDataList.append((evalName, tagFloat))
        elif tagTyp == tyFloat8Array:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyTDateTime:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            tagTime = int((tagFloat - 25569) * 86400)
            tagTime = time.gmtime(tagTime)
            tagDataList.append((evalName, tagTime))
        elif tagTyp == tyAnsiString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode("utf-8").strip("\0")
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyWideString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode(
                        "utf-16le", errors="ignore").strip("\0")
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyBinaryBlob:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        else:
            print("ERROR: Unknown tag type")
            print(tagTyp)
            raise TypeError("ERROR: Unknown tag type")
        if tagIdent == "Header_End":
            break
    tagNames = [tagDataList[i][0] for i in range(0, len(tagDataList))]
    tagValues = [tagDataList[i][1] for i in range(0, len(tagDataList))]
    tags = dict(zip(tagNames, tagValues))
    data_offset = inputfile.tell()
    inputfile.close()
    return tags, data_offset


# PicoHarp 300 specific decoding -------------------------------------------------
SUPPORTED_PICO_TYPES = {
    rtPicoHarpT2: "PicoHarp 300 T2",
    rtPicoHarpT3: "PicoHarp 300 T3",
}

def _decode_pico_t2(words):
    channels = ((words >> 28) & 0xF).astype(np.uint8, copy=False)
    timetags = (words & 0x0FFFFFFF).astype(np.uint32, copy=False)
    return channels, timetags

def _decode_pico_t3(words):
    channels = ((words >> 28) & 0xF).astype(np.uint8, copy=False)
    dtime = ((words >> 16) & 0x0FFF).astype(np.uint16, copy=False)
    nsync = (words & 0xFFFF).astype(np.uint32, copy=False)
    return channels, dtime, nsync

def _unwrap_pico_t2(channels, timetags):
    """Handle PicoHarp 300 T2 overflows/markers."""
    T2WRAPAROUND = 210698240
    oflcorrection = 0
    detector_channels = np.empty_like(channels)
    detector_timestamps = np.empty(channels.size, dtype=np.int64)
    det_idx = 0
    for idx in range(channels.size):
        channel = int(channels[idx])
        timetag = int(timetags[idx])
        if channel == 0xF:
            markers = timetag & 0xF
            if markers == 0:
                oflcorrection += T2WRAPAROUND
            # Marker events are ignored for correlation
            continue
        detector_channels[det_idx] = channel
        detector_timestamps[det_idx] = oflcorrection + timetag
        det_idx += 1
    return detector_channels[:det_idx], detector_timestamps[:det_idx]

def _unwrap_pico_t3(channels, dtime, nsync):
    """Handle PicoHarp 300 T3 overflows/markers."""
    T3WRAPAROUND = 65536
    oflcorrection = 0
    detector_channels = np.empty_like(channels)
    detector_timestamps = np.empty(channels.size, dtype=np.int64)
    det_idx = 0
    for idx in range(channels.size):
        channel = int(channels[idx])
        if channel == 0xF:
            if int(dtime[idx]) == 0:
                oflcorrection += T3WRAPAROUND
            # Marker records are ignored
            continue
        truensync = oflcorrection + int(nsync[idx])
        detector_channels[det_idx] = channel
        detector_timestamps[det_idx] = truensync
        det_idx += 1
    return detector_channels[:det_idx], detector_timestamps[:det_idx]

def convert_pico_records(words, record_type):
    """Decode a chunk of PicoHarp 300 records into detector channels/timestamps."""
    if record_type == rtPicoHarpT2:
        channels, timetags = _decode_pico_t2(words)
        return _unwrap_pico_t2(channels, timetags)
    if record_type == rtPicoHarpT3:
        channels, dtime, nsync = _decode_pico_t3(words)
        return _unwrap_pico_t3(channels, dtime, nsync)
    raise ValueError(f"Unsupported PicoHarp record type: {record_type}")

def timestamps_select_channels(timestamps, detector, mode):
    if mode == "X01":
        ts1 = timestamps[detector == 0] # reference (start)
        ts2 = timestamps[detector == 1] # correlated (stop)
    elif mode == "X10":
        ts1 = timestamps[detector == 1]
        ts2 = timestamps[detector == 0]
    elif mode == "A0":
        ts1 = ts2 = timestamps[detector == 0]
    elif mode == "A1":
        ts1 = ts2 = timestamps[detector == 1]
    elif mode == "all":
        ts1  = ts2 = timestamps
    else:
        raise ValueError(f"Unknown mode {mode}")
    return ts1, ts2

#@numba.jit(nopython=True, nogil=False, parallel=False)
def fast_correlate_mode_symmetric_fast(ts1, ts2, tmax, hist, binning, block_size= 2000):

    # Estimate for a safe and efficient block size for 16 GB RAM with 7 parallel cores.
    max_ram_per_worker = 0.8e9 # 0.8 GB per worker
    chunksize = max(len(ts1), len(ts2))
    block_size = int(max_ram_per_worker / (8* chunksize))

    N1 = ts1.size
    bins = np.arange(-tmax, tmax+0.1, binning)
    nbins = len(bins)

    for i0 in range(0, N1, block_size):
        i1 = min(N1, i0 + block_size)
        block = ts1[i0:i1]
        diffs = block[:, None] - ts2[None, :]
        mask = (np.abs(diffs) < tmax) & (diffs != 0)
        sel = diffs[mask]
        h, _ = np.histogram(sel, range = (-tmax, tmax), bins=bins)
        hist += h.astype(hist.dtype, copy=False)

        
def _process_chunk_range_pico_b(args):
    """Worker for PicoHarp binary .b chunks."""
    (start_chunk, end_chunk, filepath, total_records, chunksize,
     record_type, tau_min, tau_max, tau_binning, mode, symmetric,
     progress_step, progress_counter) = args

    if not symmetric:
        n_bins = (tau_max - tau_min) // tau_binning
    else:
        n_bins = (2 * tau_max) // tau_binning

    hist_local = np.zeros(n_bins, dtype=np.uint64)
    tpc_local = np.zeros(end_chunk - start_chunk, dtype=np.uint64)
    processed_since_report = 0

    mapped = np.memmap(filepath, mode="r", dtype=np.uint32, offset=0, shape=(total_records,))

    for j, chunk_idx in enumerate(range(start_chunk, end_chunk)):
        start = chunk_idx * chunksize
        stop = min(start + chunksize, total_records)
        if start >= stop:
            break

        chunk_words = mapped[start:stop]
        if chunk_words.size == 0:
            tpc_local[j] = 0
        else:
            channels, timestamps = convert_pico_records(chunk_words, record_type)
            if timestamps.size >= 2:
                hbuf = np.zeros(n_bins, dtype=np.uint32)
                if symmetric:
                    t1, t2 = timestamps_select_channels(timestamps, channels, mode)
                    fast_correlate_mode_symmetric_fast(t1, t2, tau_max, hbuf, tau_binning)
                else:
                    fast_correlate_mode(timestamps, channels, hbuf, tau_binning, tau_min, tau_max, mode)
                hist_local += hbuf
                tpc_local[j] = np.uint64(timestamps[-1] - timestamps[0])
            else:
                tpc_local[j] = np.uint64(0)

        processed_since_report += 1
        if progress_counter is not None and processed_since_report >= progress_step:
            try:
                progress_counter.value += processed_since_report
            except Exception:
                pass
            processed_since_report = 0

    if progress_counter is not None and processed_since_report:
        try:
            progress_counter.value += processed_since_report
        except Exception:
            pass

    del mapped
    return (start_chunk, end_chunk, hist_local, tpc_local)

def correlate_pico_b_parallel(
    input_filepath,
    record_type=rtPicoHarpT2,
    chunksize=2**17,
    chunk_bins=None,
    tau_min=0,
    tau_max=150004,
    tau_binning=44,
    debug=False,
    log=None,
    max_workers=os.cpu_count()-1,
    progress_step=10,
    chunks_per_task=100,
    mode="X01",
    chunk_range=None,
    symmetric=False,
):
    """Parallel correlation over PicoHarp binary .b files (no header)."""

    chunksize = int(chunksize)
    tau_min = int(tau_min)
    tau_max = int(tau_max)
    tau_binning = int(tau_binning)
    progress_step = max(1, int(progress_step))

    filesize = os.path.getsize(input_filepath)
    total_records = filesize // 4
    if total_records == 0:
        raise ValueError("Empty .b file.")

    n_chunks = math.ceil(total_records / chunksize)
    if not symmetric:
        n_bins = (tau_max - tau_min) // tau_binning
    else:
        n_bins = (2 * tau_max) // tau_binning

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)

    if chunks_per_task is None:
        chunks_per_task = int(n_chunks / max_workers / 10)
    chunks_per_task = max(1, int(chunks_per_task))

    if chunk_range is None:
        chunk_range = (0, n_chunks)

    if log is not None:
        log(f"[ANALYSIS - {curtime()}] Parallel start: {chunk_range[1]-chunk_range[0]} Pico .b chunks ({n_chunks} total), {max_workers} workers\n")

    print(f"[ANALYSIS - {curtime()}] Processing chunk range ", chunk_range)
    ranges = []
    for start in range(*chunk_range, chunks_per_task):
        end = min(start + chunks_per_task, chunk_range[1])
        ranges.append((start, end))

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    progress_counter = manager.Value('i', 0)
    stop_flag = threading.Event()

    def monitor_progress():
        last_reported = 0
        start_ts = time.time()
        total = chunk_range[1] - chunk_range[0]
        while not stop_flag.is_set():
            try:
                done = int(progress_counter.value)
            except Exception:
                done = last_reported
            if done >= last_reported + progress_step or done >= total:
                last_reported = done
                pct = 100.0 * min(done, total) / total if total else 100.0
                elapsed = (time.time() - start_ts) / 60.0
                msg = f"\r[ANALYSIS - {curtime()}] Progress: {min(done,total)}/{total} ({pct:.1f}%) — {elapsed:.1f} min"
                if log is not None:
                    log(msg)
                else:
                    print(msg, end="")
                if done >= total:
                    break
            time.sleep(0.5)
        if log is not None:
            log("\n")
        else:
            print()

    monitor = threading.Thread(target=monitor_progress, daemon=True)
    monitor.start()

    hist64_accum = np.zeros(n_bins, dtype=np.uint64)
    time_per_chunk = np.zeros(n_chunks, dtype=np.uint64)

    tasks = [
        (start, end, input_filepath, total_records, chunksize,
         record_type, tau_min, tau_max, tau_binning, mode, symmetric,
         progress_step, progress_counter)
        for (start, end) in ranges
    ]

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = [pool.submit(_process_chunk_range_pico_b, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            start, end, hist_local, tpc_local = res
            hist64_accum += hist_local
            span = end - start
            time_per_chunk[start:end] = tpc_local[:span]

    stop_flag.set()
    monitor.join()
    try:
        manager.shutdown()
    except Exception:
        pass

    hist = np.asarray(np.minimum(hist64_accum, np.uint64(np.iinfo(np.uint32).max)), dtype=np.uint32)

    if debug:
        return hist, time_per_chunk, n_chunks
    return hist, time_per_chunk

def _process_chunk_range_pico(args):
    """Worker for PicoHarp PTU chunks."""
    (start_chunk, end_chunk, filepath, data_offset, num_records, chunksize,
     record_type, tau_min, tau_max, tau_binning, mode, symmetric,
     progress_step, progress_counter) = args

    if not symmetric:
        n_bins = (tau_max - tau_min) // tau_binning
    else:
        n_bins = (2 * tau_max) // tau_binning

    hist_local = np.zeros(n_bins, dtype=np.uint64)
    tpc_local = np.zeros(end_chunk - start_chunk, dtype=np.uint64)
    processed_since_report = 0

    range_start_record = start_chunk * chunksize
    range_end_record = min(num_records, end_chunk * chunksize)
    if range_start_record >= range_end_record:
        return (start_chunk, end_chunk, hist_local, tpc_local)

    with open(filepath, "rb") as handle:
        handle.seek(data_offset + range_start_record * 4)
        dwords = np.fromfile(handle, dtype=np.uint32, count=range_end_record - range_start_record)

    for j, chunk_idx in enumerate(range(start_chunk, end_chunk)):
        chunk_start = chunk_idx * chunksize
        chunk_end = min(num_records, chunk_start + chunksize)
        if chunk_start >= range_end_record:
            break
        rel_start = chunk_start - range_start_record
        rel_end = chunk_end - range_start_record
        chunk_words = dwords[rel_start:rel_end]
        if chunk_words.size == 0:
            tpc_local[j] = 0
        else:
            channels, timestamps = convert_pico_records(chunk_words, record_type)
            if timestamps.size >= 2:
                hbuf = np.zeros(n_bins, dtype=np.uint32)
                if symmetric:
                    t1, t2 = timestamps_select_channels(timestamps, channels, mode)
                    fast_correlate_mode_symmetric_fast(t1, t2, tau_max, hbuf, tau_binning)
                else:
                    fast_correlate_mode(timestamps, channels, hbuf, tau_binning, tau_min, tau_max, mode)
                hist_local += hbuf
                tpc_local[j] = np.uint64(timestamps[-1] - timestamps[0])
            else:
                tpc_local[j] = np.uint64(0)

        processed_since_report += 1
        if progress_counter is not None and processed_since_report >= progress_step:
            try:
                progress_counter.value += processed_since_report
            except Exception:
                pass
            processed_since_report = 0

    if progress_counter is not None and processed_since_report:
        try:
            progress_counter.value += processed_since_report
        except Exception:
            pass

    return (start_chunk, end_chunk, hist_local, tpc_local)

def correlate_pico_ptu_parallel(
    input_filepath,
    data_offset,
    num_records,
    record_type,
    chunksize=2**17,
    chunk_bins=None,
    tau_min=0,
    tau_max=150004,
    tau_binning=44,
    debug=False,
    log=None,
    max_workers=os.cpu_count()-1,
    progress_step=10,
    chunks_per_task=100,
    mode="X01",
    chunk_range=None,
    symmetric=False,
):
    """Parallel correlation over PicoHarp PTU files."""

    chunksize = int(chunksize)
    tau_min = int(tau_min)
    tau_max = int(tau_max)
    tau_binning = int(tau_binning)
    progress_step = max(1, int(progress_step))

    if num_records <= 0:
        raise ValueError("PTU file does not contain any records.")

    n_chunks = math.ceil(num_records / chunksize)
    if not symmetric:
        n_bins = (tau_max - tau_min) // tau_binning
    else:
        n_bins = (2 * tau_max) // tau_binning

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)

    if chunks_per_task is None:
        chunks_per_task = int(n_chunks / max_workers / 10)
    chunks_per_task = max(1, int(chunks_per_task))

    if chunk_range is None:
        chunk_range = (0, n_chunks)

    if log is not None:
        log(f"[ANALYSIS - {curtime()}] Parallel start: {chunk_range[1]-chunk_range[0]} chunks ({n_chunks} total), {max_workers} workers\n")

    print(f"[ANALYSIS - {curtime()}] Processing chunk range ", chunk_range)
    ranges = []
    for start in range(*chunk_range, chunks_per_task):
        end = min(start + chunks_per_task, chunk_range[1])
        ranges.append((start, end))

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    progress_counter = manager.Value('i', 0)
    stop_flag = threading.Event()

    def monitor_progress():
        last_reported = 0
        start_ts = time.time()
        while not stop_flag.is_set():
            try:
                done = int(progress_counter.value)
            except Exception:
                done = last_reported
            total = chunk_range[1] - chunk_range[0]
            if done >= last_reported + progress_step or done >= total:
                last_reported = done
                pct = 100.0 * min(done, total) / total if total else 100.0
                elapsed = (time.time() - start_ts) / 60.0
                msg = f"\r[ANALYSIS - {curtime()}] Progress: {min(done,total)}/{total} ({pct:.1f}%) — {elapsed:.1f} min"
                if log is not None:
                    log(msg)
                else:
                    print(msg, end="")
                if done >= total:
                    break
            time.sleep(0.5)
        if log is not None:
            log("\n")
        else:
            print()

    monitor = threading.Thread(target=monitor_progress, daemon=True)
    monitor.start()

    hist64_accum = np.zeros(n_bins, dtype=np.uint64)
    time_per_chunk = np.zeros(n_chunks, dtype=np.uint64)

    tasks = [
        (start, end, input_filepath, data_offset, num_records, chunksize,
         record_type, tau_min, tau_max, tau_binning, mode, symmetric,
         progress_step, progress_counter)
        for (start, end) in ranges
    ]

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = [pool.submit(_process_chunk_range_pico, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            start, end, hist_local, tpc_local = res
            hist64_accum += hist_local
            span = end - start
            time_per_chunk[start:end] = tpc_local[:span]

    stop_flag.set()
    monitor.join()
    try:
        manager.shutdown()
    except Exception:
        pass

    hist = np.asarray(np.minimum(hist64_accum, np.uint64(np.iinfo(np.uint32).max)), dtype=np.uint32)

    if debug:
        return hist, time_per_chunk, n_chunks
    return hist, time_per_chunk


def fast_correlate_mode(timestamps, detector, histogram, binning, tmin, tmax, mode = 'X01'):
    """Calculate un-normalized g2(tau) correlation function from timestamps.

    Time differences tau are
    binned into histogram array for tmin <= tau <= tmax. Returns None
    but increments (!) argument histogram with coincidences.

    Inspired by Picoharp_T2Analysis_parallel.py script written by Thomas Fink.

    Parameters
    ----------
    timestamps : array of uint32
        Contains timestamps, typically in units of some base resolution. But
        can have any units, binning is relative to the units in timestamps.
    detector : array of integer type
        Array of same length as timestamps, containing channel indices
        indicating which detector (0 or 1) each timestamp belongs to.
        are discarded.
    histogram : array of uint32
        Array to save histogram into, must have compatible length (...)
    binning : integer
        Histogram bin width, relative to the units of the timestamps.
        Example: binning=4 means that time differences 0,1,2,3 are summed
                 into the same bin.
    tmin : integer
        Minimal time delay tau the histogram should contain.
        Must satisfy tmin >= 0, because only waiting times after
        detection at channel 0 are considered. Inclusive.
    tmax : integer
        Maximum time delay tau the histogram should contain. Exclusive.
    mode : str
        Mode of correlation, 'X01' is default, meaning photon delay in channel 2 conditioned on the detection of photon in ch. 1
    """

    start_chan, stop_chan = _mode_to_channels(mode)
    if mode == "all":
        is_start = np.ones_like(detector, dtype=np.uint8)
        is_stop = np.ones_like(detector, dtype=np.uint8)
    else:
        is_start = (detector == start_chan).astype(np.uint8)
        is_stop = (detector == stop_chan).astype(np.uint8)
    fast_correlate_mode_numba(
        timestamps.astype(np.int64, copy=False),
        is_start,
        is_stop,
        histogram,
        int(binning),
        int(tmin),
        int(tmax),
        mode == "all"
    )

def _mode_to_channels(mode):
    if mode == "X01":
        return 0, 1
    if mode == "X10":
        return 1, 0
    if mode == "A0":
        return 0, 0
    if mode == "A1":
        return 1, 1
    if mode == "all":
        return -1, -1
    raise ValueError(f"Unknown mode '{mode}'")

@numba.njit(cache=True)
def fast_correlate_mode_numba(timestamps, is_start, is_stop, histogram, binning, tmin, tmax, any_channel):
    num_events = timestamps.shape[0]
    bin_offset = tmin // binning
    for photon1idx in range(num_events):
        if not any_channel and not bool(is_start[photon1idx]):
            continue
        photon2idx = photon1idx + 1
        while photon2idx < num_events and (timestamps[photon2idx] - timestamps[photon1idx] < tmin):
            photon2idx += 1
        if photon2idx >= num_events:
            break
        while photon2idx < num_events:
            dt = timestamps[photon2idx] - timestamps[photon1idx]
            if dt >= tmax:
                break
            if any_channel or bool(is_stop[photon2idx]):
                binidx = dt // binning - bin_offset
                if 0 <= binidx < histogram.shape[0]:
                    histogram[binidx] += 1
            photon2idx += 1
        
def correlate_timestamps_Kilian(params, parallel = False):
    """
    For remote analysis, returns histogram and time per chunk (reading for the average count rate during a chunk).Parameters
    ----------

    params: Dictionary with the following elements: 


    input_filepath : string
        Path to input file (.b from run_parallel method in hydraharp_srv.py).
    data_path : 
    chunksize : integer, optional
        The data from file is read and processed in chunks of this size.
        The default is 2**11 = 2048.   -- the longer the chunk size the more costly the correlation, but the larger the longest time delay that can be checked
    chunk_bins ...
        ...
    tau_min: integer, optional
        Lowest time delay in histogram in units of
        base resolution / tau_binning.
        The default is 0.
    tau_max: integer, optional
        Highest time delay in histogram in units of
        base resolution / tau_binning.
        Example: tau_max = 15000 = 60000 // 4 corresponds to 60 ns for
                tau_binning = 4.
    tau_binning: integer, optional
        The histogram bin width in multiples of the timestamp base resoltion.
        Example: If base resolution is 4 ps, then tau_binning = 4
                leads to bins of size 16 ps.
    debug : Bool, enable more status prints during run.
    log : Log object that exists in analysis_srv.py and writes to the remote output
    mode: str, default "X01"
        The mode in which correlations are computed, can be X01 or X10 (the two cross-correlation modes), or A0/A1, 
        the two autocorrelation modes, or 'all' in which case it doesn't matter where the photons arrive
    Returns
    -------
    histograms : TYPE
        DESCRIPTION.
    time_per_chunk : TYPE
        DESCRIPTION.

    Notes
    -----
    """


    # Define default parameters
    dp = {  # default parameters
        'input_file': None,  # the user must supply this one
        'data_path': None,  # the user may want to supply extra info
        'chunksize': 2**10,  # 1024
        'chunk_bins': None,  # All data along physical time axis is included
        'tau_min': 0,  #
        'tau_max': 15000,  # in units of pico base resolution
        'tau_binning': 4,  # into 4 * 4ps = 16 ps bins
        'debug': False,  #
        'global_resolution': 1e-12,  # fixed for hydra
        'log' : None,
        'mode' : "X01",
        'chunk_range': None,
        'symmetric':False,
        'pico_record_type': rtPicoHarpT2
        }
    dp.update(params)  # update user supplied params

    print(f"[ANALYSIS - {curtime()}] Starting correlate_timestamps")


    # Analyze the timestamps and form correlation histograms
    start_time = time.time()

    input_file = dp['input_file']
    print(f'[ANALYSIS - {curtime()}] Correlate timestampfile: {input_file}')
    file_ext = os.path.splitext(input_file)[1].lower()

    dp['tau_max'] = int(dp['tau_max']/4) # if the base resolution is 4ps, then this should make sure we can pass to the function the true time in ps
    dp['tau_min'] = int(dp['tau_min']/4) # if the base resolution is 4ps, then this should make sure we can pass to the function the true time in ps
    dp['tau_binning'] = int(dp['tau_binning']/4) 
    
    

    if file_ext == ".ptu":
        tags, data_offset = read_ptu_header(input_file)
        record_type = tags.get("TTResultFormat_TTTRRecType")
        if record_type not in SUPPORTED_PICO_TYPES:
            raise ValueError(f"Record type {record_type} is not supported for PicoHarp PTU decoding.")
        num_records = int(tags.get("TTResult_NumberOfRecords", 0))
        print(f"[ANALYSIS - {curtime()}] Detected {SUPPORTED_PICO_TYPES[record_type]} with {num_records} records.")

        if parallel:
            out = correlate_pico_ptu_parallel(
                input_filepath=dp['input_file'],
                data_offset=data_offset,
                num_records=num_records,
                record_type=record_type,
                chunksize=dp['chunksize'],
                chunk_bins=dp['chunk_bins'],
                tau_min=dp['tau_min'],
                tau_max=dp['tau_max'],
                tau_binning=dp['tau_binning'],
                debug=dp['debug'],
                log=dp['log'],
                mode = dp['mode'],
                chunk_range = dp['chunk_range'],
                symmetric = dp['symmetric']
            )
        else:
            raise "SHOULD ONLY USE PARALLEL MODE!"
    elif file_ext == ".b":
        record_type = dp.get('pico_record_type', rtPicoHarpT2)
        print(f"[ANALYSIS - {curtime()}] Detected PicoHarp binary .b file (record type {record_type}).")
        if parallel:
            out = correlate_pico_b_parallel(
                input_filepath=dp['input_file'],
                record_type=record_type,
                chunksize=dp['chunksize'],
                chunk_bins=dp['chunk_bins'],
                tau_min=dp['tau_min'],
                tau_max=dp['tau_max'],
                tau_binning=dp['tau_binning'],
                debug=dp['debug'],
                log=dp['log'],
                mode = dp['mode'],
                chunk_range = dp['chunk_range'],
                symmetric = dp['symmetric']
            )
        else:
            raise "SHOULD ONLY USE PARALLEL MODE!"
    else:
        raise ValueError(f"Unsupported file extension '{file_ext}'. Expected .ptu or .b.")
    runtime = ((time.time()-start_time)/60)
    print(f'[ANALYSIS - {curtime()}] Correlated data in {runtime:.2f} min')


    h,c = out
    return h,c # histogram, time per chunk

    


def get_countrates_chunk(filepath, chunksize=2**11, chunk_range=None,
                     log = None):
                        
    """ only extracts the count rate per chunk, for long-time-scale correlations 
    or a quick overview of measurement stability
    
    Parameters
    ----------
    filepath : string
        Path to input file (file format .ptu from Picoharp GUI
                            or .b from correlation_measurement.py).
    chunksize : integer, optional
        The data from file is read and processed in chunks of this size.
        The default is 2**11 = 2048.   -- the longer the chunk size the more costly the correlation, but the larger the longest time delay that can be checked
    chunk_range ...
        ...
    log: A log function (!) that prints on the client side.
    
    Returns
    -------
    time_per_chunk : array

    Notes
    """

    # Open file and read header metainformation
    n_chunks_total = 0
    folder = os.path.dirname(filepath)
    # Slow brute force solution: Go through the whole file and find length
    with open(filepath, "rb") as inputfile:
        try:
            while True:
                data_dwords = np.frombuffer(inputfile.read(4*chunksize),
                                            dtype=np.uint32,
                                            count=chunksize)
                n_chunks_total += 1
        except ValueError:
            pass
    if chunk_range is None:
        chunk_range = np.zeros(2)
        chunk_range[0] = 0
        chunk_range[1] = n_chunks_total
    inputfile = open(filepath, "rb")  # and reopen at beginning
    
    n_chunks_slice = min(int(chunk_range[1]-chunk_range[0]), n_chunks_total)
    # Inversely proportional to current countrate, quantity for post-selection
    time_per_chunk = np.zeros(n_chunks_slice, dtype=np.uint64)  
        
        
        
    # Iterate through timestamp data in chunks (chunksize)
    for j in range(n_chunks_total):
         # Load binary data into array of size chunksize (this is very fast)
        data_dwords = np.frombuffer(inputfile.read(4*chunksize),
                                    dtype=np.uint32, count=chunksize)
        
        if j>= chunk_range[0] and j<chunk_range[1]:
            if log is not None and j%1e2 == 0:
                msg = f"\r[ANALYSIS] Progress counting chunks: {j/n_chunks_slice*100:.3f}% (chunks {j:d}/{n_chunks_slice:d})"
                log(msg)
                
                
           
            # Extract channel and timestamp information (discarding markers)
            channels, timestamps = convert_pico_records(data_dwords, rtPicoHarpT2)
            time_per_chunk[int(j-chunk_range[0])] = timestamps[-1] - timestamps[0]  # base resoltution
        if j>chunk_range[1]:
            break
    inputfile.close()
    if log is not None:
        log("\n")
    return time_per_chunk
    
    