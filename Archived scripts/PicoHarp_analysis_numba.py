# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 17:02:11 2025

@author: ohuber
"""

# Add these imports at top of your file
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count, shared_memory
from tqdm import tqdm

# Optional: Numba accelerated core
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# ---------- Constants ----------
rtPicoHarpT2 = int('00010203', 16)
T2WRAPAROUND = 210698240

# ---------- Numba core (if available) ----------
if NUMBA_AVAILABLE:
    @njit
    def _numba_bsearch_left(a, x):
        # simple binary search returning leftmost index where a[idx] >= x
        lo = 0
        hi = a.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # NB: avoid prange for histogram updates (race conditions). We'll do per-thread hist and sum outside if parallel used.
    @njit
    def correlate_numba_single(ch0, ch1, tau_max, tau_bin_width):
        n_bins = (tau_max + tau_bin_width - 1) // tau_bin_width
        hist = np.zeros(n_bins, dtype=np.uint64)
        n1 = ch1.shape[0]
        for i in range(ch0.shape[0]):
            t0 = ch0[i]
            j = _numba_bsearch_left(ch1, t0)
            # accumulate until dt >= tau_max
            while j < n1:
                dt = ch1[j] - t0
                if dt >= tau_max:
                    break
                bin_idx = dt // tau_bin_width
                hist[bin_idx] += 1
                j += 1
        return hist

# ---------- PTU Header Reader ----------
def read_ptu_header(input_filepath):
    with open(input_filepath, "rb") as f:
        magic = f.read(8).decode("utf-8").strip('\0')
        if magic != "PQTTTR":
            raise ValueError("Not a valid PicoQuant PTU file.")
        version = f.read(8).decode("utf-8").strip('\0')

        tags = {}
        while True:
            tagIdent = f.read(32).decode("utf-8").strip('\0')
            tagIdx = np.frombuffer(f.read(4), dtype=np.int32)[0]
            tagTyp = np.frombuffer(f.read(4), dtype=np.int32)[0]
            tagValue = f.read(8)
            if tagIdent == "Header_End":
                break
            if tagTyp in (0x10000008, 0x11000008, 0x12000008):  # int, bitset, color
                tags[tagIdent] = np.frombuffer(tagValue, dtype=np.int64)[0]
            elif tagTyp == 0x20000008:  # float
                tags[tagIdent] = np.frombuffer(tagValue, dtype=np.float64)[0]
            elif tagTyp == 0x4001FFFF:  # string
                strlen = np.frombuffer(tagValue, dtype=np.int64)[0]
                tags[tagIdent] = f.read(strlen).decode("utf-8").strip('\0')
            else:
                pass  # ignore unknown tag types
        return open(input_filepath, "rb"), tags


# ---------- Fast vectorized record extraction ----------
def extract_records_picoharp(data_words):
    """
    Decode 32-bit PicoHarp T2 records from uint32 array.

    Parameters
    ----------
    data_words : np.ndarray
        1D array of dtype=np.uint32 containing raw records.

    Returns
    -------
    special : np.ndarray
        1 for overflow, 2 for marker, 0 for normal photon.
    channels : np.ndarray
        Channel numbers (0–4 for photons, 15 for special).
    timestamps : np.ndarray
        Raw 28-bit timestamps (before overflow correction).
    """
    # Extract channel (upper 4 bits)
    channels = (data_words >> 28) & 0xF

    # Extract 28-bit time tag
    timestamps = data_words & 0x0FFFFFFF

    # Identify special records (channel == 15)
    is_special = channels == 0xF
    special = np.zeros_like(channels, dtype=np.uint8)

    # For special records, check the lower 4 bits of the timestamp
    timetag_low4 = timestamps & 0xF
    is_overflow = is_special & (timetag_low4 == 0)
    is_marker   = is_special & (timetag_low4 > 0)

    special[is_overflow] = 1
    special[is_marker] = 2

    return special, channels, timestamps


def unwrap_time_picoharp(special, channels, timestamps, show_progress=False):
    """
    Apply overflow correction vectorized (~100× faster than looping).
    """
    # Find indices of overflow events
    overflow_idx = np.where(special == 1)[0]
    if len(overflow_idx) == 0:
        return channels, timestamps.astype(np.uint64)

    if show_progress:
        print(f"Found {len(overflow_idx)} overflows")

    # Each overflow adds one wraparound period
    overflow_count = np.zeros_like(timestamps, dtype=np.uint64)
    overflow_count[overflow_idx] = 1
    overflow_count = np.cumsum(overflow_count)

    # Apply correction
    full_times = timestamps.astype(np.uint64) + overflow_count * T2WRAPAROUND
    return channels, full_times




# ---------- NumPy fallback core ----------
def correlate_numpy_single(ch0, ch1, tau_max, tau_bin_width):
    """
    Optimized numpy fallback: uses np.searchsorted and np.add.at on precomputed bin indices
    Processes each t0 in a python loop but avoids expensive np.histogram calls.
    """
    n_bins = int((tau_max + tau_bin_width - 1) // tau_bin_width)
    hist = np.zeros(n_bins, dtype=np.uint64)
    n1 = len(ch1)

    for t0 in ch0:
        # left start index
        j = np.searchsorted(ch1, t0, side='left')
        if j >= n1:
            continue
        # consider a limited window to avoid scanning too far; in practice we'd break when dt >= tau_max
        dt = ch1[j:] - t0
        # only keep dt in [0, tau_max)
        valid_mask = (dt >= 0) & (dt < tau_max)
        if not np.any(valid_mask):
            continue
        dt_valid = dt[valid_mask]
        bin_idx = (dt_valid // tau_bin_width).astype(np.int64)
        # accumulate
        # np.add.at is used to avoid creating many intermediate arrays
        np.add.at(hist, bin_idx, 1)
    return hist

# ---------- Worker wrapper (shared memory) ----------
def _open_shm_array(shm_name, shape, dtype):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, arr

def correlate_chunk_shared(args):
    """
    Worker invoked in subprocesses.
    args = (shm_name, shape, dtype_str, ch0_chunk, tau_max, tau_bin_width, use_numba)
    """
    shm_name, shape, dtype_str, ch0_chunk, tau_max, tau_bin_width, use_numba = args
    dtype = np.dtype(dtype_str)
    shm, ch1_times = _open_shm_array(shm_name, shape, dtype)
    # make sure we operate on a numpy array (view) with correct dtype
    try:
        if use_numba and NUMBA_AVAILABLE:
            # numba expects integer types (uint64)
            ch0 = ch0_chunk.astype(np.uint64, copy=False)
            ch1 = ch1_times.astype(np.uint64, copy=False)
            hist = correlate_numba_single(ch0, ch1, np.uint64(tau_max), np.uint64(tau_bin_width))
        else:
            hist = correlate_numpy_single(ch0_chunk, ch1_times, tau_max, tau_bin_width)
    finally:
        shm.close()
    return hist

# ---------- Main parallel correlation function (replacement) ----------
def correlate_picoharp_ptu_parallel_shared_fast(file_path,
                                               tau_bin_width=16,
                                               tau_max=10000,
                                               n_chunks=None,
                                               use_numba=True):
    """
    Faster version of your correlation pipeline using:
     - precomputed bin count
     - shared memory for ch1_times
     - ProcessPoolExecutor for parallelism
     - numba-accelerated worker if available (otherwise optimized numpy fallback)
    """
    inputfile, tags = read_ptu_header(file_path)
    record_type = tags["TTResultFormat_TTTRRecType"]
    if record_type != rtPicoHarpT2:
        raise TypeError("File is not PicoHarp T2")

    base_res = tags["MeasDesc_GlobalResolution"]
    # ---------- Read all data (same as before) ----------
    # reposition and read similarly as your current implementation
    inputfile.seek(0, 2)
    file_size = inputfile.tell()
    inputfile.seek(0)
    inputfile.read(8 + 8)  # skip magic/version
    # read body in chunks
    chunks = []
    pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading .ptu")
    while inputfile.tell() < file_size:
        chunk = inputfile.read(4 * 10**6)
        if not chunk:
            break
        chunks.append(np.frombuffer(chunk, dtype=np.uint32))
        pbar.update(len(chunk))
    pbar.close()
    inputfile.close()

    data_words = np.concatenate(chunks)
    del chunks

    special, channels, timestamps = extract_records_picoharp(data_words)
    del data_words
    channels, timestamps = unwrap_time_picoharp(special, channels, timestamps, show_progress=True)

    ch0_times = timestamps[channels == 0].astype(np.uint64)
    ch1_times = timestamps[channels == 1].astype(np.uint64)
    del special, channels, timestamps

    if n_chunks is None:
        n_chunks = max(1, cpu_count() - 1)

    # ---------- Shared memory for ch1_times ----------
    # create a shared memory block and copy ch1_times there once
    shm = shared_memory.SharedMemory(create=True, size=ch1_times.nbytes)
    shm_arr = np.ndarray(ch1_times.shape, dtype=ch1_times.dtype, buffer=shm.buf)
    np.copyto(shm_arr, ch1_times)

    ch0_chunks = np.array_split(ch0_times, n_chunks)

    # prepare args for each worker (send dtype as string to avoid pickling issues)
    args_list = []
    for chunk in ch0_chunks:
        args_list.append((shm.name, ch1_times.shape, ch1_times.dtype.str, chunk, np.uint64(tau_max), np.uint64(tau_bin_width), bool(use_numba)))

    print(f"Start correlations with {n_chunks} workers (numba_available={NUMBA_AVAILABLE}, use_numba={use_numba})...")
    start_time = time.time()
    n_bins = int((tau_max + tau_bin_width - 1) // tau_bin_width)
    hist = np.zeros(n_bins, dtype=np.uint64)

    # use ProcessPoolExecutor for nicer progress and robust scheduling
    with ProcessPoolExecutor(max_workers=n_chunks) as ex:
        futures = [ex.submit(correlate_chunk_shared, args) for args in args_list]
        for i, fut in enumerate(as_completed(futures), 1):
            chunk_hist = fut.result()
            hist += chunk_hist
            elapsed = time.time() - start_time
            frac = i / len(futures)
            if frac > 0:
                eta = elapsed / frac - elapsed
            else:
                eta = 0.0
            print(f"Progress: {frac*100:.1f}%, ETA: {eta:.1f}s", end='\r')

    print("\nCorrelation done. It took %.1fs" % (time.time() - start_time))
    print("\nThe total number of photons was %.3e." % (len(ch0_times) + len(ch1_times)))

    # cleanup
    shm.close()
    shm.unlink()

    tau = np.arange(0, tau_max, tau_bin_width) * base_res
    print('Base resolution is: %.2f ps.' % (base_res * 1e12))
    return tau, hist



# ---------- Example usage ----------
if __name__ == "__main__":
    file_path = r"Z:\Labs\G13\SSPDs\Olivier & Kilian MoTe2 Correlations\20251112\Sacher_AOM_300s_50kHz.ptu"
    tau, g2_hist = correlate_picoharp_ptu_parallel_shared_fast(file_path, tau_bin_width=int(4e3/4), tau_max=int(5000e6/4), n_chunks=None)
    
    np.savetxt(r'Z:\Labs\G13\SSPDs\Olivier & Kilian MoTe2 Correlations\20251112\Sacher_AOM_300s_50kHz_hist.txt',
               np.column_stack((tau, g2_hist)),
               header='Tau \t Histogram',
               delimiter='\t')
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(tau*1e6, g2_hist, lw=1.5)
    plt.xlabel("Time delay τ [us]")
    plt.ylabel("Coincidences")
    plt.title("g² histogram (PicoHarp T2)")
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    