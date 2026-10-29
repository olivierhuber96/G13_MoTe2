import os
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, shared_memory

# ---------------- PicoHarp settings -----------------
T2WRAPAROUND = 210698240  # 28-bit rollover for T2
BASE_RES = 4e-12          # Base resolution in seconds (4 ps typical)

# ---------------- Record extraction -----------------
def extract_records_picoharp(data_words):
    """
    Decode 32-bit PicoHarp T2 records from uint32 array.
    """
    channels = (data_words >> 28) & 0xF
    timestamps = data_words & 0x0FFFFFFF
    is_special = channels == 0xF
    special = np.zeros_like(channels, dtype=np.uint8)
    timetag_low4 = timestamps & 0xF
    is_overflow = is_special & (timetag_low4 == 0)
    is_marker = is_special & (timetag_low4 > 0)
    special[is_overflow] = 1
    special[is_marker] = 2
    return special, channels, timestamps

def unwrap_time_picoharp(special, channels, timestamps, show_progress=False):
    """
    Apply overflow correction vectorized (~100× faster than looping).
    """
    overflow_idx = np.where(special == 1)[0]
    if len(overflow_idx) == 0:
        return channels, timestamps.astype(np.uint64)
    if show_progress:
        print(f"Found {len(overflow_idx)} overflows")
    overflow_count = np.zeros_like(timestamps, dtype=np.uint64)
    overflow_count[overflow_idx] = 1
    overflow_count = np.cumsum(overflow_count)
    full_times = timestamps.astype(np.uint64) + overflow_count * T2WRAPAROUND
    return channels, full_times

# ---------------- Binary file reading -----------------
def read_picoharp_b_file(filepath, show_progress=False):
    """
    Read .b binary file and return channels and unwrapped timestamps
    """
    filesize = os.path.getsize(filepath)
    if show_progress:
        print(f"Reading {filesize} bytes from {filepath}")
    with open(filepath, "rb") as f:
        data_words = np.frombuffer(f.read(), dtype=np.uint32)
    special, channels, timestamps = extract_records_picoharp(data_words)
    channels, timestamps = unwrap_time_picoharp(special, channels, timestamps, show_progress=show_progress)
    mask = channels != 0xF  # drop special records
    return channels[mask], timestamps[mask].astype(np.uint64)

# ---------------- Correlation functions -----------------
def correlate_numpy_single(ch0, ch1, tau_max, tau_bin_width):
    n_bins = int((tau_max + tau_bin_width - 1) // tau_bin_width)
    hist = np.zeros(n_bins, dtype=np.uint64)
    n1 = len(ch1)
    for t0 in ch0:
        j = np.searchsorted(ch1, t0, side='left')
        if j >= n1:
            continue
        dt = ch1[j:] - t0
        valid_mask = (dt >= 0) & (dt < tau_max)
        if not np.any(valid_mask):
            continue
        dt_valid = dt[valid_mask]
        bin_idx = (dt_valid // tau_bin_width).astype(np.int64)
        np.add.at(hist, bin_idx, 1)
    return hist

def _open_shm_array(shm_name, shape, dtype):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, arr

def correlate_chunk_shared(args):
    shm_name, shape, dtype_str, ch0_chunk, tau_max, tau_bin_width, use_numba = args
    dtype = np.dtype(dtype_str)
    shm, ch1_times = _open_shm_array(shm_name, shape, dtype)
    try:
        # Here we always use numpy fallback
        hist = correlate_numpy_single(ch0_chunk, ch1_times, tau_max, tau_bin_width)
    finally:
        shm.close()
    return hist

# ---------------- Main parallel correlation -----------------
def correlate_picoharp_b_parallel(filepath, tau_bin_width=16, tau_max=10000, n_chunks=None, show_progress=True):
    channels, timestamps = read_picoharp_b_file(filepath, show_progress=show_progress)
    ch0_times = timestamps[channels == 0]
    ch1_times = timestamps[channels == 1]
    del channels, timestamps

    if n_chunks is None:
        n_chunks = max(1, cpu_count() - 1)

    # Shared memory for ch1_times
    shm = shared_memory.SharedMemory(create=True, size=ch1_times.nbytes)
    shm_arr = np.ndarray(ch1_times.shape, dtype=ch1_times.dtype, buffer=shm.buf)
    np.copyto(shm_arr, ch1_times)

    ch0_chunks = np.array_split(ch0_times, n_chunks)
    args_list = [(shm.name, ch1_times.shape, ch1_times.dtype.str, chunk,
                  np.uint64(tau_max), np.uint64(tau_bin_width), False)
                 for chunk in ch0_chunks]

    if show_progress:
        print(f"Starting correlation with {n_chunks} workers...")

    start_time = time.time()
    n_bins = int((tau_max + tau_bin_width - 1) // tau_bin_width)
    hist = np.zeros(n_bins, dtype=np.uint64)

    with ProcessPoolExecutor(max_workers=n_chunks) as ex:
        futures = [ex.submit(correlate_chunk_shared, args) for args in args_list]
        for i, fut in enumerate(as_completed(futures), 1):
            hist += fut.result()
            if show_progress:
                elapsed = time.time() - start_time
                frac = i / len(futures)
                eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
                print(f"Progress: {frac*100:.1f}%, ETA: {eta:.1f}s", end='\r')

    print("\nCorrelation done in %.1f s" % (time.time() - start_time))
    print(f"Total photons: ch0={len(ch0_times)}, ch1={len(ch1_times)}")

    shm.close()
    shm.unlink()

    tau = np.arange(0, tau_max, tau_bin_width) * BASE_RES  # in seconds
    return tau, hist

# ---------------- Plotting -----------------
def plot_correlation(tau, hist, title="Cross-correlation histogram"):
    plt.figure(figsize=(8,5))
    plt.plot(tau * 1e9, hist, drawstyle='steps-mid')  # convert to ns
    plt.xlabel("Delay τ [ns]")
    plt.ylabel("Counts")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ---------------- Example usage -----------------
if __name__ == "__main__":
    filepath = r"C:\Users\SSPDs\Documents\Offline SSPD Data\20251116\Toptica_AOM_50kHz_30s\tests\_chunk0.b"
    tau_bin_width = 4e3/4       # desired_width / resolution
    tau_max = 500e6/4           # desired_tau_max / resolution

    tau, hist = correlate_picoharp_b_parallel(filepath, tau_bin_width, tau_max, show_progress=True)
    plot_correlation(tau, hist)
    np.savetxt('Histogram_data_%s.txt'%filepath[-9:],[tau,hist])
