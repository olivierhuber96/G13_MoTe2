import numpy as np
import time
from multiprocessing import Pool, cpu_count, shared_memory
from tqdm import tqdm

# ---------- Constants ----------
rtPicoHarpT2 = int('00010203', 16)

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
#def extract_records_picoharp(data_words):
#    """Decode 32-bit PicoHarp T2 records"""
#    channels = (data_words & 0x0000000F).astype(np.uint8)
#    timestamps = ((data_words >> 4) & 0x0FFFFFFF).astype(np.uint32)
#    special = (data_words >> 31).astype(np.uint8)
#    return special, channels, timestamps

#def unwrap_time_picoharp(special, channels, timestamps, show_progress=False):
#    """Vectorized overflow handling (≈100× faster than loop)"""
#    overflow_events = np.where(special == 1)[0]
#    if len(overflow_events) == 0:
#        return channels, timestamps.astype(np.uint64)

#    overflow_add = np.zeros_like(timestamps, dtype=np.uint64)
#    if show_progress:
#        print("Unwrapping timestamps...")
#    overflow_add[overflow_events] = 1
#    overflow_add = np.cumsum(overflow_add) << 28
#    full_times = timestamps.astype(np.uint64) + overflow_add
#    return channels, full_times

def extract_records_picoharp(data_words):
    """Correctly decode 32-bit PicoHarp T2 records"""
    special = (data_words >> 31).astype(np.uint8)
    channels = ((data_words >> 27) & 0x0F).astype(np.uint8)
    timestamps = (data_words & 0x07FFFFFF).astype(np.uint32)  # 27-bit mask
    return special, channels, timestamps

def unwrap_time_picoharp(special, channels, timestamps, show_progress=False):
    """Correct vectorized overflow handling for PicoHarp T2 (27-bit)"""
    # Overflow records have special=1 and channel=15 (0xF)
    overflow_events_idx = np.where((special == 1) & (channels == 15))[0]
    if len(overflow_events_idx) == 0:
        # No overflows, but still filter out non-photon records
        photons_idx = np.where((special == 0) & (channels < 15))[0]
        return channels[photons_idx], timestamps[photons_idx].astype(np.uint64)
    overflow_add = np.zeros_like(timestamps, dtype=np.uint64)
    if show_progress:
        print("Unwrapping timestamps...")
    # The 'timestamps' field of an overflow record holds the *number* of overflows
    overflow_counts = timestamps[overflow_events_idx]
    # In case a 0 count is recorded, treat it as 1
    overflow_counts[overflow_counts == 0] = 1
    overflow_add[overflow_events_idx] = overflow_counts
    # Cumulatively sum the *actual* overflow counts and shift by 27 bits
    overflow_add = np.cumsum(overflow_add) << 27
    full_times = timestamps.astype(np.uint64) + overflow_add
    # Filter out the overflow records, returning only photon events
    photons_idx = np.where((special == 0) & (channels < 15))[0]
    return channels[photons_idx], full_times[photons_idx]

# ---------- Parallel correlation ----------
def correlate_chunk_shared(args):
    """Worker function (shared memory)"""
    shm_name, shape, dtype, ch0_chunk, tau_max, tau_bin_width = args
    shm = shared_memory.SharedMemory(name=shm_name)
    ch1_times = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    hist_bins = np.arange(0, tau_max + tau_bin_width, tau_bin_width)
    hist = np.zeros(len(hist_bins) - 1, dtype=np.uint64)

    for t0 in ch0_chunk:
        idx1_start = np.searchsorted(ch1_times, t0, side='left')
        dt = ch1_times[idx1_start:idx1_start+1000] - t0
        dt = dt[(dt >= 0) & (dt < tau_max)]
        hist += np.histogram(dt, bins=hist_bins)[0].astype(np.uint64)

    shm.close()
    return hist

def correlate_picoharp_ptu_parallel_shared(file_path, tau_bin_width=16, tau_max=10000, n_chunks=None):
    inputfile, tags = read_ptu_header(file_path)
    record_type = tags["TTResultFormat_TTTRRecType"]
    if record_type != rtPicoHarpT2:
        raise TypeError("File is not PicoHarp T2")

    num_records = tags["TTResult_NumberOfRecords"]
    base_res = tags["MeasDesc_GlobalResolution"]

    # ---------- Read file with progress ----------
    print("Reading PTU data...")
    inputfile.seek(0, 2)
    file_size = inputfile.tell()
    inputfile.seek(0)
    inputfile.read(8 + 8)  # skip magic/version
    pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading .ptu")
    chunks = []
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

    # ---------- Extract and unwrap ----------
    special, channels, timestamps = extract_records_picoharp(data_words)
    del data_words
    channels, timestamps = unwrap_time_picoharp(special, channels, timestamps, show_progress=True)

    ch0_times = timestamps[channels == 0]
    ch1_times = timestamps[channels == 1]
    del special, channels, timestamps

    if n_chunks is None:
        n_chunks = max(1, cpu_count() - 1)

    # ---------- Shared memory ----------
    shm = shared_memory.SharedMemory(create=True, size=ch1_times.nbytes)
    shm_array = np.ndarray(ch1_times.shape, dtype=ch1_times.dtype, buffer=shm.buf)
    np.copyto(shm_array, ch1_times)

    ch0_chunks = np.array_split(ch0_times, n_chunks)
    args_list = [(shm.name, ch1_times.shape, ch1_times.dtype, chunk, tau_max, tau_bin_width)
                 for chunk in ch0_chunks]

    print(f"Start correlations with {n_chunks} chunks...")

    start_time = time.time()
    hist = np.zeros(int(tau_max / tau_bin_width), dtype=np.uint64)

    with Pool(processes=n_chunks) as pool:
        for i, chunk_hist in enumerate(pool.imap_unordered(correlate_chunk_shared, args_list), 1):
            hist += chunk_hist
            elapsed = time.time() - start_time
            frac = i / n_chunks
            eta = elapsed / frac - elapsed
            print(f"Progress: {frac*100:.1f}%, ETA: {eta:.1f}s", end='\r')

    print("\nCorrelation done. It took %.0fs"%(time.time()-start_time))
    shm.close()
    shm.unlink()

    tau_ns = np.arange(0, tau_max, tau_bin_width) * base_res * 1e9
    return tau_ns, hist

# ---------- Example usage ----------
if __name__ == "__main__":
    file_path = r"C:\Users\SSPDs\Documents\Offline SSPD Data\251111\T2_Sacher_300s.ptu"
    tau_ns, g2_hist = correlate_picoharp_ptu_parallel_shared(file_path, tau_bin_width=40e3/4, tau_max=200e6/4, n_chunks=7)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(tau_ns, g2_hist, lw=1.5)
    plt.xlabel("Time delay τ [ns]")
    plt.ylabel("Coincidences")
    plt.title("g² histogram (PicoHarp T2)")
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
