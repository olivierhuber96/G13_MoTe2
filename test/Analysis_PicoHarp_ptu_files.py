import numpy as np
import struct
import matplotlib.pyplot as plt
import time 
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, shared_memory

# -----------------------------
# PicoHarp record types
# -----------------------------
rtPicoHarpT2 = 0x00010203

# -----------------------------
# PicoQuant tag types (little-endian)
# -----------------------------
tyEmpty8      = 0xFFFF0008
tyBool8       = 0x00000008
tyInt8        = 0x10000008
tyBitSet64    = 0x11000008
tyColor8      = 0x12000008
tyFloat8      = 0x20000008
tyTDateTime   = 0x21000008
tyFloat8Array = 0x2001FFFF
tyAnsiString  = 0x4001FFFF
tyWideString  = 0x4002FFFF
tyBinaryBlob  = 0xFFFFFFFF

# -----------------------------
# PTU header reader
# -----------------------------
def read_ptu_header(input_filepath):
    """Read PTU header file, return file handle and tag dictionary."""
    inputfile = open(input_filepath, "rb")
    magic = inputfile.read(8).decode("utf-8").strip('\0')
    if magic != "PQTTTR":
        inputfile.close()
        raise AssertionError("Not a PTU file")
    version = inputfile.read(8).decode("utf-8").strip('\0')

    tagDataList = []
    while True:
        tagIdent = inputfile.read(32).decode("utf-8").strip('\0')
        tagIdx = struct.unpack("<i", inputfile.read(4))[0]       # signed
        tagTyp = struct.unpack("<I", inputfile.read(4))[0]       # unsigned!

        evalName = tagIdent + f"({tagIdx})" if tagIdx > -1 else tagIdent

        if tagTyp == tyEmpty8:
            inputfile.read(8)
            tagDataList.append((evalName, "<empty Tag>"))

        elif tagTyp == tyBool8:
            val = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, bool(val)))

        elif tagTyp in (tyInt8, tyBitSet64, tyColor8):
            val = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, val))

        elif tagTyp == tyFloat8:
            val = struct.unpack("<d", inputfile.read(8))[0]
            tagDataList.append((evalName, val))

        elif tagTyp == tyTDateTime:
            val = struct.unpack("<d", inputfile.read(8))[0]
            tagDataList.append((evalName, val))

        elif tagTyp == tyAnsiString:
            length = struct.unpack("<q", inputfile.read(8))[0]
            val = inputfile.read(length).decode("utf-8").strip("\0")
            tagDataList.append((evalName, val))

        elif tagTyp == tyWideString:
            length = struct.unpack("<q", inputfile.read(8))[0]
            val = inputfile.read(length).decode("utf-16le", errors="ignore").strip("\0")
            tagDataList.append((evalName, val))

        elif tagTyp == tyBinaryBlob:
            length = struct.unpack("<q", inputfile.read(8))[0]
            inputfile.read(length)
            tagDataList.append((evalName, f"<binary blob, {length} bytes>"))

        else:
            raise TypeError(f"Unknown tag type: {tagTyp}")

        if tagIdent == "Header_End":
            break

    tags = dict(tagDataList)
    return inputfile, tags

# -----------------------------
# PicoHarp T2 record extraction
# -----------------------------
def extract_records_picoharp(data_words):
    """Decode 32-bit PicoHarp T2 records"""
    channels = (data_words & 0x0000000F).astype(np.uint8)      # lower 4 bits = channel
    timestamps = ((data_words >> 4) & 0x0FFFFFFF).astype(np.uint32)  # next 28 bits = time
    special = (data_words >> 31) & 0x1  # overflow flag
    return special, channels, timestamps

def unwrap_time_picoharp(special, channels, timestamps, show_progress=True):
    """Handle overflows and reconstruct full timestamps with optional progress bar"""
    full_times = np.zeros_like(timestamps, dtype=np.uint64)
    overflow_correction = 0

    if show_progress:
        iterator = tqdm(range(len(timestamps)), desc="Unwrapping timestamps")
    else:
        iterator = range(len(timestamps))

    for i in iterator:
        if special[i] == 1:  # overflow record
            overflow_correction += 1 << 28  # 28-bit counter overflow
        full_times[i] = timestamps[i] + overflow_correction
    return channels, full_times


# -----------------------------
# Simple g2 correlation
# -----------------------------
def correlate_picoharp_ptu(file_path, tau_bin_width=16, tau_max=10000):
    inputfile, tags = read_ptu_header(file_path)
    record_type = tags["TTResultFormat_TTTRRecType"]
    if record_type != rtPicoHarpT2:
        raise TypeError("File is not PicoHarp T2")

    num_records = tags["TTResult_NumberOfRecords"]
    base_res = tags["MeasDesc_GlobalResolution"]  # seconds per timestamp

    # Read all 32-bit records
    data_words = np.frombuffer(inputfile.read(num_records*4), dtype=np.uint32)
    inputfile.close()

    special, channels, timestamps = extract_records_picoharp(data_words)
    channels, timestamps = unwrap_time_picoharp(special, channels, timestamps, show_progress=True)

    # Split channels
    ch0_times = timestamps[channels==0]
    ch1_times = timestamps[channels==1]

    # g2 histogram
    hist_bins = np.arange(0, tau_max+tau_bin_width, tau_bin_width)
    hist = np.zeros(len(hist_bins)-1, dtype=np.uint64)

    # Simple cross-correlation with progress display
    idx1_start = 0
    print('Start correlations with standard method.')
    start_time = time.time()
    total = len(ch0_times)

    for i, t0 in enumerate(ch0_times):
        idx1_start = np.searchsorted(ch1_times, t0, side='left')
        dt = ch1_times[idx1_start:idx1_start+1000] - t0  # limit to nearby events
        dt = dt[(dt >= 0) & (dt < tau_max)]
        hist += np.histogram(dt, bins=hist_bins)[0].astype(np.uint64)

        # Print progress every 1%
        if i % max(1, total//100) == 0:
            elapsed = time.time() - start_time
            fraction_done = i / total
            eta = elapsed / max(1e-6, fraction_done) - elapsed
            print(f"Progress: {fraction_done*100:.1f}%, ETA: {eta:.1f}s", end='\r')

    print("\nCorrelation done. It took %.0fs"%(time.time()-start_time))

    tau_ns = hist_bins[:-1] * base_res * 1e9  # convert to ns
    return tau_ns, hist

# -----------------------------
# Fast g2 correlation using numpy
# -----------------------------
def correlate_picoharp_ptu_fast(file_path, tau_bin_width=16, tau_max=10000):
    inputfile, tags = read_ptu_header(file_path)
    record_type = tags["TTResultFormat_TTTRRecType"]
    if record_type != rtPicoHarpT2:
        raise TypeError("File is not PicoHarp T2")

    num_records = tags["TTResult_NumberOfRecords"]
    base_res = tags["MeasDesc_GlobalResolution"]

    data_words = np.frombuffer(inputfile.read(num_records*4), dtype=np.uint32)
    inputfile.close()

    special, channels, timestamps = extract_records_picoharp(data_words)
    channels, timestamps = unwrap_time_picoharp(special, channels, timestamps, show_progress=True)

    ch0 = timestamps[channels==0]
    ch1 = timestamps[channels==1]

    hist_bins = np.arange(0, tau_max+tau_bin_width, tau_bin_width, dtype=np.uint64)
    hist = np.zeros(len(hist_bins)-1, dtype=np.uint64)

    # Vectorized g2 cross-correlation with progress estimate
    idx1 = np.searchsorted(ch1, ch0)
    print('Start correlations')
    start_time = time.time()
    for i, t0 in enumerate(ch0):
        start = idx1[i]
        dt = ch1[start:]
        dt = dt[dt - t0 < tau_max] - t0
        hist += np.histogram(dt, bins=hist_bins)[0].astype(np.uint64)

        # Print progress every 1%
        if i % max(1, len(ch0)//100) == 0:
            elapsed = time.time() - start_time
            fraction_done = i / len(ch0)
            eta = elapsed / max(1e-6, fraction_done) - elapsed
            print(f"Progress: {fraction_done*100:.1f}%, ETA: {eta:.1f}s", end='\r')

    print("\nCorrelation done. It took %.0fs"%(time.time()-start_time))
    tau_ns = hist_bins[:-1] * base_res * 1e9
    return tau_ns, hist



# -----------------------------
# Fast g2 correlation using numpy and breaking computation in chunks
# -----------------------------
def correlate_chunk(args):
    """Compute g2 histogram for a chunk of ch0_times."""
    ch0_chunk, ch1_times, tau_max, tau_bin_width = args
    hist_bins = np.arange(0, tau_max + tau_bin_width, tau_bin_width)
    hist = np.zeros(len(hist_bins) - 1, dtype=np.uint64)

    for t0 in ch0_chunk:
        idx = np.searchsorted(ch1_times, t0, side='left')
        dt = ch1_times[idx:idx+1000] - t0
        dt = dt[(dt >= 0) & (dt < tau_max)]
        hist += np.histogram(dt, bins=hist_bins)[0].astype(np.uint64)
    return hist

def correlate_picoharp_ptu_parallel(file_path, tau_bin_width=16, tau_max=10000, n_chunks=None):
    inputfile, tags = read_ptu_header(file_path)
    record_type = tags["TTResultFormat_TTTRRecType"]
    if record_type != rtPicoHarpT2:
        raise TypeError("File is not PicoHarp T2")

    num_records = tags["TTResult_NumberOfRecords"]
    base_res = tags["MeasDesc_GlobalResolution"]

    data_words = np.frombuffer(inputfile.read(num_records*4), dtype=np.uint32)
    inputfile.close()

    special, channels, timestamps = extract_records_picoharp(data_words)
    channels, timestamps = unwrap_time_picoharp(special, channels, timestamps, show_progress=True)

    ch0_times = timestamps[channels==0]
    ch1_times = timestamps[channels==1]

    if n_chunks is None:
        n_chunks = cpu_count()  # use all available cores

    # Split ch0_times into chunks for parallel processing
    ch0_chunks = np.array_split(ch0_times, n_chunks)
    args_list = [(chunk, ch1_times, tau_max, tau_bin_width) for chunk in ch0_chunks]

    print('Start correlations with %.0f chunks.'%n_chunks)

    start_time = time.time()
    hist = np.zeros(int(tau_max / tau_bin_width), dtype=np.uint64)    

    with Pool(processes=n_chunks) as pool:
        for i, chunk_hist in enumerate(pool.imap_unordered(correlate_chunk, args_list), 1):
            hist += chunk_hist
            elapsed = time.time() - start_time
            fraction_done = i / n_chunks
            eta = elapsed / fraction_done - elapsed
            print(f"Progress: {fraction_done*100:.1f}%, ETA: {eta:.1f}s", end='\r')

    print("\nCorrelation done.")
    tau_ns = np.arange(0, tau_max, tau_bin_width) * base_res * 1e9
    return tau_ns, hist


# -----------------------------
# Fast g2 correlation using numpy and breaking computation in chunks and using shared memory to avoid MemoryError
# -----------------------------
def correlate_chunk_shared(args):
    """Worker function: uses shared memory instead of copying arrays"""
    shm_name, shape, dtype, ch0_chunk, tau_max, tau_bin_width = args
    # Attach to shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    ch1_times = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    hist_bins = np.arange(0, tau_max + tau_bin_width, tau_bin_width)
    hist = np.zeros(len(hist_bins) - 1, dtype=np.uint64)

    for t0 in ch0_chunk:
        idx1_start = np.searchsorted(ch1_times, t0, side='left')
        dt = ch1_times[idx1_start:idx1_start+1000] - t0
        dt = dt[(dt >= 0) & (dt < tau_max)]
        hist += np.histogram(dt, bins=hist_bins)[0].astype(np.uint64)

    existing_shm.close()
    return hist

def correlate_picoharp_ptu_parallel_shared(file_path, tau_bin_width=16, tau_max=10000, n_chunks=None):
    inputfile, tags = read_ptu_header(file_path)
    record_type = tags["TTResultFormat_TTTRRecType"]
    if record_type != rtPicoHarpT2:
        raise TypeError("File is not PicoHarp T2")

    num_records = tags["TTResult_NumberOfRecords"]
    base_res = tags["MeasDesc_GlobalResolution"]

    print("Reading PTU data...")
    data_words = np.frombuffer(inputfile.read(num_records*4), dtype=np.uint32)
    inputfile.close()

    special, channels, timestamps = extract_records_picoharp(data_words)
    channels, timestamps = unwrap_time_picoharp(special, channels, timestamps, show_progress=True)

    ch0_times = timestamps[channels==0]
    ch1_times = timestamps[channels==1]

    if n_chunks is None:
        n_chunks = max(1, cpu_count() - 1)

    # Create shared memory for ch1_times
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

    print("\nCorrelation done.")

    # Cleanup
    shm.close()
    shm.unlink()

    tau_ns = np.arange(0, tau_max, tau_bin_width) * base_res * 1e9
    return tau_ns, hist


# -----------------------------
# Calculate correlations
# -----------------------------

if __name__ == '__main__':
    file_path = "C:/Users/SSPDs/Documents/Offline SSPD Data/251111/T2_Sacher.ptu"
    
    #tau_ns, g2_hist = correlate_picoharp_ptu(file_path, tau_bin_width=400e3, tau_max=2e6/4)
    #tau_ns, g2_hist = correlate_picoharp_ptu_fast(file_path, tau_bin_width=400e3/4, tau_max=200e6/4)
    #tau_ns, g2_hist = correlate_picoharp_ptu_parallel(file_path, tau_bin_width=400e3/4, tau_max=200e6/4, n_chunks=None)
    tau_ns, g2_hist = correlate_picoharp_ptu_parallel_shared(file_path, tau_bin_width=400e3/4, tau_max=200e6/4, n_chunks=7)
    
    
    plt.figure(figsize=(6,4))
    plt.plot(tau_ns, g2_hist, drawstyle='steps-mid')
    plt.xlabel("Time delay τ [ns]")
    plt.ylabel("Coincidences)")
    plt.tight_layout()
    plt.show()