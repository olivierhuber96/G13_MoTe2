# =====================================================
# PTU Photon Correlation - Streaming + Parallel + Numba
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import multiprocessing as mp
import os
import time

# -------- Numba correlation worker --------
@njit
def correlate_block(t1, t2, max_tau, binwidth):
    nbins = int(2*max_tau/binwidth) + 1
    hist = np.zeros(nbins, dtype=np.int64)
    j_start = 0
    n1 = len(t1)
    n2 = len(t2)
    for i in range(n1):
        t = t1[i]
        while j_start < n2 and t2[j_start] < t - max_tau:
            j_start += 1
        j = j_start
        while j < n2 and t2[j] <= t + max_tau:
            dt = t2[j] - t
            bin_idx = int((dt + max_tau)/binwidth)
            if 0 <= bin_idx < nbins:
                hist[bin_idx] += 1
            j += 1
    return hist

def worker(args):
    t1_chunk, t2, max_tau, binwidth = args
    return correlate_block(t1_chunk, t2, max_tau, binwidth)

# =====================================================
# ---start_main---
if __name__ == "__main__":
    
    filename = r"C:\Users\SSPDs\Documents\Offline SSPD Data\251112\.txt"
    filename = r"Q:\PH_correlations\T2_Sacher_AOM_20s.txt"

    max_tau = 20e-3     # seconds
    binwidth = 100e-9      # seconds
    ncpus = max(1, mp.cpu_count() - 1)
    nbins = int(2 * max_tau / binwidth) + 1
    hist_total = np.zeros(nbins, dtype=np.int64)

    # We'll accumulate CH0 and CH1 photons until a chunk threshold
    CHUNK_PHOTONS = 5_000_000  # tune this to manage RAM use

    ch0_list = []
    ch1_list = []
    total_lines = 0

    print(f"Streaming from: {filename}")
    print(f"Using {ncpus} cores, tau_max = {max_tau*1e6:.1f} us, binwidth = {binwidth*1e9:.1f} ns")
    start_time = time.time()

    with open(filename, 'r') as f:
        for line in f:
            if 'CHN' not in line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                chan = int(parts[2])
                t_ps = np.uint64(parts[4])
            except (ValueError, IndexError):
                continue
            t_s = t_ps * 1e-12  # convert ps → s
            if chan == 0:
                ch0_list.append(t_s)
            elif chan == 1:
                ch1_list.append(t_s)

            total_lines += 1

            # Process in chunks to control memory
            if len(ch0_list) > CHUNK_PHOTONS or len(ch1_list) > CHUNK_PHOTONS:
                print(f"Processing block at line {total_lines:,} ...")
                t_ch0 = np.array(ch0_list, dtype=np.float64)
                t_ch1 = np.array(ch1_list, dtype=np.float64)

                # Split CH0 for multiprocessing
                chunks = np.array_split(t_ch0, ncpus)
                args = [(chunk, t_ch1, max_tau, binwidth) for chunk in chunks]

                with mp.Pool(processes=ncpus) as pool:
                    results = pool.map(worker, args)

                hist_block = np.sum(np.array(results), axis=0)
                hist_total += hist_block

                # Clear for next chunk
                ch0_list.clear()
                ch1_list.clear()

    # Process any remaining data
    if len(ch0_list) > 0 and len(ch1_list) > 0:
        print("Processing final block...")
        t_ch0 = np.array(ch0_list, dtype=np.float64)
        t_ch1 = np.array(ch1_list, dtype=np.float64)
        chunks = np.array_split(t_ch0, ncpus)
        args = [(chunk, t_ch1, max_tau, binwidth) for chunk in chunks]
        with mp.Pool(processes=ncpus) as pool:
            results = pool.map(worker, args)
        hist_total += np.sum(np.array(results), axis=0)

    print(f"Done! Processed roughly {total_lines:,} photon lines.")
    print(r"It took %.0f seconds."%(time.time() - start_time))
    bins = np.linspace(-max_tau, max_tau, len(hist_total))

    # -------- Plot --------
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    t0_index = len(bins)//2
    plt.plot(bins[t0_index:-1], hist_total[t0_index:-1])
    plt.xlabel("Delay (s)")
    plt.ylabel("Counts")
    plt.title("Cross-correlation CH0 vs CH1 (streaming mode)")
    plt.grid(True)
    plt.show()
