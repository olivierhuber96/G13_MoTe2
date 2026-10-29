import numpy as np
import matplotlib.pyplot as plt
import RemoteCorrelationAnalysis as ana

file_path = r'C:\Users\SSPDs\Documents\Offline SSPD Data\251111\T2_Sacher.ptu'
ana.read_ptu_header(file_path)
ana.correlate_hydra_ptu(file_path, chunksize=2**11, chunk_bins=None,
                    tau_min=0, tau_max=4*200e3, tau_binning=4*40,
                    post_selection_bins=None, debug=False)


''''
# For large datasets, consider using time-tagging histogramming or PyCorrFit
from itertools import combinations

diffs = []
for t1, t2 in zip(ch0_times[:-1], ch0_times[1:]):
    dt_pair = t2 - t1
    if abs(dt_pair) <= t_max:
        diffs.append(dt_pair)

diffs = np.array(diffs)

# Histogram the time differences
hist, edges = np.histogram(diffs, bins=bins)

# -----------------------------
# 4️⃣ Plot g2 histogram
# -----------------------------
plt.figure(figsize=(8,5))
plt.bar(edges[:-1]*1e6, hist, width=bin_width*1e6, align='edge')
plt.xlabel("Time lag τ [µs]")
plt.ylabel("Counts")
plt.title("g2 Autocorrelation (Channel 0)")
plt.show()
'''