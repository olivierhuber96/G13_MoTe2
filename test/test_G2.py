import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

import numpy as np

# Prepare empty lists to store the data
channel_list = []
times_list = []

# This is the name of the text file you created
filename = r"C:\Users\SSPDs\Documents\Offline SSPD Data\251111\output.txt"

print(f"Reading data from {filename}...")

try:
    with open(filename, 'r') as f:
        for line in f:
            # We only care about lines that contain 'CHN' (photon records)
            if 'CHN' in line:
                try:
                    parts = line.split()
                    # parts[2] is the integer after 'CHN'
                    channel_list.append(int(parts[2]))
                    # parts[4] is the 'truetime/ps' value
                    times_list.append(np.uint64(parts[4]))
                except (IndexError, ValueError):
                    # This will skip any malformed lines
                    pass

    # Convert the Python lists into efficient NumPy arrays
    channel = np.array(channel_list, dtype=np.uint8)
    times = np.array(times_list, dtype=np.float64)
    

    print("Data loaded successfully.")
    print(f"Loaded {len(channel)} photon records.")
    
    # --- Example of the first 10 entries ---
    print("\n--- First 10 entries ---")
    print("Channels:", channel[:10])
    print("Times (ps):", times[:10])

except FileNotFoundError:
    print(f"ERROR: The file '{filename}' was not found.")
    print("Please make sure it's in the same directory as this script.")

# Create boolean masks
mask_ch0 = (channel == 0)
mask_ch1 = (channel == 1)

# Apply the masks to the times array
t_ch0 = times[mask_ch0]*1e-12
t_ch1 = times[mask_ch1]*1e-12

# --- Verification ---
print(f"Total photons: {len(times)}")
print(f"Photons in Ch 0: {len(t_ch0)}")
print(f"Photons in Ch 1: {len(t_ch1)}")

# -------- 3. Run correlation --------
def correlate_stream(t1, t2=None, max_tau=1e-6, binwidth=1e-9):
    """
    t1, t2 in seconds
    max_tau: correlation window in seconds
    binwidth: histogram bin width in seconds
    """
    if t2 is None:
        t2 = t1
    bins = np.arange(-max_tau, max_tau + binwidth, binwidth)
    hist = np.zeros(len(bins) - 1, dtype=np.int64)
    j_start = 0
    for t in t1:
        while j_start < len(t2) and t2[j_start] < t - max_tau:
            j_start += 1
        j = j_start
        while j < len(t2) and t2[j] <= t + max_tau:
            dt = t2[j] - t
            bin_idx = int((dt + max_tau) / binwidth)
            if 0 <= bin_idx < len(hist):
                hist[bin_idx] += 1
            j += 1
    return bins[:-1] + binwidth/2, hist

taus, g = correlate_stream(t_ch0, t_ch1, max_tau=200e-6, binwidth=40e-9)

# -------- 4. Plot or inspect --------
import matplotlib.pyplot as plt
plt.plot(taus, g)
plt.xlabel("Delay (ns)")
plt.ylabel("Counts")
plt.title("Cross-correlation CH0 vs CH1")
plt.show()
