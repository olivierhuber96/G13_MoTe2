import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
@njit
def multitau_correlate(timestamps_a, timestamps_b=None, m=16, deltat=1e-9, nlevels=16):
    """
    Multi-tau photon correlation algorithm (auto or cross).
    
    Parameters
    ----------
    timestamps_a : ndarray
        Photon arrival times (seconds) for channel A
    timestamps_b : ndarray or None
        Photon arrival times for channel B (or None for autocorrelation)
    m : int
        Number of bins per level (commonly 8–32)
    deltat : float
        Base time resolution (s)
    nlevels : int
        Number of correlation levels

    Returns
    -------
    taus : ndarray
        Delay times (s)
    g : ndarray
        Correlation counts (not normalized)
    """
    if timestamps_b is None:
        timestamps_b = timestamps_a

    # ensure sorted arrays
    timestamps_a = np.sort(timestamps_a)
    timestamps_b = np.sort(timestamps_b)
    
    g = []
    taus = []
    factor = 1
    t_res = deltat
    
    for level in range(nlevels):
        # histogram bin edges for this level
        edges = np.arange(0, m+1) * t_res
        counts = np.zeros(m)
        
        # streaming correlation
        ia = 0
        for tb in timestamps_b:
            while ia < len(timestamps_a) and timestamps_a[ia] < tb - m*t_res:
                ia += 1
            ib = ia
            while ib < len(timestamps_a) and timestamps_a[ib] < tb:
                dt = tb - timestamps_a[ib]
                bin_idx = int(dt / t_res)
                if bin_idx < m:
                    counts[bin_idx] += 1
                ib += 1
        
        taus.extend(edges[:-1])
        g.extend(counts)
        
        # downsample timestamps for next level
        if len(timestamps_a) > 1:
            timestamps_a = 0.5 * (timestamps_a[::2] + timestamps_a[1::2])
        if len(timestamps_b) > 1:
            timestamps_b = 0.5 * (timestamps_b[::2] + timestamps_b[1::2])
        t_res *= 2  # double bin width
    
    return np.array(taus), np.array(g)

# load only truetime/ps and channel
df = pd.read_csv(r"C:\Users\SSPDs\Documents\Offline SSPD Data\251111\output.txt", comment='#', skiprows=78, 
                 names=["record", "chan", "nsync", "truetime_ps"])

t_ch0 = df.loc[df["chan"] == 0, "truetime_ps"].values * 1e-12
t_ch1 = df.loc[df["chan"] == 1, "truetime_ps"].values * 1e-12

# run correlation
taus, g = multitau_correlate(t_ch0, t_ch1, m=25, deltat=1e-9, nlevels=50)

import matplotlib.pyplot as plt
plt.plot(taus*1e9, g)
plt.xlabel("Delay (ns)")
plt.ylabel("Counts")
plt.title("g² correlation (multi-tau)")
plt.show()

