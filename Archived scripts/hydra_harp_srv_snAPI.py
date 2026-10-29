import Pyro4, time, os, sys
import numpy as np 
from time import gmtime, strftime
def curtime():
    return strftime("UTC %Y-%m-%d %H:%M:%S", gmtime())


Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}
Pyro4.config.SERVERTYPE = "thread"         
Pyro4.config.COMMTIMEOUT = 0     # 0 = no per-request timeout
    

from snAPI.Main import *
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())
import time

if(__name__ == "__main__"):

    sn = snAPI()
    sn.getDeviceIDs()
    sn.getDevice()
    #sn.getFileDevice(r"D:\Data\PicoQuant\CW_Shelved.ptu") # T2 File
    #sn.getFileDevice(r"\mnt\d\Data\PicoQuant\CW_Shelved.ptu") # T2 File

    sn.initDevice(MeasMode.T2)
        
    # set the configuration for your device type
    sn.loadIniConfig("hh_config\MH.ini")
    
    # 1. shift the signals to max correlation max at tau = 0
    sn.device.setInputChannelOffset(1, 1588)
    
    # only process data from 70s to end of file 
    sn.manipulators.subStream(70)
    
    # 2. set windowSize and startTime
    sn.correlation.setG2Parameters(1, 2, 500000, 250, True)
    sn.correlation.measure(0,savePTU=False)

    while True:
        finished = sn.correlation.isFinished()
        data, bins = sn.correlation.getG2Data()
        time.sleep(.3)
        
        plt.clf()
        plt.plot(bins, data, linewidth=2.0, label='g(2)')
        plt.xlabel('Time [s]')
        plt.ylabel('g(2)')
        plt.legend()
        plt.title("g(2)")
        plt.pause(0.1)
        
        if finished:
            break
    
    plt.show(block=True)




# # chunkdata must be np.uint32
# chunkdata = np.asarray(chunkdata, dtype=np.uint32)

# # bit fields (1|6|25) from the spec
# SPECIAL_SHIFT = np.uint32(31)
# CHAN_SHIFT    = np.uint32(25)
# CHAN_MASK     = np.uint32(0x3F)       # 6 bits
# TIME_MASK     = np.uint32(0x1FFFFFF)  # 25 bits

# special   = (chunkdata >> SPECIAL_SHIFT).astype(np.uint8)                    # 0 or 1
# channels  = ((chunkdata >> CHAN_SHIFT) & CHAN_MASK).astype(np.uint8)         # 0..63
# timestamps= (chunkdata & TIME_MASK).astype(np.uint32)                        # 0..(2^25-1)

# # overflow unwrapping (modern HH2/TH260/MH: overflow count is in timetag)
# # Overflow period per snAPI doc: 2^25 = 33,554,432
# T2_WRAP = np.uint64(33554432)  # modern devices; HH v1 isn’t supported by snAPI.  [oai_citation:1‡picoquant.github.io](https://picoquant.github.io/snAPI/tttr%20data%20format.html)

# def unwrap_time_hydra(special, channels, timestamps):
#     ofl = np.uint64(0)
#     N   = timestamps.size - int(special.sum())
#     det_ts  = np.zeros(N, dtype=np.uint64)
#     det_ch  = np.zeros(N, dtype=np.uint8)
#     k = 0
#     for i in range(timestamps.size):
#         if special[i]:
#             # channel 63 => overflow; timetag holds the number of wraps
#             if channels[i] == 63:
#                 ofl += T2_WRAP * np.uint64(timestamps[i])
#             # other special codes (sync=0, markers=1..15) are ignored here.  [oai_citation:2‡picoquant.github.io](https://picoquant.github.io/snAPI/tttr%20data%20format.html)
#         else:
#             det_ch[k] = channels[i]
#             det_ts[k] = np.uint64(timestamps[i]) + ofl
#             k += 1
#     return det_ch, det_ts