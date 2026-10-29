# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:40:17 2024

@author: QPG
"""

import time
import numpy as np
from datetime import datetime
import os
import sys
from collections import deque
import logging
import ctypes
import Device_picoharp as ph

# Either auto-detect:
pico = ph.picoharp()
pico.opendefaults()

print("Serial:", pico.GetSerialNumber())
print("Hardware Info:", pico.GetHardwareInfo())
print("Library Version:", pico.GetLibraryVersion())


'''
#!/usr/bin/env python3
"""
Example: run PicoHarp in T2 mode, collect TTTR events via ReadFiFo, parse records,
and compute a simple cross-correlation g2(tau) between channel 0 and channel 1.

Adjust parameters (measurement time, buffer sizes, binning) to your setup.
"""
# import your class (assumes the class is in picoharp.py or the same file)
# from picoharp import picoharp
# If the class is in the same file, no import needed.

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PH_T2_Corr")


def parse_t2_records(records_u32, resolution, overflow_counter):
    """
    Parse array of uint32 TTTR records from PicoHarp T2 mode.

    Returns:
      events: list of (channel, abs_timestamp_in_seconds)
      overflow_counter: updated overflow accumulator (in time-tags)
    """
    events = []
    # Masks / shifts
    CHANNEL_MASK = 0xF0000000
    TIMETAG_MASK = 0x0FFFFFFF
    for r in records_u32:
        # ensure uint32
        val = int(r & 0xFFFFFFFF)
        channel = (val >> 28) & 0xF
        timetag = val & TIMETAG_MASK

        if channel == 15:
            # special record: marker or overflow
            # Many PicoQuant demos use the 28-bit timetag field to indicate overflow increment(s)
            # The value in 'timetag' often holds how many times the time-tag counter overflowed (or the overflow magnitude).
            # We treat it as an overflow increment and add TIMETAG_MASK+1 per increment.
            # If the manual/demos for your exact device specify another encoding, adapt here.
            overflow_counter += (timetag + 1)  # +1 because a zero might indicate 1 wrap
            continue
        else:
            # absolute timetag (in device ticks) = timetag + overflow_counter * (TIMETAG_MASK+1)
            absolute_ticks = timetag + overflow_counter * (TIMETAG_MASK + 1)
            absolute_time_s = absolute_ticks * resolution
            events.append((channel, absolute_time_s))
    return events, overflow_counter


def compute_cross_correlation(tsA, tsB, bin_width_s=1e-9, max_lag_s=1e-6):
    """
    Compute cross-correlation histogram (g2) between tsA and tsB (monotonic arrays of timestamps in seconds).
    This implementation finds, for each event in the smaller list, the events in the other list within +/- max_lag
    and bins their time differences into histogram bins.

    NOTE: This is a straightforward (not highly optimized) approach intended for moderate event counts.
    For very large TTTR streams consider specialized fast correlation libraries (or chunked processing).
    """
    tsA = np.asarray(tsA)
    tsB = np.asarray(tsB)
    # choose reference as smaller array for efficiency
    if tsA.size > tsB.size:
        ref = tsB
        other = tsA
        invert = True
    else:
        ref = tsA
        other = tsB
        invert = False

    nbins = int(np.ceil((2 * max_lag_s) / bin_width_s))
    bins = np.linspace(-max_lag_s, max_lag_s, nbins + 1)
    hist = np.zeros(nbins, dtype=np.int64)

    # use searchsorted to find window in other for each reference event
    for t in ref:
        lo = np.searchsorted(other, t - max_lag_s, side='left')
        hi = np.searchsorted(other, t + max_lag_s, side='right')
        if lo >= hi:
            continue
        diffs = other[lo:hi] - t
        if invert:
            diffs = -diffs  # if we swapped A/B earlier, invert sign so tau = tA - tB consistently
        # bin diffs
        h, _ = np.histogram(diffs, bins=bins)
        hist += h

    # bin centers
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, hist


ph = picoharp()
try:
    ph.Initialize(mode="T2")
    log.info("Initialized in T2 mode.")
    ph.SetCFDDefaults()
    # Set binning irrelevant for TTTR, but keep defaults
    # ph.SetBinning(5)

    # Measurement/control parameters
    read_block_size = 200000  # number of 32-bit records to request each ReadFiFo
    total_meas_time_s = 5.0  # seconds of acquisition
    poll_interval = 0.05  # seconds between FIFO polls while running

    # start measurement (tacq in ms for StartMeas)
    ph.ClearHistMem()
    ph.SetStopOverflow(0, 0)  # don't stop on overflow
    ph.StartMeas(int(total_meas_time_s * 1000))
    log.info("Measurement started for %.2f s", total_meas_time_s)

    start_time = time.time()
    overflow_counter = 0  # counts time-tag wraps in units of (TIMETAG_MASK+1) ticks
    collected = {0: [], 1: []}  # timestamp lists per channel

    resolution = ph.GetResolution()  # base resolution in seconds per tick
    log.info("Device resolution: %s s per tick", resolution)

    # Continuously read FIFO until measurement time elapses
    while True:
        # read up to read_block_size records
        records = ph.ReadFiFo(read_block_size)
        if records is not None and records.size > 0:
            events, overflow_counter = parse_t2_records(records, resolution, overflow_counter)
            # append events for channel 0 and 1
            for ch, t in events:
                if ch in (0, 1):
                    collected[ch].append(t)

        # break if measurement finished
        if time.time() - start_time >= total_meas_time_s:
            break
        time.sleep(poll_interval)

    ph.StopMeas()
    log.info("Measurement stopped. Collected counts: ch0=%d, ch1=%d",
            len(collected[0]), len(collected[1]))

    # convert to numpy arrays
    ts0 = np.array(collected[0], dtype=float)
    ts1 = np.array(collected[1], dtype=float)

    if ts0.size == 0 or ts1.size == 0:
        log.warning("Not enough events to compute correlation.")

    # Compute cross-correlation
    bin_width_s = 1e-8  # 10 ns bins (adjust)
    max_lag_s = 1e-6  # +/- 1 microsecond (adjust)
    centers, hist = compute_cross_correlation(ts0, ts1, bin_width_s=bin_width_s, max_lag_s=max_lag_s)

    # Normalize to obtain g2 (simple normalization: divide by expected random coincidences)
    # expected coincidences per bin = rateA * rateB * bin_width * measurement_time
    rate0 = len(ts0) / total_meas_time_s
    rate1 = len(ts1) / total_meas_time_s
    expected = rate0 * rate1 * bin_width_s * total_meas_time_s
    g2 = hist / expected

    # Save / print results
    np.savetxt("g2_tau_hist.txt", np.column_stack([centers, hist, g2]),
            header="tau_s\thist_counts\tg2_normalized")
    log.info("g2 histogram saved to g2_tau_hist.txt")

    # Quick text output of a few central bins
    mid = len(centers) // 2
    for i in range(mid - 5, mid + 6):
        if 0 <= i < len(centers):
            print(f"{centers[i]:.3e} s\tcounts={hist[i]}\tg2={g2[i]:.3f}")

finally:
    ph.close()



def g2_drift_correction(params, DM):
    
    gm = DM.devices.get("daq_ao")
    ph = DM.devices.get("apd")
    
    ph.opendefaults()
    ph.ClearHistMem()
    
    measurement_time = params.get("measurement_time") *60*60*1e3
    defect_position = params.get("defect_position")
    gm.set_ao0(defect_position[0])
    gm.set_ao1(defect_position[1])

    count_time = params.get("count_time")
    optimized_xyz = None
    count_arr = []

    ph.StartMeas(tacq=int(65530e3))
    iteration = 0
    drift_check = []
    
    while True:
    
        # Perform XYZ scan and update positions            
        data_xy, DM = scan_area(params, DM)
        best_xy = data_xy.get('best_xy_loc')
        worst = data_xy.get('worst')
        best = data_xy.get('best')
        drift_check.append([worst, best])
        
        data_z, DM = scan_depth(params, DM)
        best_z = data_z.get('best_z_loc')
        

        optimized_xyz = [best_xy[0], best_xy[1], best_z]
        print(f"Optimized XYZ position: {optimized_xyz}")
        
        params["scan_center"] = best_xy
        params["scan_z_midpoint"] = best_z
            
        # get current photon counts
        count = ph.GetCountRate(chan=0)+ph.GetCountRate(chan=1)
        count_arr.append(count)
        
        print(count)
        print("Elapsed time: ", ph.GetElapsedMeasTime())
        
        if (count>400) and (ph.GetElapsedMeasTime()<measurement_time):
            print(count, iteration) 
        else:
            hist = ph.GetHistogram()
            break
        
        print('elapse_time=%d s'%ph.GetElapsedMeasTime())
        print('total_count=%d '%np.sum(ph.GetHistogram()))
        
        iteration += 1
        time.sleep(count_time)
        
    ph.StopMeas()
    data = {
        'histogram': hist,
        'iteration': iteration,
        'count_arr': np.array(count_arr),
        'drift_check': np.array(drift_check)
        
     }

    return data, DM


# Earlier version of g2 code from Liu as reference
# i=1
# anti_c=0
# while i==1:
    
#     count=0
    
#     if anti_c%1==0:
        
#         antidrift(address)
        
#         if anti_c !=0:
            
#             hist = ph.GetHistogram()
#             dataframe_hist = pd.DataFrame(hist)
#             dataframe_hist.to_csv(address+r'\ph_hist_'+name+time.strftime("%y-%m-%d-%H-%M-%S",time.localtime())+'.csv', index=False, sep=',')
   
#     anti_c=anti_c+1
    
#     for j in range(10):
        
#         count=count+ph.GetCountRate(chan=0)+ph.GetCountRate(chan=1)
#         time.sleep(0.2)
        
#     count=int(count/10)
    
    
#     if (count>400) and (ph.GetElapsedMeasTime()<65000e3):
        
#         print(count) 
        
#         with open(address+r'\CPS_with_time'+name+'.txt','a') as f:
            
#             f.write('%s\t' % (time.strftime("%y-%m-%d-%H-%M-%S",time.localtime())))
#             f.write('\t')     
#             f.write('%d\t' % (count))
#             f.write('\t')
#             f.write('\n') 
            
#         f.close
                
#     else:
        
#         hist = ph.GetHistogram()
#         dataframe_hist = pd.DataFrame(hist)
#         dataframe_hist.to_csv(address+r'\ph_hist_'+name+time.strftime("%y-%m-%d-%H-%M-%S",time.localtime())+'.csv', index=False, sep=',')
#         sys.exit()
        
    
#     print('elapse_time=%d s'%ph.GetElapsedMeasTime())
#     print('total_count=%d '%np.sum(ph.GetHistogram()))
#     time.sleep(600)
'''