# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:07:15 2023
Copy the necessary functions from G8's correlation_analysis.py such that it can be 
called by the Hydraharp_srv to convert the raw T2 files into histograms. 
@author: GM
"""
import numpy as np
from time import gmtime, strftime
import time
import numba

def curtime():
    return strftime("UTC %Y-%m-%d %H:%M:%S", gmtime())
import datetime
import os
import struct
import matplotlib as mpl
mpl.use("Agg")   # non-interactive backend to be able to save plots in a thread that is not the main thread
import matplotlib.pyplot as plt
import pickle
#from scipy.signal import find_peaks
from multiprocessing import Process
import sys, multiprocessing as mp
mp.set_start_method("spawn", force=True)  
mp.set_executable(sys.executable)          
#print(f"[analysis server] sys.executable = {sys.executable}", flush=True)


import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


import glob
import threading 
import hashlib

# Picoquant Tag Types
tyEmpty8      = struct.unpack(">i", bytes.fromhex("FFFF0008"))[0]
tyBool8       = struct.unpack(">i", bytes.fromhex("00000008"))[0]
tyInt8        = struct.unpack(">i", bytes.fromhex("10000008"))[0]
tyBitSet64    = struct.unpack(">i", bytes.fromhex("11000008"))[0]
tyColor8      = struct.unpack(">i", bytes.fromhex("12000008"))[0]
tyFloat8      = struct.unpack(">i", bytes.fromhex("20000008"))[0]
tyTDateTime   = struct.unpack(">i", bytes.fromhex("21000008"))[0]
tyFloat8Array = struct.unpack(">i", bytes.fromhex("2001FFFF"))[0]
tyAnsiString  = struct.unpack(">i", bytes.fromhex("4001FFFF"))[0]
tyWideString  = struct.unpack(">i", bytes.fromhex("4002FFFF"))[0]
tyBinaryBlob  = struct.unpack(">i", bytes.fromhex("FFFFFFFF"))[0]
# Picoquant Record types
rtPicoHarpT3     = struct.unpack(">i", bytes.fromhex('00010303'))[0]
rtPicoHarpT2     = struct.unpack(">i", bytes.fromhex('00010203'))[0]
rtHydraHarpT3    = struct.unpack(">i", bytes.fromhex('00010304'))[0]
rtHydraHarpT2    = struct.unpack(">i", bytes.fromhex('00010204'))[0]
rtHydraHarp2T3   = struct.unpack(">i", bytes.fromhex('01010304'))[0]
rtHydraHarp2T2   = struct.unpack(">i", bytes.fromhex('01010204'))[0]
rtTimeHarp260NT3 = struct.unpack(">i", bytes.fromhex('00010305'))[0]
rtTimeHarp260NT2 = struct.unpack(">i", bytes.fromhex('00010205'))[0]
rtTimeHarp260PT3 = struct.unpack(">i", bytes.fromhex('00010306'))[0]
rtTimeHarp260PT2 = struct.unpack(">i", bytes.fromhex('00010206'))[0]
rtMultiHarpNT3   = struct.unpack(">i", bytes.fromhex('00010307'))[0]
rtMultiHarpNT2   = struct.unpack(">i", bytes.fromhex('00010207'))[0]

def read_ptu_header(input_filepath):
    """Read PTU header file, return open file handle and some useful data.

    Can proceed reading the file to obtain timestamps (1 DWORD = 4 byte each).
    This takes 200 milliseconds.
    """
    # Open file
    inputfile = open(input_filepath, "rb")
    # Read file header
    magic = inputfile.read(8).decode("utf-8").strip('\0')
    if magic != "PQTTTR":
        print("ERROR: Magic invalid, this is not a PTU file.")
        inputfile.close()
        raise AssertionError
    version = inputfile.read(8).decode("utf-8").strip('\0')
    tagDataList = []    # Contains tuples of (tagName, tagValue)
    while True:
        tagIdent = inputfile.read(32).decode("utf-8").strip('\0')
        tagIdx = struct.unpack("<i", inputfile.read(4))[0]
        tagTyp = struct.unpack("<i", inputfile.read(4))[0]
        if tagIdx > -1:
            evalName = tagIdent + '(' + str(tagIdx) + ')'
        else:
            evalName = tagIdent
        if tagTyp == tyEmpty8:
            inputfile.read(8)
            tagDataList.append((evalName, "<empty Tag>"))
        elif tagTyp == tyBool8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            if tagInt == 0:
                tagDataList.append((evalName, "False"))
            else:
                tagDataList.append((evalName, "True"))
        elif tagTyp == tyInt8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyBitSet64:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyColor8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyFloat8:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            tagDataList.append((evalName, tagFloat))
        elif tagTyp == tyFloat8Array:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyTDateTime:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            tagTime = int((tagFloat - 25569) * 86400)
            tagTime = time.gmtime(tagTime)
            tagDataList.append((evalName, tagTime))
        elif tagTyp == tyAnsiString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode("utf-8").strip("\0")
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyWideString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode(
                        "utf-16le", errors="ignore").strip("\0")
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyBinaryBlob:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagDataList.append((evalName, tagInt))
        else:
            print("ERROR: Unknown tag type")
            print(tagTyp)
            raise TypeError("ERROR: Unknown tag type")
        if tagIdent == "Header_End":
            break
    # Reformat the saved data for easier access
    tagNames = [tagDataList[i][0] for i in range(0, len(tagDataList))]
    tagValues = [tagDataList[i][1] for i in range(0, len(tagDataList))]
    tags = dict(zip(tagNames, tagValues))
    return inputfile, tags

@numba.jit(nopython=True, parallel=False)
def extract_records_hydra(chunkdata):
    """Process chunks of hydraharp DWORDS into channels and timestamps."""
    timestamp_mask = np.uint32(-1) >> np.uint32(7)  # mask 000000011...11 to isolate the
    timestamps = chunkdata & timestamp_mask  # timestamp via logical and
    special = chunkdata >> 31
    channel_mask = ((1 << 6) - 1) << 25
    channels = (chunkdata & channel_mask) >> 25 
    return special, channels, timestamps 

def unwrap_time_hydra(special, channels, timestamps):
    """Unwrap overflow markers and overwrite them with adjacent timestamp.

    Danger: uint32 overflow for low countrates, this is not checked!
    """
    T2WRAPAROUND = np.uint64(33554432)  
    oflcorrection = np.uint64(0)
    N = special.shape[0] - np.sum(special)
    detector_timestamps = np.zeros(N, dtype=np.uint64)
    detector_channels = np.zeros(N, dtype=np.uint8)
    det_idx = 0
    for idx in range(timestamps.shape[0]):
        if special[idx] == 1:  # Special record
            if channels[idx] == 63:  # overflow marker
                oflcorrection += T2WRAPAROUND * timestamps[idx]
                # Fix leading bits for the sake of "time per chunk" calculation
                #if idx > 0:
                #    timestamps[idx] = timestamps[idx - 1]
                #else:
                #    timestamps[idx] = timestamps[idx + 1] + oflcorrection
        else:
            #timestamps[idx] += oflcorrection
            detector_channels[det_idx] = channels[idx]
            detector_timestamps[det_idx] = timestamps[idx] + oflcorrection
            det_idx += 1
    return detector_channels, detector_timestamps

@numba.jit(nopython=True, nogil=False, parallel=False)
def fast_correlate(timestamps, detector, histogram, binning, tmin, tmax):
    """Calculate un-normalized g2(tau) correlation function from timestamps.

    Time differences tau = timestamp detector==1 - timestamp detector==0 are
    binned into histogram array for tmin <= tau <= tmax. Returns None
    but increments (!) argument histogram with coincidences.

    Inspired by Picoharp_T2Analysis_parallel.py script written by Thomas Fink.

    Parameters
    ----------
    timestamps : array of uint32
        Contains timestamps, typically in units of some base resolution. But
        can have any units, binning is relative to the units in timestamps.
    detector : array of integer type
        Array of same length as timestamps, containing channel indices
        indicating which detector (0 or 1) each timestamp belongs to.
        are discarded.
    histogram : array of uint32
        Array to save histogram into, must have compatible length (...)
    binning : integer
        Histogram bin width, relative to the units of the timestamps.
        Example: binning=4 means that time differences 0,1,2,3 are summed
                 into the same bin.
    tmin : integer
        Minimal time delay tau the histogram should contain.
        Must satisfy tmin >= 0, because only waiting times after
        detection at channel 0 are considered. Inclusive.
    tmax : integer
        Maximum time delay tau the histogram should contain. Exclusive.
    """
    bin_offset = tmin // binning
    for photon1idx in range(timestamps.shape[0]):
        # We only calculate photon1-photon0
        # if not detector[photon1idx] == 0:
        if not detector[photon1idx] == 1:
            continue
        # Timestamps being sorted, this is the first candidate to form pair
        photon2idx = photon1idx + 1
        # Discard photon2 until the time difference exceeds tmin
        while timestamps[photon2idx] - timestamps[photon1idx] < tmin:
            photon2idx += 1
        # Iterate photon2 until we fall outside tmax or exhaust all timestamps
        while ((timestamps[photon2idx] - timestamps[photon1idx] < tmax)
               and (photon2idx < timestamps.shape[0])):
            # If timestamp from detector 1, save the difference into histogram
            # if detector[photon2idx] == 1:
            if detector[photon2idx] == 1:
                photon2idx += 1
                continue
            binidx = ((timestamps[photon2idx] - timestamps[photon1idx])
                      // binning)
            assert binidx-bin_offset >= 0, 'idx too small'
            assert binidx-bin_offset < histogram.size, 'idx too big'
            histogram[binidx - bin_offset] += 1
            photon2idx += 1


@numba.jit(nopython=True, nogil=False, parallel=False)
def fast_correlate_mode(timestamps, detector, histogram, binning, tmin, tmax, mode = 'X01'):
    """Calculate un-normalized g2(tau) correlation function from timestamps.

    Time differences tau are
    binned into histogram array for tmin <= tau <= tmax. Returns None
    but increments (!) argument histogram with coincidences.

    Inspired by Picoharp_T2Analysis_parallel.py script written by Thomas Fink.

    Parameters
    ----------
    timestamps : array of uint32
        Contains timestamps, typically in units of some base resolution. But
        can have any units, binning is relative to the units in timestamps.
    detector : array of integer type
        Array of same length as timestamps, containing channel indices
        indicating which detector (0 or 1) each timestamp belongs to.
        are discarded.
    histogram : array of uint32
        Array to save histogram into, must have compatible length (...)
    binning : integer
        Histogram bin width, relative to the units of the timestamps.
        Example: binning=4 means that time differences 0,1,2,3 are summed
                 into the same bin.
    tmin : integer
        Minimal time delay tau the histogram should contain.
        Must satisfy tmin >= 0, because only waiting times after
        detection at channel 0 are considered. Inclusive.
    tmax : integer
        Maximum time delay tau the histogram should contain. Exclusive.
    mode : str
        Mode of correlation, 'X01' is default, meaning photon delay in channel 2 conditioned on the detection of photon in ch. 1
    """

    def ph1_condition(detector, mode):
        if mode =="X01": 
            if detector == 0: 
                #print("Mode X01: Ph1 detectedin det. 0")
                return True 
        elif mode =="X10":
            if detector == 1:
                return True
        elif mode =="A0":
            if detector == 0:
                return True
        elif mode =="A1":
            if detector == 1:
                return True
        elif mode =="all":
            return True
        return False
        
    def ph2_condition(detector, mode):
        if mode =="X01": 
            if detector == 1: 
                ##print("Mode X01: Ph2 detected in det. 1")
                return True 
        elif mode =="X10":
            if detector == 0:
                return True
        elif mode =="A0":
            if detector == 0:
                return True
        elif mode =="A1":
            if detector == 1:
                return True
        elif mode =="all":
            return True
        return False


    bin_offset = tmin // binning
    #print(timestamps.shape[0])
    for photon1idx in range(timestamps.shape[0]):
        
        if not ph1_condition(detector[photon1idx], mode):
            continue
        # Timestamps being sorted, this is the first candidate to form pair
        photon2idx = photon1idx + 1
        # Discard photon2 until the time difference exceeds tmin
        # if photon1idx ==1:
        #     print(timestamps[photon2idx] - timestamps[photon1idx], timestamps[photon2idx], timestamps[photon2idx] - timestamps[photon1idx] < tmin )
        while timestamps[photon2idx] - timestamps[photon1idx] < tmin   :
            photon2idx += 1
        # Iterate photon2 until we fall outside tmax or exhaust all timestamps
        while ((timestamps[photon2idx] - timestamps[photon1idx] < tmax*1.1)
               and (photon2idx < timestamps.shape[0])):
            if not ph2_condition(detector[photon2idx], mode):
                photon2idx += 1
                continue
            binidx = (timestamps[photon2idx] - timestamps[photon1idx]) // binning
            # if binidx ==0:
            #     print(timestamps[photon2idx], timestamps[photon1idx], (timestamps[photon2idx] - timestamps[photon1idx],  (timestamps[photon2idx] - timestamps[photon1idx])// binning))
            #assert binidx-bin_offset >= 0, 'idx too small'
            #assert binidx-bin_offset < histogram.size, 'idx too big'
            if binidx-bin_offset < histogram.size and  binidx-bin_offset >= 0:
                histogram[binidx - bin_offset] += 1
            #print(f"Ph1/ph2 index: {photon1idx}, {photon2idx} Incremented histogram")
            photon2idx += 1

@numba.jit(nopython=True, nogil=False, parallel=False)
def fast_correlate_mode_symmetric(timestamps, detector, histogram, binning,tmax, mode = 'X01'):
    """Calculate un-normalized g2(tau) correlation function from timestamps.

    Time differences tau are
    binned into histogram array for tmin <= tau <= tmax. Returns None
    but increments (!) argument histogram with coincidences.

    Inspired by Picoharp_T2Analysis_parallel.py script written by Thomas Fink.

    Parameters
    ----------
    timestamps : array of uint32
        Contains timestamps, typically in units of some base resolution. But
        can have any units, binning is relative to the units in timestamps.
    detector : array of integer type
        Array of same length as timestamps, containing channel indices
        indicating which detector (0 or 1) each timestamp belongs to.
        are discarded.
    histogram : array of uint32
        Array to save histogram into, must have compatible length (...)
    binning : integer
        Histogram bin width, relative to the units of the timestamps.
        Example: binning=4 means that time differences 0,1,2,3 are summed
                 into the same bin.
    tmax : integer
        Maximum time delay tau the histogram should contain. Exclusive.
    mode : str
        Mode of correlation, 'X01' is default, meaning photon delay in channel 2 conditioned on the detection of photon in ch. 1
    """

    def ph1_condition(detector, mode):
        if mode =="X01": 
            if detector == 0: 
                return True 
        elif mode =="X10":
            if detector == 1:
                return True
        elif mode =="A0":
            if detector == 0:
                return True
        elif mode =="A1":
            if detector == 1:
                return True
        elif mode =="all":
            return True
        return False
        
    def ph2_condition(detector, mode):
        if mode =="X01": 
            if detector == 1: 
                return True 
        elif mode =="X10":
            if detector == 0:
                return True
        elif mode =="A0":
            if detector == 0:
                return True
        elif mode =="A1":
            if detector == 1:
                return True
        elif mode =="all":
            return True
        return False
    
    for photon1idx in range(timestamps.shape[0]):
        
        if not ph1_condition(detector[photon1idx], mode):
            continue
        
        # all time delays
        for photon2idx in range(timestamps.shape[0]):
            # symmetric maximum time exclusion
            if np.abs(timestamps[photon2idx] - timestamps[photon1idx]) < tmax*1.1:  
                # Check photon 2 condition
                if not ph2_condition(detector[photon2idx], mode):
                    continue

                binidx = (timestamps[photon2idx] - timestamps[photon1idx]) // binning + int(histogram.size/2)
                # if timestamps[photon2idx] - timestamps[photon1idx]<0:
                #     print(f"ph1 idx {photon1idx}, ph2 idx {photon2idx},  time diff {(timestamps[photon2idx] - timestamps[photon1idx])}, total bins {histogram.size}, assigned bin:  {binidx}")
                ###### 
                if binidx < histogram.size and  binidx >= 0:
                    histogram[binidx] += 1
                

def timestamps_select_channels(timestamps, detector, mode):
    if mode == "X01":
        ts1 = timestamps[detector == 0]   # reference (start)
        ts2 = timestamps[detector == 1]   # correlated (stop)
    elif mode == "X10":
        ts1 = timestamps[detector == 1]
        ts2 = timestamps[detector == 0]
    elif mode == "A0":
        ts1 = ts2 = timestamps[detector == 0]
    elif mode == "A1":
        ts1 = ts2 = timestamps[detector == 1]
    elif mode == "all":
        ts1 = ts2 = timestamps
    else:
        raise ValueError(f"Unknown mode {mode}")
    return ts1, ts2

#@numba.jit(nopython=True, nogil=False, parallel=False)
def fast_correlate_mode_symmetric_fast(ts1, ts2, tmax, hist, binning, block_size= 2000):
    
    # Estimate for a safe and efficient block size for 16 GB RAM with 7 parallel cores.
    max_ram_per_worker = 0.8e9 # 0.8 GB per worker
    chunksize = max(len(ts1), len(ts2))
    block_size = int(max_ram_per_worker / (8* chunksize))

    N1 = ts1.size
    bins = np.arange(-tmax, tmax+0.1, binning)
    nbins = len(bins)

    for i0 in range(0, N1, block_size):
        i1 = min(N1, i0 + block_size)
        block = ts1[i0:i1]
        diffs = block[:, None] - ts2[None, :]
        mask = (np.abs(diffs) < tmax) & (diffs != 0)
        sel = diffs[mask]
        h, _ = np.histogram(sel, range = (-tmax, tmax), bins=bins)
        hist += h.astype(hist.dtype, copy=False)



@numba.jit(nopython=True)
def min_index_ex(array, item):
    """Find index of first array element > item."""
    for idx, val in np.ndenumerate(array):
        if val > item:
            return idx[0]
'''
def correlate_hydra_ptu(input_filepath, chunksize=2**11, chunk_bins=None,
                    tau_min=0, tau_max=4*15000, tau_binning=4*4,
                    post_selection_bins=None, debug=False, log = None):
                        
    """Correlate photons from .ptu using fast_correlate and extract_records.
    
    Parameters
    ----------
    input_filepath : string
        Path to input file (file format .ptu from Picoharp GUI
                            or .b from correlation_measurement.py).
    chunksize : integer, optional
        The data from file is read and processed in chunks of this size.
        The default is 2**11 = 2048.   -- the longer the chunk size the more costly the correlation, but the larger the longest time delay that can be checked
    chunk_bins ...
        ...
    tau_min: integer, optional
        Lowest time delay in histogram in units of
        base resolution / tau_binning.
        The default is 0.
    tau_max: integer, optional
        Highest time delay in histogram in units of
        base resolution / tau_binning.
        Example: tau_max = 15000 = 60000 // 4 corresponds to 60 ns for
                 tau_binning = 4.
    tau_binning: integer, optional
        The histogram bin width in multiples of the timestamp base resoltion.
        Example: If base resolution is 4 ps, then tau_binning = 4
                 leads to bins of size 16 ps.
    post_selection_bins: array, optional
        Defines bin edges for post-selecting histograms according to the
        measurement time it took for each chunk to complete.
        The default is None, meaning that no post-selection is performed
        and the data from all chunks is summed up into one histogram.
        If bins are supplied, separate histograms will be accumulated from
        measurements with completion time within the corresponding bins.
        Measurements with time outside the supplied bins will be discarded.
    log: A log function (!) that prints on the client side.
    
    Returns
    -------
    histograms : TYPE
        DESCRIPTION.
    time_per_chunk : TYPE
        DESCRIPTION.
    [if debug]:
    also returns post_selection_indices
    Notes
    -----
    Ideal chunksize on lab pc for reading is about 2**17 but this
    could change when taking operations into account...
    Performance degraded a bit after introducing post-selection. There
    is probably some room to make that part faster also...
    """
    # Open file and read header metainformation
    extension = os.path.splitext(input_filepath)[1]
    if extension == '.ptu':
        inputfile, tags = read_ptu_header(input_filepath)
        recordType = tags["TTResultFormat_TTTRRecType"]
        if not recordType == rtHydraHarp2T2:
            raise TypeError('Wrong record type - expected Picoharp T2')
        numRecords = tags["TTResult_NumberOfRecords"]
        global_resolution = tags["MeasDesc_GlobalResolution"]  # 4e-12s = 4ps
        # We discard the last numrecords%chunksize entries here (fix later?)
        n_chunks = len(range(0, numRecords-chunksize, chunksize))
    elif extension == '.b':
        global_resolution = 1e-12  # is this 1ps for hydra??
        # Problem: this value is not really used, so need to patch code below!
        n_chunks = 0
        # Slow brute force solution: Go through the whole file and find length
        with open(input_filepath, "rb") as inputfile:
            try:
                while True:
                    data_dwords = np.frombuffer(inputfile.read(4*chunksize),
                                                dtype=np.uint32,
                                                count=chunksize)
                    n_chunks += 1
            except ValueError:
                pass
        inputfile = open(input_filepath, "rb")  # and reopen at beginning
    else:
        raise NotImplementedError('Input file type not supported.')
    tau_bins = np.arange(tau_min, tau_max, tau_binning, dtype=np.uint32)
    
    
    if post_selection_bins is None:
        histograms = np.zeros(tau_bins.size, dtype=np.uint32)
    else:
        histograms = np.zeros((tau_bins.size, post_selection_bins.size - 1),
                              dtype=np.uint32)
    # Inversely proportional to current countrate, quantity for post-selection
    time_per_chunk = np.zeros(n_chunks, dtype=np.uint64)  # dtype=np.uint32)
    if debug:
        post_selection_indices = np.zeros(n_chunks, dtype=np.int32)
        
        
        
        
    # Iterate through timestamp data in chunks (chunksize)
    for j in range(n_chunks):
        
        if log is not None and j%1e2 == 0:
            msg = f"\r[ANALYSIS] Progress: {j/n_chunks*100:.3f}% (chunks {j:d}/{n_chunks:d})"
            log(msg)
            
            
        # Load binary data into array of size chunksize (this is very fast)
        data_dwords = np.frombuffer(inputfile.read(4*chunksize),
                                    dtype=np.uint32, count=chunksize)
        # Check if chunk j is inside chunk_bins, if not: discard data
        if chunk_bins is not None:
            chunk_cond = (chunk_bins[:, 0] <= j) & (j < chunk_bins[:, 1])
            if not np.any(chunk_cond):
                time_per_chunk[j] = 0  # Inidcates discarded chunk
                continue
        # Extract channel and timestamp information (discarding markers)
        special, channels, wrapped_times = extract_records_hydra(data_dwords)
        channels, timestamps = unwrap_time_hydra(special, channels, wrapped_times)
        # Check current countrate and store histogram accordingly
        # Problem: this appears to overflow uint32 sometimes?
        # For now, we will just extend it to 64 bit but such
        # a time per chunk means >17 ms for one chunk: very low countrate.
        time_per_chunk[j] = timestamps[-1] - timestamps[0]  # base resoltution
        if post_selection_bins is None:
            # FastCorrelate will increment coincidence histogram directly
            fast_correlate(timestamps, channels, histograms,
                           tau_binning, tau_min, tau_max)
        else:
            countrate = chunksize / (time_per_chunk[j] * global_resolution)
            post_select = min_index_ex(post_selection_bins, countrate)
            if post_select is None or post_select == 0:
                if debug:
                    post_selection_indices[j] = -1 if post_select == 0 \
                                                   else -2
                continue  # outside post-selection range, discard data
            fast_correlate(timestamps, channels,
                           histograms[:, post_select - 1],
                           tau_binning, tau_min, tau_max)
            if debug:
                post_selection_indices[j] = post_select - 1
    inputfile.close()
    if log is not None:
        log("\n")
    if debug:
        return histograms, time_per_chunk, post_selection_indices
    else:
        return histograms, time_per_chunk
    '''


def get_countrates_chunk(folder, chunksize=2**11, chunk_bins=None,
                     log = None):
                        
    """ only extracts the count rate per chunk, for long-time-scale correlations 
    or a quick overview of measurement stability
    
    Parameters
    ----------
    input_filepath : string
        Path to input file (file format .ptu from Picoharp GUI
                            or .b from correlation_measurement.py).
    chunksize : integer, optional
        The data from file is read and processed in chunks of this size.
        The default is 2**11 = 2048.   -- the longer the chunk size the more costly the correlation, but the larger the longest time delay that can be checked
    chunk_bins ...
        ...
    log: A log function (!) that prints on the client side.
    
    Returns
    -------
    time_per_chunk : array

    Notes
    """

    # Open file and read header metainformation
    n_chunks = 0
    # Slow brute force solution: Go through the whole file and find length
    with open(os.path.join(folder,"chunk_merged.b"), "rb") as inputfile:
        try:
            while True:
                data_dwords = np.frombuffer(inputfile.read(4*chunksize),
                                            dtype=np.uint32,
                                            count=chunksize)
                n_chunks += 1
        except ValueError:
            pass
    inputfile = open(os.path.join(folder,"chunk_merged.b"), "rb")  # and reopen at beginning
    
    # Inversely proportional to current countrate, quantity for post-selection
    time_per_chunk = np.zeros(n_chunks, dtype=np.uint64)  
        
        
        
    # Iterate through timestamp data in chunks (chunksize)
    for j in range(n_chunks):
        
        if log is not None and j%1e4 == 0:
            msg = f"\r[ANALYSIS] Progress counting chunks: {j/n_chunks*100:.3f}% (chunks {j:d}/{n_chunks:d})"
            log(msg)
            
            
        # Load binary data into array of size chunksize (this is very fast)
        data_dwords = np.frombuffer(inputfile.read(4*chunksize),
                                    dtype=np.uint32, count=chunksize)
        # Check if chunk j is inside chunk_bins, if not: discard data
        if chunk_bins is not None:
            chunk_cond = (chunk_bins[:, 0] <= j) & (j < chunk_bins[:, 1])
            if not np.any(chunk_cond):
                time_per_chunk[j] = 0  # Inidcates discarded chunk
                continue
        # Extract channel and timestamp information (discarding markers)
        special, channels, wrapped_times = extract_records_hydra(data_dwords)
        channels, timestamps = unwrap_time_hydra(special, channels, wrapped_times)
        # Check current countrate and store histogram accordingly
        # Problem: this appears to overflow uint32 sometimes?
        # For now, we will just extend it to 64 bit but such
        # a time per chunk means >17 ms for one chunk: very low countrate.
        time_per_chunk[j] = timestamps[-1] - timestamps[0]  # base resoltution
        
    inputfile.close()
    if log is not None:
        log("\n")
    return time_per_chunk
    
    

def _process_chunk_range(args):
    """
    Worker: process a contiguous range of chunks and return:
      (start_chunk, end_chunk, hist_local_uint64, time_per_chunk_local_uint64)

    args:
      start_chunk, end_chunk, filepath, total_u32, chunksize,
      tau_min, tau_max, tau_binning,
      progress_step, progress_counter  (Manager.Value or None)
    """
    (start_chunk, end_chunk, filepath, total_u32, chunksize,
     tau_min, tau_max, tau_binning, mode,symmetric,
     progress_step, progress_counter) = args


    # Map once per worker. (If this ever trips 1455 on your system, fall back to open/seek/read.)
    mapped = np.memmap(filepath, mode="r", dtype=np.uint32, offset=0, shape=(total_u32,))

    if not symmetric: 
        n_bins = (tau_max - tau_min) // tau_binning 
    else:
        n_bins = (2*tau_max) // tau_binning 
        
    hist_local = np.zeros(n_bins, dtype=np.uint64)              # accumulator for this worker
    tpc_local  = np.zeros(end_chunk - start_chunk, dtype=np.uint64)

    processed_since_report = 0

    for j, chunk_idx in enumerate(range(start_chunk, end_chunk)):
        start = chunk_idx * chunksize
        stop  = min(start + chunksize, total_u32)
        if start >= stop:
            break

        data_dwords = mapped[start:stop]

        special, channels, wrapped = extract_records_hydra(data_dwords)
        channels, timestamps = unwrap_time_hydra(special, channels, wrapped)
        timestamps = timestamps.astype(np.int64)
        if timestamps.size >= 2:  ### THIS IS USING THE PREVIOUS UTILITY, FAST CORRELATE
            # Local temp histogram (uint32 is sufficient per chunk) -> add into uint64 accumulator
            hbuf = np.zeros(n_bins, dtype=np.uint32)
            if symmetric: 
                #fast_correlate_mode_symmetric(timestamps, channels, hbuf, tau_binning, tau_max, mode)
                t1, t2 = timestamps_select_channels(timestamps, channels, mode)
                fast_correlate_mode_symmetric_fast(t1,t2, tau_max, hbuf, tau_binning)
            else:
                fast_correlate_mode(timestamps, channels, hbuf, tau_binning, tau_min, tau_max, mode)
            # fast_correlate(timestamps, channels, hbuf, tau_binning, tau_min, tau_max)
            hist_local += hbuf  
            tpc_local[j] = np.uint64(timestamps[-1] - timestamps[0])
        else:
            tpc_local[j] = np.uint64(0)
        
        
        # Progress reporting to parent every 'progress_step' chunks
        processed_since_report += 1
        if progress_counter is not None and processed_since_report >= progress_step:
            try:
                # Manager.Value proxy is process-safe; no get_lock() exists/needed
                progress_counter.value += processed_since_report
            except Exception:
                pass
            processed_since_report = 0

    # flush any remaining progress
    if progress_counter is not None and processed_since_report:
        try:
            progress_counter.value += processed_since_report
        except Exception:
            pass

    del mapped
    return (start_chunk, end_chunk, hist_local, tpc_local)
    
    
def correlate_hydra_b_parallel(
    input_filepath,
    chunksize=2**17,
    chunk_bins=None,              # kept for signature symmetry (unused here)
    tau_min=0,
    tau_max=15000*4,
    tau_binning=4*4,
    post_selection_bins=None,     # not applied in this parallel path
    debug=False,
    log=None,
    max_workers=os.cpu_count()-1,
    progress_step=10,            # workers report every N chunks
    chunks_per_task=2000,          # size of each chunk-range batch per worker task
    mode = "X01",
    chunk_range = None, # tupel, indices of first and last chunk
    symmetric = False
):
    """
    Parallel correlation over .b files using chunk-range tasks.
    - Workers return ONE histogram and tpc array per range (dramatically less IPC).
    - Parent maintains a single uint64 accumulator and sums worker results as they complete.
    - Progress reported every `progress_step` chunks (from workers via Manager.Value).
    """

    # ---- sanitize ints (prevents '... got 2000000.0' errors) ----
    chunksize      = int(chunksize)
    tau_min        = int(tau_min)
    tau_max        = int(tau_max)
    tau_binning    = int(tau_binning)
    progress_step  = max(1, int(progress_step))
   

    # ---- file/plan ----
    filesize  = os.path.getsize(input_filepath)
    total_u32 = filesize // 4
    if total_u32 == 0:
        raise ValueError("Empty file.")
    n_chunks  = math.ceil(total_u32 / chunksize)

    if not symmetric: 
        n_bins    = (tau_max - tau_min) // tau_binning
    else: 
        n_bins = (2 * tau_max) // tau_binning

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)
    
    if chunks_per_task is None: 
        chunks_per_task = int(n_chunks/max_workers/10)
    chunks_per_task= max(1, int(chunks_per_task))

    if chunk_range is None: 
        chunk_range = (0, n_chunks)

    if log is not None:
        log(f"[ANALYSIS - {curtime()}] Parallel start: { (chunk_range[1]-chunk_range[0])} chunks ({n_chunks} total in file), {max_workers} workers\n")

    # ---- partition into contiguous ranges ----
   
    print(f"[ANALYSIS - {curtime()}] Processing chunk range ", chunk_range)
    ranges = []
    for start in range(*chunk_range, chunks_per_task):
        try: 
            end = min(start + chunks_per_task, chunk_range[1])
            ranges.append( (start, end) )
        except: 
            print(f"[ANALYSIS - {curtime()}] Likely have too large chunk size for too few overall counts: Either integrate longer or choose smaller chunk size.")


    # ---- shared progress counter via Manager (spawn-safe) ----
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    progress_counter = manager.Value('i', 0)
    stop_flag = threading.Event()

    # monitor thread: emits when the manager counter crosses multiples of progress_step
    def monitor_progress():
        last_reported = 0
        start_ts = time.time()
        while not stop_flag.is_set():
            try:
                done = int(progress_counter.value)
            except Exception:
                done = last_reported
            if done >= last_reported + progress_step or done >= chunk_range[1]-chunk_range[0]:
                last_reported = done
                pct = 100.0 * min(done, (chunk_range[1]-chunk_range[0])) /(chunk_range[1]-chunk_range[0])
                elapsed = (time.time() - start_ts) / 60.0
                msg = f"\r[ANALYSIS - {curtime()}] Progress: {min(done,(chunk_range[1]-chunk_range[0]))}/{(chunk_range[1]-chunk_range[0])} ({pct:.1f}%) — {elapsed:.1f} min"
                if log is not None:
                    log(msg)
                else:
                    print(msg, end="")
                if done >= (chunk_range[1]-chunk_range[0]):
                    break
            time.sleep(0.5)
        if log is not None:
            log("\n")
        else:
            print()

    monitor = threading.Thread(target=monitor_progress, daemon=True)
    monitor.start()

    # ---- parent accumulators (bounded memory) ----
    hist64_accum   = np.zeros(n_bins, dtype=np.uint64)
    time_per_chunk = np.zeros(n_chunks, dtype=np.uint64)

    # ---- submit tasks ----
    tasks = [
        (start, end, input_filepath, total_u32, chunksize,
         tau_min, tau_max, tau_binning, mode,symmetric,
         progress_step, progress_counter)
        for (start, end) in ranges
    ]

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = [pool.submit(_process_chunk_range, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            if isinstance(res, tuple) and len(res) >= 3 and isinstance(res[2], str):
                # optional: if you change worker to return ("ERROR", ... , "msg")
                if log: log(f"[ERROR - worker] {res[2]}")
                else: print(f"[ERROR - worker] {res[2]}")
                continue

            start, end, hist_local, tpc_local = res
            # bounded reduce (just += one array)
            hist64_accum += hist_local
            # stitch time_per_chunk into the right span
            span = end - start
            time_per_chunk[start:end] = tpc_local[:span]

    stop_flag.set()
    monitor.join()
    try:
        manager.shutdown()
    except Exception:
        pass

    # ---- finalize result dtype ----
    # clamp to uint32 in case any bin exceeds 4,294,967,295
    hist = np.asarray(np.minimum(hist64_accum, np.uint64(np.iinfo(np.uint32).max)), dtype=np.uint32)

    if debug:
        return hist, time_per_chunk, n_chunks
    else:
        return hist, time_per_chunk        
        
def window_mean(a, window_size, mode='same'):
    window = np.ones(window_size)/float(window_size)
    return np.convolve(a, window, mode)

# def correlate_lasersweep(input_filepath, post_selection_bins, Sweep_npoints,
                         # Sweep_dt, chunksize=2**11, 
                         # tau_min=0, tau_max=4*15000,tau_binning=4*4):
    # """Correlate photons from .ptu of a measurement where the laser wavelength was swept.
    
    # Based on correlate_hydra_ptu.
    # Assumes the laser sweep starts from red side and alternates the detuning 
    # after each detected maxima. 
    # Creats two set of histograms per postselection bin.
    # """
    # global_resolution = 1e-12
    # n_chunks = 0
    ##### Read out the number of chunks
    # with open(input_filepath, 'rb') as inputfile:
        # try:
            # while True:
                # data_dwords = np.frombuffer(inputfile.read(4*chunksize),
                                            # dtype=np.uint32,
                                            # count=chunksize)
                # n_chunks += 1
        # except ValueError:
            # pass
    ##### Determin detuing side of each chunk 1=red; 0=blue
    # inputfile = open(input_filepath, 'rb')
    #### Get countrate first
    # try:
        # time_per_chunk = np.zeros(n_chunks, dtype=np.uint64)
        # for j in range(n_chunks):
            # data_dwords = np.frombuffer(inputfile.read(4*chunksize),
                                        # dtype=np.uint32, count=chunksize)
            # special, channels, wrapped_times = extract_records_hydra(data_dwords)
            # channels, timestamps = unwrap_time_hydra(special, channels, wrapped_times)
            # time_per_chunk[j] = timestamps[-1] - timestamps[0]
    # except:
        # inputfile.close()
        # raise
    # inputfile.close()
    # countrate = chunksize/(time_per_chunk*global_resolution)
    # countrate_avg = window_mean(countrate, 1000)
    
    # tpc_mean = np.mean(time_per_chunk)*1e-12
    # trep = Sweep_npoints*Sweep_dt
    # avgrep = trep/tpc_mean/2
    # maxidx = find_peaks(countrate_avg, height = np.mean(countrate_avg),
                        # width = avgrep*0.1, distance=avgrep*0.5)[0]
    # avgmaxdiff = int(np.round(np.mean(np.diff(maxidx))))
    # newmaxidx = []
    # for i in range(len(maxidx)):
        # if i == 0:
            # newmaxidx.append(maxidx[0])
        # elif np.diff(maxidx)[i-1] > 1.5*avgmaxdiff:
            # newmaxidx.append(maxidx[i-1]+avgmaxdiff)
        # else:
            # newmaxidx.append(maxidx[i])
    # width = avgrep*2/3
    # Maxidx = []
    # for i, idx in enumerate(maxidx):
        # if idx-avgrep/2<=0:
            # continue
        # yidx = [idx-int(width/2), idx+int(width/2)]
        # if yidx[1] > len(countrate_avg):
            # break
        # x = np.linspace(0,yidx[1]-yidx[0], yidx[1]-yidx[0])
        # y = countrate_avg[yidx[0]:yidx[1]]
        # param = np.polyfit(x, y, 2)
        # x0 = -param[1]/2/param[0]
        # Maxidx.append(int(x0+yidx[0]))
    # if np.max(np.diff(Maxidx)) > 1.5*avgmaxdiff:
        # try:
            # print('failed to find all maximas')
            # np.savetxt(input_filepath[:-2]+ '_Peakselection.txt', np.diff(Maxidx))
        # except:
            # print('failed to find all maximas')
        # plt.figure(figsize=(12,4.5))  
        # plt.plot(np.diff(Maxidx),'.')
        # ax = plt.gca()
        # ax.axhline(np.mean(np.diff(maxidx))+np.mean(np.diff(maxidx))/2,
                   # linestyle='--', color='k')
        # ax.set_xlabel('Index')
        # ax.set_ylabel('Maxidx difference')
        # plt.savefig(input_filepath[:-2]+ '_Peakselection.png', dpi=300)
    ##### Would be good to come up with some safety check for all the maximas♥
        
    ### calculate histogramms and assign them to the given detuning
    # tau_bins = np.arange(tau_min, tau_max, tau_binning, dtype=np.uint32)
    # histograms = np.zeros((tau_bins.size, post_selection_bins.size-1, 2),
                          # dtype=np.uint32)
    # inputfile = open(input_filepath, 'rb')
    # SweepIdx = 1
    # Side = 0
    # for j in range(n_chunks):
        # data_dwords = np.frombuffer(inputfile.read(4*chunksize),
                                    # dtype=np.uint32, count=chunksize)
        # if j <= Maxidx[0]:
            # continue
        # if j == Maxidx[SweepIdx]:
            # SweepIdx += 1
            # Side += 1
            # Side = Side%2
        # if SweepIdx >=len(Maxidx):
            # break
        # special, channels, wrapped_times = extract_records_hydra(data_dwords)
        # channels, timestamps = unwrap_time_hydra(special, channels, wrapped_times)
        # post_select_idx = min_index_ex(post_selection_bins, countrate[j])
        # if countrate[j] >= post_selection_bins[-1]:
            # post_select_idx = len(post_selection_bins)-1
        # if post_select_idx == 0:
            # continue
        # try:
            # fast_correlate(timestamps, channels,
                           # histograms[:, post_select_idx-1, Side],
                           # tau_binning, tau_min, tau_max)
        # except TypeError:
            # print(countrate[j], post_selection_bins[-1], post_select_idx-1)
            # raise
    # inputfile.close()
    # return histograms, time_per_chunk
        

def plot_correlations(hist, tpc, plot_path, params={}):
    """Plot to inspect process_pico_ptu outputs."""
    if params.get('plot_normalize'):
        h_plot = hist / np.mean(hist, axis=0)
    else:
        h_plot = hist
    global_resolution = params.get('global_resolution', 4e-12)
    tau_min = params.get('tau_min', 0)
    tau_max = params.get('tau_max', 15000)
    tau_binning = params.get('tau_binning', 4)
    tau_bins = np.arange(tau_min, tau_max, tau_binning, dtype=np.uint32)\
        * global_resolution
    params['tau_bins'] = tau_bins
    chunksize = params.get('chunksize')
    # Version with multiple post_selection windows not implemented yet.
    """
    if post_selection_bins is not None:
        post_selection_labels = [r'[{:1.2f}, {:1.2f}) MHz'.format(
                                 post_selection_bins[i]/1e6,
                                 post_selection_bins[i+1]/1e6)
                                 for i in range(post_selection_bins.size-1)]
    """
    times_per_chunk = np.array(tpc) * global_resolution
    countrate = np.divide(chunksize, times_per_chunk,
                          out=np.zeros_like(times_per_chunk),
                          where=times_per_chunk != 0)
    countrate[np.isnan(countrate)] = 0
    f, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].set_title('Correlations {}'.format(params.get('filename')))
    if len(h_plot.shape) == 1:
        ax[0].plot(tau_bins*1e9, h_plot, 'k.-', label='All countrates')
    """
    elif len(h.shape) == 2:
        for idx in range(h.shape[1]):
            ax[0].plot(tau_bins*1e9, h_plot[:, idx], '.-',
                       label=post_selection_labels[idx])
    """
    ax[0].legend(loc='best')
    ax[0].set_xlabel(r'$\tau$ (ns)')
    ax[0].set_ylabel('Correlations')
    # ax[0].set_xlim(29, 38)
    ax[0].grid()
    ax[1].plot(times_per_chunk*1e3, 'b.-', lw=1, ms=1, alpha=0.4)
    ax[1].set_ylabel('Time per chunk (ms)', color='b')
    ax[1].set_xlabel('Chunk number')
    ax[1].set_title('Chunksize {}'.format(chunksize))
    axt = ax[1].twinx()
    axt.plot(countrate, 'r.-', lw=1, ms=1, alpha=0.8)
    """
    if len(h.shape) == 2:
        axt.hlines(post_selection_bins, 0, countrate.size*1.05,
                   lw=1, alpha=1)
    if post_selection_bins is not None:
        axt.set_ylim(0, 1.05 * max(max(countrate),
                                   max(post_selection_bins)))
    """
    axt.set_xlim(0, countrate.size*1.05)
    axt.set_ylabel('Chunk countrate (Hz)', color='r')
    f.tight_layout()
    mpl.rcParams['agg.path.chunksize'] = 10000
    f.savefig(plot_path + '.png', dpi=300)
    
def pickle_save(obj, file_path):
    """Save object obj that can be pickled at file_path."""
    if os.path.splitext(file_path)[-1] == '.p':
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    else:
        with open(file_path + '.p', 'wb') as file:
            pickle.dump(obj, file)
            
def correlate_parallel(argument):
    proc = Process(target=correlate_timestamps, args=(argument,))
    proc.start()

    

def correlate_timestamps_Kilian(params, parallel = False):
    """
    For remote analysis, returns histogram and time per chunk (reading for the average count rate during a chunk).
    
    
   
    Parameters
    ----------
    
    params: Dictionary with the following elements: 
    
    
    input_filepath : string
        Path to input file (.b from run_parallel method in hydraharp_srv.py).
    data_path : 
    chunksize : integer, optional
        The data from file is read and processed in chunks of this size.
        The default is 2**11 = 2048.   -- the longer the chunk size the more costly the correlation, but the larger the longest time delay that can be checked
    chunk_bins ...
        ...
    tau_min: integer, optional
        Lowest time delay in histogram in units of
        base resolution / tau_binning.
        The default is 0.
    tau_max: integer, optional
        Highest time delay in histogram in units of
        base resolution / tau_binning.
        Example: tau_max = 15000 = 60000 // 4 corresponds to 60 ns for
                 tau_binning = 4.
    tau_binning: integer, optional
        The histogram bin width in multiples of the timestamp base resoltion.
        Example: If base resolution is 4 ps, then tau_binning = 4
                 leads to bins of size 16 ps.
    post_selection_bins: array, optional
        Defines bin edges for post-selecting histograms according to the
        measurement time it took for each chunk to complete.
        The default is None, meaning that no post-selection is performed
        and the data from all chunks is summed up into one histogram.
        If bins are supplied, separate histograms will be accumulated from
        measurements with completion time within the corresponding bins.
        Measurements with time outside the supplied bins will be discarded.
    debug : Bool, enable more status prints during run.
    log : Log object that exists in analysis_srv.py and writes to the remote output
    mode: str, default "X01"
        The mode in which correlations are computed, can be X01 or X10 (the two cross-correlation modes), or A0/A1, 
        the two autocorrelation modes, or 'all' in which case it doesn't matter where the photons arrive
    Returns
    -------
    histograms : TYPE
        DESCRIPTION.
    time_per_chunk : TYPE
        DESCRIPTION.
    
    Notes
    -----
    """
    
    
    # Define default parameters
    dp = {  # default parameters
        'input_file': None,  # the user must supply this one
        'data_path': None,  # the user may want to supply extra info
        'chunksize': 2**10,  # 1024
        'chunk_bins': None,  # All data along physical time axis is included
        'tau_min': 0,  #
        'tau_max': 15000,  # in units of pico base resolution
        'tau_binning': 4,  # into 4 * 4ps = 16 ps bins
        'post_selection_bins': np.linspace(0, 4, 81) * 1e6,  # MHz
        'debug': False,  #
        'global_resolution': 1e-12,  # fixed for hydra
        'log' : None,
        'mode' : "X01",
        'chunk_range': None,
        'symmetric':False
        }
    dp.update(params)  # update user supplied params
    
    print(f"[ANALYSIS - {curtime()}] Starting correlate_timestamps")
    
    
    # Analyze the timestamps and form correlation histograms
    start_time = time.time()
    
    input_file = dp['input_file']
    print(f'[ANALYSIS - {curtime()}] Correlate timestampfile: {input_file}')

    dp['tau_max'] = 4 * dp['tau_max'] # base resolution? 
    dp['tau_binning'] = 4 * dp['tau_binning']
    if dp['debug']: 
        print("Input file: ", dp['input_file'])
    
    if parallel:
        out = correlate_hydra_b_parallel(
            input_filepath=dp['input_file'],
            chunksize=dp['chunksize'],
            chunk_bins=dp['chunk_bins'],
            tau_min=dp['tau_min'],
            tau_max=dp['tau_max'],
            tau_binning=dp['tau_binning'],
            post_selection_bins=dp['post_selection_bins'],
            debug=dp['debug'],
            log=dp['log'],
            mode = dp['mode'],
            chunk_range = dp['chunk_range'],
            symmetric = dp['symmetric']
        )
    else:
        # if dp['mode'] != "X01":
        #     raise ValueError("In non-parallel architecture only X01 is implemented!")
        # out = correlate_hydra_ptu(
        #     dp['input_file'],
        #     chunksize=dp['chunksize'],
        #     chunk_bins=dp['chunk_bins'],
        #     tau_min=dp['tau_min'],
        #     tau_max=dp['tau_max'],
        #     tau_binning=dp['tau_binning'],
        #     post_selection_bins=dp['post_selection_bins'],
        #     debug=dp['debug'],
        #     log=dp['log']
        # )
        raise "SHOULD ONLY USE PARALLEL MODE!"
    runtime = ((time.time()-start_time)/60)
    print(f'[ANALYSIS - {curtime()}] Correlated data in {runtime:.2f} min')
    
    
    if dp['debug']:
        h, c, post_selection_indices = out
        return h,c, post_selection_indices # histogram, time per chunk, post_selection_indices
    else: 
        h,c = out
        return h,c # histogram, time per chunk
    
    
    
   
def merge_chunk_files(filename_prefix):
    """
    Merge all chunk files starting with filename_prefix into a single file
    called 'chunk_merged.b' in the 'merged_chunks' subfolder. Overwrites existing file.
    Returns the merged file path and verifies integrity by comparing hashes.
    """
    folder = os.path.dirname(filename_prefix)
    if os.path.dirname(filename_prefix) is not None:
        os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)
    base_prefix = os.path.basename(filename_prefix)

    # Find all matching chunk files (fix: use base_prefix to match properly)
    chunk_files = glob.glob(os.path.join(folder, f"{base_prefix}_chunk*.b"))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found with prefix '{filename_prefix}'.")

    # Sort files by chunk index
    def chunk_index(filename):
        base = os.path.basename(filename)
        idx_part = base.split("_chunk")[-1].split(".b")[0]
        return int(idx_part)

    chunk_files.sort(key=chunk_index)

    merged_file_path = os.path.join(folder, "chunk_merged.b")

    # Initialize a hash object to verify merged content
    merged_hash = hashlib.sha256()

    with open(merged_file_path, 'wb') as outfile:
        for chunk_file in chunk_files:
            with open(chunk_file, 'rb') as infile:
                while True:
                    buf = infile.read(1024 * 1024)  # 1 MB chunks
                    if not buf:
                        break
                    outfile.write(buf)
                    merged_hash.update(buf)

    # Verify integrity: recompute hash from merged file and compare
    verify_hash = hashlib.sha256()
    with open(merged_file_path, 'rb') as merged:
        for block in iter(lambda: merged.read(1024 * 1024), b""):
            verify_hash.update(block)

    if merged_hash.digest() != verify_hash.digest():
        raise ValueError("Integrity check failed: merged file does not match input chunks.")

    total_input_size = sum(os.path.getsize(f) for f in chunk_files)
    merged_size = os.path.getsize(merged_file_path)
    print(total_input_size,merged_size)
    if total_input_size != merged_size:
        # raise Warning("Merged file size does not match total input size.") #ValueError("Merged file size does not match total input size.")
        print("WARNING: Merged file size does not match total input size.")
    #return merged_file_path

def correlate_timestamps(arguments):
    correlate_parallel(arguments)
    
def create_new_folder(folder_name):
    try:
        print('Create a new folder for the measurement.')
        print(folder_name)
        if folder_name is not None:
            os.makedirs(folder_name, exist_ok=True)
    except:
        print('New folder was not created')

def look_for_files(filename_prefix):
    return sorted(glob.glob(f"{filename_prefix}_chunk*.b"))