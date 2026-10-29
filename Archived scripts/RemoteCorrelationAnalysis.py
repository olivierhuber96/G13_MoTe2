# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:07:15 2023
Copy the necessary functions from G8's correlation_analysis.py such that it can be 
called by the Hydraharp_srv to convert the raw T2 files into histograms. 
@author: GM
"""
import numpy as np
import time
import datetime
import os
import struct
import numba
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks
from multiprocessing import Process
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
            if detector[photon2idx] == 0:
                binidx = ((timestamps[photon2idx] - timestamps[photon1idx])
                          // binning)
                assert binidx-bin_offset >= 0, 'idx too small'
                assert binidx-bin_offset < histogram.size, 'idx too big'
                histogram[binidx - bin_offset] += 1
            photon2idx += 1

@numba.jit(nopython=True)
def min_index_ex(array, item):
    """Find index of first array element > item."""
    for idx, val in np.ndenumerate(array):
        if val > item:
            return idx[0]

def correlate_hydra_ptu(input_filepath, chunksize=2**11, chunk_bins=None,
                    tau_min=0, tau_max=4*15000, tau_binning=4*4,
                    post_selection_bins=None, debug=False):
    """Correlate photons from .ptu using fast_correlate and extract_records.
    
    Parameters
    ----------
    input_filepath : string
        Path to input file (file format .ptu from Picoharp GUI
                            or .b from correlation_measurement.py).
    chunksize : integer, optional
        The data from file is read and processed in chunks of this size.
        The default is 2**11 = 2048.
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
    
    Returns
    -------
    histograms : TYPE
        DESCRIPTION.
    time_per_chunk : TYPE
        DESCRIPTION.
    
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
    if debug:
        return histograms, time_per_chunk, post_selection_indices
    return histograms, time_per_chunk

def window_mean(a, window_size, mode='same'):
    window = np.ones(window_size)/float(window_size)
    return np.convolve(a, window, mode)

def correlate_lasersweep(input_filepath, post_selection_bins, Sweep_npoints,
                         Sweep_dt, chunksize=2**11, 
                         tau_min=0, tau_max=4*15000,tau_binning=4*4):
    """Correlate photons from .ptu of a measurement where the laser wavelength was swept.
    
    Based on correlate_hydra_ptu.
    Assumes the laser sweep starts from red side and alternates the detuning 
    after each detected maxima. 
    Creats two set of histograms per postselection bin.
    """
    global_resolution = 1e-12
    n_chunks = 0
    # Read out the number of chunks
    with open(input_filepath, 'rb') as inputfile:
        try:
            while True:
                data_dwords = np.frombuffer(inputfile.read(4*chunksize),
                                            dtype=np.uint32,
                                            count=chunksize)
                n_chunks += 1
        except ValueError:
            pass
    # Determin detuing side of each chunk 1=red; 0=blue
    inputfile = open(input_filepath, 'rb')
    # Get countrate first
    try:
        time_per_chunk = np.zeros(n_chunks, dtype=np.uint64)
        for j in range(n_chunks):
            data_dwords = np.frombuffer(inputfile.read(4*chunksize),
                                        dtype=np.uint32, count=chunksize)
            special, channels, wrapped_times = extract_records_hydra(data_dwords)
            channels, timestamps = unwrap_time_hydra(special, channels, wrapped_times)
            time_per_chunk[j] = timestamps[-1] - timestamps[0]
    except:
        inputfile.close()
        raise
    inputfile.close()
    countrate = chunksize/(time_per_chunk*global_resolution)
    countrate_avg = window_mean(countrate, 1000)
    
    tpc_mean = np.mean(time_per_chunk)*1e-12
    trep = Sweep_npoints*Sweep_dt
    avgrep = trep/tpc_mean/2
    maxidx = find_peaks(countrate_avg, height = np.mean(countrate_avg),
                        width = avgrep*0.1, distance=avgrep*0.5)[0]
    avgmaxdiff = int(np.round(np.mean(np.diff(maxidx))))
    newmaxidx = []
    for i in range(len(maxidx)):
        if i == 0:
            newmaxidx.append(maxidx[0])
        elif np.diff(maxidx)[i-1] > 1.5*avgmaxdiff:
            newmaxidx.append(maxidx[i-1]+avgmaxdiff)
        else:
            newmaxidx.append(maxidx[i])
    width = avgrep*2/3
    Maxidx = []
    for i, idx in enumerate(maxidx):
        if idx-avgrep/2<=0:
            continue
        yidx = [idx-int(width/2), idx+int(width/2)]
        if yidx[1] > len(countrate_avg):
            break
        x = np.linspace(0,yidx[1]-yidx[0], yidx[1]-yidx[0])
        y = countrate_avg[yidx[0]:yidx[1]]
        param = np.polyfit(x, y, 2)
        x0 = -param[1]/2/param[0]
        Maxidx.append(int(x0+yidx[0]))
    if np.max(np.diff(Maxidx)) > 1.5*avgmaxdiff:
        try:
            print('failed to find all maximas')
            np.savetxt(input_filepath[:-2]+ '_Peakselection.txt', np.diff(Maxidx))
        except:
            print('failed to find all maximas')
    #     plt.figure(figsize=(12,4.5))  
    #     plt.plot(np.diff(Maxidx),'.')
    #     ax = plt.gca()
    #     ax.axhline(np.mean(np.diff(maxidx))+np.mean(np.diff(maxidx))/2,
    #                linestyle='--', color='k')
    #     ax.set_xlabel('Index')
    #     ax.set_ylabel('Maxidx difference')
    #     plt.savefig(input_filepath[:-2]+ '_Peakselection.png', dpi=300)
    # Would be good to come up with some safety check for all the maximas♥
        
    # calculate histogramms and assign them to the given detuning
    tau_bins = np.arange(tau_min, tau_max, tau_binning, dtype=np.uint32)
    histograms = np.zeros((tau_bins.size, post_selection_bins.size-1, 2),
                          dtype=np.uint32)
    inputfile = open(input_filepath, 'rb')
    SweepIdx = 1
    Side = 0
    for j in range(n_chunks):
        data_dwords = np.frombuffer(inputfile.read(4*chunksize),
                                    dtype=np.uint32, count=chunksize)
        if j <= Maxidx[0]:
            continue
        if j == Maxidx[SweepIdx]:
            SweepIdx += 1
            Side += 1
            Side = Side%2
        if SweepIdx >=len(Maxidx):
            break
        special, channels, wrapped_times = extract_records_hydra(data_dwords)
        channels, timestamps = unwrap_time_hydra(special, channels, wrapped_times)
        post_select_idx = min_index_ex(post_selection_bins, countrate[j])
        if countrate[j] >= post_selection_bins[-1]:
            post_select_idx = len(post_selection_bins)-1
        if post_select_idx == 0:
            continue
        try:
            fast_correlate(timestamps, channels,
                           histograms[:, post_select_idx-1, Side],
                           tau_binning, tau_min, tau_max)
        except TypeError:
            print(countrate[j], post_selection_bins[-1], post_select_idx-1)
            raise
    inputfile.close()
    return histograms, time_per_chunk
        

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

def correlate_timestamps(params):
    """Worker for subprocess, performing timestamp correlation.

    Supply params dictionary with arguments for correlta_ptu,
    it requires at least an input file path to locate the timestamp data.
    """
    print('Starting correlate_timestamps') #Olivier debugging
    # dp = {  # default parameters
        # 'input_file': None,  # the user must supply this one
        # 'data_path': None,  # the user may want to supply extra info
        # 'plot_path': None,  #
        # 'chunksize': 2**10,  # 1024
        # 'chunk_bins': None,  # All data along physical time axis is included
        # 'tau_min': 0,  #
        # 'tau_max': 15000,  # in units of pico base resolution
        # 'tau_binning': 4,  # into 4 * 4ps = 16 ps bins
        # 'post_selection_bins': np.linspace(0, 4, 81) * 1e6,  # MHz
        # 'debug': False,  #
        # 'global_resolution': 1e-12,  # fixed for hydra
        # 'plot_normalize': False,
        # 'laser_sweep': False,#
        # }
    # dp.update(params)  # update user supplied params
    # # Analyze the timestamps and form correlation histograms
    # start_time = time.time()
    # print('Correlate timestampfile: {:s}, start time: {:s}'.format(dp['input_file'], datetime.datetime.now().strftime('%H:%M:%S')))
    # # Patching up for the transition pico to hydra... :-S
    # dp['tau_max'] = 4 * dp['tau_max']
    # dp['tau_binning'] = 4 * dp['tau_binning']
    # if 1:#dp['laser_sweep'] == False:
        # print(dp['input_file']) #added by Olivier on 12.09.2025 to debug
        # h, c = correlate_hydra_ptu(dp['input_file'], chunksize=dp['chunksize'],
                                   # chunk_bins=dp['chunk_bins'],
                                   # tau_min=dp['tau_min'], tau_max=dp['tau_max'],
                                   # tau_binning=dp['tau_binning'],
                                   # post_selection_bins=dp['post_selection_bins'],
                                   # debug=dp['debug'])
    # else:
        # h, c = correlate_lasersweep(dp['input_file'], dp['post_selection_bins'],
                                    # dp['Sweep_npoints'], dp['Sweep_dt'],
                                    # chunksize=dp['chunksize'],
                                    # tau_min=dp['tau_min'], tau_max=dp['tau_max'],
                                    # tau_binning=dp['tau_binning'])

    # print('Correlated data in {:1.2f} min'.format((time.time()-start_time)/60))
    # dp['histograms'] = h
    # dp['time_per_chunk'] = c
    # # Plot summed histograms for rough inspection
    # dp['filename'] = os.path.basename(dp['input_file'])
    # plot_path = os.path.join(dp['plot_path'], dp['filename'])
    # if np.sum(dp['post_selection_bins']) == None:
        # plot_correlations(h, c, plot_path, params=dp)
    # else:
        # hist_summed = np.sum(h, axis=1)
        # plot_correlations(hist_summed, c, plot_path, params=dp)
    # # Save data somewhere
    # pickle_save(dp, os.path.join(dp['data_path'], dp['filename']))
    print('Finished correlate_timestamps.')
    
def correlate_timestamps_Oli(params):
    """Worker for subprocess, performing timestamp correlation.

    Supply params dictionary with arguments for correlta_ptu,
    it requires at least an input file path to locate the timestamp data.
    """
    print('Starting correlate_timestamps') #Olivier debugging
    dp = {  # default parameters
        'input_file': None,  # the user must supply this one
        'data_path': None,  # the user may want to supply extra info
        'plot_path': None,  #
        'chunksize': 2**10,  # 1024
        'chunk_bins': None,  # All data along physical time axis is included
        'tau_min': 0,  #
        'tau_max': 15000,  # in units of pico base resolution
        'tau_binning': 4,  # into 4 * 4ps = 16 ps bins
        'post_selection_bins': np.linspace(0, 4, 81) * 1e6,  # MHz
        'debug': False,  #
        'global_resolution': 1e-12,  # fixed for hydra
        'plot_normalize': False,
        'laser_sweep': False,#
        }
    dp.update(params)  # update user supplied params
    # Analyze the timestamps and form correlation histograms
    start_time = time.time()
    print('Correlate timestampfile: {:s}, start time: {:s}'.format(dp['input_file'], datetime.datetime.now().strftime('%H:%M:%S')))
    # Patching up for the transition pico to hydra... :-S
    dp['tau_max'] = 4 * dp['tau_max']
    dp['tau_binning'] = 4 * dp['tau_binning']
    if 1:#dp['laser_sweep'] == False:
        print(dp['input_file']) #added by Olivier on 12.09.2025 to debug
        h, c = correlate_hydra_ptu(dp['input_file'], chunksize=dp['chunksize'],
                                   chunk_bins=dp['chunk_bins'],
                                   tau_min=dp['tau_min'], tau_max=dp['tau_max'],
                                   tau_binning=dp['tau_binning'],
                                   post_selection_bins=dp['post_selection_bins'],
                                   debug=dp['debug'])
    else:
        h, c = correlate_lasersweep(dp['input_file'], dp['post_selection_bins'],
                                    dp['Sweep_npoints'], dp['Sweep_dt'],
                                    chunksize=dp['chunksize'],
                                    tau_min=dp['tau_min'], tau_max=dp['tau_max'],
                                    tau_binning=dp['tau_binning'])

    print('Correlated data in {:1.2f} min'.format((time.time()-start_time)/60))
    dp['histograms'] = h
    dp['time_per_chunk'] = c
    # Plot summed histograms for rough inspection
    dp['filename'] = os.path.basename(dp['input_file'])
    plot_path = os.path.join(dp['plot_path'], dp['filename'])
    if np.sum(dp['post_selection_bins']) == None:
        plot_correlations(h, c, plot_path, params=dp)
    else:
        hist_summed = np.sum(h, axis=1)
        plot_correlations(hist_summed, c, plot_path, params=dp)
    # Save data somewhere
    pickle_save(dp, os.path.join(dp['data_path'], dp['filename']))
    print('Finished correlate_timestamps.')
    
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
        raise ValueError("Merged file size does not match total input size.")
    
    #return merged_file_path

def correlate_timestamps(arguments):
    correlate_parallel(arguments)
    
def correlate_timestamps_seq(arguments):
    c_timestamps(arguments)

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