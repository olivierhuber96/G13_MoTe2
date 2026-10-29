# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 09:55:01 2017

@author: Thomas Fink, Quantum Photonics Group, ETH Zurich
modified by GM Schnüringer, Olivier Huber & Kilian Kuhlbrodt
"""

#from __future__ import division,print_function,unicode_literals,absolute_import

import ctypes
import numpy as np
import functools
import logging
import sys
import itertools
import Pyro4
import Pyro4.naming
import time
from time import gmtime, strftime
def curtime():
    return strftime("UTC %Y-%m-%d %H:%M:%S", gmtime())

import shutil
# from RemoteCorrelationAnalysis import correlate_parallel, merge_chunk_files, create_new_folder, look_for_files, correlate_timestamps, correlate_timestamps_Oli
from RemoteCorrelationAnalysis_edited_Kilian import correlate_parallel, merge_chunk_files, create_new_folder, look_for_files, correlate_timestamps, correlate_timestamps_Kilian
import glob
import threading 
from collections import deque
import os
import hashlib
import datetime

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}
Pyro4.config.SERVERTYPE = "thread"         
Pyro4.config.COMMTIMEOUT = 0     # 0 = no per-request timeout
    

############################## ALLOW PYTHON TO TALK TO THE C API FROM HH
class FunctionHelper(object):
    '''
    Goal: Wrap the Hydra Harp DLL (hhlib.dll or hhlib64.dll on 64 bit systems) C API functions in python methods so that we can call the python methods
    of an object of the class hydraharpsrv and forward this to the hydra DLL.
    Prepends "HH_" to function names fun that can then be looked up as HH_fun in the DLL.
    '''
    def __init__(self,
                 library,
                 prefix,
                 return_type,
                 return_handler, # will raise exception on non-zero return
                 DevIdx,
                 ):

        self._library = library
        self._prefix = prefix
        self._return_type = return_type
        self._return_handler = return_handler
        self._DevIdx = DevIdx
        
    def wrap(self, *sig):
        
        def decorator(fun, *args, **kwargs):
            libfun = getattr(self._library, self._prefix+fun.__name__)
            libfun.restype = self._return_type
            libfun.argtypes = (ctypes.c_int,) + sig

            @functools.wraps(fun)
            def wrapper(*args, **kwargs):

                if self._return_handler is not None:

                    def DevIdxWrap(*args, **kwargs):
                        self._return_handler(libfun(self._DevIdx,
                                                    *args, **kwargs))

                    return fun(DevIdxWrap, *args, **kwargs)

                else:

                    def DevIdxWrap(*args, **kwargs):
                        return libfun(self._DevIdx, *args, **kwargs)

                    return fun(DevIdxWrap, *args, **kwargs)

            return staticmethod(wrapper)

        return decorator




############################## CLASS WITH THE HYDRA UTILITIES
@Pyro4.expose
class hydraharpsrv(object):

    def __init__(self):

        self._logger = logging.getLogger(self.__class__.__name__)
        self._log = deque(maxlen=10000)
        self._cv = threading.Condition()
        self._hh_lock = threading.Lock()

    def test(self):
        print('a')
        return 'a'


    ############################## MAKE THE DLL WRAPPER
    global _hhlib
    if sys.maxsize > 2**32:
        _hhlib = ctypes.windll.LoadLibrary('hhlib64')
    else:
        _hhlib = ctypes.windll.LoadLibrary('hhlib')

    def GetLibraryVersion(self):
        buf = ctypes.create_string_buffer(b'',8)
        ret = _hhlib.HH_GetLibraryVersion(ctypes.byref(buf))
        if ret:
            raise RuntimeError(self._GetErrorString(ret))
        return buf.value

    @property
    def hydradict(self):
        dic = {}
        dic['LibraryVersion']   = self.GetLibraryVersion()
        dic['SerialNumber']     = self.GetSerialNumber()
        dic['HardwareInfo']     = self.GetHardwareInfo()
        dic['Resolution']       = self.GetResolution()
        return dic
        
    def _GetErrorString(self,errcode):
        buf = ctypes.create_string_buffer(b'',40)
        _hhlib.HH_GetErrorString(ctypes.byref(buf),errcode)
        return buf.value
        
        
    def _return_handler(ret):
        if ret:
            buf = ctypes.create_string_buffer(b'',40)
            _hhlib.HH_GetErrorString(ctypes.byref(buf),ret)
            raise RuntimeError(buf.value)


    _HH = FunctionHelper(
                         _hhlib,                         
                         'HH_',
                         ctypes.c_int,
                         _return_handler,
                         0,
                         )
    ##############################


    ######################################################################
    ############# BASIC UTILITIES: INITIALIZATION & CLOSING  #############
    
    @_HH.wrap(ctypes.c_int,ctypes.c_int)
    def Initialize(_wrapped,mode='hist',refsource=0):
        """Note: This routine must be called before any of the other routines below can be used. Note that some of them depend on the
        measurement mode you select here. See the HydraHarp manual for more information on the measurement modes.
        """
        mode = dict(
                hist = 0,
                T2 = 2,
                T3 = 3,
                cont = 8,
            )[mode]
        _wrapped(mode,refsource)


    def close(self):
        self.CloseDevice()
        self._logger.info(f"[SETUP - {curtime()}] Hydraharp connection closed.")
        print(f"[SETUP - {curtime()}] Hydraharp connection closed.")

   
    @_HH.wrap(ctypes.c_char_p)
    def OpenDevice(_wrapped):  
        serial = ctypes.create_string_buffer(b'',8)
        _wrapped(serial)
        return serial.value.decode('ascii')      

                     
    @_HH.wrap()
    def CloseDevice(_wrapped):
        """Note: Closes and releases the device for use by other programs.
        """
        _wrapped()


    def SetCFDDefaults(self):
        """ Set channel CFD default values:
            Ch0 -- Zero Cross 20mV / Discriminator 260mV
            Ch1 -- Zero Cross 20mV / Discriminator 252mV
            Sync Divider 1 (None)
            Offset -55000ps
        """
        #self.SetInputCFD(0, 210, 20) #testing a higher discriminator voltage on 4 Nov 2024
        #self.SetInputCFD(1, 210, 20)
        self.SetInputCFD(0, 400, 20)
        self.SetInputCFD(1, 400, 20)
        self.SetSyncCFD(210, 20)
        self.SetSyncDiv(1)
        self.SetSyncChannelOffset(10000)


    def SetMode(self, mode='hist'):
        self.last_mode = mode
        self.Initialize(mode=mode)
        self.Calibrate()
        

    def GetMode(self):
        return self.last_mode


    def opendefaults(self):
        self.OpenDevice()
        self.SetMode()
        self.Calibrate()
        self.SetCFDDefaults()  # HELL NO!
        self.GetCountRate(channel=0)
        self.GetCountRate(channel=1)
        self._logger.warning(self.GetWarningsText(self.GetWarnings()))
        self.SetBinning(0)
        setup_msg = f"[SETUP - {curtime()}] Hydraharp connection opened."
        print(setup_msg)
        self._append_log("\n")
        self._append_log(setup_msg)
        self._append_log("\n")
        



    ###########################################################################################
    ############# BASIC UTILITIES: Calibration & hardware side configuration      #############


    @_HH.wrap()
    def Calibrate(_wrapped):
        _wrapped()
        
        
    @_HH.wrap(ctypes.c_int)
    def SetSyncDiv(_wrapped, div):
        """Note: The sync divider must be used to keep the effective sync rate at values ≤ 12.5 MHz. It should only be used with sync sources
        of stable period. Using a larger divider than strictly necessary does not do great harm but it may result in slightly larger timing
        jitter. The readings obtained with HH_GetCountRate are internally corrected for the divider setting and deliver the external
        (undivided) rate. The sync divider should not be changes whilee a measurement is running.
        """
        logging.getLogger(__name__).warning('SetSyncDiv {}'.format(div))
        _wrapped(div)


    @_HH.wrap(ctypes.c_int,ctypes.c_int)
    def SetSyncCFD(_wrapped,level,zerocross):
        """Note: Value is given as a positive number although the electrical signals are actually negative.
        After a firmware update to version 2.0 the CFD settings may require slight adjustments.
        """
        logging.getLogger(__name__).warning('SetSyncCFD {}'.format((level, zerocross)))
        _wrapped(level,zerocross)


    @_HH.wrap(ctypes.c_int)
    def SetSyncChannelOffset(_wrapped,off):
        logging.getLogger(__name__).warning('SetSyncOffset {}'.format(off))
        _wrapped(off)

            
    @_HH.wrap(ctypes.c_int,ctypes.c_int,ctypes.c_int)
    def SetInputCFD(_wrapped,channel,level,zerocross):
        """Note: Value is given as a positive number although the electrical signals are actually negative.
        The maximum input channel index must correspond to nchannels-1 as obtained through HH_GetNumOfInputChannels().
        After a firmware update to version 2.0 the CFD settings may require slight adjustments.
        """
        logging.getLogger(__name__).warning('SetInputCFD {}'.format((channel, level, zerocross)))
        _wrapped(channel,level,zerocross)
        
        
    @_HH.wrap(ctypes.c_int,ctypes.c_int)
    def SetInputChannelOffset(_wrapped,channel,offset):
        """Note: The maximum input channel index must correspond to nchannels-1 as obtained through HH_GetNumOfInputChannels().
        """
        logging.getLogger(__name__).warning('SetChannelOffset {}'.format((channel, offset)))
        _wrapped(channel,offset)        

        
    @_HH.wrap(ctypes.c_int, ctypes.c_int)
    def SetInputChannelEnable(_wrapped,channel,enable):
        """Note: The maximum channel index must correspond to nchannels-1 as obtained through HH_GetNumOfInputChannels().
        """
        _wrapped(channel,enable)
        
    
    @_HH.wrap(ctypes.c_int)
    def SetBinning(_wrapped,binning):
        """Note: binning code works to the power of 2, i.e.
        0 = 1x base resolution,
        1 = 2x base resolution,
        2 = 4x base resolution,
        3 = 8x base resolution, and so on.
        """
        _wrapped(binning)
        
        
    @_HH.wrap(ctypes.c_int)
    def SetOffset(_wrapped,offset):
        _wrapped(offset)



    @_HH.wrap(ctypes.c_int, ctypes.c_uint)
    def SetStopOverflow(_wrapped,stop_ovfl,stopcount):
        """Note: This setting determines if a measurement run will stop if any channel reaches the maximum set by stopcount. If stop_ofl
        is 0 the measurement will continue but counts above STOPCNTMAX in any bin will be clipped.
        """
        _wrapped(stop_ovfl,stopcount)
        
        
    

    @_HH.wrap(ctypes.c_int,ctypes.POINTER(ctypes.c_int))
    def SetHistoLen(_wrapped,lencode):
        actuallen = ctypes.c_int()
        _wrapped(lencode,ctypes.byref(actuallen))
        return actuallen.value
        

    ######################################################################
    #############     BASIC UTILITIES: MEASUREMENT CONTROL   #############
        
    @_HH.wrap(ctypes.c_int)
    def ClearHistMem(_wrapped,Block=0):
        _wrapped(Block)

        
    @_HH.wrap(ctypes.c_int, ctypes.c_int, ctypes.c_int)
    def SetMeasControl(_wrapped,meascontrol,startedge,stopedge):
        _wrapped(meascontrol,startedge,stopedge)
        

    @_HH.wrap(ctypes.c_int)
    def StartMeas(_wrapped,tacq=1000):
        """  Start measurement for tacq in [ms]
        """
        _wrapped(tacq)
            
            
    @_HH.wrap()
    def StopMeas(_wrapped):
        _wrapped()
            
    

    @_HH.wrap(ctypes.POINTER(ctypes.c_float))
    def GetResolution(_wrapped):
        resolution = ctypes.c_float()
        _wrapped(ctypes.byref(resolution))
        return resolution.value


    ######## GET COUNT RATES
    @_HH.wrap(ctypes.POINTER(ctypes.c_int))
    def GetSyncRate(_wrapped):
        syncrate = ctypes.c_int()
        _wrapped(ctypes.byref(syncrate))
        return syncrate.value
        
        
    @_HH.wrap(ctypes.c_int,ctypes.POINTER(ctypes.c_int))
    def GetCountRate(_wrapped,channel=0):
        """ Note: Allow at least 100 ms after HH_Initialize or HH_SetSyncDivider to get a stable rate meter reading.
        Similarly, wait at least 100 ms to get a new reading. This is the gate time of the counters.
        The maximum input channel index must correspond to nchannels-1 as obtained through HH_GetNumOfInputChannels().
        """
        rate = ctypes.c_int()
        _wrapped(channel,ctypes.byref(rate))
        return rate.value
        
    ####################################
    ###### DIAGNOSING & DEBUGGING ######

    @_HH.wrap(ctypes.POINTER(ctypes.c_int))
    def GetFlags(_wrapped):
        """Note: Use the predefined bit mask values in hhdefin.h (e.g. FLAG_OVERFLOW) to extract individual bits through a bitwise AND.
        """
        flags = ctypes.c_int()
        _wrapped(ctypes.byref(flags))
        flags = flags.value
        overflow = flags & 0x0001
        fifofull = flags & 0x0002
        synclost = flags & 0x0004
        reflost  = flags & 0x0008
        syserror = flags & 0x0010
        active   = flags & 0x0020
        return overflow,fifofull,synclost,reflost,syserror,active


    @_HH.wrap(ctypes.POINTER(ctypes.c_int))
    def CTCStatus(_wrapped):
        ### CHECKS "measurement complete" flag
        ctcstatus = ctypes.c_int()
        _wrapped(ctypes.byref(ctcstatus))
        return ctcstatus.value   
        
    @_HH.wrap(ctypes.POINTER(ctypes.c_double))
    def GetElapsedMeasTime(_wrapped):
        elapsed = ctypes.c_double()
        _wrapped(ctypes.byref(elapsed))
        return elapsed.value
        
    
    @_HH.wrap(ctypes.POINTER(ctypes.c_int))
    def GetWarnings(_wrapped):
        """note: You must call HH_GetCoutRate and HH_GetCoutRate for all channels prior to this call.
        """
        warnings = ctypes.c_int()
        _wrapped(ctypes.byref(warnings))
        return warnings
        
        
    @_HH.wrap(ctypes.c_char_p,ctypes.c_int)
    def GetWarningsText(_wrapped,warnings):
        """Get Warnings."""
        text = ctypes.create_string_buffer(b'',16384)
        _wrapped(text,warnings)
        return text
        
    @_HH.wrap(ctypes.POINTER(ctypes.c_double))
    def GetSyncPeriod(_wrapped):
        """Get the Sync Period."""
        period = ctypes.c_double()
        _wrapped(ctypes.byref(period))
        return period.value
        
        
    @_HH.wrap(ctypes.POINTER(ctypes.c_uint),ctypes.c_int,ctypes.POINTER(ctypes.c_int))
    def ReadFiFo(_wrapped,count):
        """ Read FiFo from the buffer.
        Note: CPU time during wait for completion will be yielded to other 
        processes / threads. Function will return after a timeout period of
        ~10 ms, if not all data could be fetched. Buffer must not be accessed
        until the function returns.
        """
        buf = np.empty(count,dtype=np.uint32)
        nactual = ctypes.c_int()
        _wrapped(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),count,ctypes.byref(nactual))
        return buf.ravel()[:nactual.value].astype(np.uint32)        
    
    



    ####################################
    ###### MEASUREMENT UTILITIES ######

    @_HH.wrap(ctypes.POINTER(ctypes.c_uint), ctypes.c_int, ctypes.c_int)
    def GetHistogram(_wrapped,channel,clear=1):
        chcount = np.empty(2**16,dtype=np.uint32)
        _wrapped(chcount.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),channel,clear)
        return chcount
        

    def GetT2(self, tacq):  #NOT TESTED! Generates FIFO overruns!
        """In T2 mode, start measurement for tacq [ms] and return FiFo buffer.
        
        Problem: Overflow occurs if not reading fast enough...
        This function takes about 3 ms + 30% overhead...
        """
        if self.last_mode != 'T2':
            self.SetMode('T2')
        if int(tacq) == 0:
            raise ValueError('Hydra acq time must be integer > 0 [ms]')
        # Divide long measurement times into of size ~50ms
        # This helps to prevent FiFo overflows
        t_chunked = np.diff(np.linspace(0, tacq, int(tacq//100)+2))
        if not np.isclose(np.sum(t_chunked), tacq):
            print(np.sum(t_chunked), tacq)
            raise AssertionError
        chunks = []
        for i, ti in enumerate(t_chunked):
            self.StartMeas(int(ti))
            done = False
            data = []
            while done != True :
                if int(self.GetFlags()[1]) != 0:
                    raise RuntimeError(f"[ERROR - {curtime()}] FiFo overrun.")
                buf = self.ReadFiFo(int(2**17))
                data.extend(buf)
                if (buf.size == 0) & (self.CTCStatus() == 1):
                    done = True
            chunks.append(data.copy())
        return list(itertools.chain.from_iterable(chunks))

    def GetCounts(self, tacq):  #NOT TESTED!
        """In T2 mode, start measurement for tacq in [ms] and return counts."""   
        data = self.GetT2(tacq)
        counts = len(data)  # we make some error here counting markers/ofl
        return counts        
    
    ### PARSE RAW 32 BIT RECORDS into chunks of (special, channels, timestamps)
    #@numba.jit(nopython=True, parallel=False)
    def extract_records_hydra(self, chunkdata):
        """Process chunks of hydraharp DWORDS into channels and timestamps.
        
        Taken from correlation_analzsis.py
        """
        timestamp_mask = np.uint32(-1) >> np.uint32(7)  # mask 000000011...11 to isolate the
        timestamps = chunkdata & timestamp_mask  # timestamp via logical and
        special = chunkdata >> np.uint32(31)
        channel_mask = ((np.uint32(1) << np.uint32(6)) - np.uint32(1)) << np.uint(25)
        channels = (chunkdata & channel_mask) >> np.uint32(25)
        return special, channels, timestamps 
    
    #@numba.jit(nopython=True, parallel=False, nogil=False)
    def unwrap_time_hydra(self, special, channels, timestamps):
        """Unwrap overflow markers and overwrite them with adjacent timestamp.
        
        Taken from correlation_analzsis.py
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
    
    def GetTimestamps(self, tacq):
        """Get Timestamps by using getT2 and extracting the data.
        
        Units should be in global resolution = 1ps
        """
        rawdata = self.GetT2(tacq)
        special, channels, timestamps = self.extract_records_hydra(np.array(rawdata, dtype=np.uint32))
        channels, timestamps = self.unwrap_time_hydra(special, channels, timestamps)
        return channels, timestamps








    #########################################################
    ############## MEASUREMENT ROUTINES #####################

    '''
    def measure_timestamps_T2(self, acq_time, filename):
        """Acquire acq_time (ms) of T2 and save to local SSD
        
        Intended to be used remotely with correlation_measurementv2.py from G8.
        """
        if self.last_mode != 'T2':
            self.SetMode('T2')
        OVERRUN_MARKER_HYDRA = np.array(127 << 25, dtype=np.uint32)
        file = open(filename, mode='ba+')
        completed = False
        completed_time = 0.0
        while not completed:
            if completed_time >= acq_time:
                break
            # Start measurement for acquisition time that is remaining
            self.StartMeas(tacq=int(acq_time - completed_time))
            start_time = self._milliseconds()
            acq_done = False
            while not acq_done:
                if int(self.GetFlags()[1]) != 0:
                    completed_time += self._milliseconds() - start_time
                    self._logger.warning(f'[ERROR - {curtime()}] Hydraharp FiFo overrun!')
                    # We place our own overrun marker when restarting◙
                    # to ensure that timestamps increase monotonically
                    # pickle.dump(self.OVERRUN_MARKER, file)
                    OVERRUN_MARKER_HYDRA.tofile(file)
                    self.StopMeas()  # manual suggests to do this
                    break  # will attempt to measure the remaining time
                buffer = self.ReadFiFo(int(2**17))  # array uint32
                if (buffer.size == 0) & (self.CTCStatus() == 1):
                    completed_time += self._milliseconds() - start_time
                    acq_done = True
                    completed = True
                else:
                    # pickle.dump(buffer, file)
                    buffer.tofile(file)
            self.StopMeas()  # manual suggests to do this, unnecessary?
        file.close()
        
     
    def measure_timestamps_T2_chunked_old(self, acq_time, filename_prefix, chunk_time, state, cancel_event=None):
        if self.last_mode != 'T2':
            self.SetMode('T2')
            
        print(f"[SETUP - {curtime()}] Starting chunked T2 measurements with parameters")
        print(f"Total acquisition time: {acq_time/(60e3*60)} hours")
        print(f"Individual chunk time: {chunk_time/(60e3*60)} hours")
        print(f"Filename prefix: {filename_prefix}")
        
        start_chunk_idx = state["next_chunk_idx"]
        
        OVERRUN_MARKER_HYDRA = np.array(127 << 25, dtype=np.uint32)
        if start_chunk_idx is None:
            folder = os.path.dirname(filename_prefix) or "."
            base = os.path.basename(filename_prefix)
            existing = glob.glob(os.path.join(folder, f"{base}_chunk*.b"))
            if existing:
                def idx(p):
                    b = os.path.basename(p)
                    return int(b.split("_chunk")[-1].split(".b")[0])
                start_chunk_idx = max(idx(p) for p in existing) + 1
            else:
                start_chunk_idx = 0
                
                
        # Prepare acquisition loop        
        total_acquired_time = 0.0
        acquisition_start_time = self._milliseconds()
        chunk_idx = start_chunk_idx
        last_emit = 0
        EMIT_PERIOD = 1 # 1s between progress updates
        
        # ACQUISITION LOOP
        while total_acquired_time < acq_time and not (cancel_event is not None and cancel_event.is_set()):
            
            # This is for each chunk
            remaining_global = max(0.0, acq_time - total_acquired_time)
            remaining_time = min(chunk_time, remaining_global)
            chunk_filename = f"{filename_prefix}_chunk{chunk_idx}.b"
            chunk_elapsed = 0.0
            buffer_sleeps = 0
            
            chunk_anchor = self._milliseconds()
            
            msg_start = f"[ACQUISITION - {curtime()}] Starting chunk {chunk_idx}, target {remaining_time:.2f} ms"
            self._logger.info(msg_start)
            print(msg_start)
            state["last_chunk_time"] = time.time()
            
            
            with open(chunk_filename, 'ab+') as file:
                while chunk_elapsed < remaining_time and not (cancel_event is not None and cancel_event.is_set()):
                    # This is within each chunk
                    
                    state["buffer_sleeps"] = 0
                    tacq = int(max(1, int(remaining_time - chunk_elapsed)))
                    measurement_timeout = tacq * 1.2
                    
                    try:
                        self.StopMeas() # Make sure nothing is running before starting a new meas
                    except:
                        pass
                        
                    self.StartMeas(tacq=tacq)
                    attempt_anchor = self._milliseconds()   # only for timeouts
                    acquisition_done = False
                    
                    while not acquisition_done and not (cancel_event is not None and cancel_event.is_set()):
                        # This is within the acquisition
                        now = self._milliseconds()
                        
                        # progress vs fixed chunk anchor
                        chunk_elapsed = now - chunk_anchor
                        
                        
                        ## Send progress update
                        if (now / 1000.0 - last_emit) >= EMIT_PERIOD:
                            den = max(1.0, float(remaining_time))
                            msg = (f"\r[ACQUISITION - {curtime()}] chunk {chunk_idx} "
                                   f"{int(chunk_elapsed/1e3):d} s / {int(den/1e3):d} s ({chunk_elapsed/den*100:.2f} %)  "
                                   f"total {int(total_acquired_time/1e3):d} s / {int(acq_time/1e3):d} s ({total_acquired_time/max(1.0, float(acq_time))*100:.2f} %)")
                            self._append_log(msg)
                            state["last_chunk_time"] = time.time()
                            last_emit = now / 1000.0
                        
                        if (now - attempt_anchor) > measurement_timeout:
                            msg_timeout = (f"[ACQUISITION - {curtime()}] Acquisition time "
                                           f"tacq={tacq} in chunk {chunk_idx} reached, proceeding. "
                                           f"flags={self.GetFlags()}, CTC={self.CTCStatus()}")
                            self._logger.error(msg_timeout)
                            self._append_log("\n" + msg_timeout + "\n")
                            print(msg_timeout)
                            
                            # self.StopMeas()
                            acquisition_done = True
                            break
                            
                        # CHECK FOR FIFO OVERRUNw  
                        fifo_overrun_flag = int(self.GetFlags()[1])
                        if fifo_overrun_flag != 0:
                            msg_overrun = (f"[WARNING - {curtime()}] Hydraharp FIFO overrun in chunk {chunk_idx}. "
                                           f"flags={self.GetFlags()}, CTC={self.CTCStatus()}")
                            self._logger.warning(msg_overrun)
                            self._append_log("\n" + msg_overrun + "\n")
                            OVERRUN_MARKER_HYDRA.tofile(file)
                            # self.StopMeas()
                            break
                        
                        
                        buffer = self.ReadFiFo(2**17)
                        if buffer.size == 0 and self.CTCStatus() == 1:
                            acquisition_done = True
                        else:
                            buffer.tofile(file)
                            state["last_chunk_time"] = time.time()
                            
                            if buffer.size == 0:
                                buffersleeptimems = 25
                                time.sleep(buffersleeptimems/1e3)
                                # acquisition_start_time += buffersleeptimems
                                # chunk_start_time += buffersleeptimems
                                # remaining_time += buffersleeptimems
                                # acq_time += buffersleeptimems
                                state["buffer_sleeps"] +=1
                        
                        
                    self.StopMeas()
            chunk_elapsed = min(chunk_elapsed, remaining_time)
            total_acquired_time += chunk_elapsed
        
            done_msg = (f"\n[ACQUISITION - {curtime()}] Chunk {chunk_idx} done, "
                        f"acquired {int(chunk_elapsed)} ms  "
                        f"(total {int(total_acquired_time)}/{int(acq_time)} ms)")
            self._logger.info(done_msg); self._append_log(done_msg + "\n"); print(done_msg)

            state["last_chunk_time"] = time.time()
            chunk_idx += 1
            
        end_msg = f"\n[ACQUISITION - {curtime()}] Done acquiring up to chunk {chunk_idx-1}."
        self._append_log(end_msg + "\n"); print(end_msg)
        return float(total_acquired_time), int(chunk_idx)
    '''


    def run_acquire_only(self, acq_time, chunk_time, filename_prefix, metadata = None, reinit_fn=None, appendix=None):
        """
        Single-threaded acquisition of T2 timestamps into chunk files.
        No analysis.

        Appendix: If integer will be incremented to the smalles number such that metadata_appendix does not exist yet.
        """
        if metadata is not None: 
            self._save_metadata(metadata, os.path.dirname(filename_prefix),appendix)
        try:
            total, next_idx = self.acquire_t2_chunked_simple(
                acq_time=acq_time,
                chunk_time=chunk_time,
                filename_prefix=filename_prefix,
                reinit_fn=reinit_fn,
                soft_retry_once=True,
                allow_one_reopen=True
            )
            print(f"[ACQUISITION - {curtime()}] Finished: {int(total)}/{int(acq_time)} ms, last chunk index {next_idx-1} \n")
            return total, next_idx
        except KeyboardInterrupt:
            print(f"\n[ABORT - {curtime()}] Ctrl+C received. Stopping measurement…")
            try:
                with self._hh_lock:
                    self.StopMeas()
            except Exception:
                pass
            self._append_log(f"\n[ABORT - {curtime()}] User abort.\n")
            return 0.0, 0
    
    def merge_chunk_files(self, filename_prefix):
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
            raise FileNotFoundError(f"[ANALYSIS ERROR - {curtime()}] No chunk files found with prefix '{filename_prefix}'.")

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
            raise ValueError(f"[ANALYSIS ERROR - {curtime()}] Integrity check failed: merged file does not match input chunks.")

        total_input_size = sum(os.path.getsize(f) for f in chunk_files)
        merged_size = os.path.getsize(merged_file_path)
        print(total_input_size,merged_size)
        if total_input_size != merged_size:
            raise ValueError(f"[ANALYSIS ERROR - {curtime()}] Merged file size does not match total input size.")
        
        return merged_file_path

    
    def acquire_t2_chunked_simple(
        self,
        acq_time,
        chunk_time,
        filename_prefix,
        reinit_fn=None,
        soft_retry_once=True,
        allow_one_reopen=True
    ):
        """
        Conservative chunked T2 acquisition with attempt to handle of USB/BULK read failures.
        Writes raw DWORDs to <prefix>_chunk{idx}.b.
        """

        # Configuration
        STALL_TIMEOUT_MS = 15_000
        MAX_STALL_ESCALATIONS = 2
        EMIT_PERIOD = 1.0
        OVERRUN_MARKER_HYDRA = np.array(127 << 25, dtype=np.uint32)

        if self.last_mode != 'T2':
            self.SetMode('T2')

        # --- determine next chunk index ---
        folder = os.path.dirname(filename_prefix) or "."
        os.makedirs(folder, exist_ok=True)
        base = os.path.basename(filename_prefix)
        existing = glob.glob(os.path.join(folder, f"{base}_chunk*.b"))
        if existing:
            def idx(p): return int(os.path.basename(p).split("_chunk")[-1].split(".b")[0])
            chunk_idx = max(idx(p) for p in existing) + 1
        else:
            chunk_idx = 0

        total_acquired_time = 0.0
        last_emit_s = 0.0

        # --- local helpers, all protected by lock ---
        def _stop_quiet():
            try:
                with self._hh_lock:
                    self.StopMeas()
            except Exception:
                pass

        def _start_safe(tacq):
            with self._hh_lock:
                self.StartMeas(tacq=int(max(1, int(tacq))))

        def _read_fifo(count):
            with self._hh_lock:
                return self.ReadFiFo(count)

        def _ctc():
            with self._hh_lock:
                return self.CTCStatus()

        def _flags():
            with self._hh_lock:
                return self.GetFlags()

        def _reopen_once():
            with self._hh_lock:
                try:
                    self.CloseDevice()
                except Exception:
                    pass
            time.sleep(3)
            with self._hh_lock:
                self.opendefaults()
            
        print(f"[SETUP - {curtime()}] Starting T2 chunked acquisition")
        print(f"  Total target: {acq_time/1e3:.1f} s   Chunk: {chunk_time/1e3:.1f} s   Prefix: {filename_prefix}")

        while total_acquired_time < acq_time:
            remaining_global = max(0.0, acq_time - total_acquired_time)
            remaining_time = min(chunk_time, remaining_global)
            if remaining_time <= 0:
                break

            chunk_file = f"{filename_prefix}_chunk{chunk_idx}.b"
            chunk_anchor = self._milliseconds()
            chunk_elapsed = 0.0

            print(f"[ACQ  - {curtime()}] Start chunk {chunk_idx}, target {int(remaining_time)} ms -> {chunk_file} \n")
            soft_retry_used = False
            reopen_used = False
            
            #
            #Oli 17.10.2025: was previously inside the while chunk_elapsed < remaining_time: loop
            _stop_quiet()
            time.sleep(5)
            _start_safe(tacq=chunk_time) #previously _start_safe(tacq=tacq) 
            #
            happened_immediately_after_again = False
            with open(chunk_file, "ab+") as f:
                while chunk_elapsed < remaining_time:
                    tacq = int(max(1, int(remaining_time - chunk_elapsed)))
                
                    attempt_anchor = self._milliseconds()
                    last_activity = attempt_anchor
                    stall_escalations = 0
                    prev_flags = _flags()

                    while True:
                        now = self._milliseconds()
                        chunk_elapsed = now - chunk_anchor

                        # Progress log to remote
                        now_s = now / 1000.0
                        if now_s - last_emit_s >= EMIT_PERIOD:
                            den = max(1.0, float(remaining_time))
                            pct_chunk = 100.0 * (chunk_elapsed / den)
                            pct_total = 100.0 * (total_acquired_time / max(1.0, float(acq_time)))
                            sysmsg = (f"\r[ACQ  - {curtime()}] chunk {chunk_idx}: "
                                      f"{int(chunk_elapsed/1e3)}/{int(den/1e3)} s ({pct_chunk:5.1f}%)  "
                                      f"total {int(total_acquired_time/1e3)}/{int(acq_time/1e3)} s ({pct_total:5.1f}%)")
                            self._append_log(sysmsg)
                            last_emit_s = now_s

                        # attempt timeout
                        if (now - attempt_anchor) > (tacq * 1.2):
                            break

                        # Flag changes count as activity
                        flags_now = _flags()
                        if flags_now != prev_flags:
                            prev_flags = flags_now
                            last_activity = now

                        overflow, fifofull, synclost, reflost, syserror, active = flags_now
                        if int(fifofull) != 0:
                            warn = (f"[WARNING - {curtime()}] FIFO overrun in chunk {chunk_idx}. "
                                    f"flags={flags_now}, CTC={_ctc()}, consecutive = {happened_immediately_after_again}")
                            self._logger.warning(warn)
                            self._append_log("\n" + warn + "\n")
                            OVERRUN_MARKER_HYDRA.tofile(f)
                            if happened_immediately_after_again:
                                #print('Sleeping and restarting')
                                time.sleep(5)
                                _stop_quiet()
                                time.sleep(5)
                                try:
                                    _start_safe(tacq=tacq)
                                    #time.sleep(5) # Oli: commented out on 21.10.2025
                                    continue
                                except Exception:
                                    pass
                            happened_immediately_after_again = True
                            break
                        happened_immediately_after_again = False  # if Fifo wasn't full, happened_right_after_again is reset.
                        # Try to read FIFO
                        try:
                            buf = _read_fifo(2**17)
                        except RuntimeError as e:
                            msg = e.args[0] if e.args else b""
                            print("FIFO read failed once: ", msg, " just trying again.")
                        #time.sleep(5) # Oli: commented out on 21.10.2025, was overflowing FiFo, use a short
                        try:
                            buf = _read_fifo(2**17)
                        except RuntimeError as e:
                            msg = e.args[0] if e.args else b""
                            if isinstance(msg, (bytes, bytearray)) and b'HH_ERROR_USB_BULKRD_FAIL' in msg:
                                # Bulk transfer fail recovery
                                if soft_retry_once and not soft_retry_used:
                                    soft_retry_used = True
                                    time.sleep(5)
                                    _stop_quiet()
                                    time.sleep(5)
                                    try:
                                        _start_safe(tacq=tacq)
                                        #time.sleep(5)
                                        continue
                                    except Exception:
                                        pass
                                if allow_one_reopen and not reopen_used:
                                    print(f"[RECOVER - {curtime()}] BULKRD_FAIL -> reopening device once…")
                                    time.sleep(5)
                                    _stop_quiet()
                                    time.sleep(5)
                                    _reopen_once()
                                    time.sleep(5)
                                    reopen_used = True
                                    _start_safe(tacq=tacq)
                                    time.sleep(5)
                                    continue
                                print(f"[ERROR - {curtime()}] BULKRD_FAIL (no more retries). Aborting chunk.")
                                raise
                            else:
                                print(f"[ERROR - {curtime()}] ReadFiFo failed: {e}")
                                raise
                        else:
                            if buf.size > 0:
                                buf.tofile(f)
                                last_activity = now
                            else:
                                if _ctc() == 1:
                                    last_activity = now
                                    break
                                time.sleep(0.025)
                        """ 
                        # ---- STALL WATCHDOG ----
                        if (now - last_activity) > STALL_TIMEOUT_MS:
                            stall_escalations += 1
                            note = f"[STALL - {curtime()}] No activity for {STALL_TIMEOUT_MS/1000:.1f}s (chunk {chunk_idx}, attempt {stall_escalations})."
                            self._append_log("\n" + note + "\n"); print(note)
                            _stop_quiet()
                            time.sleep(5)
                            if stall_escalations == 1:
                                try:
                                    _start_safe(tacq=tacq)
                                    time.sleep(5)
                                    attempt_anchor = self._milliseconds()
                                    last_activity = attempt_anchor
                                    continue
                                except Exception as ee:
                                    self._append_log(f"[STALL-RECOVER] Soft restart failed: {ee}\n")
                            elif stall_escalations == 2:
                                try:
                                    print(f"[STALL-RECOVER - {curtime()}] Reopening device…")
                                    _reopen_once()
                                    time.sleep(5)
                                    _start_safe(tacq=tacq)
                                    time.sleep(5)
                                    attempt_anchor = self._milliseconds()
                                    last_activity = attempt_anchor
                                    continue
                                except Exception as ee:
                                    self._append_log(f"[STALL-RECOVER] Reopen failed: {ee}\n")
                            err = f"[ERROR - {curtime()}] Stall persists after {stall_escalations} escalations. Aborting chunk."
                            self._append_log("\n" + err + "\n"); print(err)
                            raise RuntimeError("Acquisition stalled without activity.")
                    """
            _stop_quiet()
            time.sleep(5)

            # finalize chunk
            chunk_elapsed = min(chunk_elapsed, remaining_time)
            total_acquired_time += chunk_elapsed
            print(f"[ACQ  - {curtime()}] Chunk {chunk_idx} done, acquired {int(chunk_elapsed)} ms  "
                  f"(total {int(total_acquired_time)}/{int(acq_time)} ms) \n")
            chunk_idx += 1

        print(f"[ACQ  - {curtime()}] Done acquiring up to chunk {chunk_idx-1}.")
        return float(total_acquired_time), int(chunk_idx)
    


    #########################################################
    ############## SYSTEM / HARDWARE UTILITIES ##############

    @_HH.wrap(ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p)
    def GetHardwareInfo(_wrapped):
        model = ctypes.create_string_buffer(b'',16)
        partno = ctypes.create_string_buffer(b'',8)
        vers = ctypes.create_string_buffer(b'',8)
        _wrapped(model,partno,vers)
        return (
                model.value.decode('ascii'),
                partno.value.decode('ascii'),
                vers.value.decode('ascii')
                )
                

    @_HH.wrap(ctypes.POINTER(ctypes.c_int))
    def GetFeatures(_wrapped):
        """Note: You do not really need this function. It is mainly for integration in PicoQuant system software such as SymPhoTime in order to
        figure out what capabilities the device has. If you want it anyway, use the bit masks from hhdefin.h to evaluate individual bits
        in the pattern.
        """
        features = ctypes.c_int()
        _wrapped(ctypes.byref(features))
        features = features.value
        dll     = features & 0x0001
        tttr    = features & 0x0002
        markers = features & 0x0004
        lowres  = features & 0x0008
        trigout = features & 0x0010
        return dll,tttr,markers,lowres,trigout

                
    @_HH.wrap(ctypes.c_char_p)
    def GetSerialNumber(_wrapped):
        serial = ctypes.create_string_buffer(b'',8)
        _wrapped(serial)
        return serial.value.decode('ascii')


    @_HH.wrap(ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int))
    def GetBaseResolution(_wrapped):
        """Note: Use the value returned in binsteps as maximum value for the HH_SetBinning function.
        """
        resolution = ctypes.c_double()
        binsteps   = ctypes.c_int(0)
        _wrapped(ctypes.byref(resolution),ctypes.byref(binsteps))
        return resolution.value, binsteps.value


    @_HH.wrap(ctypes.POINTER(ctypes.c_int))
    def GetNumOfInputChannels(_wrapped):
        nchannels = ctypes.c_int(0)
        _wrapped(ctypes.byref(nchannels))
        return nchannels.value


    @_HH.wrap(ctypes.POINTER(ctypes.c_int))
    def GetNumOfModules(_wrapped):
        nummod = ctypes.c_int(0)
        _wrapped(ctypes.byref(nummod))
        return nummod.value


    @_HH.wrap(ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int))
    def GetModuleInfo(_wrapped,modidx):
        modelcode   = ctypes.c_int(0)
        versioncode = ctypes.c_int(0)
        _wrapped(modidx,ctypes.byref(modelcode),ctypes.byref(versioncode))
        return modelcode.value, versioncode.value


    @_HH.wrap(ctypes.c_int,ctypes.POINTER(ctypes.c_int))
    def GetModuleIndex(_wrapped,channel):
        """Note: The maximum input channel index must correspond to nchannels-1 as obtained through HH_GetNumOfInputChannels().
        """
        modidx = ctypes.c_int(0)
        _wrapped(channel,ctypes.byref(modidx))
        return modidx.value

        
    @_HH.wrap(ctypes.c_char_p)
    def GetHardwareDebugInfo(_wrapped):
        """Note: Use this call to obtain debug information for support enquiries if you detect FLAG_SYSERROR or HH_ERROR_STATUS_FAIL.
        """
        debuginfo = ctypes.create_string_buffer(b'',65536)
        _wrapped(debuginfo)


    def get_disk_space(self, drive='Q:/'):
        """Print available disk space."""
        total, used, free = shutil.disk_usage(drive)
        print("Free disk space: %d GiB" % (free // (2**30)))
        return free
        

    def _milliseconds(self):
        """Return time since epoch in miliseconds."""
        return time.time() * 1000
        

    def run_with_timeout(self, func, timeout, *args, **kwargs):
        """Run func with timeout (in seconds)."""
        result = [None]
        exc = [None]
        t = threading.Thread(target=lambda: self._run_target(func, result, exc, *args, **kwargs))
        t.start()
        t.join(timeout)
        if t.is_alive():
            # didn’t finish in time
            return TimeoutError(f"{func.__name__} exceeded {timeout} seconds")
        if exc[0]:
            return exc[0]
        return result[0]

    def _run_target(self, func, result, exc, *args, **kwargs):
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exc[0] = e
            
    

    def _append_log(self, msg: str):
        
        # print(msg, flush=True) # outcomment to avoid server side flooding
        with self._cv:
            self._log.append(msg)
            self._cv.notify_all()

    def get_logs(self, last_idx=0, timeout=5.0):
        end = time.time() + timeout
        with self._cv:
            while last_idx >= len(self._log):
                remaining = end - time.time()
                if remaining <= 0:
                    return last_idx, []
                self._cv.wait(remaining)
            new = list(self._log)[last_idx:]
            return len(self._log), new
        
    def _save_metadata(self, metadata: dict, folder: str, index = None):
        """
        metadata : dict
            Dictionary containing key–value pairs of metadata.
            Values can be strings, numbers, list....
        folder : str
            Path (excluding filename) where the .txt file will be saved.
            Example: "C:/Users/SSPDs/Documents/Offline SSPD Data/20251010/"

        index : integer, numbering metadata files
        """
        # Ensure the output folder exists
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        def format_value(value, indent=0):
            """Recursively format lists/dicts for readable output."""
            if isinstance(value, dict):
                lines = []
                for k, v in value.items():
                    lines.append(f"{'  '*indent}{k}: {format_value(v, indent+1)}")
                return "\n".join(lines)
            elif isinstance(value, list):
                return "[" + ", ".join(str(v) for v in value) + "]"
            else:
                return str(value)
        if index is not None:
            try: 
                with open(os.path.join(folder,f"metadata_{index}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"# Measurement Metadata\n\n")
                    for key, value in metadata.items():
                        formatted = format_value(value)
                        f.write(f"{key}: {formatted}\n")
            except: 
                print("[SETUP - WARNING] Error trying to write metadata file: Path doesn't exist or file is locked from previous run! Delete/rename old file or move to new folder.")
                self._save_metadata(metadata, folder, index = index+1)
            os.chmod(os.path.join(folder,f"metadata_{index}.txt"), 0o444) # lock the file from accidental editing, can open lock via 0o666
        else:
            # Write to file
            index = 0
            try: 
                with open(os.path.join(folder,"metadata.txt"), "w", encoding="utf-8") as f:
                    f.write(f"# Measurement Metadata\n\n")
                    for key, value in metadata.items():
                        formatted = format_value(value)
                        f.write(f"{key}: {formatted}\n")
            except: 
                print("[SETUP - WARNING] Error trying to write metadata file: Path doesn't exist or file is locked from previous run! Delete/rename old file or move to new folder.")
                self._save_metadata(metadata, folder, index = index)
            os.chmod(os.path.join(folder,"metadata.txt"), 0o444) # lock the file from accidental editing, can open lock via 0o666

if __name__ == "__main__":
    hydra = hydraharpsrv()
    daemon = Pyro4.Daemon(host='g13-sspd.dhcp-int.phys.ethz.ch', port=9090)
    uri = daemon.register(hydra, objectId='Hydraharp_g13')
    print('Obj URI:', uri)
    daemon.requestLoop()
    
    




###### NOT USED ANYMORE?

# def measure_timestamps_T2_Oli(self, acq_time, filename_prefix, chunk_idx):
    #     """Acquire acq_time (ms) of T2 and save to local SSD
        
    #     Intended to be used remotely with correlation_measurementv2.py from G8.
    #     """
    #     if self.last_mode != 'T2':
    #         self.SetMode('T2')
    #     OVERRUN_MARKER_HYDRA = np.array(127 << 25, dtype=np.uint32)
    #     chunk_filename = f"{filename_prefix}_chunk{chunk_idx}.b"
    #     file = open(chunk_filename, mode='ba+')
    #     completed = False
    #     completed_time = 0.0
    #     while not completed:
    #         if completed_time >= acq_time:
    #             break
    #         # Start measurement for acquisition time that is remaining
    #         self.StartMeas(tacq=int(acq_time - completed_time))
    #         start_time = self._milliseconds()
    #         acq_done = False
    #         while not acq_done:
    #             if int(self.GetFlags()[1]) != 0:
    #                 completed_time += self._milliseconds() - start_time
    #                 self._logger.warning('Hydraharp FiFo overrun!')
    #                 # We place our own overrun marker when restarting◙
    #                 # to ensure that timestamps increase monotonically
    #                 # pickle.dump(self.OVERRUN_MARKER, file)
    #                 OVERRUN_MARKER_HYDRA.tofile(file)
    #                 self.StopMeas()  # manual suggests to do this
    #                 break  # will attempt to measure the remaining time
    #             buffer = self.ReadFiFo(int(2**17))  # array uint32
    #             if (buffer.size == 0) & (self.CTCStatus() == 1):
    #                 completed_time += self._milliseconds() - start_time
    #                 acq_done = True
    #                 completed = True
    #             else:
    #                 # pickle.dump(buffer, file)
    #                 buffer.tofile(file)
    #         self.StopMeas()  # manual suggests to do this, unnecessary?
    #     file.close()   