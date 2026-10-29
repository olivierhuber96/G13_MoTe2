# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 09:55:01 2017

@author: Thomas Fink, Quantum Photonics Group, ETH Zurich
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
import shutil
from RemoteCorrelationAnalysis import correlate_parallel, merge_chunk_files, create_new_folder, look_for_files, correlate_timestamps, correlate_timestamps_Oli
#from RemoteCorrelationAnalysis import correlate_timestamps as c_timestamps
import glob
import threading 
import os
import hashlib
import datetime

# Pyro4.config.SERIALIZERS_ACCEPTED = set(['pickle', 'json', 'marshal', 'serpent']) #Kilian changed this to the line below for debugging
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}

class FunctionHelper(object):

    def __init__(self,
                 library,
                 prefix,
                 return_type,
                 return_handler,
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

@Pyro4.expose
class hydraharpsrv(object):

    global _hhlib
    if sys.maxsize > 2**32:
        _hhlib = ctypes.windll.LoadLibrary('hhlib64')
    else:
        _hhlib = ctypes.windll.LoadLibrary('hhlib')

    def __init__(self):

        self._logger = logging.getLogger(self.__class__.__name__)
        
    def test(self):
        print('a')
        return 'a'

    def SetCFDDefaults(self):
        """ Set channel CFD default values:
            Ch0 -- Zero Cross 20mV / Discriminator 260mV
            Ch1 -- Zero Cross 20mV / Discriminator 252mV
            Sync Divider 1 (None)
            Offset -55000ps
        """
        self.SetInputCFD(0, 210, 20)
        self.SetInputCFD(1, 210, 20)
        self.SetSyncCFD(210, 20)
        self.SetSyncDiv(1)
        self.SetSyncChannelOffset(10000)


    def opendefaults(self):
        self.OpenDevice()
        self.SetMode()
        self.Calibrate()
        self.SetCFDDefaults()  # HELL NO!
        self.GetCountRate(channel=0)
        self.GetCountRate(channel=1)
        self._logger.warning(self.GetWarningsText(self.GetWarnings()))
        self.SetBinning(0)
        self._logger.info("Hydraharp connection opened.")
        print("Hydraharp connection opened.")


    def close(self):
        self.CloseDevice()
        self._logger.info("Hydraharp connection closed.")
        print("Hydraharp connection closed.")


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
        

    def GetLibraryVersion(self):
        buf = ctypes.create_string_buffer(b'',8)
        ret = _hhlib.HH_GetLibraryVersion(ctypes.byref(buf))
        if ret:
            raise RuntimeError(self._GetErrorString(ret))
        return buf.value


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

    def SetMode(self, mode='hist'):
        self.last_mode = mode
        self.Initialize(mode=mode)
        self.Calibrate()
        
    def GetMode(self):
        return self.last_mode

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
        
        
    @_HH.wrap(ctypes.c_int, ctypes.c_uint)
    def SetStopOverflow(_wrapped,stop_ovfl,stopcount):
        """Note: This setting determines if a measurement run will stop if any channel reaches the maximum set by stopcount. If stop_ofl
        is 0 the measurement will continue but counts above STOPCNTMAX in any bin will be clipped.
        """
        _wrapped(stop_ovfl,stopcount)
        
        
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


    @_HH.wrap(ctypes.c_int,ctypes.POINTER(ctypes.c_int))
    def SetHistoLen(_wrapped,lencode):
        actuallen = ctypes.c_int()
        _wrapped(lencode,ctypes.byref(actuallen))
        return actuallen.value
        
        
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
            
            
    @_HH.wrap(ctypes.POINTER(ctypes.c_int))
    def CTCStatus(_wrapped):
        ctcstatus = ctypes.c_int()
        _wrapped(ctypes.byref(ctcstatus))
        return ctcstatus.value
            
            
    @_HH.wrap(ctypes.POINTER(ctypes.c_uint), ctypes.c_int, ctypes.c_int)
    def GetHistogram(_wrapped,channel,clear=1):
        chcount = np.empty(2**16,dtype=np.uint32)
        _wrapped(chcount.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),channel,clear)
        return chcount
        

    @_HH.wrap(ctypes.POINTER(ctypes.c_float))
    def GetResolution(_wrapped):
        resolution = ctypes.c_float()
        _wrapped(ctypes.byref(resolution))
        return resolution.value


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
                    raise RuntimeError("FiFo overrun.")
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

## For correlation measurements

    def get_disk_space(self, drive='Q:/'):
        """Print available disk space."""
        total, used, free = shutil.disk_usage(drive)
        print("Free disk space: %d GiB" % (free // (2**30)))
        return free
        
    def _milliseconds(self):
        """Return time since epoch in miliseconds."""
        return time.time() * 1000
        
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
                    self._logger.warning('Hydraharp FiFo overrun!')
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
        
        
    def measure_timestamps_T2_chunked(self, acq_time, filename_prefix, chunk_time):
        """
        Acquire T2 timestamps for acq_time (ms) and save to multiple files every chunk_time (ms).
        Each chunk contains exactly chunk_time ms of data, accounting for overruns.
        """
        if self.last_mode != 'T2':
            self.SetMode('T2')
        print("[Hydra] Starting chunked T2 measurements with parameters")
        print(f"Total acquisition time: {acq_time}")
        print(f"Individual chunk time: {chunk_time}")
        print(f"Filename prefix: {filename_prefix}")
        
        
        OVERRUN_MARKER_HYDRA = np.array(127 << 25, dtype=np.uint32)

        total_acquired_time = 0.0
        chunk_idx = 0

        while total_acquired_time < acq_time:
            remaining_time = min(chunk_time, acq_time - total_acquired_time)
            chunk_filename = f"{filename_prefix}_chunk{chunk_idx}.b"
            chunk_elapsed = 0.0

            self._logger.info(f"Starting chunk {chunk_idx}, target {remaining_time:.2f} ms")
            with open(chunk_filename, 'ab+') as file:
                while chunk_elapsed < remaining_time:
                    # ensure we don't request 0 ms
                    tacq = max(1, int(remaining_time - chunk_elapsed))
                    self.StopMeas()  # make sure device is idle before starting
                    time.sleep(5)
                    self.StartMeas(tacq=tacq)

                    start_time = self._milliseconds()
                    acquisition_done = False
                    timeout = start_time + tacq + 500  # small margin

                    while not acquisition_done:
                        now = self._milliseconds()
                        if now > timeout:
                            self._logger.error(
                                f"Timeout in chunk {chunk_idx}, tacq={tacq}, "
                                f"flags={self.GetFlags()}, CTC={self.CTCStatus()}"
                            )
                            self.StopMeas()
                            acquisition_done = True  # exit and restart chunk loop
                            break

                        flags = int(self.GetFlags()[1])
                        if flags != 0:
                            overrun_time = now - start_time
                            chunk_elapsed += overrun_time
                            total_acquired_time += overrun_time
                            self._logger.warning(
                                f"Hydraharp FIFO overrun in chunk {chunk_idx} at {total_acquired_time:.2f} ms"
                            )
                            OVERRUN_MARKER_HYDRA.tofile(file)
                            self.StopMeas()
                            break

                        buffer = self.ReadFiFo(2**17)
                        if buffer.size == 0 and self.CTCStatus() == 1:
                            elapsed = now - start_time
                            chunk_elapsed += elapsed
                            total_acquired_time += elapsed
                            acquisition_done = True
                        else:
                            buffer.tofile(file)
                            if buffer.size == 0:
                                time.sleep(0.001)

                    self.StopMeas()
                    time.sleep(10) # Sleep to avoid racing between Hydra system commands Stop and Start
                    
                

            self._logger.info(f"Chunk {chunk_idx} done, acquired {chunk_elapsed:.2f} ms")
            print('Chunk_%s processed successfully.'%chunk_idx)
            chunk_idx += 1
        print(f"Done processing {chunk_idx} chunks.")
    
    def measure_timestamps_T2_Oli(self, acq_time, filename_prefix, chunk_idx):
        """Acquire acq_time (ms) of T2 and save to local SSD
        
        Intended to be used remotely with correlation_measurementv2.py from G8.
        """
        if self.last_mode != 'T2':
            self.SetMode('T2')
        OVERRUN_MARKER_HYDRA = np.array(127 << 25, dtype=np.uint32)
        chunk_filename = f"{filename_prefix}_chunk{chunk_idx}.b"
        file = open(chunk_filename, mode='ba+')
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
                    self._logger.warning('Hydraharp FiFo overrun!')
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
        

    def correlate_all_chunks(self, filename_prefix, dp_chunk):# plot_path, data_path,
        chunk_files = sorted(glob.glob(f"{filename_prefix}_chunk*.b"))
        if not chunk_files:
            print(f"No chunk files found with prefix {filename_prefix}")
            return
        
        print(f"Found {len(chunk_files)} chunk files. Processing...")

        for chunk_file in chunk_files:
            params = {}
            if dp_chunk:
                params.update(dp_chunk)  # defaults
            # params.update({
                # 'input_file': chunk_file,
                # 'plot_path': plot_path,
                # 'data_path': data_path
            # })
            print(f"Processing chunk: {chunk_file}")
            correlate_parallel(params)
            
    
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
        
        return merged_file_path

    
    def correlate_timestamps_call(self, arguments):
        #correlate_parallel(arguments) #Original version:
        correlate_timestamps_Oli(arguments)
        
    
    '''    
    def correlate_timestamps_seq(self, arguments):
        c_timestamps(arguments)
    
    def create_new_folder(self, folder_name):
        try:
            print('Create a new folder for the measurement.')
            print(folder_name)
            if folder_name is not None:
                os.makedirs(folder_name, exist_ok=True)
        except:
            print('New folder was not created')
    
    def look_for_files(self,filename_prefix):
        return sorted(glob.glob(f"{filename_prefix}_chunk*.b"))
    '''
    
    def run_parallel(self, acq_time, chunk_time, filename_prefix, dp_merged):
        stop_event = threading.Event()
        folder_name = dp_merged['data_path']
        create_new_folder(folder_name)

        def analysis_loop():
            print("[ANALYSIS] Loop started, waiting for chunks...")
            processed_chunks = set()
            
            while not stop_event.is_set():
                new_chunks = []
                #print('Files seen: ',sorted(glob.glob(f"{filename_prefix}_chunk*.b")))
                #try:
                chunk_files = look_for_files(filename_prefix)
                new_chunks = [f for f in chunk_files if f not in processed_chunks]
                #except FileNotFoundError:
                #    print("No files found yet.")
                #except Exception as e:
                #    print(f"[ERROR] while looking for files: {e}")

                if new_chunks:
                    print(f"[ANALYSIS] Found {len(new_chunks)} new chunks: {new_chunks}")
                    try:
                        #time.sleep(5)
                        print('Merge the chunks found')
                        merge_chunk_files(filename_prefix)
                    except Exception as e:
                        print(f"[ERROR] during analysis: {e}")
                    
                    print('Correlate the data in the merged file')
                    correlate_timestamps_Oli(dp_merged)
                    processed_chunks.update(new_chunks)

                #time.sleep(5)  # optional: adjust based on chunk_time

            # Final pass after acquisition stops
            #chunk_files = look_for_files(filename_prefix)
            #new_chunks = [f for f in chunk_files if f not in processed_chunks]
            #if new_chunks:
            merge_chunk_files(filename_prefix)
                #correlate_timestamps_Oli(dp_merged) # This function is not working, same problem for correalte_timestamps
            correlate_timestamps_Oli(dp_merged) # This function is not working, same problem for correalte_timestamps
            print("[ANALYSIS] Analysis finished. Total processed chunks:", len(processed_chunks))

        def acquisition_loop():
            self.measure_timestamps_T2_chunked(
                acq_time=acq_time,
                filename_prefix=filename_prefix,
                chunk_time=chunk_time
            )
            stop_event.set()

        # Start both threads
        analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
        acquisition_thread = threading.Thread(target=acquisition_loop)
        analysis_thread.start()
        time.sleep(11)
        acquisition_thread.start()

        acquisition_thread.join()
        analysis_thread.join()

if __name__ == "__main__":
    hydra = hydraharpsrv()
    daemon = Pyro4.Daemon(host='g13-sspd.dhcp-int.phys.ethz.ch', port=9090)
    uri = daemon.register(hydra, objectId='Hydraharp_g13')
    print('Obj URI:', uri)
    daemon.requestLoop()
    
    