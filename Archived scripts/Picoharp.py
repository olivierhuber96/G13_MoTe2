# -*- coding: utf-8 -*-
"""
Created on Wed Feb 03 09:28:01 2016

# Picoharp
# Inhist mode, there are always 65536 time bins
# T2 event buffer: 256k (262144 FIFO)

@author: 
    Thomas Fink, Quantum Photonics Group, ETH Zurich
    Patrick Knüppel
"""

import ctypes
import numpy as np
import functools
import logging
import sys
import itertools


class FunctionHelper(object):
    """"""

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
        
    def wrap(self,*sig):
        
        def decorator(fun,*args,**kwargs):
            libfun = getattr(self._library,self._prefix+fun.__name__)
            libfun.restype = self._return_type
            libfun.argtypes = (ctypes.c_int,) + sig
            
            @functools.wraps(fun)
            def wrapper(*args,**kwargs):
                
                if self._return_handler is not None:
                    
                    def DevIdxWrap(*args,**kwargs):
                        self._return_handler(libfun(self._DevIdx,*args,**kwargs))
                        
                    return fun(DevIdxWrap,*args,**kwargs)
                    
                else:
                    
                    def DevIdxWrap(*args,**kwargs):
                        return libfun(self._DevIdx,*args,**kwargs)
                        
                    return fun(DevIdxWrap,*args,**kwargs)

            return staticmethod(wrapper)    
            
        return decorator


class picoharp(object):
    
    T2WRAPAROUND = 210698240
    
    global _phlib
    if sys.maxsize > 2**32:
        _phlib = ctypes.windll.LoadLibrary('phlib64')
    else:
        _phlib = ctypes.windll.LoadLibrary('phlib')
        
        
    def __init__(self):

        self._logger = logging.getLogger(self.__class__.__name__)


    def SetCFDDefaults(self):
        """ Set channel CFD default values:
            Ch0 -- Zero Cross 20mV / Discriminator 260mV
            Ch1 -- Zero Cross 20mV / Discriminator 252mV
            Sync Divider 1 (None)
            Offset -55000ps
        """        
        self.SetInputCFD(0,210,20)
        self.SetInputCFD(1,170,20)
        self.SetSyncDiv(1)
        self.SetSyncOffset(-10000)
        
    
    def opendefaults(self):
        self.OpenDevice()
        self.SetMode()
        self.Calibrate()
        self.SetCFDDefaults()
        self.GetCountRate(chan=0)
        self.GetCountRate(chan=1)
        self._logger.debug(self.GetWarningsText(self.GetWarnings()))
        self.SetBinning(0)
        self._logger.info("Picoharp connection opened.")
        
        
    def close(self):
        self.CloseDevice()
        self._logger.info("Picoharp connection closed.")
        
        
    @property
    def picodict(self):
        dic = {}
        dic['LibraryVersion']   = self.GetLibraryVersion()
        dic['SerialNumber']     = self.GetSerialNumber()
        dic['HardwareInfo']     = self.GetHardwareInfo()
        dic['Resolution']       = self.GetResolution()
        return dic    

    def _GetErrorString(self,errcode):
        buf = ctypes.create_string_buffer(b'',40)
        _phlib.PH_GetErrorString(ctypes.byref(buf),errcode)
        return buf.value


    def _return_handler(ret):
        if ret:
            buf = ctypes.create_string_buffer(b'',40)
            _phlib.PH_GetErrorString(ctypes.byref(buf),ret)
            raise RuntimeError(buf.value)

            
    _PH = FunctionHelper(_phlib,
                         'PH_',
                         ctypes.c_int,
                         _return_handler,
                         0,
                         )
        

    def GetLibraryVersion(self):
        buf = ctypes.create_string_buffer(b'',8)
        ret = _phlib.PH_GetLibraryVersion(ctypes.byref(buf))
        if ret:
            raise RuntimeError(self._GetErrorString(ret))
        return buf.value


    @_PH.wrap(ctypes.c_char_p)
    def OpenDevice(_wrapped):  
        serial = ctypes.create_string_buffer(b'',8)
        _wrapped(serial)
        return serial.value.decode('ascii')      

                     
    @_PH.wrap()
    def CloseDevice(_wrapped):
        """Note: Closes and releases the device for use by other programs.
        """
        _wrapped()
#        print("Picoharp connection closed.")
        
    def SetMode(self, mode='hist'):
        self.last_mode = mode
        self.Initialize(mode=mode)
    
    @_PH.wrap(ctypes.c_int)
    def Initialize(_wrapped,mode='hist'):
        """Note: This call will fail with error code -19 (ERROR_INVALID_OPTION) if you have no license for using the library.
        It will also fail with this error code if you try to initialize for TTTR mode without having a license for TTTR mode.
        """
        mode = dict(
                hist = 0,
                T2 = 2,
                T3 = 3,
            )[mode]
        _wrapped(mode)
               
        
    @_PH.wrap(ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p)
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
                
                
    @_PH.wrap(ctypes.c_char_p)
    def GetSerialNumber(_wrapped):
        serial = ctypes.create_string_buffer(b'',8)
        _wrapped(serial)
        return serial.value.decode('ascii')


    @_PH.wrap(ctypes.POINTER(ctypes.c_double))
    def GetBaseResolution(_wrapped):
        resolution = ctypes.c_double()
        _wrapped(ctypes.byref(resolution))
        return resolution.value


    @_PH.wrap(ctypes.POINTER(ctypes.c_int))
    def GetFeatures(_wrapped):
        """Note: Use the predefined bit mask values in phdefin.h to probe for a specific feature
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
        
        
    @_PH.wrap()
    def Calibrate(_wrapped):
        _wrapped()
        
        
    @_PH.wrap(ctypes.c_int,ctypes.c_int,ctypes.c_int)
    def SetInputCFD(_wrapped,channel,level,zerocross):
        logging.getLogger(__name__).warning('SetInputCFD {}'.format((channel, level, zerocross)))
        _wrapped(channel,level,zerocross)
        
        
    @_PH.wrap(ctypes.c_int)
    def SetSyncDiv(_wrapped,div):
        """Note: The sync divider must be used to keep the effective sync rate at values <= 10 MHz. It should only be used with sync sources
        of stable period. The readings obtained with PH_GetCountRate are corrected for the divider setting and deliver the external
        (undivided) rate.
        """
        logging.getLogger(__name__).warning('SetSyncDiv {}'.format(div))
        _wrapped(div)

        
    @_PH.wrap(ctypes.c_int)
    def SetSyncOffset(_wrapped,off):
        """Note: This function can replace an adjustable cable delay. A positive offset corresponds to inserting a cable in the sync input.
        See also PH_SetRoutingChannelOffset.
        """
        logging.getLogger(__name__).warning('SetSyncOffset {}'.format(off))
        _wrapped(off)
        
        
    @_PH.wrap(ctypes.c_int,ctypes.c_int)
    def SetStopOverflow(_wrapped,stop_ovfl,stopcount):
        """Note: This setting determines if a measurement run will stop if any channel reaches the maximum set by stopcount. If stop_ofl is 0
        the measurement will continue but counts above 65,535 in any bin will be clipped.
        """
        _wrapped(stop_ovfl,stopcount)        


    @_PH.wrap(ctypes.c_int)
    def SetMultistopEnable(_wrapped,enable):
        """Note: This is only for special applications where the multistop feature of the PicoHarp is causing complications in statistical analysis.
        Ususally it is not required to call this funktion. By default, multistop is enabled after PH_Initialize.
        """
        _wrapped(enable)

        
    @_PH.wrap(ctypes.c_int)
    def SetBinning(_wrapped,binning):
        _wrapped(binning)
        
        
    @_PH.wrap(ctypes.c_int)
    def ClearHistMem(_wrapped,Block=0):
        _wrapped(Block)


    @_PH.wrap(ctypes.c_int)
    def StartMeas(_wrapped,tacq=1000):
        """  Start measurement for tacq in [ms]
        """
        _wrapped(tacq)
            
            
    @_PH.wrap()
    def StopMeas(_wrapped):
        """Note: Can also be used before the CTC expires but for internal 
        housekeeping it MUST be called any time you finish a measure -
        ment, even if data collection was stopped internally, e.g. by 
        expiration of the CTC or an overflow.
        """
        _wrapped()
            
            
    @_PH.wrap(ctypes.POINTER(ctypes.c_int))
    def CTCStatus(_wrapped):
        ctcstatus = ctypes.c_int()
        _wrapped(ctypes.byref(ctcstatus))
        return ctcstatus.value
            
            
    @_PH.wrap(ctypes.POINTER(ctypes.c_uint32),ctypes.c_int)
    def GetHistogram(_wrapped,Block=0):
        """Note: The current version counts only up to 65,535 (16 bits). This may change in the future.
        """
        chcount = np.empty(2**16,dtype=np.uint32)
        _wrapped(chcount.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),Block)
        return chcount
        

    @_PH.wrap(ctypes.POINTER(ctypes.c_double))
    def GetResolution(_wrapped):
        resolution = ctypes.c_double()
        _wrapped(ctypes.byref(resolution))
        return resolution.value


    @_PH.wrap(ctypes.c_int,ctypes.POINTER(ctypes.c_int))
    def GetCountRate(_wrapped,chan=0):
        """ Note: The hardware rate meters employ a gate time of 100 ms.
        You must allow at least 100 ms after PH_Initialize or PH_SetDync-
        Divider to get a valid rate meter reading. Similarly, wait at least
        100 ms to get a new reading. The readings are corrected for
        the sync divider setting and deliver the external (undivided) rate.
        The gate time cannot be changed. The readings may therefore
        be inaccurate or fluctuating when the rates are very low. If accurate
        rates are needed you must perform a full blown
        measurement and sum up the recorded events.
        """
        rate = ctypes.c_int()
        _wrapped(chan,ctypes.byref(rate))
        return rate.value
        
        
    @_PH.wrap(ctypes.POINTER(ctypes.c_int))
    def GetFlags(_wrapped):
        """ Note: Returns FIFOFull, overflow, syserror marker, respectively.
        """
        flags = ctypes.c_int32()
        _wrapped(ctypes.byref(flags))
        flags = flags.value
        fifofull = flags & 0x0003
        overflow = flags & 0x0040
        syserror = flags & 0x0100
        return fifofull, overflow, syserror
        
        
    @_PH.wrap(ctypes.POINTER(ctypes.c_double))
    def GetElapsedMeasTime(_wrapped):
        elapsed = ctypes.c_double()
        _wrapped(ctypes.byref(elapsed))
        return elapsed.value

        
    @_PH.wrap(ctypes.POINTER(ctypes.c_int))
    def GetWarnings(_wrapped):
        """ Note: Must call PH_GetCoutRates for all channels prior to this call.
        """
        warnings = ctypes.c_int()
        _wrapped(ctypes.byref(warnings))
        return warnings.value
        
        
    @_PH.wrap(ctypes.c_char_p,ctypes.c_int)
    def GetWarningsText(_wrapped,warning):
        text = ctypes.create_string_buffer(b'',16384)
        _wrapped(text,warning)
        return text.value.decode('ascii')
        

    @_PH.wrap(ctypes.POINTER(ctypes.c_uint),ctypes.c_int,ctypes.POINTER(ctypes.c_int))
    def ReadFiFo(_wrapped,count):
        """ Note: Must not be called with count larger than buffer size permits.
        CPU time during wait for completion will be yielded to other processes/
        threads. Function will return after a timeout period of ~80 ms, even
        if not all data could be fetched. Return value indicates how many records
        were fetched. Buffer must not be accessed until the function returns.
        Make sure to call PH_ReadFiFo in a loop until all data is retrieved.
        You cannot always rely on getting all data in one flush, even if it
        is less than count.
        """
        buf = np.empty(count,dtype=np.uint32)
        nactual = ctypes.c_int()
        _wrapped(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),count,ctypes.byref(nactual))
        return buf.ravel()[:nactual.value].astype(np.uint32)

    
    def GetT2(self, tacq):  #NOT TESTED! Generates FIFO overruns!
        """In T2 mode, start measurement for tacq [ms] and return FiFo buffer
        
        Problem: Overflow occurs if not reading fast enough...
        This function takes about 3 ms + 30% overhead...
        """
        if self.last_mode != 'T2':
            self.SetMode('T2')
        if int(tacq) == 0:
            raise ValueError('Pico acq time must be integer > 0 [ms]')
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
                if int(self.GetFlags()[0]) != 0:
                    raise RuntimeError("FiFo overrun.")
                buf = self.ReadFiFo(int(2**17))
                data.extend(buf)
                if (buf.size == 0) & (self.CTCStatus() == 1):
                    done = True
            chunks.append(data.copy())
        return list(itertools.chain.from_iterable(chunks))

    def GetCounts(self, tacq):  #NOT TESTED!
        """In T2 mode, start measurement for tacq in [ms] and return counts"""   
        data = self.GetT2(tacq)
        counts = len(data)  # we make some error here counting markers/ofl
        return counts
    
    def GetCountsPerChannel(self, tacq):
        """In T2 mode, start measurement for tacq in [ms] and return counts per channel"""
        data = self.GetT2(tacq)
        binary = [np.binary_repr(dword, width=32) for dword in data]
        channels = []
        overflow_correction = 0
        for idx, b in enumerate(binary):
            current_channel = int(b[0:4], base=2)
            if current_channel == 15:  # special record
                if int(b[28:32], base=2) == 0:  # overflow marker
                    overflow_correction += self.T2WRAPAROUND
            else:  # normal timestamp record  
                channels.append(current_channel)                        
        channels = np.array(channels, dtype=np.uint8) # use that ch0 markers are 0 and ch1 markers are 1
        Ct1 = np.sum(channels)
        Ct0 = len(channels) - Ct1
        return Ct0, Ct1
    
    
    def GetChannelsTimestamps(self, tacq):  #NOT TESTED!
        """In T2 mode, start measurement for tacq in [ms] and return
        channels, timestamps
        Problems: *maximum length of data list...? *takes long time to run...
        """
        data = self.GetT2(tacq)
        resolution = self.GetResolution()
        binary = [np.binary_repr(dword, width=32) for dword in data]
        channels = []
        timestamps = []
        overflow_correction = 0
        for idx, b in enumerate(binary):
            current_channel = int(b[0:4], base=2)
            if current_channel == 15:  # special record
                if int(b[28:32], base=2) == 0:  # overflow marker
                    overflow_correction += self.T2WRAPAROUND
            else:  # normal timestamp record  
                channels.append(current_channel)
                record = int(b[4:32], base=2) + overflow_correction
                timestamps.append(record * resolution)
        channels = np.array(channels, dtype=np.uint8)
        return channels, timestamps
    
    def GetTimestamps(self, tacq, useFIFO=False):  #NOT TESTED!
        """In T2 mode, start measurement for tacq in [ms] and return timestamps
        
        Problems: maximum length of data list...?
        """
        if useFIFO:
            if self.last_mode != 'T2':
                self.SetMode('T2')
            self.StartMeas(int(tacq))
            done = False
            data = []
            while done != True :
                if int(self.GetFlags()[0]) != 0:
                    raise RuntimeError("FiFo overrun.")
                buf = self.ReadFiFo(int(2**17))
                data.extend(buf)
                if (buf.size == 0) & (self.CTCStatus() == 1):
                    done = True
       
        else:
            data = self.GetT2(tacq)
        resolution = self.GetResolution()
        binary = [np.binary_repr(dword, width=32) for dword in data]
        timestamps = []
        overflow_correction = 0
        for idx, b in enumerate(binary):
            current_channel = int(b[0:4], base=2)
            if current_channel == 15:  # special record
                if int(b[28:32], base=2) == 0:  # overflow marker
                    overflow_correction += self.T2WRAPAROUND
            else:  # normal timestamp record
                record = int(b[4:32], base=2) + overflow_correction
                timestamps.append(record * resolution)
        # timestamps = np.array(timestamps) # careful size limitations!
        return timestamps

       
            
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import tqdm
    import pickle
    pico = picoharp()
    pico.opendefaults()
    
    

#    print(pico._parse_errorcode(-10))
#    print(pico.GetLibraryVersion())
#    print(pico._GetErrorString(-70))
#    pico.OpenDevice()
#    pico.SetMode()
#    pico.SetMode('T2')
#    time.sleep(1)
#    print(pico.GetHardwareInfo())
#    print(pico.GetSerialNumber())
#    print(pico.GetResolution())
#    pico.SetInputCFD(0,260,20)
#    pico.SetInputCFD(1,252,20)
#    pico.SetSyncDiv(1)
#    pico.SetSyncOffset(-55000)
#    cts = np.zeros((601,2),dtype=np.int)
#    discr = np.linspace(0,600,cts.shape[0],dtype=np.int)
#    for Idx in tqdm.tqdm(range(len(discr))):
#        pico.SetInputCFD(0,discr[Idx],20)
#        pico.SetInputCFD(1,discr[Idx],20)        
#        time.sleep(.2)
#        cts[Idx,0] = pico.GetCountRate(chan=0)
#        cts[Idx,1] = pico.GetCountRate(chan=1)
#    plt.plot(discr,cts)       
#    print(pico.GetCountRate(chan=0))
#    print(pico.GetCountRate(chan=1))
#    print(pico.GetFeatures())
#    pico.ClearHistMem()
#    pico.StartMeas(tacq=int(1e3))
#    print(pico.CTCStatus())
#    time.sleep(1)
#    print(pico.CTCStatus())
#    pico.StopMeas()
#    hist = pico.GetHistogram()
#    fl = pico.GetFlags()
#
#    with open('test.p', mode='a+b') as fid:
#        pico.StartMeas(tacq=int(10*1e3))
#        done = False
#        while done != True :
#            if int(pico.GetFlags()[0]) != 0:
#                raise RuntimeError("FiFo overrun.")
#            buf = pico.ReadFiFo(int(2**17))
#            if (buf.size == 0) & (pico.CTCStatus() == 1):
#                done = True
#            pickle.dump(buf,fid)
#    
#    data = []
#    with open('test.p', 'rb') as fid:
#        while True:
#            try:
#                data.append(pickle.load(fid))
#            except EOFError:
#                break
#    print(pico.picodict)
#    pico.CloseDevice()

#    time.sleep(1)
#    pico.close()
    
    
    
    
"""
pico.SetMode('T2')
pico.StartMeas(tacq=int(1000))
done = False
data = []
while done != True :
    if int(pico.GetFlags()[0]) != 0:
        raise RuntimeError("FiFo overrun.")
    buf = pico.ReadFiFo(int(2**17))
    data.append(buf)
    if (buf.size == 0) & (pico.CTCStatus() == 1):
        done = True
"""
