# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:54:11 2024
@author: Johannes Eberle
@functionality: Slightly adjusted code for picoharp.
                I changed the GetCountRate method
@changes: 
line 133: _PH = FunctionHelper(_phlib, "PH_", ctypes.c_int, _return_handler, 1) 
the last digit corresponds to the device index (determined by serial channel), may need to be changed.
"""

import ctypes
import numpy as np
import functools
import logging
import sys
import ctypes
import os
import ctypes.util

class FunctionHelper(object):
    def __init__(
        self, library, prefix, return_type, return_handler, DevIdx,
    ):
        self._library = library
        self._prefix = prefix
        self._return_type = return_type
        self._return_handler = return_handler
        self._DevIdx = DevIdx

    def wrap(self, *sig):
        def decorator(fun, *args, **kwargs):
            libfun = getattr(self._library, self._prefix + fun.__name__)
            libfun.restype = self._return_type
            libfun.argtypes = (ctypes.c_int,) + sig

            @functools.wraps(fun)
            def wrapper(*args, **kwargs):
                if self._return_handler is not None:
                    def DevIdxWrap(*args, **kwargs):
                        self._return_handler(libfun(self._DevIdx, *args, **kwargs))
                    return fun(DevIdxWrap, *args, **kwargs)
                else:
                    def DevIdxWrap(*args, **kwargs):
                        return libfun(self._DevIdx, *args, **kwargs)
                    return fun(DevIdxWrap, *args, **kwargs)
            return staticmethod(wrapper)
        return decorator

class picoharp(object):
    def __init__(self, dev_idx=None):
        self._logger = logging.getLogger(self.__class__.__name__)

        # If not given, try to auto-detect
        if dev_idx is None:
            dev_idx = self._find_first_device()
        self._DevIdx = dev_idx
        print('dev_idx = ',dev_idx)

        # Recreate the FunctionHelper with the correct device index
        self._PH = FunctionHelper(_phlib, "PH_", ctypes.c_int, self._return_handler, self._DevIdx)

        self.opendefaults()

    def _find_first_device(self):
        for i in range(8):
            serial = ctypes.create_string_buffer(8)
            ret = _phlib.PH_OpenDevice(i, serial)
            if ret == 0:
                _phlib.PH_CloseDevice(i)
                self._logger.info(f"Found PicoHarp at index {i}, serial {serial.value.decode()}")
                return i
        raise RuntimeError("No PicoHarp devices found.")


    global _phlib
    if sys.maxsize > 2**32:
        os.add_dll_directory(r"C:\Program Files\PicoQuant\PH300-PHLibv30")
        _phlib = ctypes.windll.LoadLibrary("phlib64")
        #print('DLL for 64bit Windows correctly imported')
    else:
        _phlib = ctypes.windll.LoadLibrary("phlib")

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.opendefaults()

    def SetCFDDefaults(self):
        self.SetInputCFD(0, 550, 10)
        self.SetInputCFD(1, 550, 10)
        self.SetSyncDiv(1)
        self.SetSyncOffset(-30000)

    def opendefaults(self):
        self.OpenDevice()
        self.Initialize()
        self.Calibrate()
        self.SetCFDDefaults()
        self.GetCountRate(chan=0)
        self.GetCountRate(chan=1)
        self._logger.warning(self.GetWarningsText(self.GetWarnings()))
        self.SetBinning(5)  # 5=128ns
        self._logger.info("Picoharp connection opened.")

    def close(self):
        self.CloseDevice()
        self._logger.info("Picoharp connection closed.")

    @property
    def picodict(self):
        dic = {}
        dic["LibraryVersion"] = self.GetLibraryVersion()
        dic["SerialNumber"] = self.GetSerialNumber()
        dic["HardwareInfo"] = self.GetHardwareInfo()
        dic["Resolution"] = self.GetResolution()
        return dic

    def _GetErrorString(self, errcode):
        buf = ctypes.create_string_buffer(b"", 40)
        _phlib.PH_GetErrorString(ctypes.byref(buf), errcode)
        return buf.value

    def _return_handler(ret):
        if ret:
            buf = ctypes.create_string_buffer(b"", 40)
            _phlib.PH_GetErrorString(ctypes.byref(buf), ret)
            raise RuntimeError(buf.value)

    _PH = FunctionHelper(_phlib, "PH_", ctypes.c_int, _return_handler, 1)

    def GetLibraryVersion(self):
        buf = ctypes.create_string_buffer(b"", 8)
        ret = _phlib.PH_GetLibraryVersion(ctypes.byref(buf))
        if ret:
            raise RuntimeError(self._GetErrorString(ret))
        return buf.value

    @_PH.wrap(ctypes.c_char_p)
    def OpenDevice(_wrapped):
        serial = ctypes.create_string_buffer(b"", 8)
        _wrapped(serial)
        return serial.value.decode("ascii")

    @_PH.wrap()
    def CloseDevice(_wrapped):
        _wrapped()

    @_PH.wrap(ctypes.c_int)
    def Initialize(_wrapped, mode="hist"):
        mode = dict(hist=0, T2=2, T3=3)[mode]
        _wrapped(mode)

    @_PH.wrap(ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p)
    def GetHardwareInfo(_wrapped):
        model = ctypes.create_string_buffer(b"", 16)
        partno = ctypes.create_string_buffer(b"", 8)
        vers = ctypes.create_string_buffer(b"", 8)
        _wrapped(model, partno, vers)
        return (
            model.value.decode("ascii"),
            partno.value.decode("ascii"),
            vers.value.decode("ascii"),
        )

    @_PH.wrap(ctypes.c_char_p)
    def GetSerialNumber(_wrapped):
        serial = ctypes.create_string_buffer(b"", 8)
        _wrapped(serial)
        return serial.value.decode("ascii")

    @_PH.wrap(ctypes.POINTER(ctypes.c_double))
    def GetBaseResolution(_wrapped):
        resolution = ctypes.c_double()
        _wrapped(ctypes.byref(resolution))
        return resolution.value

    @_PH.wrap(ctypes.POINTER(ctypes.c_int))
    def GetFeatures(_wrapped):
        features = ctypes.c_int()
        _wrapped(ctypes.byref(features))
        features = features.value
        dll = features & 0x0001
        tttr = features & 0x0002
        markers = features & 0x0004
        lowres = features & 0x0008
        trigout = features & 0x0010
        return dll, tttr, markers, lowres, trigout

    @_PH.wrap()
    def Calibrate(_wrapped):
        _wrapped()

    @_PH.wrap(ctypes.c_int, ctypes.c_int, ctypes.c_int)
    def SetInputCFD(_wrapped, channel, level, zerocross):
        _wrapped(channel, level, zerocross)

    @_PH.wrap(ctypes.c_int)
    def SetSyncDiv(_wrapped, div):
        _wrapped(div)

    @_PH.wrap(ctypes.c_int)
    def SetSyncOffset(_wrapped, off):
        _wrapped(off)

    @_PH.wrap(ctypes.c_int, ctypes.c_int)
    def SetStopOverflow(_wrapped, stop_ovfl, stopcount):
        _wrapped(stop_ovfl, stopcount)

    @_PH.wrap(ctypes.c_int)
    def SetMultistopEnable(_wrapped, enable):
        _wrapped(enable)

    @_PH.wrap(ctypes.c_int)
    def SetBinning(_wrapped, binning):
        _wrapped(binning)

    @_PH.wrap(ctypes.c_int)
    def ClearHistMem(_wrapped, Block=0):
        _wrapped(Block)

    @_PH.wrap(ctypes.c_int)
    def StartMeas(_wrapped, tacq=1000):
        _wrapped(tacq)

    @_PH.wrap()
    def StopMeas(_wrapped):
        _wrapped()

    @_PH.wrap(ctypes.POINTER(ctypes.c_int))
    def CTCStatus(_wrapped):
        ctcstatus = ctypes.c_int()
        _wrapped(ctypes.byref(ctcstatus))
        return ctcstatus.value

    @_PH.wrap(ctypes.POINTER(ctypes.c_uint32), ctypes.c_int)
    def GetHistogram(_wrapped, Block=0):
        chcount = np.empty(2**16, dtype=np.uint32)
        _wrapped(chcount.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), Block)
        return chcount

    @_PH.wrap(ctypes.POINTER(ctypes.c_double))
    def GetResolution(_wrapped):
        resolution = ctypes.c_double()
        _wrapped(ctypes.byref(resolution))
        return resolution.value

    @_PH.wrap(ctypes.c_int, ctypes.POINTER(ctypes.c_int))
    def GetCountRate(_wrapped, chan=0):
        rate = ctypes.c_int()
        _wrapped(chan, ctypes.byref(rate))
        return rate.value    
    

    @_PH.wrap(ctypes.POINTER(ctypes.c_int))
    def GetFlags(_wrapped):
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
        warnings = ctypes.c_int()
        _wrapped(ctypes.byref(warnings))
        return warnings.value

    @_PH.wrap(ctypes.c_char_p, ctypes.c_int)
    def GetWarningsText(_wrapped, warning):
        text = ctypes.create_string_buffer(b"", 16384)
        _wrapped(text, warning)
        return text.value.decode("ascii")

    #@_PH.wrap(ctypes.POINTER(ctypes.c_uint), ctypes.c_int, ctypes.POINTER(ctypes.c_int))
    #def ReadFiFo(_wrapped, count):
    #    buf = np.empty(count, dtype=np.uint32)
    #    nactual = ctypes.c_int()
    #    _wrapped(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), count, ctypes.byref(nactual))
    #    return buf.ravel()[: nactual.value].astype(np.uint32)
    @_PH.wrap(ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.POINTER(ctypes.c_int))
    def ReadFiFo(_wrapped, count):
        buf = np.empty(count, dtype=np.uint32)
        nactual = ctypes.c_int()
        _wrapped(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), count, ctypes.byref(nactual))
        # return ONLY the valid part
        return buf[: nactual.value].copy()


 