# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:40:17 2024

@author: QPG
"""

import ctypes
import numpy as np
import functools
import logging
import sys
import threading
import time
from time import gmtime, strftime
import Pyro4
import os
import glob
import hashlib
from collections import deque

def curtime():
    return strftime("UTC %Y-%m-%d %H:%M:%S", gmtime())

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}
Pyro4.config.SERVERTYPE = "thread"
Pyro4.config.COMMTIMEOUT = 0

##############################
# PicoHarp DLL wrapper
##############################
global _phlib
if sys.maxsize > 2**32:
    _phlib = ctypes.windll.LoadLibrary('phlib64')  # 64-bit DLL
else:
    _phlib = ctypes.windll.LoadLibrary('phlib')    # 32-bit DLL

class FunctionHelper:
    def __init__(self, library, prefix, return_type, return_handler, DevIdx):
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
                def DevIdxWrap(*args, **kwargs):
                    res = libfun(self._DevIdx, *args, **kwargs)
                    if self._return_handler:
                        self._return_handler(res)
                    return res
                return fun(DevIdxWrap, *args, **kwargs)
            return staticmethod(wrapper)
        return decorator

############################## CLASS WITH THE PicoHarp UTILITIES
@Pyro4.expose
class picoharpsrv:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._log = deque(maxlen=10000)
        self._cv = threading.Condition()
        self._ph_lock = threading.Lock()
        self.last_mode = None

    def GetLibraryVersion(self):
        buf = ctypes.create_string_buffer(b"", 8)
        ret = _phlib.PH_GetLibraryVersion(ctypes.byref(buf))
        if ret:
            raise RuntimeError(self._GetErrorString(ret))
        return buf.value

    def _milliseconds(self):
        return int(time.time() * 1000)

    def opendefaults(self):
        self.OpenDevice()
        self.Initialize()
        #self.Calibrate()
        self.SetCFDDefaults()
        self.GetCountRate(chan=0)
        self.GetCountRate(chan=1)
        self._logger.warning(self.GetWarningsText(self.GetWarnings()))
        self.SetBinning(5)  # 5=128ns
        self._logger.info("Picoharp connection opened.")
    
    def _return_handler(ret):
        if ret:
            buf = ctypes.create_string_buffer(b"", 40)
            _phlib.PH_GetErrorString(ctypes.byref(buf), ret)
            raise RuntimeError(buf.value)

    def SetCFDDefaults(self):
        self.SetInputCFD(0, 400, 10)
        self.SetInputCFD(1, 400, 10)
        # self.SetInputCFD(0, 210, 20)
        # self.SetInputCFD(1, 210, 20)

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
    
    @_PH.wrap(ctypes.c_int, ctypes.c_int, ctypes.c_int)
    def SetInputCFD(_wrapped, channel, level, zerocross):
        _wrapped(channel, level, zerocross)
    
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

    @_PH.wrap(ctypes.c_int)
    def SetBinning(_wrapped, binning):
        _wrapped(binning)

    #@_PH.wrap(ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.POINTER(ctypes.c_int))
    #def ReadFiFo(_wrapped, count):
        #buf = np.empty(count, dtype=np.uint32)
        #nactual = ctypes.c_int()
        #_wrapped(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), count, ctypes.byref(nactual))
        # return ONLY the valid part
        #return buf[: nactual.value].copy()
        #buf = np.empty(count,dtype=np.uint32)
        #nactual = ctypes.c_int()
        #_wrapped(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),count,ctypes.byref(nactual))
        #return buf.ravel()[:nactual.value].astype(np.uint32)   
    @_PH.wrap(
    ctypes.c_int,                               # devIdx
    ctypes.POINTER(ctypes.c_uint32),            # buffer
    ctypes.c_int,                               # count
    ctypes.POINTER(ctypes.c_int))               # nactual

    def ReadFiFo(_wrapped, count):
        buf = np.empty(count, dtype=np.uint32)
        nactual = ctypes.c_int()
        # ALWAYS pass devIdx (= 0 for most users)
        dev_idx = self.dev  # or 0 if you store it differently
        _wrapped(dev_idx,buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),count,ctypes.byref(nactual))
        return buf[:nactual.value]



    def SetMode(self, mode):
        # PicoHarp: simplified placeholder
        self.last_mode = mode
        print(f"[INFO] Setting mode to {mode}")

    def StartMeas(self, tacq):
        print(f"[INFO] Starting measurement for {tacq} ms")
        # Call PicoHarp StartMeas here
        # _phlib.PH_StartMeas(ctypes.c_int(tacq))

    def StopMeas(self):
        print("[INFO] Stopping measurement")
        # Call PicoHarp StopMeas here
        # _phlib.PH_StopMeas()

    def GetFlags(self):
        # PicoHarp flags placeholder: overflow, fifofull, synclost, reflost, syserror, active
        return (0, 0, 0, 0, 0, 1)

    def CTCStatus(self):
        return 1  # Assume always finished

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

    def close(self):
        self.CloseDevice()
        self._logger.info(f"[SETUP - {curtime()}] Picoharp connection closed.")
        print(f"[SETUP - {curtime()}] Picoharp connection closed.")


    ####################################################################################
    # ------------------------ Acquisition functions ----------------------------------
    ####################################################################################

    def run_acquire_only(self, acq_time, chunk_time, filename_prefix, metadata=None, appendix=None):
        """Acquire T2 timestamps in chunks and save to files"""
        if metadata is not None:
            # Placeholder: save metadata
            pass

        try:
            total, next_idx = self.acquire_t2_chunked_simple(acq_time=acq_time,
                                                             chunk_time=chunk_time,
                                                             filename_prefix=filename_prefix)
            print(f"[ACQUISITION] Finished: {int(total)}/{int(acq_time)} ms, last chunk index {next_idx-1}")
            return total, next_idx
        except KeyboardInterrupt:
            print("\n[ABORT] Ctrl+C received. Stopping measurement…")
            try:
                with self._ph_lock:
                    self.StopMeas()
            except Exception:
                pass
            return 0.0, 0

    def acquire_t2_chunked_simple(self, acq_time, chunk_time, filename_prefix):
        """Conservative chunked T2 acquisition with simple error handling"""
        #if self.last_mode != 'T2':
        #    self.SetMode('T2')
        self.Initialize(mode="T2")
        
        folder = os.path.dirname(filename_prefix) or "."
        os.makedirs(folder, exist_ok=True)
        base = os.path.basename(filename_prefix)

        # Determine next chunk index
    
        existing = glob.glob(os.path.join(folder, f"{base}_chunk*.b"))
        if existing:
            def idx(p): return int(os.path.basename(p).split("_chunk")[-1].split(".b")[0])
            chunk_idx = max(idx(p) for p in existing) + 1
        else:
            chunk_idx = 0

        total_acquired_time = 0.0
        EMIT_PERIOD = 1.0

        print(f"[SETUP] Starting T2 chunked acquisition: target {acq_time/1e3:.1f}s, chunk {chunk_time/1e3:.1f}s")

        while total_acquired_time < acq_time:
            remaining_global = max(0.0, acq_time - total_acquired_time)
            remaining_time = min(chunk_time, remaining_global)
            if remaining_time <= 0:
                break
            
            chunk_file = f"{filename_prefix}_chunk{chunk_idx}.b"
            chunk_anchor = self._milliseconds()
            chunk_elapsed = 0.0

            print(f"[ACQ] Start chunk {chunk_idx}, target {int(remaining_time)} ms -> {chunk_file}")

            with open(chunk_file, "ab+") as f:
                self.StartMeas(tacq=int(chunk_time))
                while chunk_elapsed < remaining_time:
                    now = self._milliseconds()
                    chunk_elapsed = now - chunk_anchor

                    # Periodic log
                    if (chunk_elapsed / 1000.0) % EMIT_PERIOD < 0.05:
                        pct_chunk = 100.0 * (chunk_elapsed / remaining_time)
                        pct_total = 100.0 * (total_acquired_time / max(1.0, float(acq_time)))
                        sysmsg = f"[ACQ] chunk {chunk_idx}: {pct_chunk:.1f}%   total {pct_total:.1f}%"
                        self._append_log(sysmsg)

                    # Read T2 timestamps
                    print('About to read FiFo....')
                    buf = self.ReadFiFo(2**17)
                    print('...read FiFo')
                    if buf.size > 0:
                        #buf.tofile(f) # this does not work
                        f.write(buf.tobytes())  # Use this instead of buf.tofile(f)
                    else:
                        if self.CTCStatus() == 1:
                            break
                        time.sleep(0.025)

            self.StopMeas()
            total_acquired_time += min(chunk_elapsed, remaining_time)
            print(f"[ACQ] Chunk {chunk_idx} done, total acquired {int(total_acquired_time)} ms")
            chunk_idx += 1

        print(f"[ACQ] Done acquiring up to chunk {chunk_idx-1}.")
        return float(total_acquired_time), int(chunk_idx)



    def merge_chunk_files(self, filename_prefix):
        """Merge all chunk files starting with filename_prefix into a single file"""
        folder = os.path.dirname(filename_prefix) or "."
        os.makedirs(folder, exist_ok=True)
        base = os.path.basename(filename_prefix)

        chunk_files = glob.glob(os.path.join(folder, f"{base}_chunk*.b"))
        
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found with prefix '{filename_prefix}'")

        chunk_files.sort(key=lambda f: int(os.path.basename(f).split("_chunk")[-1].split(".b")[0]))
        merged_file_path = os.path.join(folder, "chunk_merged.b")   
        
        merged_hash = hashlib.sha256()
        with open(merged_file_path, 'wb') as outfile:
            for chunk_file in chunk_files:
                with open(chunk_file, 'rb') as infile:
                    while True:
                        buf = infile.read(1024*1024)
                        if not buf:
                            break
                        outfile.write(buf)
                        merged_hash.update(buf)

        # Verify integrity
        verify_hash = hashlib.sha256()
        with open(merged_file_path, 'rb') as merged:
            for block in iter(lambda: merged.read(1024*1024), b""):
                verify_hash.update(block)

        if merged_hash.digest() != verify_hash.digest():
            raise ValueError("Integrity check failed: merged file does not match input chunks")

        total_input_size = sum(os.path.getsize(f) for f in chunk_files)
        merged_size = os.path.getsize(merged_file_path)
        if total_input_size != merged_size:
            raise ValueError("Merged file size does not match total input size")

        print(f"[INFO] Merged {len(chunk_files)} chunks -> {merged_file_path}")
        return merged_file_path


    ########################
    # Logging
    ########################
    def _append_log(self, msg):
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

if __name__ == "__main__":
    pico = picoharpsrv()
    daemon = Pyro4.Daemon(host='g13-sspd.dhcp-int.phys.ethz.ch', port=9091)
    uri = daemon.register(pico, objectId='Picoharp_g13')
    print('Obj URI:', uri)
    daemon.requestLoop()

