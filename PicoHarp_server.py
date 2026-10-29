"""
Created on Sun 16.11.2025 09:55:01
@author: Olivier
Notes: 

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
import ctypes
from pathlib import Path


def curtime():
    return strftime("UTC %Y-%m-%d %H:%M:%S", gmtime())

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}
Pyro4.config.SERVERTYPE = "thread"
Pyro4.config.COMMTIMEOUT = 0


# ----------------------------------------------------------------------------
# Load the library depending on Python bitness
# ----------------------------------------------------------------------------
if os.name != "nt":
    raise RuntimeError("This wrapper targets Windows DLLs (PicoHarp).")

if ctypes.sizeof(ctypes.c_void_p) > 4:
    _dll_name = "phlib64"
else:
    _dll_name = "phlib"

try:
    #_phlib = ctypes.windll.LoadLibrary(_dll_name)
    dll_path = Path(r"C:\Users\SSPDs\conda-envs\hydra_env_test\DLLs\phlib64.dll")
    _phlib = ctypes.WinDLL(str(dll_path))
except Exception as e:
    raise RuntimeError(f"Failed to load { _dll_name }: {e}")

# Helper to raise on non-zero return
def _check_ret(ret):
    if ret != 0:
        # fetch error string from DLL if available
        try:
            buf = ctypes.create_string_buffer(256)
            _phlib.PH_GetErrorString(ctypes.byref(buf), ctypes.c_int(ret))
            msg = buf.value.decode(errors="ignore")
        except Exception:
            msg = ""
        if msg:
            raise RuntimeError(f"PHlib error {ret}: {msg}")
        raise RuntimeError(f"PHlib error {ret}")

# ----------------------------------------------------------------------------
# Prototype the PHlib functions we use
# ----------------------------------------------------------------------------
# int PH_GetLibraryVersion(char *buf)
_phlib.PH_GetLibraryVersion.restype = ctypes.c_int
_phlib.PH_GetLibraryVersion.argtypes = [ctypes.c_char_p]

# int PH_OpenDevice(int devIdx, char *serial)
_phlib.PH_OpenDevice.restype = ctypes.c_int
_phlib.PH_OpenDevice.argtypes = [ctypes.c_int, ctypes.c_char_p]

# int PH_CloseDevice(int devIdx)
_phlib.PH_CloseDevice.restype = ctypes.c_int
_phlib.PH_CloseDevice.argtypes = [ctypes.c_int]

# int PH_Initialize(int devIdx, int mode)
_phlib.PH_Initialize.restype = ctypes.c_int
_phlib.PH_Initialize.argtypes = [ctypes.c_int, ctypes.c_int]

# int PH_GetHardwareInfo(int devIdx, char *model, char *partno, char *version)
_phlib.PH_GetHardwareInfo.restype = ctypes.c_int
_phlib.PH_GetHardwareInfo.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]

# int PH_GetSerialNumber(int devIdx, char *serial)
_phlib.PH_GetSerialNumber.restype = ctypes.c_int
_phlib.PH_GetSerialNumber.argtypes = [ctypes.c_int, ctypes.c_char_p]

# int PH_GetBaseResolution(int devIdx, double *resolution)
_phlib.PH_GetBaseResolution.restype = ctypes.c_int
_phlib.PH_GetBaseResolution.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]

# int PH_GetCountRate(int devIdx, int channel, int *count)
_phlib.PH_GetCountRate.restype = ctypes.c_int
_phlib.PH_GetCountRate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

# int PH_GetFlags(int devIdx, int *flags)
_phlib.PH_GetFlags.restype = ctypes.c_int
_phlib.PH_GetFlags.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

# int PH_GetElapsedMeasTime(int devIdx, double *elapsed)
_phlib.PH_GetElapsedMeasTime.restype = ctypes.c_int
_phlib.PH_GetElapsedMeasTime.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]

# int PH_GetWarnings(int devIdx, int *warnings)
_phlib.PH_GetWarnings.restype = ctypes.c_int
_phlib.PH_GetWarnings.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

# int PH_GetWarningsText(int devIdx, char *text, int warning)
_phlib.PH_GetWarningsText.restype = ctypes.c_int
_phlib.PH_GetWarningsText.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

# int PH_SetBinning(int devIdx, int binning)
_phlib.PH_SetBinning.restype = ctypes.c_int
_phlib.PH_SetBinning.argtypes = [ctypes.c_int, ctypes.c_int]

# int PH_SetInputCFD(int devIdx, int channel, int level, int zerocross)
_phlib.PH_SetInputCFD.restype = ctypes.c_int
_phlib.PH_SetInputCFD.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

# int PH_StartMeas(int devIdx, int tacq)
_phlib.PH_StartMeas.restype = ctypes.c_int
_phlib.PH_StartMeas.argtypes = [ctypes.c_int, ctypes.c_int]

# int PH_StopMeas(int devIdx)
_phlib.PH_StopMeas.restype = ctypes.c_int
_phlib.PH_StopMeas.argtypes = [ctypes.c_int]

# int PH_CTCStatus(int devIdx, int *status)
_phlib.PH_CTCStatus.restype = ctypes.c_int
_phlib.PH_CTCStatus.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

# int PH_ReadFiFo(int devIdx, uint32_t *buffer, int count, int *nactual) #Note how that it is different than ReadFiFo of HH400
_phlib.PH_ReadFiFo.restype = ctypes.c_int
_phlib.PH_ReadFiFo.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

# ----------------------------------------------------------------------------
# Wrapper class
# ----------------------------------------------------------------------------

@Pyro4.expose
class picoharpsrv:
    def __init__(self, dev_idx=0): #dev_idx = 0 in current configuration (17.11.2025)
        self.dev = int(dev_idx)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._log = deque(maxlen=10000)
        self._cv = threading.Condition()
        self._ph_lock = threading.Lock()
        self.last_mode = None

    def _milliseconds(self):
        return int(time.time() * 1000)

    # Basic helper
    def _check(self, ret):
        _check_ret(ret)

    def GetLibraryVersion(self):
        buf = ctypes.create_string_buffer(32)
        ret = _phlib.PH_GetLibraryVersion(buf)
        self._check(ret)
        return buf.value.decode(errors="ignore")

    def opendefaults(self):
        serial = self.OpenDevice()
        self.Initialize()
        #self.Calibrate()
        self.SetCFDDefaults()
        count_rate_0 = self.GetCountRate(chan=0)
        count_rate_1 = self.GetCountRate(chan=1)
        warning_text = self.GetWarningsText(self.GetWarnings())
        self._logger.warning(warning_text)
        #self.SetBinning(5)  # 5=128ns / irrelevant in T2 mode
        self._logger.info("Picoharp connection opened.")
        return {
            "serial": serial,
            "count_rate_0": count_rate_0,
            "count_rate_1": count_rate_1,
            "warnings": warning_text,
        }

    def OpenDevice(self):
        serial = ctypes.create_string_buffer(32)
        # PH_OpenDevice usually returns 0 on success, and writes serial
        ret = _phlib.PH_OpenDevice(ctypes.c_int(self.dev), serial)
        self._check(ret)
        return serial.value.decode(errors="ignore")

    def CloseDevice(self):
        ret = _phlib.PH_CloseDevice(ctypes.c_int(self.dev))
        self._check(ret)

    def Initialize(self, mode="T2"):
        modes = {"hist": 0, "T2": 2, "T3": 3}
        if mode not in modes:
            raise ValueError("Unknown mode: %r" % mode)
        ret = _phlib.PH_Initialize(ctypes.c_int(self.dev), ctypes.c_int(modes[mode]))
        self._check(ret)
        print('PH initialized in ', mode, ' mode')

    def SetCFDDefaults(self):
        #self.SetInputCFD(0, 400, 10)
        #self.SetInputCFD(1, 400, 10)
        self.SetInputCFD(0, 210, 20)
        self.SetInputCFD(1, 210, 20)

    def GetHardwareInfo(self):
        model = ctypes.create_string_buffer(64)
        partno = ctypes.create_string_buffer(64)
        vers = ctypes.create_string_buffer(64)
        ret = _phlib.PH_GetHardwareInfo(ctypes.c_int(self.dev), model, partno, vers)
        self._check(ret)
        return model.value.decode(errors="ignore"), partno.value.decode(errors="ignore"), vers.value.decode(errors="ignore")

    def GetSerialNumber(self):
        serial = ctypes.create_string_buffer(32)
        ret = _phlib.PH_GetSerialNumber(ctypes.c_int(self.dev), serial)
        self._check(ret)
        return serial.value.decode(errors="ignore")

    def GetBaseResolution(self):
        res = ctypes.c_double()
        ret = _phlib.PH_GetBaseResolution(ctypes.c_int(self.dev), ctypes.byref(res))
        self._check(ret)
        return float(res.value)

    def GetCountRate(self, chan=0):
        cnt = ctypes.c_int()
        ret = _phlib.PH_GetCountRate(ctypes.c_int(self.dev), ctypes.c_int(chan), ctypes.byref(cnt))
        self._check(ret)
        return int(cnt.value)

    def GetFlags(self):
        flags = ctypes.c_int()
        ret = _phlib.PH_GetFlags(ctypes.c_int(self.dev), ctypes.byref(flags))
        self._check(ret)
        return int(flags.value)

    def GetElapsedMeasTime(self):
        elapsed = ctypes.c_double()
        ret = _phlib.PH_GetElapsedMeasTime(ctypes.c_int(self.dev), ctypes.byref(elapsed))
        self._check(ret)
        return float(elapsed.value)

    def GetWarnings(self):
        warnings = ctypes.c_int()
        ret = _phlib.PH_GetWarnings(ctypes.c_int(self.dev), ctypes.byref(warnings))
        self._check(ret)
        return int(warnings.value)

    def GetWarningsText(self, warning_mask):
        buf = ctypes.create_string_buffer(16384)
        ret = _phlib.PH_GetWarningsText(ctypes.c_int(self.dev), buf, ctypes.c_int(warning_mask))
        self._check(ret)
        return buf.value.decode(errors="ignore")

    def SetBinning(self, binning):
        ret = _phlib.PH_SetBinning(ctypes.c_int(self.dev), ctypes.c_int(binning))
        self._check(ret)

    def SetInputCFD(self, channel, level, zerocross):
        ret = _phlib.PH_SetInputCFD(ctypes.c_int(self.dev), ctypes.c_int(channel), ctypes.c_int(level), ctypes.c_int(zerocross))
        self._check(ret)

    def StartMeas(self, tacq_ms):
        ret = _phlib.PH_StartMeas(ctypes.c_int(self.dev), ctypes.c_int(int(tacq_ms)))
        self._check(ret)

    def StopMeas(self):
        ret = _phlib.PH_StopMeas(ctypes.c_int(self.dev))
        self._check(ret)

    def CTCStatus(self):
        status = ctypes.c_int()
        ret = _phlib.PH_CTCStatus(ctypes.c_int(self.dev), ctypes.byref(status))
        self._check(ret)
        return int(status.value)  # status: 0=running, 1=finished

    def close(self):
        self.CloseDevice()
        self._logger.info(f"[SETUP - {curtime()}] Picoharp connection closed.")
        print(f"[SETUP - {curtime()}] Picoharp connection closed.")

    def ReadFiFo(self, count):
        """Read up to `count` records from the device FIFO.
        Returns a NumPy array view of length `nactual` (dtype=uint32).
        The returned array is a copy of the valid slice to ensure the
        underlying C buffer lifetime is not an issue.
        """
        if count <= 0:
            return np.empty(0, dtype=np.uint32)

        buf = np.empty(count, dtype=np.uint32)
        nactual = ctypes.c_int()

        ret = _phlib.PH_ReadFiFo(ctypes.c_int(self.dev),
                                 buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                                 ctypes.c_int(count),
                                 ctypes.byref(nactual))
        self._check(ret)
        n = int(nactual.value)
        if n <= 0:
            return np.empty(0, dtype=np.uint32)
        # Return a copy of the valid slice (safe across calls)
        return buf[:n].copy()


    ####################################################################################
    # ------------------------ Acquisition functions ----------------------------------
    ####################################################################################

    def run_acquire_only(self, acq_time, chunk_time, filename_prefix, metadata=None, appendix=None):
        """Acquire T2 timestamps in chunks and save to files
        acq_time [ms]  : total acquisition time
        chunk_time [ms]: measureent time per chunk
        """
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
        self.Initialize(mode="T2")
        
        folder = os.path.dirname(filename_prefix) or "."
        os.makedirs(folder, exist_ok=True)
        #base = os.path.basename(filename_prefix)

        # Determine next chunk index
        existing = glob.glob(os.path.join(folder, f"_chunk*.b"))
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
                self.StartMeas(tacq_ms=int(chunk_time))
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
                    buf = self.ReadFiFo(2**17)
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
    daemon = Pyro4.Daemon(host='g13-sspd.dhcp-int.phys.ethz.ch', port=9089)
    uri = daemon.register(pico, objectId='Picoharp_g13')
    print('Obj URI:', uri)
    daemon.requestLoop()

'''
if __name__ == "__main__":
    ph = PicoHarpPHLib(dev_idx=1)
    print("Library version:", ph.GetLibraryVersion())
    try:
        serial = ph.OpenDevice()
        print("Opened device serial:", serial)

        ph.Initialize(mode="T2")
        #ph.SetBinning(5)
        #ph.SetInputCFD(0, 400, 10)
        #ph.SetInputCFD(1, 400, 10)
        print('ok')
        #a = ph.GetBaseResolution()
        #print("Base resolution (ps):", a)
        print("Starting short measurement (2000 ms)...")
        ph.StartMeas(2000)  # milliseconds

        # poll read
        t0 = time.time()
        total = 0
        while time.time() - t0 < 2.5:
            arr = ph.ReadFiFo(2**14)
            if arr.size:
                total += arr.size
                print(f"Read {arr.size} records (total {total})")
            else:
                time.sleep(0.01)

        ph.StopMeas()
        print("Done, total records read:", total)
    finally:
        try:
            ph.CloseDevice()
        except Exception:
            pass
'''
