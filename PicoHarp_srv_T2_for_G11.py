"""
Minimal Pyro4 PicoHarp server for single-file T2 acquisitions.
Adapted for PicoHarp Device 2 used from G11 for MoTe2 project.
"""

import ctypes
import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import numpy as np
import Pyro4

try:
    import numpy.core as _numpy_core
    sys.modules.setdefault("numpy._core", _numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
except Exception:
    pass

HOST = "g13-sspd.dhcp-int.phys.ethz.ch"
# CHANGED: Use port 9090
PORT = 9090
# CHANGED: Use a distinct object ID for the second device
OBJECT_ID = "Picoharp2_g13" 
DLL_PATH = Path(r"C:\Users\SSPDs\conda-envs\hydra_env_test\DLLs\phlib64.dll")

MODE_T2 = 2
FIFO_READ_SIZE = 2**17

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}
Pyro4.config.SERVERTYPE = "thread"
Pyro4.config.COMMTIMEOUT = 0


if os.name != "nt":
    raise RuntimeError("This server targets the Windows PicoHarp DLL.")

_phlib = ctypes.WinDLL(str(DLL_PATH))


def _prototype_functions():
    _phlib.PH_GetLibraryVersion.restype = ctypes.c_int
    _phlib.PH_GetLibraryVersion.argtypes = [ctypes.c_char_p]

    _phlib.PH_GetErrorString.restype = ctypes.c_int
    _phlib.PH_GetErrorString.argtypes = [ctypes.c_char_p, ctypes.c_int]

    _phlib.PH_OpenDevice.restype = ctypes.c_int
    _phlib.PH_OpenDevice.argtypes = [ctypes.c_int, ctypes.c_char_p]

    _phlib.PH_CloseDevice.restype = ctypes.c_int
    _phlib.PH_CloseDevice.argtypes = [ctypes.c_int]

    _phlib.PH_Initialize.restype = ctypes.c_int
    _phlib.PH_Initialize.argtypes = [ctypes.c_int, ctypes.c_int]

    _phlib.PH_SetInputCFD.restype = ctypes.c_int
    _phlib.PH_SetInputCFD.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    _phlib.PH_GetCountRate.restype = ctypes.c_int
    _phlib.PH_GetCountRate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

    _phlib.PH_GetWarnings.restype = ctypes.c_int
    _phlib.PH_GetWarnings.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

    _phlib.PH_GetWarningsText.restype = ctypes.c_int
    _phlib.PH_GetWarningsText.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

    _phlib.PH_StartMeas.restype = ctypes.c_int
    _phlib.PH_StartMeas.argtypes = [ctypes.c_int, ctypes.c_int]

    _phlib.PH_StopMeas.restype = ctypes.c_int
    _phlib.PH_StopMeas.argtypes = [ctypes.c_int]

    _phlib.PH_CTCStatus.restype = ctypes.c_int
    _phlib.PH_CTCStatus.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

    _phlib.PH_ReadFiFo.restype = ctypes.c_int
    _phlib.PH_ReadFiFo.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]


def _check(ret):
    if ret == 0:
        return

    buf = ctypes.create_string_buffer(256)
    try:
        _phlib.PH_GetErrorString(buf, ctypes.c_int(ret))
        msg = buf.value.decode(errors="ignore")
    except Exception:
        msg = ""

    if msg:
        raise RuntimeError(f"PHlib error {ret}: {msg}")
    raise RuntimeError(f"PHlib error {ret}")


_prototype_functions()


@Pyro4.expose
class PicoHarpT2Server:
    def __init__(self, dev_idx=0): #CHANGED !!!!!!!  CHANGED: defaults to 1 for the second device NO after reboot is now 1 for the G11 PH
        self.dev = int(dev_idx)
        self._lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._is_open = False
        self._is_acquiring = False
        self._is_measuring = False
        self._worker = None
        self._status = self._idle_status()
        self._log = []
        self._cv = threading.Condition()

    def GetLibraryVersion(self):
        buf = ctypes.create_string_buffer(32)
        with self._lock:
            _check(_phlib.PH_GetLibraryVersion(buf))
        return buf.value.decode(errors="ignore")

    def opendefaults(self):
        serial = self.open()
        self.initialize_t2()
        self.set_cfd_defaults()
        return {
            "serial": serial,
            "count_rate_0": self.get_count_rate(0),
            "count_rate_1": self.get_count_rate(1),
            "warnings": self.get_warnings_text(),
        }

    def open(self):
        serial = ctypes.create_string_buffer(32)
        with self._lock:
            _check(_phlib.PH_OpenDevice(ctypes.c_int(self.dev), serial))
            self._is_open = True
        return serial.value.decode(errors="ignore")

    def initialize_t2(self):
        with self._lock:
            _check(_phlib.PH_Initialize(ctypes.c_int(self.dev), ctypes.c_int(MODE_T2)))

    def set_cfd_defaults(self):
        self.set_input_cfd(channel=0, level=210, zerocross=20)
        self.set_input_cfd(channel=1, level=210, zerocross=20)

    def set_input_cfd(self, channel, level, zerocross):
        with self._lock:
            _check(
                _phlib.PH_SetInputCFD(
                    ctypes.c_int(self.dev),
                    ctypes.c_int(channel),
                    ctypes.c_int(level),
                    ctypes.c_int(zerocross),
                )
            )

    def get_count_rate(self, channel):
        count = ctypes.c_int()
        with self._lock:
            _check(_phlib.PH_GetCountRate(ctypes.c_int(self.dev), ctypes.c_int(channel), ctypes.byref(count)))
        return int(count.value)

    def get_warnings_text(self):
        warnings = ctypes.c_int()
        text = ctypes.create_string_buffer(16384)
        with self._lock:
            _check(_phlib.PH_GetWarnings(ctypes.c_int(self.dev), ctypes.byref(warnings)))
            _check(_phlib.PH_GetWarningsText(ctypes.c_int(self.dev), text, warnings))
        return text.value.decode(errors="ignore")

    def go(self, acq_time_ms, filename, metadata=None, progress_interval_s=5.0):
        if acq_time_ms <= 0:
            raise ValueError("acq_time_ms must be positive")
        if not filename:
            raise ValueError("filename must be a non-empty server-side path")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict or None")
        if progress_interval_s <= 0:
            raise ValueError("progress_interval_s must be positive")
        metadata = self._json_safe(metadata)

        with self._lock:
            if self._is_acquiring:
                raise RuntimeError("A T2 acquisition is already running")
            path = self._resolve_acquisition_path(filename)
            metadata_path = self._metadata_path(path)
            self._reserve_acquisition_file(path)
            self._is_acquiring = True
            self._stop_requested.clear()
            self._status = {
                "state": "starting",
                "filename": str(path),
                "metadata_filename": str(metadata_path),
                "acq_time_ms": int(acq_time_ms),
                "elapsed_ms": 0,
                "progress": 0.0,
                "records": 0,
                "bytes": 0,
                "stopped_by_user": False,
                "error": None,
                "result": None,
            }

        try:
            self._write_metadata(metadata_path, metadata, acq_time_ms, path)
        except Exception:
            with self._lock:
                self._is_acquiring = False
                self._status = self._idle_status()
            raise

        with self._lock:
            self._worker = threading.Thread(
                target=self._run_go,
                args=(int(acq_time_ms), path, metadata_path, float(progress_interval_s)),
                daemon=True,
            )
            self._worker.start()

        return {
            "started": True,
            "filename": str(path),
            "metadata_filename": str(metadata_path),
            "acq_time_ms": int(acq_time_ms),
        }

    def _run_go(self, acq_time_ms, path, metadata_path, progress_interval_s):
        start_time = time.time()
        next_progress_time = start_time
        records = 0
        stopped_by_user = False
        buffer = np.empty(FIFO_READ_SIZE, dtype=np.uint32)
        nactual = ctypes.c_int()

        try:
            with self._lock:
                self._status["state"] = "running"
                self._status["started_unix_time"] = start_time
                _check(_phlib.PH_StartMeas(ctypes.c_int(self.dev), ctypes.c_int(int(acq_time_ms))))
                self._is_measuring = True

            with open(path, "r+b", buffering=0) as output:
                while not self._stop_requested.is_set():
                    now = time.time()
                    if now >= next_progress_time:
                        self._update_progress(start_time, acq_time_ms, records)
                        self._emit_progress(start_time, acq_time_ms, records)
                        next_progress_time = now + float(progress_interval_s)

                    with self._lock:
                        _check(
                            _phlib.PH_ReadFiFo(
                                ctypes.c_int(self.dev),
                                buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                                ctypes.c_int(buffer.size),
                                ctypes.byref(nactual),
                            )
                        )
                        nread = int(nactual.value)

                    if nread:
                        output.write(buffer[:nread].tobytes())
                        records += nread
                        continue

                    if self.ctc_status() == 1:
                        break
                    time.sleep(0.01)

            stopped_by_user = self._stop_requested.is_set()
            self._update_progress(start_time, acq_time_ms, records)
            self._emit_progress(start_time, acq_time_ms, records, final=True)
            result = {
                "filename": str(path),
                "metadata_filename": str(metadata_path),
                "acq_time_ms": int(acq_time_ms),
                "elapsed_ms": int((time.time() - start_time) * 1000),
                "records": int(records),
                "bytes": int(records * 4),
                "stopped_by_user": stopped_by_user,
            }
            with self._lock:
                self._status.update(result)
                self._status["state"] = "stopped" if stopped_by_user else "done"
                self._status["progress"] = min(1.0, self._status["elapsed_ms"] / max(1, int(acq_time_ms)))
                self._status["result"] = dict(result)
        except Exception as exc:
            with self._lock:
                self._status["state"] = "error"
                self._status["error"] = repr(exc)
            self._append_log(f"\n[T2] acquisition failed: {exc}\n")
            print(f"\n[T2] acquisition failed: {exc}", flush=True)
        finally:
            self._stop_measurement_device()

    def status(self):
        with self._lock:
            status = dict(self._status)
            status["busy"] = self._is_acquiring
        status["progress_bar"] = self._progress_bar(status.get("progress", 0.0))
        return status

    def wait(self, timeout_s=None):
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=None if timeout_s is None else float(timeout_s))
        return self.status()

    def lock_file(self, filename, include_metadata=True):
        if not filename:
            raise ValueError("filename must be a non-empty server-side path")

        path = Path(filename)
        with self._lock:
            active_file = self._status.get("filename")
            if self._is_acquiring and active_file and Path(active_file) == path:
                raise RuntimeError("Cannot lock a file while its acquisition is still running; call wait() first")

        paths = [path]
        metadata_path = self._metadata_path(path)
        if include_metadata and metadata_path.exists():
            paths.append(metadata_path)

        locked = []
        for item in paths:
            if not item.exists():
                raise FileNotFoundError(str(item))
            before = item.stat().st_mode
            item.chmod(before & ~stat.S_IWRITE & ~stat.S_IWGRP & ~stat.S_IWOTH)
            after = item.stat().st_mode
            locked.append(
                {
                    "path": str(item),
                    "readonly": not bool(after & stat.S_IWRITE),
                    "mode_before": int(before),
                    "mode_after": int(after),
                }
            )

        self._append_log(f"[T2] locked read-only: {', '.join(entry['path'] for entry in locked)}\n")
        return {"locked": locked}

    def unlock_file(self, filename, include_metadata=True):
        if not filename:
            raise ValueError("filename must be a non-empty server-side path")

        path = Path(filename)
        paths = [path]
        metadata_path = self._metadata_path(path)
        if include_metadata and metadata_path.exists():
            paths.append(metadata_path)

        unlocked = []
        for item in paths:
            if not item.exists():
                raise FileNotFoundError(str(item))
            before = item.stat().st_mode
            item.chmod(before | stat.S_IWRITE)
            after = item.stat().st_mode
            unlocked.append(
                {
                    "path": str(item),
                    "readonly": not bool(after & stat.S_IWRITE),
                    "mode_before": int(before),
                    "mode_after": int(after),
                }
            )

        self._append_log(f"[T2] unlocked writable: {', '.join(entry['path'] for entry in unlocked)}\n")
        return {"unlocked": unlocked}

    def _idle_status(self):
        return {
            "state": "idle",
            "filename": None,
            "metadata_filename": None,
            "acq_time_ms": 0,
            "elapsed_ms": 0,
            "progress": 0.0,
            "records": 0,
            "bytes": 0,
            "stopped_by_user": False,
            "error": None,
            "result": None,
        }

    def _update_progress(self, start_time, acq_time_ms, records):
        elapsed_ms = int((time.time() - start_time) * 1000)
        progress = min(1.0, elapsed_ms / max(1, int(acq_time_ms)))
        with self._lock:
            self._status["elapsed_ms"] = elapsed_ms
            self._status["progress"] = progress
            self._status["records"] = int(records)
            self._status["bytes"] = int(records * 4)

    def _metadata_path(self, data_path):
        return data_path.with_name(f"{data_path.stem}_metadata.json")

    def _reserve_acquisition_file(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb"):
            pass

    def _resolve_acquisition_path(self, target):
        target = Path(target)
        if target.suffix.lower() == ".b":
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite existing acquisition file: {target}")
            return target

        target.mkdir(parents=True, exist_ok=True)
        next_index = self._next_acquisition_index(target)
        return target / f"acq_{next_index}.b"

    def _next_acquisition_index(self, directory):
        highest = 0
        for path in directory.glob("acq_*.b"):
            index = self._acquisition_index_from_stem(path.stem)
            if index is not None:
                highest = max(highest, index)
        for path in directory.glob("acq_*_metadata.json"):
            stem = path.stem.removesuffix("_metadata")
            index = self._acquisition_index_from_stem(stem)
            if index is not None:
                highest = max(highest, index)
        return highest + 1

    def _acquisition_index_from_stem(self, stem):
        prefix = "acq_"
        if not stem.startswith(prefix):
            return None
        suffix = stem[len(prefix):]
        if not suffix.isdigit():
            return None
        return int(suffix)

    def _json_safe(self, value):
        if value is None:
            return None
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        return value

    def _write_metadata(self, metadata_path, metadata, acq_time_ms, data_path):
        payload = {
            "data_filename": str(data_path),
            "acq_time_ms": int(acq_time_ms),
            "mode": "T2",
            "metadata": metadata or {},
            "created_unix_time": time.time(),
        }
        with metadata_path.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)

    def _emit_progress(self, start_time, acq_time_ms, records, final=False):
        elapsed_s = max(0.0, time.time() - start_time)
        total_s = max(0.001, float(acq_time_ms) / 1000.0)
        fraction = min(1.0, elapsed_s / total_s)
        width = 30
        filled = int(round(width * fraction))
        bar = "#" * filled + "-" * (width - filled)
        prefix = "\n" if final else "\r"
        suffix = "\n" if final else ""
        msg = (
            f"{prefix}[T2] [{bar}] {100.0 * fraction:5.1f}% "
            f"{self._format_duration(elapsed_s)} / {self._format_duration(total_s)} "
            f"records={int(records)}{suffix}"
        )
        print(msg, end="", flush=True)
        self._append_log(msg)

    def _progress_bar(self, progress):
        fraction = min(1.0, max(0.0, float(progress)))
        width = 30
        filled = int(round(width * fraction))
        return f"[{'#' * filled}{'-' * (width - filled)}] {100.0 * fraction:5.1f}%"

    def _format_duration(self, seconds):
        seconds = int(round(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

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

    def ctc_status(self):
        status = ctypes.c_int()
        with self._lock:
            _check(_phlib.PH_CTCStatus(ctypes.c_int(self.dev), ctypes.byref(status)))
        return int(status.value)

    def stop(self):
        self._stop_requested.set()
        with self._lock:
            if self._is_acquiring and self._status.get("state") == "running":
                self._status["state"] = "stopping"
        self._stop_measurement_device()

    def _stop_measurement_device(self):
        with self._lock:
            if self._is_measuring:
                _check(_phlib.PH_StopMeas(ctypes.c_int(self.dev)))
                self._is_measuring = False
            if self._is_acquiring and self._status.get("state") in {"done", "stopped", "error"}:
                self._is_acquiring = False

    def close(self):
        self.stop()
        self.wait(timeout_s=10.0)
        with self._lock:
            if self._is_open:
                _check(_phlib.PH_CloseDevice(ctypes.c_int(self.dev)))
                self._is_open = False

if __name__ == "__main__":
    pico = PicoHarpT2Server(dev_idx=1)
    daemon = Pyro4.Daemon(host=HOST, port=PORT)
    uri = daemon.register(pico, objectId=OBJECT_ID)
    print("Obj URI:", uri)
    daemon.requestLoop()