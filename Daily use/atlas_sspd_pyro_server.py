"""
Pyro4 bridge for a Single Quantum Atlas/WebSQ SSPD controller.

Run this on the PC that can reach the Atlas web driver locally. The lab PC can
then call the exposed methods through Pyro4 without being on the Atlas private
192.168.1.x network.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import time

import Pyro4


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from single_quantum_atlas import AtlasSSPD
from atlas_websq_counts import AtlasTCPCounts, log_tcp_count_rates_csv


Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle", "serpent"}
Pyro4.config.SERVERTYPE = "thread"


@Pyro4.expose
class AtlasSSPDProxy:
    def __init__(
        self,
        atlas_host: str = "192.168.1.1",
        atlas_port: int = 12000,
        atlas_count_port: int = 12345,
        count_measurement_period_ms: int = 2000,
    ):
        self.atlas_host = atlas_host
        self.atlas_port = atlas_port
        self.atlas_count_port = int(atlas_count_port)
        self.count_measurement_period_ms = int(count_measurement_period_ms)
        self._count_logger_stop = threading.Event()
        self._count_logger_thread = None
        self._count_logger_status = {"state": "idle"}

    def ping(self) -> str:
        return "AtlasSSPDProxy alive"

    def enable_detectors(self, state: bool) -> bool:
        with AtlasSSPD(host=self.atlas_host, port=self.atlas_port) as atlas:
            atlas.enable_detectors(bool(state))
        return bool(state)

    def set_bias_current(self, currents_uA: list[float]) -> list[float]:
        currents = [float(current) for current in currents_uA]
        with AtlasSSPD(host=self.atlas_host, port=self.atlas_port) as atlas:
            atlas.set_bias_current(currents)
        return currents

    def set_count_measurement_period(self, period_ms: int | float = 2000) -> int:
        period_ms = int(round(float(period_ms)))
        if period_ms <= 0:
            raise ValueError("period_ms must be positive")
        with AtlasSSPD(host=self.atlas_host, port=self.atlas_port) as atlas:
            atlas.set_measurement_period(period_ms)
        self.count_measurement_period_ms = period_ms
        return period_ms

    def protect(self, n_channels: int = 4, settle: float = 0.5) -> bool:
        with AtlasSSPD(host=self.atlas_host, port=self.atlas_port) as atlas:
            atlas.protect(n_channels=int(n_channels), settle=float(settle))
        return True

    def restore(
        self,
        currents_uA: list[float],
        settle: float = 0.5,
        enable: bool = True,
    ) -> list[float]:
        currents = [float(current) for current in currents_uA]
        with AtlasSSPD(host=self.atlas_host, port=self.atlas_port) as atlas:
            atlas.restore(currents, settle=float(settle), enable=bool(enable))
        return currents

    def get_count_rates(self, timeout_s: float = 10.0, measurement_period_ms: int | float | None = None) -> dict:
        measurement_period_ms = self._resolve_count_measurement_period(measurement_period_ms)
        self.set_count_measurement_period(measurement_period_ms)
        with AtlasTCPCounts(
            host=self.atlas_host,
            port=self.atlas_count_port,
            timeout=float(timeout_s),
        ) as counts:
            sample = counts.next_counts(timeout_s=float(timeout_s))
        sample["measurement_period_ms"] = measurement_period_ms
        return sample

    def start_count_rate_logger(
        self,
        output_path: str,
        duration_s: float | None = None,
        measurement_period_ms: int | float | None = None,
    ) -> dict:
        if not output_path:
            raise ValueError("output_path must be a non-empty server-side CSV path")
        if self._count_logger_thread is not None and self._count_logger_thread.is_alive():
            raise RuntimeError("Atlas count-rate logger is already running")
        measurement_period_ms = self._resolve_count_measurement_period(measurement_period_ms)
        self.set_count_measurement_period(measurement_period_ms)

        self._count_logger_stop.clear()
        self._count_logger_status = {
            "state": "starting",
            "output_path": str(output_path),
            "duration_s": None if duration_s is None else float(duration_s),
            "measurement_period_ms": measurement_period_ms,
            "rows": 0,
            "last_sample": None,
            "error": None,
        }

        def status_update(update: dict) -> None:
            self._count_logger_status.update(update)
            self._count_logger_status["state"] = "running"

        def worker() -> None:
            try:
                result = log_tcp_count_rates_csv(
                    output_path,
                    host=self.atlas_host,
                    port=self.atlas_count_port,
                    duration_s=None if duration_s is None else float(duration_s),
                    stop_event=self._count_logger_stop,
                    status_callback=status_update,
                )
                self._count_logger_status.update(result)
                self._count_logger_status["state"] = "stopped" if self._count_logger_stop.is_set() else "done"
            except Exception as exc:
                self._count_logger_status["state"] = "error"
                self._count_logger_status["error"] = repr(exc)

        self._count_logger_thread = threading.Thread(target=worker, daemon=True)
        self._count_logger_thread.start()
        return dict(self._count_logger_status)

    def stop_count_rate_logger(self) -> dict:
        self._count_logger_stop.set()
        thread = self._count_logger_thread
        if thread is not None:
            thread.join(timeout=5.0)
        return self.count_rate_logger_status()

    def count_rate_logger_status(self) -> dict:
        status = dict(self._count_logger_status)
        thread = self._count_logger_thread
        status["busy"] = bool(thread is not None and thread.is_alive())
        return status

    def _resolve_count_measurement_period(self, period_ms: int | float | None) -> int:
        if period_ms is None:
            period_ms = self.count_measurement_period_ms
        period_ms = int(round(float(period_ms)))
        if period_ms <= 0:
            raise ValueError("measurement_period_ms must be positive")
        return period_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Pyro bind host")
    parser.add_argument("--port", default=9097, type=int, help="Pyro bind port")
    parser.add_argument("--name", default="AtlasSSPD", help="Pyro object name")
    parser.add_argument("--atlas-host", default="192.168.1.1", help="Atlas control host")
    parser.add_argument("--atlas-port", default=12000, type=int, help="Atlas control port")
    parser.add_argument("--atlas-count-port", default=12345, type=int, help="Atlas TCP count-rate stream port")
    parser.add_argument("--count-measurement-period-ms", default=2000, type=int, help="Default Atlas count-rate integration period")
    args = parser.parse_args()

    proxy = AtlasSSPDProxy(
        atlas_host=args.atlas_host,
        atlas_port=args.atlas_port,
        atlas_count_port=args.atlas_count_port,
        count_measurement_period_ms=args.count_measurement_period_ms,
    )
    daemon = Pyro4.Daemon(host=args.host, port=args.port)
    uri = daemon.register(proxy, objectId=args.name)

    print(f"Atlas SSPD Pyro proxy ready: {uri}")
    print(f"Forwarding Atlas/WebSQ control commands to {args.atlas_host}:{args.atlas_port}")
    print(
        "From the lab PC, use "
        f"PYRO:{args.name}@g13-sspd.dhcp-int.phys.ethz.ch:{args.port}"
    )
    print("Press Ctrl+C to stop.")
    try:
        daemon.requestLoop()
    finally:
        daemon.close()


if __name__ == "__main__":
    main()
