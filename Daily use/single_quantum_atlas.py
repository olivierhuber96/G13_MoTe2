from __future__ import annotations

import json
import socket
import time
from contextlib import contextmanager
from typing import Iterable, Sequence


class AtlasSSPD:
    """
    Minimal Single Quantum Atlas control helper.

    This sends the same JSON commands as the Single Quantum WebSQControl driver:
    ``DetectorEnable``, ``SetAllBiasCurrents``, and related settings on the
    Atlas control port.
    """

    def __init__(self, host: str = "localhost", port: int = 12000, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: socket.socket | None = None

    def __enter__(self) -> "AtlasSSPD":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        if self.socket is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        self.socket = sock

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def _send(self, payload: dict) -> None:
        if self.socket is None:
            raise RuntimeError("AtlasSSPD is not connected")
        self.socket.send(bytes(json.dumps(payload), "utf-8"))

    def enable_detectors(self, state: bool) -> None:
        self._send(
            {
                "command": "DetectorEnable",
                "label": "DetectorEnable",
                "value": bool(state),
            }
        )

    def set_bias_current(self, currents_uA: Sequence[float]) -> None:
        self._send(
            {
                "command": "SetAllBiasCurrents",
                "label": "BiasCurrent",
                "value": [float(current) for current in currents_uA],
            }
        )

    def set_measurement_period(self, period_ms: int | float) -> None:
        """
        Set the Atlas input/count-rate integration period in ms.
        """
        period_ms = int(round(float(period_ms)))
        if period_ms <= 0:
            raise ValueError("period_ms must be positive")
        self._send(
            {
                "command": "SetMeasurementPeriod",
                "label": "InptMeasurementPeriod",
                "value": period_ms,
            }
        )

    def protect(self, n_channels: int = 4, settle: float = 0.5) -> None:
        """
        Disable detection and set all bias currents to 0 uA.
        """
        self.enable_detectors(False)
        self.set_bias_current([0.0] * n_channels)
        if settle > 0:
            time.sleep(settle)

    def restore(
        self,
        currents_uA: Sequence[float],
        *,
        settle: float = 0.5,
        enable: bool = True,
    ) -> None:
        """
        Restore bias currents and optionally enable detectors.
        """
        self.set_bias_current(currents_uA)
        if settle > 0:
            time.sleep(settle)
        if enable:
            self.enable_detectors(True)


@contextmanager
def protected_sspds(
    host: str,
    restore_currents_uA: Sequence[float],
    *,
    port: int = 12000,
    n_channels: int,
    protect_settle: float = 0.5,
    restore_settle: float = 0.5,
):
    """
    Context manager that disables Atlas SSPDs, then restores them afterwards.

    Use this around Rigol setup/output-enable code. If an exception occurs, the
    SSPDs are still restored to ``restore_currents_uA`` before the exception is
    re-raised.
    """
    if n_channels is None:
        n_channels = len(restore_currents_uA)
    with AtlasSSPD(host=host, port=port) as atlas:
        atlas.protect(n_channels=n_channels, settle=protect_settle)
        try:
            yield atlas
        finally:
            atlas.restore(restore_currents_uA, settle=restore_settle)
