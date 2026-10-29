from __future__ import annotations

import csv
import json
import re
import socket
import time
from datetime import datetime
from pathlib import Path
from threading import Event
from urllib.parse import urljoin

import requests
import websocket


def parse_tcp_counts_line(line: str) -> dict | None:
    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) < 5:
        return None
    timestamp = float(parts[0])
    channels = {channel: float(parts[channel]) for channel in range(1, 5)}
    return {
        "local_time_s": timestamp,
        "local_iso": datetime.fromtimestamp(timestamp).isoformat(timespec="milliseconds"),
        "websq_time": None,
        "channels_hz": channels,
        "raw": line.strip(),
    }


class AtlasTCPCounts:
    def __init__(self, host: str = "192.168.1.1", port: int = 12345, timeout: float = 10.0):
        self.host = re.sub(r"^https?://", "", str(host)).strip("/")
        self.port = int(port)
        self.timeout = float(timeout)
        self.socket: socket.socket | None = None
        self._buffer = b""

    def __enter__(self) -> "AtlasTCPCounts":
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

    def next_counts(self, *, timeout_s: float | None = None) -> dict:
        self.connect()
        assert self.socket is not None
        old_timeout = self.socket.gettimeout()
        if timeout_s is not None:
            self.socket.settimeout(float(timeout_s))
        try:
            while True:
                if b"\n" in self._buffer:
                    line, self._buffer = self._buffer.split(b"\n", 1)
                    sample = parse_tcp_counts_line(line.decode(errors="replace"))
                    if sample is not None:
                        return sample
                    continue
                chunk = self.socket.recv(4096)
                if not chunk:
                    raise ConnectionError("Atlas TCP count stream closed")
                self._buffer += chunk
        finally:
            self.socket.settimeout(old_timeout)

    def iter_counts(self, *, duration_s: float | None = None, stop_event: Event | None = None):
        start = time.time()
        while True:
            if stop_event is not None and stop_event.is_set():
                return
            if duration_s is not None and time.time() - start >= float(duration_s):
                return
            yield self.next_counts(timeout_s=2.0)


def _csrf_token(html: str) -> str | None:
    match = re.search(r'name=["\']csrf_token["\'][^>]*value=["\']([^"\']+)["\']', html)
    if not match:
        match = re.search(r'value=["\']([^"\']+)["\'][^>]*name=["\']csrf_token["\']', html)
    return None if match is None else match.group(1)


def _cookie_header(session: requests.Session) -> str:
    return "; ".join(f"{cookie.name}={cookie.value}" for cookie in session.cookies)


def parse_counts_payload(payload: dict, *, local_time_s: float | None = None) -> dict | None:
    if not isinstance(payload, dict) or payload.get("label") != "counts":
        return None
    entries = payload.get("counts")
    if not isinstance(entries, list) or not entries:
        return None

    entry = entries[-1]
    if not isinstance(entry, dict):
        return None

    local_time_s = time.time() if local_time_s is None else float(local_time_s)
    channels = {}
    for channel in range(1, 5):
        value = entry.get(f"y{channel}")
        channels[channel] = None if value is None else float(value)

    return {
        "local_time_s": local_time_s,
        "local_iso": datetime.fromtimestamp(local_time_s).isoformat(timespec="milliseconds"),
        "websq_time": entry.get("x"),
        "channels_hz": channels,
        "raw": payload,
    }


class AtlasWebSQCounts:
    def __init__(
        self,
        host: str = "192.168.1.1",
        *,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 10.0,
    ):
        self.host = re.sub(r"^https?://", "", str(host)).strip("/")
        self.base_url = f"http://{self.host}"
        self.username = username
        self.password = password
        self.timeout = float(timeout)
        self.session = requests.Session()
        self.ws = None

    def __enter__(self) -> "AtlasWebSQCounts":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def login(self) -> None:
        if not self.username or self.password is None:
            return

        login_url = urljoin(self.base_url, "/login")
        response = self.session.get(login_url, timeout=self.timeout)
        response.raise_for_status()

        payload = {
            "username": self.username,
            "password": self.password,
        }
        token = _csrf_token(response.text)
        if token:
            payload["csrf_token"] = token

        # WebSQ redirects /login -> /index -> /login with requests, while still
        # setting the cookies needed by the websocket. Avoid following the loop.
        response = self.session.post(login_url, data=payload, timeout=self.timeout, allow_redirects=False)
        response.raise_for_status()
        if not any(cookie.name in {"login", "session", "remember_token"} for cookie in self.session.cookies):
            raise RuntimeError("WebSQ login did not return authentication cookies")

    def connect(self) -> None:
        if self.ws is not None:
            return
        self.login()
        headers = []
        cookie = _cookie_header(self.session)
        if cookie:
            headers.append(f"Cookie: {cookie}")
        ws_url = f"ws://{self.host}/websocket"
        self.ws = websocket.create_connection(ws_url, timeout=self.timeout, header=headers, origin=self.base_url)
        self.ws.settimeout(min(2.0, self.timeout))

    def close(self) -> None:
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    def send_pong(self) -> None:
        if self.ws is not None:
            self.ws.send(json.dumps({"request": "pong"}))

    def next_counts(self, *, timeout_s: float | None = None, send_pong: bool = True) -> dict:
        self.connect()
        deadline = None if timeout_s is None else time.time() + float(timeout_s)
        while deadline is None or time.time() < deadline:
            try:
                msg = self.ws.recv()
            except Exception:
                if send_pong:
                    self.send_pong()
                continue
            if isinstance(msg, bytes):
                continue
            try:
                payload = json.loads(msg)
            except json.JSONDecodeError:
                continue
            sample = parse_counts_payload(payload)
            if sample is not None:
                return sample
        raise TimeoutError("Timed out waiting for a WebSQ counts websocket frame")

    def iter_counts(
        self,
        *,
        duration_s: float | None = None,
        stop_event: Event | None = None,
        send_pong: bool = True,
    ):
        start = time.time()
        while True:
            if stop_event is not None and stop_event.is_set():
                return
            if duration_s is not None and time.time() - start >= float(duration_s):
                return
            try:
                yield self.next_counts(timeout_s=2.0, send_pong=send_pong)
            except TimeoutError:
                if send_pong:
                    self.send_pong()


def log_count_rates_csv(
    output_path: str | Path,
    *,
    host: str = "192.168.1.1",
    username: str | None = None,
    password: str | None = None,
    duration_s: float | None = None,
    stop_event: Event | None = None,
    status_callback=None,
) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "local_time_s",
        "local_iso",
        "websq_time",
        "ch1_hz",
        "ch2_hz",
        "ch3_hz",
        "ch4_hz",
        "raw_json",
    ]

    start = time.time()
    rows = 0
    with AtlasWebSQCounts(host=host, username=username, password=password) as counts:
        with output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if handle.tell() == 0:
                writer.writeheader()
                handle.flush()
            for sample in counts.iter_counts(duration_s=duration_s, stop_event=stop_event):
                channels = sample["channels_hz"]
                row = {
                    "local_time_s": f"{sample['local_time_s']:.6f}",
                    "local_iso": sample["local_iso"],
                    "websq_time": sample["websq_time"],
                    "ch1_hz": channels.get(1),
                    "ch2_hz": channels.get(2),
                    "ch3_hz": channels.get(3),
                    "ch4_hz": channels.get(4),
                    "raw_json": json.dumps(sample["raw"], separators=(",", ":")),
                }
                writer.writerow(row)
                handle.flush()
                rows += 1
                if status_callback is not None:
                    status_callback(
                        {
                            "rows": rows,
                            "last_sample": sample,
                            "elapsed_s": time.time() - start,
                        }
                    )

    return {
        "output_path": str(output_path),
        "rows": rows,
        "elapsed_s": time.time() - start,
    }


def log_tcp_count_rates_csv(
    output_path: str | Path,
    *,
    host: str = "192.168.1.1",
    port: int = 12345,
    duration_s: float | None = None,
    stop_event: Event | None = None,
    status_callback=None,
) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "local_time_s",
        "local_iso",
        "ch1_hz",
        "ch2_hz",
        "ch3_hz",
        "ch4_hz",
        "raw_line",
    ]

    start = time.time()
    rows = 0
    with AtlasTCPCounts(host=host, port=port) as counts:
        with output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if handle.tell() == 0:
                writer.writeheader()
                handle.flush()
            for sample in counts.iter_counts(duration_s=duration_s, stop_event=stop_event):
                channels = sample["channels_hz"]
                row = {
                    "local_time_s": f"{sample['local_time_s']:.6f}",
                    "local_iso": sample["local_iso"],
                    "ch1_hz": channels.get(1),
                    "ch2_hz": channels.get(2),
                    "ch3_hz": channels.get(3),
                    "ch4_hz": channels.get(4),
                    "raw_line": sample["raw"],
                }
                writer.writerow(row)
                handle.flush()
                rows += 1
                if status_callback is not None:
                    status_callback(
                        {
                            "rows": rows,
                            "last_sample": sample,
                            "elapsed_s": time.time() - start,
                        }
                    )

    return {
        "output_path": str(output_path),
        "rows": rows,
        "elapsed_s": time.time() - start,
    }
