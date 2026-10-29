import argparse
import getpass
import json
import os
import re
import time
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests
import websocket


class _ScriptParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "script":
            return
        attrs = dict(attrs)
        src = attrs.get("src")
        if src:
            self.scripts.append(src)


def _csrf_token(html):
    match = re.search(r'name=["\']csrf_token["\'][^>]*value=["\']([^"\']+)["\']', html)
    if not match:
        match = re.search(r'value=["\']([^"\']+)["\'][^>]*name=["\']csrf_token["\']', html)
    return None if match is None else match.group(1)


def _cookie_header(session):
    return "; ".join(f"{cookie.name}={cookie.value}" for cookie in session.cookies)


def login(base_url, username, password):
    session = requests.Session()
    login_url = urljoin(base_url, "/login")
    response = session.get(login_url, timeout=10)
    response.raise_for_status()

    token = _csrf_token(response.text)
    payload = {
        "username": username,
        "password": password,
    }
    if token:
        payload["csrf_token"] = token

    response = session.post(login_url, data=payload, timeout=10, allow_redirects=True)
    response.raise_for_status()
    index = session.get(urljoin(base_url, "/index"), timeout=10)
    index.raise_for_status()
    if "Please enter your username" in index.text or 'name="login"' in index.text:
        raise RuntimeError("Login did not reach the authenticated dashboard.")
    return session, index.text


def print_dashboard_hints(base_url, session, index_html):
    parser = _ScriptParser()
    parser.feed(index_html)
    print("Authenticated dashboard scripts:")
    for src in parser.scripts:
        print(" ", src)

    interesting = []
    for src in parser.scripts:
        try:
            js = session.get(urljoin(base_url, src), timeout=10).text
        except Exception as exc:
            print(f"Could not fetch {src}: {exc}")
            continue
        for line in js.splitlines():
            lower = line.lower()
            if any(key in lower for key in ("onmessage", "websocket", "count", "rate", "updateData", "request", "command")):
                interesting.append((src, line.strip()))

    print("\nInteresting JS lines:")
    for src, line in interesting[:200]:
        print(f"[{src}] {line}")
    if len(interesting) > 200:
        print(f"... {len(interesting) - 200} more lines omitted")


def listen_websocket(base_url, session, seconds, send_pong):
    host = re.sub(r"^https?://", "", base_url).rstrip("/")
    ws_url = f"ws://{host}/websocket"
    headers = []
    cookie = _cookie_header(session)
    if cookie:
        headers.append(f"Cookie: {cookie}")
    headers.append(f"Origin: {base_url.rstrip('/')}")

    ws = websocket.create_connection(ws_url, timeout=10, header=headers)
    ws.settimeout(2)
    print(f"\nConnected to {ws_url}")
    deadline = time.time() + float(seconds)
    n = 0
    while time.time() < deadline:
        try:
            msg = ws.recv()
        except Exception as exc:
            if send_pong:
                try:
                    ws.send(json.dumps({"request": "pong"}))
                except Exception:
                    pass
            continue
        n += 1
        if isinstance(msg, bytes):
            print(f"\nMESSAGE {n}: <{len(msg)} bytes>")
            continue
        try:
            payload = json.loads(msg)
            if isinstance(payload, dict):
                print(f"\nMESSAGE {n}: keys={sorted(payload)}")
                print(json.dumps(payload, indent=2)[:3000])
            else:
                print(f"\nMESSAGE {n}: {payload!r}"[:3000])
        except Exception:
            print(f"\nMESSAGE {n}: {msg[:3000]}")
    ws.close()
    print(f"\nReceived {n} websocket messages in {seconds:g} s")


def main():
    parser = argparse.ArgumentParser(description="Probe authenticated Single Quantum WebSQ count-rate data.")
    parser.add_argument("--host", default="192.168.1.1")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--username")
    parser.add_argument("--password-env")
    parser.add_argument("--no-js-hints", action="store_true")
    parser.add_argument("--no-pong", action="store_true")
    args = parser.parse_args()

    base_url = f"http://{args.host}"
    username = args.username or input("WebSQ username: ")
    if args.password_env:
        password = os.environ.get(args.password_env)
        if password is None:
            raise SystemExit(f"Environment variable {args.password_env!r} is not set.")
    else:
        password = getpass.getpass("WebSQ password: ")

    session, index_html = login(base_url, username, password)
    print("Login OK")
    print("Cookies:", ", ".join(cookie.name for cookie in session.cookies) or "<none>")
    if not args.no_js_hints:
        print_dashboard_hints(base_url, session, index_html)
    listen_websocket(base_url, session, args.seconds, send_pong=not args.no_pong)


if __name__ == "__main__":
    main()
