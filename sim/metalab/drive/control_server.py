"""Standalone trajectory control + telemetry server (mirrors ``sim/_runtime/telemetry.py``).

Runs a small ``ThreadingHTTPServer`` on a background daemon thread INSIDE the standalone runner
process, giving the browser one channel in each direction:

- ``GET  /``         → the trajectory control + live-plot dashboard page (``drive/dashboard_page.PAGE`` —
                        shared with the Launchpad's offline ``/simui/`` preview; Launchpad embeds this via iframe)
- ``GET  /describe`` → static run info ``{engine, task, joints, channels, groups, control_hz, gravcomp,
                        controls}`` (once). ``channels`` = the plot tabs (``drive/monitor.describe``): per
                        tab ``{key, title, unit, labels, digits}``. ``controls`` = the Joint Control rows:
                        per joint ``{name, lo, hi, bounded, init}``, all DEGREES.
- ``GET  /stream``   → SSE live snapshots (down): ``{t, playing, paused, finished, duration, group,
                        ch: {channel key: [values...]}}`` — EVERY channel every snapshot, so the page
                        buffers all tabs at once (see the page comment on freeze-on-pause).
- ``POST /cmd``      → commands (up), drained by the runner loop: ``play|pause|stop|mode``. The transport
                        acts on the SIMULATOR (pause halts physics), not just on trajectory playback.

Kept SEPARATE from ``telemetry.py`` — that one is env_driver's train/eval obs/reward/action dashboard.
Standalone bypasses env_driver, so it owns this dedicated down(SSE)+up(POST) channel. The runner:
``publish(snapshot)`` each throttled step, ``drain_commands()`` each loop.

Command schema (POST /cmd JSON):
    {"cmd": "play", "group": "<repo-relative CSV-group dir>", "hz": 1000}   # run; start `group` if idle
    {"cmd": "pause"}                                        # freeze the sim (no physics steps)
    {"cmd": "stop"}                                         # reset to init pose, stay frozen
    {"cmd": "mode", "torque": true|false}                   # Position ↔ Torque (float)
    {"cmd": "joint_target", "targets": {"<joint>": <deg>}}  # Joint Control: the FULL hand-driven set
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import threading
import time

from sim.metalab.drive.dashboard_page import PAGE


class TrajControlServer:
    """See module docstring. Start once in the runner; ``publish``/``drain_commands`` from its loop.

    Never opens a browser: the Launchpad embeds this URL in its Standalone tab (it scrapes the printed
    "[traj] control dashboard →" line), and that is the ONLY viewer — no window of our own."""

    def __init__(self, describe: dict, host: str = "127.0.0.1", port: int = 8771):
        self.describe = describe
        self.latest: dict | None = None
        self.clients = 0                              # active /stream connections
        self.commands: queue.Queue = queue.Queue()    # web → runner commands
        self._stop = threading.Event()
        self._lock = threading.Lock()

        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):               # silence per-request stderr logging
                pass

            def _send(self, body: bytes, ctype: str):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/index"):
                    self._send(PAGE.encode(), "text/html; charset=utf-8")   # drive/dashboard_page.py
                elif self.path == "/describe":
                    self._send(json.dumps(server.describe).encode(), "application/json")
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    with server._lock:
                        server.clients += 1
                    last = None
                    try:
                        while not server._stop.is_set():
                            snap = server.latest
                            if snap is not None and snap is not last:
                                last = snap
                                self.wfile.write(b"data: " + json.dumps(snap).encode() + b"\n\n")
                                self.wfile.flush()
                            time.sleep(0.01)          # poll > publish rate so ~60 Hz snaps reach the browser
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass                          # client closed the tab — normal
                    finally:
                        with server._lock:
                            server.clients -= 1
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path == "/cmd":
                    try:
                        n = int(self.headers.get("Content-Length", 0) or 0)
                        server.commands.put(json.loads(self.rfile.read(n) or b"{}"))
                        self.send_response(204)
                        self.end_headers()
                    except Exception:
                        self.send_error(400)
                else:
                    self.send_error(404)

        # Bind the preferred port; fall back to an OS-assigned free port if taken.
        try:
            self._httpd = ThreadingHTTPServer((host, port), Handler)
        except OSError:
            self._httpd = ThreadingHTTPServer((host, 0), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        url = f"http://{host}:{self.port}"
        # Launchpad scrapes this exact marker from the run log to embed the URL in the Standalone trajectory tab.
        print(f"[traj] control dashboard → {url}   (standalone trajectory player)", flush=True)

    def has_clients(self) -> bool:
        return self.clients > 0

    def drain_commands(self) -> list:
        """Pop all pending web→runner commands (called by the runner loop each step)."""
        out = []
        try:
            while True:
                out.append(self.commands.get_nowait())
        except queue.Empty:
            pass
        return out

    def publish(self, snapshot: dict) -> None:
        self.latest = snapshot                        # atomic ref swap (GIL) — SSE threads read it

    def close(self) -> None:
        self._stop.set()
        self._httpd.shutdown()
