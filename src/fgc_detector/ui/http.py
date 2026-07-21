"""A deliberately dumb static file server for the config page.

It serves one HTML file and nothing else. Every read and write the page
performs goes over the WebSocket, so there is exactly one protocol to maintain
and the page can do nothing the dashboard cannot also do.
"""

from __future__ import annotations

import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger(__name__)

_PAGE_PATH = Path(__file__).parent / "index.html"

# Shared with index.html: that file must contain this exact literal for the
# port substitution below to take effect. If either side is renamed without
# the other, this becomes a silent no-op — see test_page_no_longer_contains_
# the_placeholder in tests/test_ui_http.py, which is the only thing that
# catches that.
WS_PORT_PLACEHOLDER = "__WS_PORT__"


def find_free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _build_handler(ws_port: int) -> type[BaseHTTPRequestHandler]:
    page = _PAGE_PATH.read_text().replace(WS_PORT_PLACEHOLDER, str(ws_port))
    body = page.encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802  (stdlib naming)
            if self.path not in ("/", "/index.html"):
                self.send_error(404, "not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            log.debug("ui http: " + format, *args)

    return Handler


def serve_ui(
    host: str, port: int, ws_port: int
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the config page on a daemon thread. Returns (server, thread)."""
    server = ThreadingHTTPServer((host, port), _build_handler(ws_port))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info("config page at http://%s:%s", host, port)
    return server, thread
