import urllib.error
import urllib.request

import pytest

from fgc_detector.ui.http import find_free_port, serve_ui


@pytest.fixture
def ui():
    port = find_free_port()
    server, thread = serve_ui("127.0.0.1", port, ws_port=6600)
    yield port
    server.shutdown()
    thread.join(timeout=5)


def _get(port: int, path: str = "/") -> tuple[int, str]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
        return response.status, response.read().decode()


def test_serves_the_page_at_root(ui):
    status, body = _get(ui)
    assert status == 200
    assert "<title>" in body


def test_page_is_told_which_websocket_port_to_use(ui):
    _, body = _get(ui)
    assert "6600" in body, "the ws port must be injected into the page"


def test_page_no_longer_contains_the_placeholder(ui):
    """Guards the producer/consumer contract: if the placeholder constant is
    renamed in http.py but not in index.html (or vice versa), substitution
    silently no-ops and the served page still contains the literal
    placeholder string instead of the real port. This test fails in exactly
    that case."""
    _, body = _get(ui)
    assert "__WS_PORT__" not in body


def test_unknown_path_returns_404(ui):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(ui, "/nope")
    assert excinfo.value.code == 404


def test_find_free_port_returns_a_usable_port():
    assert 1024 < find_free_port() < 65536
