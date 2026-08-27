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


# --- the page's half of the protocol ---------------------------------------
#
# The page is the one consumer of the websocket protocol that no other test
# exercises: it is HTML, so a renamed command or a renamed config-event key
# breaks it silently. These tests read the page as text and check that both
# halves still refer to the same names.


def _page() -> str:
    from fgc_detector.ui.http import _PAGE_PATH

    return _PAGE_PATH.read_text()


def test_every_command_the_page_sends_is_a_real_command():
    import re

    from fgc_detector.types import Command

    # Deliberately matches any word, not just well-formed command names: a
    # regex that only recognizes valid names would skip a typo'd one and the
    # assertion below would pass on a page that cannot work.
    sent = set(re.findall(r'cmd:\s*"(\w+)"', _page()))
    assert sent, "found no commands in the page; the regex has drifted"
    known = {item.value for item in Command}
    assert sent <= known, f"page sends unknown commands: {sorted(sent - known)}"


def test_the_page_reads_the_capture_and_confirmer_keys_the_server_publishes():
    """Both blocks are addressed by name in the page. If a key is renamed in
    events.py's ConfigEvent, the page would quietly render `undefined`."""
    from datetime import datetime, timezone

    from fgc_detector.config import ObsConfig
    from fgc_detector.confirmer import ConfirmerConfig
    from fgc_detector.events import ConfigEvent
    from fgc_detector.types import EventType, Game, RuntimeSettings

    published = ConfigEvent(
        settings=RuntimeSettings(
            active_game=Game.SF6,
            enabled_games=frozenset({Game.SF6}),
            enabled_events=frozenset({EventType.MATCH_END}),
        ),
        available_games=[Game.SF6],
        supported_events=frozenset({EventType.MATCH_END}),
        obs=ObsConfig(source_name="Game Capture"),
        confirmer=ConfirmerConfig(),
        ts=datetime(2026, 7, 21, 20, 0, 0, tzinfo=timezone.utc),
    ).to_dict()

    import re

    page = _page()
    for section in ("obs", "confirmer"):
        for key in published[section]:
            # Whole-word: `password_setX` must not satisfy `password_set`.
            assert re.search(rf"\b{key}\b", page), (
                f"the page never mentions {section}.{key}"
            )
