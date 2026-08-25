"""Detector implementations register themselves with the registry on import.

Importing this package (which any submodule import already triggers, e.g.
`from .detectors.registry import get_detector`) is what makes every game's
detector available via `registry.get_detector`.
"""

from __future__ import annotations

from . import avatar  # noqa: F401
from . import sf6  # noqa: F401
