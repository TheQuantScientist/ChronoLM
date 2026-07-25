"""Utilities for accessing the vendored APN code from ChronoLM."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APN_ROOT = PROJECT_ROOT / "APN"


def add_apn_to_path() -> None:
    """Make APN imports available without changing the working directory."""
    apn_path = str(APN_ROOT)
    if apn_path not in sys.path:
        sys.path.insert(0, apn_path)
