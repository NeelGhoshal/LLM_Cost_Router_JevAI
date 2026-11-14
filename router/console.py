from __future__ import annotations

import sys


def utf8_stdout() -> None:
    """Model output contains characters the default Windows console encoding cannot print."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
