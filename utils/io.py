"""Byte-stable text output.

``Path.write_text`` opens in text mode, so on Windows every ``\\n`` becomes
``\\r\\n``. That would make an artifact's bytes — and therefore its hash —
depend on the operating system that produced it, which breaks the one claim this
project exists to defend: clone, run, get the identical thing.

Everything the pipeline writes goes through here and is LF, everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def write_json(path: Path, obj: Any, indent: int = 2) -> Path:
    return write_text(path, json.dumps(obj, indent=indent) + "\n")
