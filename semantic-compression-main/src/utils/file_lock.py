"""Міжпроцесне блокування файлу (Linux / Windows)."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lockf:
        lockf.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(lockf.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            lockf.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(lockf.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
