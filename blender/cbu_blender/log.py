"""Logging and stage timing.

Where output actually goes, which trips everyone up the first time:

  * ``self.report()`` shows in the status bar and the Info editor. Good for
    one-line results, useless for debugging.
  * ``print()`` and tracebacks go to the SYSTEM console, which is a separate
    window from Blender's Python Console. On Windows: Window > Toggle System
    Console. On macOS and Linux there is no such menu item - you have to have
    launched Blender from a terminal, or you will never see any of it.
  * Blender's built-in Python Console (Scripting workspace) is interactive but
    does NOT capture output from operators running elsewhere.

So this logger writes to stderr (the system console) and, optionally, to a
file - which is the only way to get output when Blender was launched by
double-clicking an icon.
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

NAME = "cbu"
_file_handler = None

LEVELS = [
    ("ERROR", "Errors only", "Only failures"),
    ("WARNING", "Warnings", "Failures and warnings (default)"),
    ("INFO", "Info", "Stage-by-stage progress"),
    ("DEBUG", "Debug", "Everything, including per-island detail"),
]


def get() -> logging.Logger:
    log = logging.getLogger(NAME)
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("[cbu] %(levelname)-7s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.WARNING)
        log.propagate = False
    return log


def set_level(name: str) -> None:
    get().setLevel(getattr(logging, name, logging.WARNING))


def set_file(path: str | None) -> str | None:
    """Mirror the log to a file. The only reliable way to see output when
    Blender was started from a desktop icon rather than a terminal."""
    global _file_handler
    log = get()
    if _file_handler is not None:
        log.removeHandler(_file_handler)
        try:
            _file_handler.close()
        except Exception:                                   # noqa: BLE001
            pass
        _file_handler = None
    if not path:
        return None
    try:
        h = logging.FileHandler(path, mode="w", encoding="utf-8")
    except OSError as exc:
        log.error("cannot open log file %s: %s", path, exc)
        return None
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    _file_handler = h
    log.info("logging to %s", path)
    return path


class Timings(dict):
    """Per-stage wall clock, so a slow step is obvious as features land."""

    def report(self) -> str:
        if not self:
            return ""
        total = sum(self.values())
        parts = " ".join(f"{k} {v * 1000:.0f}ms" for k, v in self.items())
        return f"{parts}  (total {total * 1000:.0f}ms)"


@contextmanager
def stage(timings: Timings | None, name: str):
    log = get()
    t0 = time.perf_counter()
    log.debug("-> %s", name)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if timings is not None:
            timings[name] = timings.get(name, 0.0) + dt
        log.info("%-14s %6.0f ms", name, dt * 1000)
