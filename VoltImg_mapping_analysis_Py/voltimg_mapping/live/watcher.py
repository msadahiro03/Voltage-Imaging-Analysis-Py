"""Folder watcher for incoming trial TIFFs (stdlib only).

Rig trials land in a common folder as ``<base>_<NNNNN>_mc.tif`` and are written
in place (single large file, tens to ~100 MB). A file appearing is NOT a signal
that it is complete, so we detect completion by **size stability**: the file's
size must be unchanged for ``stable_seconds`` before we hand it off. Optionally
we also verify the TIFF header/IFDs parse (cheap -- it does not read pixels).

Polling is used deliberately: rig data folders are frequently network/SMB shares
where filesystem-event APIs (inotify/FSEvents) are unreliable. Polling ``stat``
is robust everywhere and cheap for the trial cadence here.

The 1-based trial index ``tt`` is parsed from the filename's numeric field, which
aligns with the ExpStruct trial order (the imaging<->ephys alignment assumption).
"""

from __future__ import annotations

import glob
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional


@dataclass(frozen=True)
class TrialFile:
    trial_number: int
    path: str
    size: int


class TrialFileWatcher:
    """Watch ``folder`` for complete ``*_mc.tif`` trial files.

    Parameters
    ----------
    folder : str
        Directory to watch.
    pattern : str
        Glob for candidate files (default ``"*_mc.tif"``).
    trial_regex : str
        Regex with one capture group for the 1-based trial number. Default
        matches ``..._00042_mc.tif`` -> 42.
    stable_seconds : float
        A file must keep the same size for this long to count as complete.
    poll_interval : float
        Seconds between folder scans in :meth:`watch`.
    verify_tiff : bool
        If True (and tifffile is importable), also require the TIFF to open and
        report >= 1 page before emitting.
    """

    def __init__(
        self,
        folder: str,
        pattern: str = "*_mc.tif",
        trial_regex: str = r"_(\d+)_mc\.tif$",
        stable_seconds: float = 3.0,
        poll_interval: float = 2.0,
        verify_tiff: bool = True,
    ):
        self.folder = folder
        self.pattern = pattern
        self.trial_re = re.compile(trial_regex)
        self.stable_seconds = float(stable_seconds)
        self.poll_interval = float(poll_interval)
        self.verify_tiff = verify_tiff

        # path -> (last_size, first_seen_at_this_size)
        self._pending: dict = {}
        self._emitted: set = set()  # paths already yielded

    # ------------------------------------------------------------------
    def parse_trial_number(self, path: str) -> Optional[int]:
        m = self.trial_re.search(os.path.basename(path))
        return int(m.group(1)) if m else None

    def _tiff_ok(self, path: str) -> bool:
        if not self.verify_tiff:
            return True
        try:
            import tifffile
        except ImportError:
            return True
        try:
            with tifffile.TiffFile(path) as tf:
                return len(tf.pages) >= 1
        except Exception:
            return False

    # ------------------------------------------------------------------
    def scan(self) -> List[TrialFile]:
        """One folder scan. Return newly-complete, not-yet-emitted trial files,
        sorted by trial number. Non-blocking."""
        now = time.time()
        ready: List[TrialFile] = []

        for path in glob.glob(os.path.join(self.folder, self.pattern)):
            if path in self._emitted:
                continue
            if self.parse_trial_number(path) is None:
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue  # vanished mid-scan; retry next poll

            prev = self._pending.get(path)
            if prev is None or prev[0] != size:
                # New file, or still growing -> (re)start the stability clock.
                self._pending[path] = (size, now)
                continue

            stable_for = now - prev[1]
            if stable_for >= self.stable_seconds and size > 0:
                if self._tiff_ok(path):
                    ready.append(TrialFile(self.parse_trial_number(path),
                                           path, size))
                    self._emitted.add(path)
                    self._pending.pop(path, None)
                else:
                    # Header not parseable yet; keep waiting.
                    self._pending[path] = (size, now)

        ready.sort(key=lambda tf: tf.trial_number)
        return ready

    # ------------------------------------------------------------------
    def watch(
        self,
        stop_predicate: Optional[Callable[[], bool]] = None,
        max_idle_seconds: Optional[float] = None,
    ) -> Iterator[TrialFile]:
        """Blocking generator: yield trial files as they complete.

        Stops when ``stop_predicate()`` returns True, or after
        ``max_idle_seconds`` elapse with no new complete file (None = never).
        """
        last_emit = time.time()
        while True:
            if stop_predicate is not None and stop_predicate():
                return
            ready = self.scan()
            for tf in ready:
                yield tf
            if ready:
                last_emit = time.time()
            elif max_idle_seconds is not None and \
                    (time.time() - last_emit) >= max_idle_seconds:
                return
            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    def existing_files(self) -> List[TrialFile]:
        """List parseable candidate files already present (size as-is), sorted
        by trial number. Useful to backfill a session already in progress."""
        out = []
        for path in glob.glob(os.path.join(self.folder, self.pattern)):
            n = self.parse_trial_number(path)
            if n is None:
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            out.append(TrialFile(n, path, size))
        out.sort(key=lambda tf: tf.trial_number)
        return out

    def mark_emitted(self, path: str) -> None:
        """Suppress a path from future emission (e.g. already processed)."""
        self._emitted.add(path)
        self._pending.pop(path, None)
