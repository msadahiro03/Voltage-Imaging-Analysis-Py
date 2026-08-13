"""Real-time (streaming) driver for the voltage-imaging mapping analysis.

While an experiment runs on the rig, each mapping trial's motion-corrected TIFF
lands in a common folder. This subpackage watches that folder and, the moment a
trial's ``*_mc.tif`` is fully written, computes its dF/F and appends the
stim-window-cleaved, condition/holo-sorted dF/F into a growing result struct --
the same struct :func:`voltimg_mapping.pipeline.run_dff` produces in batch, but
built incrementally trial-by-trial.

Pieces
------
- :mod:`.context`     -- one-time session setup: ExpStruct params + ROIs, and a
                         per-trial stim-metadata resolver.
- :mod:`.accumulator` -- incremental condition/holo-sorted accumulator; its
                         ``snapshot()`` equals ``run_dff`` over the trials seen.
- :mod:`.watcher`     -- stdlib polling folder watcher with complete-file
                         detection (rig files are large and written in place).
- :mod:`.runner`      -- wires watcher -> per-trial dF/F -> accumulator, and
                         persists a rolling snapshot after every trial. Has a CLI.

See ``voltimg_mapping/live/README.md`` for the end-to-end usage.
"""

from . import accumulator, context, runner, watcher

__all__ = ["accumulator", "context", "runner", "watcher"]
