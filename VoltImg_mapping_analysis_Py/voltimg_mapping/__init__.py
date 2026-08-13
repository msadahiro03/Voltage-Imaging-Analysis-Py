"""Python port of the voltage-imaging all-optical connectivity mapping pipeline.

Faithful translation of
VoltImg_mapping_analysis_MultiCell_newDFF_021226_MCfineROI_TrialSpecROIandNeuropil_laserRowArtifact.m
and its local helpers. See README.md for the stage map, parity notes, and the
list of pieces that require external libraries (NoRMCorre, fibermetric).
"""

from . import (
    artifact,
    dff,
    ephys,
    expstruct,
    fextract,
    fibermetric,
    matlab_compat,
    motion_correction,
    pipeline,
    roi,
    sampling,
    tiff_io,
    trial_excluder,
)
from . import live  # noqa: E402  (imports the siblings above; keep last)

__all__ = [
    "artifact",
    "dff",
    "ephys",
    "expstruct",
    "fextract",
    "fibermetric",
    "live",
    "matlab_compat",
    "motion_correction",
    "pipeline",
    "roi",
    "sampling",
    "tiff_io",
    "trial_excluder",
]
