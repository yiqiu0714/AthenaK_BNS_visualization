#!/usr/bin/env python3
"""Model discovery and panel selection for grid figures.

A grid figure is a cartesian product over three dimensions -- model, diagnostic
and frame -- so this module's job is to turn a model spec plus a list of frame
indices into concrete files, and to report clearly when a combination does not
exist rather than leaving a blank panel unexplained.

Discovery reuses BNS_all.py: it guards main() under __main__, so importing it
costs the matplotlib import and nothing else.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.environ.get("ATHPLOT_PATH", os.path.join(_REPO, "plot-tools"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import BNS_all  # noqa: E402  (find_output_dirs / _collect_frames_by_time / overlays)

import diagnostics as diags  # noqa: E402


@dataclass(frozen=True)
class Model:
    """One simulation directory and the label it carries in the figure."""

    path: str
    label: str
    jobname: str = "bns"

    @classmethod
    def parse(cls, spec: str, jobname: str = "bns") -> "Model":
        """Parse the driver's "path|label" form; label defaults to the basename."""
        path, _, label = spec.partition("|")
        path = os.path.abspath(os.path.expanduser(path.strip()))
        if not os.path.isdir(path):
            raise ValueError(f"Model path is not a directory: {path}")
        return cls(path=path, label=label.strip() or os.path.basename(path),
                   jobname=jobname)


@dataclass(frozen=True)
class Panel:
    """One cell of the grid: which model, which diagnostic, which frame.

    `request` is what the caller asked for -- a frame index, or a simulation
    time when selecting by time -- and is what keys the grid.  `index` is the
    frame it actually resolved to, which differs per model.
    """

    model: Model
    diagnostic: diags.Diagnostic
    request: float
    index: Optional[int]
    path: Optional[str]      # None when this combination has no file
    time: Optional[float]


class FrameIndex:
    """Per-(model, diagnostic) map of frame index -> file, built once and cached.

    Frame index means position in the time-sorted, restart-deduplicated list, so
    it matches the numbering BNS_all.py uses for its PNG sequence rather than the
    raw NNNNN in the filename (which repeats across output-XXXX segments).
    """

    def __init__(self):
        self._outputs: Dict[str, List[str]] = {}
        self._frames: Dict[Tuple[str, str, str], List[Tuple[float, str]]] = {}

    def output_dirs(self, model: Model) -> List[str]:
        if model.path not in self._outputs:
            dirs = BNS_all.find_output_dirs(model.path)
            if not dirs:
                raise ValueError(
                    f"No output-XXXX directories under {model.path}"
                )
            self._outputs[model.path] = dirs
        return self._outputs[model.path]

    def frames(self, model: Model, diag: diags.Diagnostic,
               plane: str) -> List[Tuple[float, str]]:
        """Time-sorted (time, path) for one model/diagnostic, deduplicated."""
        key = (model.path, diag.name, plane)
        if key in self._frames:
            return self._frames[key]

        dirs = self.output_dirs(model)
        candidates = diag.candidate_prefixes(model.jobname, plane)
        prefix = BNS_all._detect_prefix_all(dirs, *candidates)
        time_to_path = BNS_all._collect_frames_by_time(dirs, prefix) if prefix else {}
        ordered = [(t, time_to_path[t]) for t in sorted(time_to_path)]
        self._frames[key] = ordered
        return ordered

    def count(self, model: Model, diag: diags.Diagnostic, plane: str) -> int:
        return len(self.frames(model, diag, plane))


def select_frames(index: FrameIndex, model: Model, diag: diags.Diagnostic,
                  requests: Sequence[float], plane: str,
                  by: str = "index") -> List[Panel]:
    """Resolve requests to panels, either by frame index or by simulation time.

    Selection lives here alone, so the two modes stay a local concern.  Index
    mode is exact but run-relative: the same index is a different time in runs
    with different output cadences.  Time mode picks each model's nearest frame
    to the requested time, which is what makes a physical comparison honest.
    Negative indices count from the end, so -1 is the last frame of a run.
    """
    available = index.frames(model, diag, plane)
    panels = []
    for want in requests:
        if not available:
            panels.append(Panel(model, diag, want, None, None, None))
            continue

        if by == "time":
            times = [t for t, _ in available]
            idx = min(range(len(times)), key=lambda k: abs(times[k] - want))
        else:
            want_int = int(want)
            idx = want_int + len(available) if want_int < 0 else want_int
            if not 0 <= idx < len(available):
                print(f"  [skip] {model.label} / {diag.name}: frame {want_int} "
                      f"outside 0..{len(available) - 1}")
                panels.append(Panel(model, diag, want, None, None, None))
                continue

        time, path = available[idx]
        panels.append(Panel(model, diag, want, idx, path, time))
    return panels


def build_panels(models: Sequence[Model], diagnostic_names: Sequence[str],
                 requests: Sequence[float], plane: str, by: str = "index"):
    """Return (index, {(model_path, diag_name, request): Panel}) for the grid."""
    index = FrameIndex()
    table = {}
    for model in models:
        for name in diagnostic_names:
            diag = diags.get(name)
            for panel in select_frames(index, model, diag, requests, plane, by):
                table[(model.path, name, panel.request)] = panel
    return index, table
