#!/usr/bin/env python3
"""Registry of 2D slice diagnostics, shared by every panel in a grid.

BNS_all.py spells out the same five facts for each diagnostic -- which dump
prefix holds it, which variable to read, which colormap, which colour limits,
and any derived-variable hook -- once per ``if enabled(...)`` block.  A grid
figure needs those facts for several diagnostics at once, so they live here as
data instead.

Colour limits and labels still come from :func:`athenak_units.plot_params`, so a
panel drawn here and a frame drawn by BNS_all.py agree on the scale.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from matplotlib.colors import LogNorm, Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.environ.get("ATHPLOT_PATH", os.path.join(_REPO, "plot-tools"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import athenak_units as au  # noqa: E402


def bmag_from_components(bx, by, bz):
    """|B| from its three components (matches BNS_all._bmag_from_components)."""
    return np.sqrt(bx**2 + by**2 + bz**2)


@dataclass(frozen=True)
class Diagnostic:
    """Everything needed to draw one diagnostic into one axes."""

    name: str
    var: str
    prefixes: Tuple[str, ...]
    cmap: str
    unit_var: Optional[str] = None      # key for plot_params; defaults to var
    vmin: Optional[float] = None        # explicit override of plot_params
    vmax: Optional[float] = None
    log: Optional[bool] = None          # explicit override of plot_params
    use_abs: bool = False
    derived: Optional[dict] = None      # {'func': callable, 'inputs': (...)}
    legacy_prefixes: Tuple[str, ...] = field(default_factory=tuple)

    # -- naming -------------------------------------------------------------
    def candidate_prefixes(self, jobname: str, plane: str) -> Tuple[str, ...]:
        """Dump prefixes to try, most modern first.

        The legacy ``rad_m1_*`` dumps are spherical-surface output, not slices,
        so they are only a valid fallback for xz -- offering them for xy would
        silently plot the wrong geometry.
        """
        names = [p.format(job=jobname, plane=plane) for p in self.prefixes]
        if plane == "xz":
            names += [p.format(job=jobname) for p in self.legacy_prefixes]
        return tuple(names)

    @property
    def plot_var(self) -> str:
        """Variable name after any derived/abs wrapping is applied."""
        name = f"derived:{self.name}" if self.derived else self.var
        return f"derived:abs_{name}" if self.use_abs else name

    # -- scale --------------------------------------------------------------
    def unit_key(self) -> str:
        return self.unit_var or self.var

    def params(self, units: str):
        """(scale, label, use_log, vmin, vmax) with registry overrides applied."""
        scale, label, use_log, lo, hi = au.plot_params(self.unit_key(), units)
        if self.log is not None:
            use_log = self.log
        if self.vmin is not None:
            lo = self.vmin
        if self.vmax is not None:
            hi = self.vmax
        return scale, label, use_log, lo, hi

    def norm(self, units: str, clim: Optional[Sequence[float]] = None):
        """One norm per diagnostic -- this is what makes a shared colorbar valid.

        `clim` overrides the registry limits (the driver's CLIM[name]="lo,hi").
        """
        _, _, use_log, lo, hi = self.params(units)
        if clim is not None:
            lo, hi = float(clim[0]), float(clim[1])
        return LogNorm(vmin=lo, vmax=hi) if use_log else Normalize(vmin=lo, vmax=hi)

    def label(self, units: str) -> str:
        return self.params(units)[1]

    def scale(self, units: str) -> float:
        return self.params(units)[0]


# ---------------------------------------------------------------------------
# The registry.  Names match the SECTIONS entries in BNS_all.sh so the two
# drivers can be read side by side.
# ---------------------------------------------------------------------------
_PRIM = ("{job}.prim_{plane}",)
_M1_MOM = ("{job}.m1_{plane}", "{job}.m1_mom_{plane}", "{job}.rad_E_{plane}")
_M1_NUM = ("{job}.m1_{plane}", "{job}.m1_mom_{plane}", "{job}.rad_N_{plane}")
_M1_OPAC = ("{job}.m1opac_{plane}", "{job}.m1_opac_{plane}")

DIAGNOSTICS = {
    "density": Diagnostic(
        name="density", var="dens", prefixes=_PRIM, cmap="inferno",
    ),
    "temperature": Diagnostic(
        name="temperature", var="temperature", prefixes=_PRIM, cmap="viridis",
        vmin=0.0, vmax=70.0, log=False,
    ),
    "s00": Diagnostic(
        name="s00", var="s_00", prefixes=_PRIM, cmap="plasma",
        vmin=0.0, vmax=0.5, log=False,
    ),
    "bnorm": Diagnostic(
        name="bnorm", var="bcc1", prefixes=_PRIM, cmap="plasma",
        unit_var="bmag",
        derived={"func": bmag_from_components, "inputs": ("bcc1", "bcc2", "bcc3")},
    ),
    "z4c": Diagnostic(
        name="z4c", var="con_H", prefixes=("{job}.con_{plane}",), cmap="inferno",
        use_abs=True,
    ),
    "rad_E": Diagnostic(
        name="rad_E", var="E:0", prefixes=_M1_MOM, cmap="inferno",
        legacy_prefixes=("{job}.rad_m1_E",),
    ),
    "rad_N": Diagnostic(
        name="rad_N", var="N:0", prefixes=_M1_NUM, cmap="inferno",
        legacy_prefixes=("{job}.rad_m1_N",),
    ),
    "rad_N2": Diagnostic(
        name="rad_N2", var="N:2", prefixes=_M1_NUM, cmap="inferno",
        legacy_prefixes=("{job}.rad_m1_N",),
    ),
    "rad_abs": Diagnostic(
        name="rad_abs", var="abs_1:0", prefixes=(*_M1_OPAC, "{job}.abs_1_{plane}"),
        cmap="inferno", legacy_prefixes=("{job}.rad_m1_abs_1",),
    ),
    "rad_abs0": Diagnostic(
        name="rad_abs0", var="abs_0:0", prefixes=(*_M1_OPAC, "{job}.abs_0_{plane}"),
        cmap="inferno", legacy_prefixes=("{job}.rad_m1_abs_0",),
    ),
    "rad_eta0": Diagnostic(
        name="rad_eta0", var="eta_0:0", prefixes=(*_M1_OPAC, "{job}.eta_0_{plane}"),
        cmap="inferno", legacy_prefixes=("{job}.rad_m1_eta_0",),
    ),
    "rad_eta1": Diagnostic(
        name="rad_eta1", var="eta_1:0", prefixes=(*_M1_OPAC, "{job}.eta_1_{plane}"),
        cmap="inferno", legacy_prefixes=("{job}.rad_m1_eta_1",),
    ),
}


def get(name: str) -> Diagnostic:
    """Look up a diagnostic, failing with the full menu rather than a KeyError."""
    try:
        return DIAGNOSTICS[name]
    except KeyError:
        raise KeyError(
            f"Unknown diagnostic '{name}'; available: {', '.join(sorted(DIAGNOSTICS))}"
        ) from None


def apply_to(data, diag: Diagnostic) -> str:
    """Register any derived/abs variables on a BinaryData and return the var name.

    Mirrors the derived-variable handling in BNS_all._render_frame_worker_all.
    """
    var = diag.var
    if diag.derived:
        var = f"derived:{diag.name}"
        data.register_derived_variable(var, diag.derived["func"],
                                       *diag.derived["inputs"])
    if diag.use_abs:
        abs_var = f"derived:abs_{var}"
        data.register_derived_variable(abs_var, np.abs, var)
        var = abs_var
    return var
