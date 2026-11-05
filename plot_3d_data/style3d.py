#!/usr/bin/env python3
"""Shared style and derived-field helpers for BNS/AthenaK 3D plots."""

import math
import os
import re
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


FIGSIZE = (5, 3)

# ---------------------------------------------------------------------------
# AthenaK BNS unit conversions (matching athenak_units.py)
# Code units: G = c = M_sun = 1.  Temperature already in MeV, Ye dimensionless.
# ---------------------------------------------------------------------------
_c_cgs = 2.99792458e10
_G_cgs = 6.67408e-8
_Msun_g = 1.98892e33
_unit_len_cm = _G_cgs * _Msun_g / _c_cgs**2
CODE2CGS_DENS = _Msun_g / _unit_len_cm**3          # ≈ 6.178e17 g/cm^3
_unit_time_s = _G_cgs * _Msun_g / _c_cgs**3       # ≈ 4.926e-6 s
CODE_TIME_TO_MS = _unit_time_s * 1e3               # code time → milliseconds
# Magnetic field: code (geometric_solar, b^2/2 convention) → Gauss.
# Matches athenak_units.CODE2CGS_BFIELD; B_Gauss = b_code*sqrt(4*pi*rho_unit)*c.
CODE2CGS_BFIELD = _c_cgs * math.sqrt(4.0 * math.pi * CODE2CGS_DENS)  # ≈ 8.35e19 G

# Per-field conversion: field_name → {units_name: (scale, label)}
# Fields not listed here are unit-independent (temperature [MeV], Ye, etc.)
_UNIT_TABLE = {
    "dens": {
        "code": (1.0, r"$\rho$ [code]"),
        "cgs":  (CODE2CGS_DENS, r"$\rho$ [g cm$^{-3}$]"),
    },
    "log_dens": {
        "code": (1.0, r"$\log_{10}\rho$ [code]"),
        "cgs":  (1.0, r"$\log_{10}\rho$ [g cm$^{-3}$]"),
    },
    "temperature": {
        "code": (1.0, r"$T$ [MeV]"),
        "cgs":  (1.0, r"$T$ [MeV]"),
    },
    # Raw cell-centered magnetic field components: scaled to Gauss in cgs mode
    # (convert_fields() applies these BEFORE |B|, log|B|, current are derived).
    "velx": {"code": (1.0, r"$v_x$ [c]"), "cgs": (1.0, r"$v_x$ [c]")},
    "vely": {"code": (1.0, r"$v_y$ [c]"), "cgs": (1.0, r"$v_y$ [c]")},
    "velz": {"code": (1.0, r"$v_z$ [c]"), "cgs": (1.0, r"$v_z$ [c]")},
    "bcc1": {"code": (1.0, r"$B_x$ [code]"), "cgs": (CODE2CGS_BFIELD, r"$B_x$ [G]")},
    "bcc2": {"code": (1.0, r"$B_y$ [code]"), "cgs": (CODE2CGS_BFIELD, r"$B_y$ [G]")},
    "bcc3": {"code": (1.0, r"$B_z$ [code]"), "cgs": (CODE2CGS_BFIELD, r"$B_z$ [G]")},
    # Derived B diagnostics: scale stays 1.0 (components are already converted);
    # these entries supply unit-aware labels only.
    # NOTE: use \Vert..\Vert (‖B‖), not |..| — VTK's mathtext (PyVista scalar
    # bars) fails to parse pipe characters, while matplotlib renders both.
    "bmag":      {"code": (1.0, r"$\Vert B\Vert$ [code]"),          "cgs": (1.0, r"$\Vert B\Vert$ [G]")},
    "log_bmag":  {"code": (1.0, r"$\log_{10}\Vert B\Vert$ [code]"), "cgs": (1.0, r"$\log_{10}\Vert B\Vert$ [G]")},
    "current":     {"code": (1.0, r"$\Vert\nabla\times B\Vert$ [code]"),          "cgs": (1.0, r"$\Vert\nabla\times B\Vert$ [G/len]")},
    # Unit-independent quantities: same label in both systems.
    "ye":          {"code": (1.0, r"$Y_e$"),                       "cgs": (1.0, r"$Y_e$")},
    "press":       {"code": (1.0, r"$P$ [code]"),                  "cgs": (1.0, r"$P$ [code]")},
    "divb":        {"code": (1.0, r"$\nabla\cdot B$"),             "cgs": (1.0, r"$\nabla\cdot B$")},
    "vorticity":   {"code": (1.0, r"$\Vert\nabla\times v\Vert$"), "cgs": (1.0, r"$\Vert\nabla\times v\Vert$")},
    "qcriterion":  {"code": (1.0, r"$Q$"),                         "cgs": (1.0, r"$Q$")},
    "log_current": {"code": (1.0, r"$\log_{10}\Vert\nabla\times B\Vert$ [code]"), "cgs": (1.0, r"$\log_{10}\Vert\nabla\times B\Vert$ [G/len]")},
}


def unit_scale(field_name: str, units: str = "code") -> float:
    """Multiplicative scale factor for a raw (non-log) field."""
    entry = _UNIT_TABLE.get(field_name.lower(), {}).get(units)
    return entry[0] if entry else 1.0


# Every spelling the CLI accepts maps onto one canonical table entry, so
# "--scalar log_rho" gets the same typeset label as "--scalar log_dens"
# instead of falling through to the raw field name.
_UNIT_ALIASES = {
    "rho": "dens", "density": "dens",
    "log_rho": "log_dens", "log_density": "log_dens",
    "temp": "temperature", "t": "temperature",
    "s_00": "ye", "electron_fraction": "ye",
    "magnetic": "bmag", "b": "bmag", "bfield": "bmag",
    "jmag": "current", "curl_b": "current",
    "bx": "bcc1", "by": "bcc2", "bz": "bcc3",
    "vx": "velx", "vy": "vely", "vz": "velz",
    "omega": "vorticity", "q": "qcriterion",
}


def unit_label(field_name: str, units: str = "code") -> str:
    """Colorbar / axis label for a field in the given unit system."""
    key = field_name.lower()
    key = _UNIT_ALIASES.get(key, key)
    entry = _UNIT_TABLE.get(key, {}).get(units)
    return entry[1] if entry else field_name


def convert_fields(fields: Dict[str, np.ndarray], units: str = "code") -> None:
    """Scale raw fields in-place from code units to the target unit system.

    Must be called BEFORE deriving log fields so that log10(dens) gives
    log10(dens_cgs) automatically.
    """
    if units == "code":
        return
    for name in list(fields.keys()):
        s = unit_scale(name, units)
        if s != 1.0:
            fields[name] = fields[name] * np.float32(s)


def apply_bns_matplotlib_style() -> None:
    """Apply the central BNS house style (see repo-root plot_style.py).

    Delegates to plot_style so the look is defined in one place; falls back to
    a minimal inline style if plot_style is not importable.
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import plot_style
        plot_style.use(panel_w=FIGSIZE[0], panel_h=FIGSIZE[1])
    except ImportError:
        import matplotlib as mpl
        mpl.rcParams.update({
            "font.family": "serif",
            "font.size": 16.0,
            "figure.figsize": FIGSIZE,
        })


def parse_csv_floats(value: Optional[str]) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    return tuple(float(x.strip()) for x in value.split(",") if x.strip())


def parse_domain_size(value: Optional[str]) -> Optional[Tuple[float, float, float]]:
    parts = parse_csv_floats(value)
    if parts is None:
        return None
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    raise ValueError("--domain-size must be one value or three comma-separated values")


def parse_center(value: Optional[str]) -> Optional[Tuple[float, float, float]]:
    parts = parse_csv_floats(value)
    if parts is None:
        return None
    if len(parts) != 3:
        raise ValueError("--center must be three comma-separated values")
    return (parts[0], parts[1], parts[2])


def parse_crop(value: Optional[str]) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Parse --crop xmin,xmax,ymin,ymax,zmin,zmax or a single half-width L (→ -L,L,-L,L,-L,L)."""
    parts = parse_csv_floats(value)
    if parts is None:
        return None
    if len(parts) == 1:
        L = parts[0]
        return (-L, L, -L, L, -L, L)
    if len(parts) == 6:
        return (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
    raise ValueError("--crop must be one half-width or six comma-separated values (xmin,xmax,ymin,ymax,zmin,zmax)")


def finite_percentile_limits(values: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> Tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    vmin = float(np.nanpercentile(finite, lo))
    vmax = float(np.nanpercentile(finite, hi))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if vmin == vmax:
        pad = abs(vmin) * 0.05 if vmin != 0 else 1.0
        return (vmin - pad, vmax + pad)
    return (vmin, vmax)


def default_scalar_style(name: str, units: str = "code") -> Dict[str, object]:
    """Return conservative defaults matching the BNS plotting palette."""
    key = name.lower()
    label = unit_label(name, units)
    if key in {"density", "dens", "rho", "log_density", "log_dens", "log_rho"}:
        return {"cmap": "inferno", "label": label, "log": False}
    if key in {"temperature", "temp"}:
        return {"cmap": "viridis", "label": label, "log": False}
    if key in {"ye", "electron_fraction", "s_00"}:
        return {"cmap": "plasma", "label": r"$Y_e$", "log": False}
    if key in {"bmag", "log_bmag", "magnetic"}:
        return {"cmap": "plasma", "label": label, "log": False}
    if key in {"current", "curl_b", "jmag", "log_current"}:
        return {"cmap": "inferno", "label": label, "log": False}
    if key in {"divb", "curlx", "curly", "curlz", "vx", "vy", "vz", "velx", "vely", "velz"}:
        return {"cmap": "RdBu_r", "label": label, "log": False}
    if key in {"vorticity", "omega", "qcriterion", "q"}:
        return {"cmap": "magma", "label": label, "log": False}
    return {"cmap": "inferno", "label": label, "log": False}


# VTK's scalar bar draws plain strings, so the Matplotlib mathtext labels have
# to be flattened before they reach it -- otherwise the bar reads "\log_{10}\rho".
_PLAIN_LABEL_SUBS = (
    (r"\log_{10}", "log10 "),
    (r"\Vert", "|"),
    (r"\nabla\times", "curl "),
    (r"\rho", "rho"),
    (r"\odot", "sun"),
    (r"\mathrm", ""),
    (r"\times", "x"),
    (r"\,", " "),
    ("^{-3}", "^-3"),
    ("^{-2}", "^-2"),
    ("_{10}", "10"),
    ("$", ""),
    ("{", ""),
    ("}", ""),
    ("~", " "),
)


def plain_text_label(label: str) -> str:
    """Flatten a mathtext label into something a VTK scalar bar can show."""
    text = str(label)
    for pattern, replacement in _PLAIN_LABEL_SUBS:
        text = text.replace(pattern, replacement)
    text = " ".join(text.split())
    # "\Vert B\Vert" lands as "| B|"; tighten only inside the bars.
    return re.sub(r"\|\s*([^|]*?)\s*\|", r"|\1|", text)


def positive_log10(values: np.ndarray, floor: Optional[float] = None) -> np.ndarray:
    arr = np.asarray(values)
    finite_pos = arr[np.isfinite(arr) & (arr > 0)]
    if floor is None:
        floor = float(np.nanmin(finite_pos)) if finite_pos.size else 1e-300
    return np.log10(np.clip(arr, floor, None))


def get_scalar_field(fields: Dict[str, np.ndarray], name: str, spacing: Sequence[float]) -> np.ndarray:
    """Resolve raw and derived scalar fields."""
    key = name.lower()
    aliases = {
        "rho": ("dens", "rho"),
        "density": ("dens", "rho"),
        "dens": ("dens", "rho"),
        "temp": ("temperature", "temp"),
        "temperature": ("temperature", "temp"),
        "vx": ("velx", "vx"),
        "velx": ("velx", "vx"),
        "vy": ("vely", "vy"),
        "vely": ("vely", "vy"),
        "vz": ("velz", "vz"),
        "velz": ("velz", "vz"),
        "bx": ("bcc1", "bx"),
        "bcc1": ("bcc1", "bx"),
        "by": ("bcc2", "by"),
        "bcc2": ("bcc2", "by"),
        "bz": ("bcc3", "bz"),
        "bcc3": ("bcc3", "bz"),
        # AthenaK prim3D output carries the electron fraction as the passive
        # scalar s_00; the coarse-grained HDF5 boxes call it 'ye'.
        "ye": ("ye", "s_00", "Ye"),
        "electron_fraction": ("ye", "s_00", "Ye"),
        "s_00": ("s_00", "ye"),
    }
    for raw_key in aliases.get(key, (name,)):
        if raw_key in fields:
            return fields[raw_key]

    if key in {"log_density", "log_dens", "log_rho"}:
        return positive_log10(get_scalar_field(fields, "dens", spacing))
    if key in {"bmag", "magnetic"}:
        bx = get_scalar_field(fields, "bcc1", spacing)
        by = get_scalar_field(fields, "bcc2", spacing)
        bz = get_scalar_field(fields, "bcc3", spacing)
        return np.sqrt(bx * bx + by * by + bz * bz)
    if key == "log_bmag":
        return positive_log10(get_scalar_field(fields, "bmag", spacing))
    if key in {"current", "curl_b", "jmag"}:
        cx, cy, cz = curl(
            get_scalar_field(fields, "bcc1", spacing),
            get_scalar_field(fields, "bcc2", spacing),
            get_scalar_field(fields, "bcc3", spacing),
            spacing,
        )
        return np.sqrt(cx * cx + cy * cy + cz * cz)
    if key == "log_current":
        return positive_log10(get_scalar_field(fields, "current", spacing))
    if key in {"vorticity", "omega"}:
        wx, wy, wz = curl(
            get_scalar_field(fields, "velx", spacing),
            get_scalar_field(fields, "vely", spacing),
            get_scalar_field(fields, "velz", spacing),
            spacing,
        )
        return np.sqrt(wx * wx + wy * wy + wz * wz)
    if key in {"qcriterion", "q"}:
        return qcriterion(
            get_scalar_field(fields, "velx", spacing),
            get_scalar_field(fields, "vely", spacing),
            get_scalar_field(fields, "velz", spacing),
            spacing,
        )
    if key == "divb":
        return divergence(
            get_scalar_field(fields, "bcc1", spacing),
            get_scalar_field(fields, "bcc2", spacing),
            get_scalar_field(fields, "bcc3", spacing),
            spacing,
        )

    available = ", ".join(sorted(fields.keys()))
    raise KeyError(f"Unknown scalar '{name}'. Available raw fields: {available}")


def get_vector_field(fields: Dict[str, np.ndarray], name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = name.lower()
    if key in {"b", "magnetic", "bfield"}:
        return fields["bcc1"], fields["bcc2"], fields["bcc3"]
    if key in {"v", "vel", "velocity"}:
        return fields["velx"], fields["vely"], fields["velz"]
    raise KeyError("--streamlines must be one of: B, velocity")


def divergence(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    dx, dy, dz = spacing
    dfx_dx = np.gradient(fx, dx, axis=0)
    dfy_dy = np.gradient(fy, dy, axis=1)
    dfz_dz = np.gradient(fz, dz, axis=2)
    return dfx_dx + dfy_dy + dfz_dz


def curl(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray, spacing: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx, dy, dz = spacing
    dfx_dy = np.gradient(fx, dy, axis=1)
    dfx_dz = np.gradient(fx, dz, axis=2)
    dfy_dx = np.gradient(fy, dx, axis=0)
    dfy_dz = np.gradient(fy, dz, axis=2)
    dfz_dx = np.gradient(fz, dx, axis=0)
    dfz_dy = np.gradient(fz, dy, axis=1)
    return (dfz_dy - dfy_dz, dfx_dz - dfz_dx, dfy_dx - dfx_dy)


def qcriterion(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    dx, dy, dz = spacing
    grad = {
        (0, 0): np.gradient(vx, dx, axis=0),
        (0, 1): np.gradient(vx, dy, axis=1),
        (0, 2): np.gradient(vx, dz, axis=2),
        (1, 0): np.gradient(vy, dx, axis=0),
        (1, 1): np.gradient(vy, dy, axis=1),
        (1, 2): np.gradient(vy, dz, axis=2),
        (2, 0): np.gradient(vz, dx, axis=0),
        (2, 1): np.gradient(vz, dy, axis=1),
        (2, 2): np.gradient(vz, dz, axis=2),
    }
    s2 = 0.0
    omega2 = 0.0
    for i in range(3):
        for j in range(3):
            sij = 0.5 * (grad[(i, j)] + grad[(j, i)])
            oij = 0.5 * (grad[(i, j)] - grad[(j, i)])
            s2 = s2 + sij * sij
            omega2 = omega2 + oij * oij
    return 0.5 * (omega2 - s2)


def auto_levels(values: np.ndarray, nlevels: int = 4, lo: float = 70.0, hi: float = 99.5) -> np.ndarray:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, nlevels)
    v0 = float(np.nanpercentile(finite, lo))
    v1 = float(np.nanpercentile(finite, hi))
    if v0 == v1:
        v0, v1 = finite_percentile_limits(finite)
    return np.linspace(v0, v1, nlevels)


def set_pyvista_camera(
    plotter,
    grid,
    elevation: float = 30.0,
    azimuth: float = 0.0,
    distance: Optional[float] = None,
    view_angle: float = 30.0,
    orthographic: bool = False,
    parallel_scale: Optional[float] = None,
) -> None:
    center = np.asarray(grid.center)
    bounds = np.asarray(grid.bounds)
    if distance is None:
        diagonal = np.linalg.norm(bounds[1::2] - bounds[::2])
        distance = 0.8 * diagonal

    elev = np.deg2rad(elevation)
    azim = np.deg2rad(azimuth)
    camera_x = center[0] + distance * np.cos(elev) * np.sin(azim)
    camera_y = center[1] + distance * np.cos(elev) * np.cos(azim)
    camera_z = center[2] + distance * np.sin(elev)

    plotter.camera.position = [camera_x, camera_y, camera_z]
    plotter.camera.focal_point = center
    plotter.camera.up = [0, 0, 1]
    plotter.camera.view_angle = view_angle
    plotter.camera.parallel_projection = orthographic
    if orthographic:
        if parallel_scale is None:
            parallel_scale = 0.55 * float(np.max(bounds[1::2] - bounds[::2]))
        if parallel_scale <= 0:
            raise ValueError("parallel_scale must be positive")
        plotter.camera.parallel_scale = float(parallel_scale)


def make_coordinate_axes(
    shape: Sequence[int],
    domain_size: Sequence[float],
    coordinates: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x/y/z cell-center axes, preserving explicit input coordinates."""
    if coordinates is not None:
        axes = tuple(np.asarray(axis, dtype=float) for axis in coordinates)
        if tuple(len(axis) for axis in axes) != tuple(shape):
            raise ValueError("Coordinate-axis lengths do not match data shape")
        return axes

    nx, ny, nz = shape
    lx, ly, lz = domain_size
    return (
        (np.arange(nx, dtype=float) + 0.5) * lx / nx - 0.5 * lx,
        (np.arange(ny, dtype=float) + 0.5) * ly / ny - 0.5 * ly,
        (np.arange(nz, dtype=float) + 0.5) * lz / nz - 0.5 * lz,
    )


def make_coordinates(
    shape: Sequence[int],
    domain_size: Sequence[float],
    coordinates: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = make_coordinate_axes(shape, domain_size, coordinates)
    return np.meshgrid(*axes, indexing="ij")


def relative_time_ms(time_code: float, t0_code: Optional[float] = None) -> float:
    """Convert code time to ms, optionally relative to a merger time."""
    return (float(time_code) - float(t0_code or 0.0)) * CODE_TIME_TO_MS


def time_label(time_ms: float, relative: bool, mathtext: bool = False) -> str:
    """Compact frame clock: 't_merger +X.XXX ms', or 't = X.XXX ms' absolute."""
    if not relative:
        return (
            rf"$t = {time_ms:.3f}$ ms" if mathtext else f"t = {time_ms:.3f} ms"
        )
    if mathtext:
        return rf"$t_\mathrm{{merger}} {time_ms:+.3f}$ ms"
    return f"t_merger {time_ms:+.3f} ms"


# Bundled with matplotlib, so these need no system font packages.  DejaVu Sans
# is matplotlib's own default sans; STIX is its Times-alike; cmr10 is the
# Computer Modern face LaTeX papers are set in.
_FONT_FILES = {
    "sans": "DejaVuSans.ttf",
    "serif": "STIXGeneral.ttf",
    "cm": "cmr10.ttf",
}
# VTK ships these three internally.
_FONT_BUILTIN = {"arial", "times", "courier"}


def resolve_font(name: str):
    """Return (font_family, font_file) for a --font choice.

    Either element may be None: VTK takes a built-in family name OR a TTF path.
    Falls back to Arial if the bundled file cannot be located.
    """
    key = (name or "sans").lower()
    if key in _FONT_BUILTIN:
        return key, None
    if key.endswith((".ttf", ".otf")):
        if not os.path.isfile(name):
            raise ValueError(f"Font file not found: {name}")
        return None, os.path.abspath(name)
    filename = _FONT_FILES.get(key)
    if filename is None:
        raise ValueError(
            f"Unknown font '{name}'; choose from "
            f"{sorted(set(_FONT_FILES) | _FONT_BUILTIN)} or pass a .ttf path"
        )
    try:
        import matplotlib
    except ImportError:
        return "arial", None
    path = os.path.join(
        os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", filename
    )
    return (None, path) if os.path.isfile(path) else ("arial", None)


def apply_font_to_scalar_bars(plotter, font_family, font_file) -> None:
    """Point scalar-bar title/label text at the same face as the annotations.

    add_scalar_bar() only accepts a built-in family name, so a bundled TTF has
    to be pushed onto the actor's text properties after the fact.
    """
    if not font_file:
        return
    for bar in getattr(plotter, "scalar_bars", {}).values():
        for prop in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
            prop.SetFontFamily(4)  # VTK_FONT_FILE
            prop.SetFontFile(font_file)


def font_kwargs(font_family, font_file) -> Dict[str, object]:
    """add_text() kwargs for a resolved font."""
    if font_file:
        return {"font_file": font_file}
    return {"font": font_family} if font_family else {}



def load_camera_track(path: str):
    """Load time-ms, parallel-scale, optional-ruler columns from a track file.

    Accepted data rows are ``time scale`` or the reference workflow's
    ``time r99 half-height ruler``. Comment and summary lines are ignored.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("#") or ":" in line or "stage" in line:
                continue
            parts = line.replace(",", " ").split()
            try:
                values = [float(value) for value in parts]
            except ValueError:
                continue
            if len(values) >= 3:
                ruler = values[3] if len(values) > 3 else np.nan
                rows.append((values[0], values[2], ruler))
            elif len(values) == 2:
                rows.append((values[0], values[1], np.nan))
    if not rows:
        raise ValueError(f"No camera-track rows found in {path}")
    data = np.asarray(sorted(rows), dtype=float)
    if np.any(data[:, 1] <= 0):
        raise ValueError("Camera-track parallel scales must be positive")
    return data[:, 0], data[:, 1], data[:, 2]


def camera_track_values(path: str, time_ms: float):
    times, scales, rulers = load_camera_track(path)
    scale = float(np.interp(time_ms, times, scales))
    finite = np.isfinite(rulers)
    ruler = None
    if np.any(finite):
        ruler = float(np.interp(time_ms, times[finite], rulers[finite]))
    return scale, ruler


def nice_scale_bar_length(view_width: float, fraction: float = 0.22) -> float:
    """Choose a 1/2/5 x 10^n ruler length near a target view fraction."""
    target = max(float(view_width) * fraction, np.finfo(float).tiny)
    exponent = math.floor(math.log10(target))
    base = target / 10.0**exponent
    mantissa = max(value for value in (1.0, 2.0, 5.0) if value <= base)
    return mantissa * 10.0**exponent


def add_orthographic_scale_bar(
    plotter,
    length: Optional[float],
    parallel_scale: float,
    window_size: Sequence[int],
    color="white",
    position="lower_right",
    text_font=None,
) -> float:
    """Add an exact screen-space ruler for an orthographic camera."""
    import pyvista as pv
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkRenderingCore import vtkActor2D, vtkCoordinate, vtkPolyDataMapper2D

    width, height = (int(window_size[0]), int(window_size[1]))
    aspect = width / float(height)
    view_width = 2.0 * float(parallel_scale) * aspect
    if length is None or length <= 0:
        length = nice_scale_bar_length(view_width)
    fraction_x = float(length) / view_width
    if not 0 < fraction_x < 0.8:
        raise ValueError("Scale-bar length must be smaller than 80% of view width")

    x1 = 0.95 if position == "lower_right" else 0.05 + fraction_x
    x0 = x1 - fraction_x
    y0 = 0.075
    cap = 0.012
    points = vtkPoints()
    screen_points = (
        (x0, y0, 0), (x1, y0, 0),
        (x0, y0 - cap, 0), (x0, y0 + cap, 0),
        (x1, y0 - cap, 0), (x1, y0 + cap, 0),
    )
    for point in screen_points:
        points.InsertNextPoint(*point)
    lines = vtkCellArray()
    for first, second in ((0, 1), (2, 3), (4, 5)):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(first)
        lines.InsertCellPoint(second)
    poly = vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)
    coordinate = vtkCoordinate()
    coordinate.SetCoordinateSystemToNormalizedViewport()
    mapper = vtkPolyDataMapper2D()
    mapper.SetInputData(poly)
    mapper.SetTransformCoordinate(coordinate)
    actor = vtkActor2D()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*pv.Color(color).float_rgb)
    actor.GetProperty().SetLineWidth(3.0)
    plotter.renderer.AddActor2D(actor)
    plotter.add_text(
        f"{length:g} M",
        position=((x0 + x1) / 2.0 - 0.025, y0 + 0.018),
        viewport=True,
        color=color,
        font_size=11,
        **(text_font or {}),
    )
    return float(length)


def nearest_index(coords: Iterable[np.ndarray], point: Sequence[float]) -> Tuple[int, int, int]:
    coords = tuple(coords)
    if all(np.asarray(coord).ndim == 1 for coord in coords):
        axes = coords
    else:
        axes = [
            c[:, 0, 0] if i == 0 else c[0, :, 0] if i == 1 else c[0, 0, :]
            for i, c in enumerate(coords)
        ]
    return tuple(int(np.argmin(np.abs(axis - p))) for axis, p in zip(axes, point))
