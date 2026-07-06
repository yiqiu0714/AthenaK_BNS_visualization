#!/usr/bin/env python3
"""Shared style and derived-field helpers for BNS/AthenaK 3D plots."""

import math
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
    "log_current": {"code": (1.0, r"$\log_{10}\Vert\nabla\times B\Vert$ [code]"), "cgs": (1.0, r"$\log_{10}\Vert\nabla\times B\Vert$ [G/len]")},
}


def unit_scale(field_name: str, units: str = "code") -> float:
    """Multiplicative scale factor for a raw (non-log) field."""
    entry = _UNIT_TABLE.get(field_name.lower(), {}).get(units)
    return entry[0] if entry else 1.0


def unit_label(field_name: str, units: str = "code") -> str:
    """Colorbar / axis label for a field in the given unit system."""
    entry = _UNIT_TABLE.get(field_name.lower(), {}).get(units)
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
    """Apply the compact Matplotlib style used by BNS_all.py."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "lines.markersize": 6,
        "font.family": "serif",
        "font.size": 10.0,
        "text.usetex": False,
        "axes.labelsize": "large",
        "xtick.major.size": 10,
        "xtick.minor.size": 6,
        "xtick.minor.visible": True,
        "xtick.labelsize": "large",
        "xtick.direction": "in",
        "xtick.top": True,
        "ytick.major.size": 10,
        "ytick.minor.size": 6,
        "ytick.minor.visible": True,
        "ytick.labelsize": "large",
        "ytick.direction": "in",
        "ytick.right": True,
        "legend.numpoints": 1,
        "legend.fontsize": "medium",
        "legend.frameon": False,
        "legend.scatterpoints": 1,
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
    if key in {"ye", "electron_fraction"}:
        return {"cmap": "plasma", "label": r"$Y_e$", "log": False}
    if key in {"bmag", "log_bmag", "magnetic"}:
        return {"cmap": "plasma", "label": label, "log": False}
    if key in {"current", "curl_b", "jmag", "log_current"}:
        return {"cmap": "inferno", "label": label, "log": False}
    if key in {"divb", "curlx", "curly", "curlz", "vx", "vy", "vz", "velx", "vely", "velz"}:
        return {"cmap": "RdBu_r", "label": name, "log": False}
    if key in {"vorticity", "omega", "qcriterion", "q"}:
        return {"cmap": "magma", "label": name, "log": False}
    return {"cmap": "inferno", "label": name, "log": False}


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


def make_coordinates(shape: Sequence[int], domain_size: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = shape
    lx, ly, lz = domain_size
    x = np.linspace(-0.5 * lx, 0.5 * lx, nx)
    y = np.linspace(-0.5 * ly, 0.5 * ly, ny)
    z = np.linspace(-0.5 * lz, 0.5 * lz, nz)
    return np.meshgrid(x, y, z, indexing="ij")


def nearest_index(coords: Iterable[np.ndarray], point: Sequence[float]) -> Tuple[int, int, int]:
    axes = [c[:, 0, 0] if i == 0 else c[0, :, 0] if i == 1 else c[0, 0, :] for i, c in enumerate(coords)]
    return tuple(int(np.argmin(np.abs(axis - p))) for axis, p in zip(axes, point))
