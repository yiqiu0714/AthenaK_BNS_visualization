#!/usr/bin/env python3
"""Load AthenaK 3D data into consistent numpy arrays."""

import os
import sys
from typing import Dict, Iterable, Optional, Sequence, Tuple

import h5py
import numpy as np


DEFAULT_ATHPLOT_PATHS = (
    # Repo-local plot-tools (this file lives in <repo>/plot_3d_data/).
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plot-tools"),
    # Legacy machine-specific locations (kept as fallbacks).
    "/global/homes/y/yiqiu/plot-tools/",
    "/global/homes/y/yiqiu/work/visualize/incite-bns-zoom-2025/plot-tools-zoom/",
)


class Data3D:
    """Container for loaded 3D fields and grid metadata."""

    def __init__(
        self,
        fields: Dict[str, np.ndarray],
        shape: Tuple[int, int, int],
        domain_size: Tuple[float, float, float],
        spacing: Tuple[float, float, float],
        time: float = 0.0,
        source: str = "",
        metadata: Optional[Dict[str, object]] = None,
    ):
        self.fields = fields
        self.shape = shape
        self.domain_size = domain_size
        self.spacing = spacing
        self.time = time
        self.source = source
        self.metadata = metadata or {}


def is_hdf5_file(path: str) -> bool:
    return path.endswith(".h5") or path.endswith(".hdf5")


def install_athplot_paths(extra_paths: Optional[Sequence[str]] = None) -> None:
    for path in tuple(extra_paths or ()) + DEFAULT_ATHPLOT_PATHS:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _domain_from_attrs(attrs, shape: Sequence[int], override: Optional[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if override is not None:
        return override

    if all(k in attrs for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")):
        return (
            float(attrs["xmax"] - attrs["xmin"]),
            float(attrs["ymax"] - attrs["ymin"]),
            float(attrs["zmax"] - attrs["zmin"]),
        )

    if "domain_size" in attrs:
        raw = attrs["domain_size"]
        if np.isscalar(raw):
            val = float(raw)
            return (val, val, val)
        vals = tuple(float(x) for x in raw)
        if len(vals) == 1:
            return (vals[0], vals[0], vals[0])
        if len(vals) >= 3:
            return (vals[0], vals[1], vals[2])

    if all(k in attrs for k in ("Lx", "Ly", "Lz")):
        return (float(attrs["Lx"]), float(attrs["Ly"]), float(attrs["Lz"]))

    if "dx" in attrs:
        dx = float(attrs["dx"])
        return (shape[0] * dx, shape[1] * dx, shape[2] * dx)

    return (8.0, 8.0, 8.0)


def _spacing(domain_size: Sequence[float], shape: Sequence[int], dx_override: Optional[float]) -> Tuple[float, float, float]:
    if dx_override is not None:
        return (dx_override, dx_override, dx_override)
    return (
        float(domain_size[0]) / float(shape[0]),
        float(domain_size[1]) / float(shape[1]),
        float(domain_size[2]) / float(shape[2]),
    )


def load_hdf5_data(
    path: str,
    variables: Optional[Iterable[str]] = None,
    domain_size: Optional[Tuple[float, float, float]] = None,
    dx: Optional[float] = None,
) -> Data3D:
    fields: Dict[str, np.ndarray] = {}
    metadata: Dict[str, object] = {}

    with h5py.File(path, "r") as f:
        for key, value in f.attrs.items():
            metadata[key] = _decode_attr(value)

        available = [key for key in f.keys() if isinstance(f[key], h5py.Dataset)]
        selected = list(variables) if variables else available
        selected = [v for v in selected if v in available]
        if not selected:
            selected = available
        if not selected:
            raise ValueError(f"No datasets found in {path}")

        for var in selected:
            arr = np.asarray(f[var][:], dtype=np.float32)
            if arr.ndim != 3:
                continue
            # Coarse-graining scripts in this project store HDF5 as (z, y, x).
            # If a file declares axis_order='xyz', leave it as-is.
            axis_order = str(metadata.get("axis_order", "zyx")).lower()
            if axis_order != "xyz":
                arr = np.transpose(arr, (2, 1, 0))
            fields[var] = np.ascontiguousarray(arr)

    if not fields:
        raise ValueError(f"No 3D datasets found in {path}")

    sample = next(iter(fields.values()))
    shape = tuple(int(n) for n in sample.shape)
    dom = _domain_from_attrs(metadata, shape, domain_size)
    return Data3D(
        fields=fields,
        shape=shape,
        domain_size=dom,
        spacing=_spacing(dom, shape, dx),
        time=float(metadata.get("time", 0.0)),
        source=path,
        metadata=metadata,
    )


def _domain_extent_from_blocks(blocks):
    """Compute physical domain extent from block coordinate metadata."""
    x_lo = min(b.get_extent()[0] for b in blocks)
    x_hi = max(b.get_extent()[1] for b in blocks)
    y_lo = min(b.get_extent()[2] for b in blocks)
    y_hi = max(b.get_extent()[3] for b in blocks)
    z_lo = min(b.get_extent()[4] for b in blocks)
    z_hi = max(b.get_extent()[5] for b in blocks)
    return x_lo, x_hi, y_lo, y_hi, z_lo, z_hi


def load_binary_data(
    path: str,
    variables: Iterable[str],
    domain_size: Optional[Tuple[float, float, float]] = None,
    dx: Optional[float] = None,
    athplot_paths: Optional[Sequence[str]] = None,
    resolution: int = 0,
    crop: Optional[Tuple[float, float, float, float, float, float]] = None,
) -> Data3D:
    install_athplot_paths(athplot_paths)
    from athplot.load_ath_bin import BinaryData

    bindata = BinaryData(path)
    variables = tuple(variables)
    levels = sorted(set(b.level for b in bindata.blocks))
    is_amr = len(levels) > 1

    if is_amr or resolution > 0:
        return _load_binary_amr(bindata, variables, domain_size, dx,
                                resolution, crop)

    # Original uniform-grid assembly (single AMR level)
    block_size = bindata.blocks[0].size
    nx = int(block_size[0] * (max(b.i for b in bindata.blocks) + 1))
    ny = int(block_size[1] * (max(b.j for b in bindata.blocks) + 1))
    nz = int(block_size[2] * (max(b.k for b in bindata.blocks) + 1))
    shape = (nx, ny, nz)

    fields = {var: np.zeros(shape, dtype=np.float32) for var in variables}
    for block, values in bindata.get_block_data(list(variables)):
        idx = block.restrict()
        for var, arr in zip(variables, values):
            fields[var][idx] = arr

    if domain_size is None:
        try:
            dx_block = tuple(float(x) for x in bindata.blocks[0].dx)
            domain_size = (dx_block[0] * nx, dx_block[1] * ny, dx_block[2] * nz)
        except Exception:
            domain_size = (8.0, 8.0, 8.0)

    return Data3D(
        fields=fields,
        shape=shape,
        domain_size=domain_size,
        spacing=_spacing(domain_size, shape, dx),
        time=float(getattr(bindata, "time", 0.0)),
        source=path,
        metadata={"time": float(getattr(bindata, "time", 0.0))},
    )


def _load_binary_amr(bindata, variables, domain_size_override, dx_override,
                     resolution, crop):
    """Assemble AMR binary data onto a uniform grid.

    Paints blocks from coarsest to finest level so that higher-resolution
    data overwrites lower-resolution data in overlapping regions.
    """
    # Physical domain extent
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = _domain_extent_from_blocks(
        bindata.blocks)

    # Apply crop if requested
    if crop is not None:
        x_lo = max(x_lo, crop[0])
        x_hi = min(x_hi, crop[1])
        y_lo = max(y_lo, crop[2])
        y_hi = min(y_hi, crop[3])
        z_lo = max(z_lo, crop[4])
        z_hi = min(z_hi, crop[5])

    Lx, Ly, Lz = x_hi - x_lo, y_hi - y_lo, z_hi - z_lo
    if resolution <= 0:
        resolution = 256

    # Scale resolution proportionally for non-cubic domains
    L_max = max(Lx, Ly, Lz)
    nx = max(1, int(round(resolution * Lx / L_max)))
    ny = max(1, int(round(resolution * Ly / L_max)))
    nz = max(1, int(round(resolution * Lz / L_max)))
    shape = (nx, ny, nz)

    dx_out = Lx / nx
    dy_out = Ly / ny
    dz_out = Lz / nz

    fields = {var: np.zeros(shape, dtype=np.float32) for var in variables}

    # Process levels from coarsest to finest so finer data overwrites
    levels = sorted(set(b.level for b in bindata.blocks))
    for level in levels:
        for block, values in bindata.get_block_data(
            list(variables),
            block_filter=lambda b, lv=level: b.level == lv,
        ):
            ext = block.get_extent()
            bx0, bx1 = ext[0], ext[1]
            by0, by1 = ext[2], ext[3]
            bz0, bz1 = ext[4], ext[5]

            # Skip blocks outside the crop region
            if bx1 <= x_lo or bx0 >= x_hi:
                continue
            if by1 <= y_lo or by0 >= y_hi:
                continue
            if bz1 <= z_lo or bz0 >= z_hi:
                continue

            # Output cell range covered by this block
            i0 = max(0, int((bx0 - x_lo) / dx_out))
            i1 = min(nx, int(round((bx1 - x_lo) / dx_out)))
            j0 = max(0, int((by0 - y_lo) / dy_out))
            j1 = min(ny, int(round((by1 - y_lo) / dy_out)))
            k0 = max(0, int((bz0 - z_lo) / dz_out))
            k1 = min(nz, int(round((bz1 - z_lo) / dz_out)))

            if i1 <= i0 or j1 <= j0 or k1 <= k0:
                continue

            block_nx, block_ny, block_nz = block.size
            block_dx = (bx1 - bx0) / block_nx
            block_dy = (by1 - by0) / block_ny
            block_dz = (bz1 - bz0) / block_nz

            # Output cell centers → nearest block cell (nearest-neighbor)
            out_cx = x_lo + (np.arange(i0, i1) + 0.5) * dx_out
            out_cy = y_lo + (np.arange(j0, j1) + 0.5) * dy_out
            out_cz = z_lo + (np.arange(k0, k1) + 0.5) * dz_out

            bi = np.clip(((out_cx - bx0) / block_dx).astype(int),
                         0, block_nx - 1)
            bj = np.clip(((out_cy - by0) / block_dy).astype(int),
                         0, block_ny - 1)
            bk = np.clip(((out_cz - bz0) / block_dz).astype(int),
                         0, block_nz - 1)

            bii, bjj, bkk = np.meshgrid(bi, bj, bk, indexing="ij")

            for var, arr in zip(variables, values):
                # get_var returns (nz, ny, nx); transpose to (nx, ny, nz)
                arr_xyz = np.asarray(arr, dtype=np.float32).transpose(2, 1, 0)
                fields[var][i0:i1, j0:j1, k0:k1] = arr_xyz[bii, bjj, bkk]

    domain_size = domain_size_override or (Lx, Ly, Lz)
    return Data3D(
        fields=fields,
        shape=shape,
        domain_size=domain_size,
        spacing=_spacing(domain_size, shape, dx_override),
        time=float(getattr(bindata, "time", 0.0)),
        source=bindata.filename,
        metadata={"time": float(getattr(bindata, "time", 0.0)),
                  "amr_levels": len(levels), "crop": crop},
    )


def load_3d_data(
    path: str,
    variables: Optional[Iterable[str]] = None,
    domain_size: Optional[Tuple[float, float, float]] = None,
    dx: Optional[float] = None,
    athplot_paths: Optional[Sequence[str]] = None,
    resolution: int = 0,
    crop: Optional[Tuple[float, float, float, float, float, float]] = None,
) -> Data3D:
    if is_hdf5_file(path):
        return load_hdf5_data(path, variables=variables, domain_size=domain_size, dx=dx)

    default_vars = ("dens", "temperature", "velx", "vely", "velz", "bcc1", "bcc2", "bcc3")
    return load_binary_data(
        path,
        variables=tuple(variables or default_vars),
        domain_size=domain_size,
        dx=dx,
        athplot_paths=athplot_paths,
        resolution=resolution,
        crop=crop,
    )
