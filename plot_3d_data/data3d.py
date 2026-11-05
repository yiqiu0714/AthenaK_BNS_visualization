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
        coordinates: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ):
        self.fields = fields
        self.shape = shape
        self.domain_size = domain_size
        self.spacing = spacing
        self.time = time
        self.source = source
        self.metadata = metadata or {}
        self.coordinates = coordinates

    @property
    def gradient_spacing(self):
        """Coordinate arrays for gradients on nonuniform grids, else dx/dy/dz."""
        return self.coordinates if self.coordinates is not None else self.spacing

    @property
    def is_uniform(self) -> bool:
        """Whether all coordinate axes are uniformly spaced."""
        if self.coordinates is None:
            return True
        for axis in self.coordinates:
            delta = np.diff(axis)
            if delta.size and not np.allclose(
                delta, delta[0], rtol=1e-5, atol=1e-7
            ):
                return False
        return True


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

    if "extent" in attrs:
        extent = float(attrs["extent"])
        return (2.0 * extent, 2.0 * extent, 2.0 * extent)

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


def _uniform_centers(lo: float, hi: float, n: int) -> np.ndarray:
    """Return n uniformly spaced cell centers spanning edge interval [lo, hi]."""
    edges = np.linspace(lo, hi, n + 1, dtype=np.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def _axes_from_domain(shape: Sequence[int], domain_size: Sequence[float]):
    return tuple(
        _uniform_centers(-0.5 * float(length), 0.5 * float(length), int(n))
        for length, n in zip(domain_size, shape)
    )


def _read_hdf5_axes(f, shape_xyz: Sequence[int]):
    """Read AthenaK-style x1/x2/x3 face or center coordinates."""
    axes = []
    for i, n in enumerate(shape_xyz, start=1):
        face_name = f"x{i}f"
        center_names = (f"x{i}v", f"x{i}")
        if (
            face_name in f
            and f[face_name].ndim == 1
            and len(f[face_name]) == n + 1
        ):
            faces = np.asarray(f[face_name][:], dtype=np.float64)
            axes.append(0.5 * (faces[:-1] + faces[1:]))
            continue
        center_name = next(
            (
                name
                for name in center_names
                if name in f and f[name].ndim == 1 and len(f[name]) == n
            ),
            None,
        )
        if center_name is None:
            return None
        axes.append(np.asarray(f[center_name][:], dtype=np.float64))
    return tuple(axes)


def _slice_for_crop(axis: np.ndarray, lo: float, hi: float) -> slice:
    indices = np.flatnonzero((axis >= lo) & (axis <= hi))
    if indices.size == 0:
        raise ValueError(
            f"Crop interval [{lo}, {hi}] does not intersect coordinate axis"
        )
    return slice(int(indices[0]), int(indices[-1]) + 1)


def _downsample_slices(slices, shape_xyz: Sequence[int], resolution: int):
    """Stride all axes so the longest selected axis is at most resolution."""
    lengths = [len(range(*sl.indices(int(n)))) for sl, n in zip(slices, shape_xyz)]
    if resolution <= 0 or max(lengths) <= resolution:
        return tuple(slices)
    stride = int(np.ceil(max(lengths) / float(resolution)))
    return tuple(slice(sl.start, sl.stop, stride) for sl in slices)


def load_hdf5_data(
    path: str,
    variables: Optional[Iterable[str]] = None,
    domain_size: Optional[Tuple[float, float, float]] = None,
    dx: Optional[float] = None,
    resolution: int = 0,
    crop: Optional[Tuple[float, float, float, float, float, float]] = None,
) -> Data3D:
    fields: Dict[str, np.ndarray] = {}
    metadata: Dict[str, object] = {}

    with h5py.File(path, "r") as f:
        for key, value in f.attrs.items():
            metadata[key] = _decode_attr(value)

        available = [
            key for key in f.keys()
            if isinstance(f[key], h5py.Dataset) and f[key].ndim == 3
        ]
        selected = list(variables) if variables else available
        selected = [v for v in selected if v in available]
        if not selected:
            selected = available
        if not selected:
            raise ValueError(f"No 3D datasets found in {path}")

        axis_order = str(metadata.get("axis_order", "zyx")).lower()
        if axis_order not in {"xyz", "zyx"}:
            raise ValueError("HDF5 axis_order must be 'xyz' or 'zyx'")
        file_shape = tuple(int(n) for n in f[selected[0]].shape)
        shape_xyz = file_shape if axis_order == "xyz" else file_shape[::-1]
        coordinates = None if domain_size is not None else _read_hdf5_axes(
            f, shape_xyz
        )
        initial_domain = _domain_from_attrs(metadata, shape_xyz, domain_size)
        if coordinates is None:
            coordinates = _axes_from_domain(shape_xyz, initial_domain)

        xyz_slices = (slice(None), slice(None), slice(None))
        if crop is not None:
            xyz_slices = (
                _slice_for_crop(coordinates[0], crop[0], crop[1]),
                _slice_for_crop(coordinates[1], crop[2], crop[3]),
                _slice_for_crop(coordinates[2], crop[4], crop[5]),
            )
        xyz_slices = _downsample_slices(xyz_slices, shape_xyz, resolution)
        coordinates = tuple(axis[sl] for axis, sl in zip(coordinates, xyz_slices))
        file_slices = xyz_slices if axis_order == "xyz" else xyz_slices[::-1]

        for var in selected:
            if tuple(f[var].shape) != file_shape:
                raise ValueError(
                    f"Dataset '{var}' shape {f[var].shape} does not match "
                    f"{selected[0]} shape {file_shape}"
                )
            arr = np.asarray(f[var][file_slices], dtype=np.float32)
            # Coarse-graining scripts in this project store HDF5 as (z, y, x).
            if axis_order != "xyz":
                arr = np.transpose(arr, (2, 1, 0))
            fields[var] = np.ascontiguousarray(arr)

    sample = next(iter(fields.values()))
    shape = tuple(int(n) for n in sample.shape)
    dom = tuple(
        float(
            axis[-1]
            - axis[0]
            + (np.median(np.diff(axis)) if len(axis) > 1 else initial_domain[i])
        )
        for i, axis in enumerate(coordinates)
    )
    spacing = tuple(
        float(np.median(np.diff(axis))) if len(axis) > 1 else float(dom[i])
        for i, axis in enumerate(coordinates)
    )
    if dx is not None:
        spacing = (dx, dx, dx)
    return Data3D(
        fields=fields,
        shape=shape,
        domain_size=dom,
        spacing=spacing,
        time=float(metadata.get("time", metadata.get("Time", 0.0))),
        source=path,
        metadata=metadata,
        coordinates=coordinates,
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
    interpolation: str = "linear",
) -> Data3D:
    install_athplot_paths(athplot_paths)
    from athplot.load_ath_bin import BinaryData

    bindata = BinaryData(path)
    # Callers may ask for several spellings of the same quantity (e.g. both
    # 'ye' and 's_00') because the name differs between AthenaK builds; keep
    # only the ones this file actually carries.
    available = tuple(getattr(bindata, "variable_names", ()) or ())
    requested = tuple(variables)
    if available:
        variables = tuple(v for v in requested if v in available)
        if not variables:
            raise ValueError(
                f"None of the requested variables {requested} are in {path}; "
                f"available: {list(available)}"
            )
    else:
        variables = requested
    interpolation = interpolation.lower()
    if interpolation not in {"nearest", "linear"}:
        raise ValueError("interpolation must be 'nearest' or 'linear'")

    levels = sorted(set(b.level for b in bindata.blocks))
    is_amr = len(levels) > 1

    if is_amr or resolution > 0:
        return _load_binary_amr(bindata, variables, domain_size, dx,
                                resolution, crop, interpolation)

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

    bounds = _domain_extent_from_blocks(bindata.blocks)
    domain_override = domain_size is not None
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
        coordinates=_axes_from_domain(shape, domain_size) if domain_override else (
            _uniform_centers(bounds[0], bounds[1], nx),
            _uniform_centers(bounds[2], bounds[3], ny),
            _uniform_centers(bounds[4], bounds[5], nz),
        ),
    )


def _load_binary_amr(bindata, variables, domain_size_override, dx_override,
                     resolution, crop, interpolation):
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

            # Output cell centers in this block. ``linear`` reconstructs
            # smooth hydrodynamic fields between native cell centres; keep
            # ``nearest`` for exact cell-value inspection.
            out_cx = x_lo + (np.arange(i0, i1) + 0.5) * dx_out
            out_cy = y_lo + (np.arange(j0, j1) + 0.5) * dy_out
            out_cz = z_lo + (np.arange(k0, k1) + 0.5) * dz_out

            if interpolation == "nearest":
                sample_i = np.clip(((out_cx - bx0) / block_dx).astype(int),
                                   0, block_nx - 1)
                sample_j = np.clip(((out_cy - by0) / block_dy).astype(int),
                                   0, block_ny - 1)
                sample_k = np.clip(((out_cz - bz0) / block_dz).astype(int),
                                   0, block_nz - 1)
            else:
                # Native values are cell-centred, hence the half-cell offset.
                sample_i = np.clip((out_cx - bx0) / block_dx - 0.5,
                                   0.0, block_nx - 1.0)
                sample_j = np.clip((out_cy - by0) / block_dy - 0.5,
                                   0.0, block_ny - 1.0)
                sample_k = np.clip((out_cz - bz0) / block_dz - 0.5,
                                   0.0, block_nz - 1.0)

            sample_coords = np.meshgrid(
                sample_i, sample_j, sample_k, indexing="ij"
            )

            for var, arr in zip(variables, values):
                # get_var returns (nz, ny, nx); transpose to (nx, ny, nz)
                arr_xyz = np.asarray(arr, dtype=np.float32).transpose(2, 1, 0)
                if interpolation == "nearest":
                    fields[var][i0:i1, j0:j1, k0:k1] = arr_xyz[
                        sample_coords[0], sample_coords[1], sample_coords[2]
                    ]
                else:
                    from scipy.ndimage import map_coordinates

                    fields[var][i0:i1, j0:j1, k0:k1] = map_coordinates(
                        arr_xyz,
                        sample_coords,
                        order=1,
                        mode="nearest",
                        prefilter=False,
                    )

    domain_size = domain_size_override or (Lx, Ly, Lz)
    return Data3D(
        fields=fields,
        shape=shape,
        domain_size=domain_size,
        spacing=(dx_out, dy_out, dz_out) if dx_override is None else (
            dx_override, dx_override, dx_override
        ),
        time=float(getattr(bindata, "time", 0.0)),
        source=bindata.filename,
        metadata={"time": float(getattr(bindata, "time", 0.0)),
                  "amr_levels": len(levels), "crop": crop,
                  "interpolation": interpolation},
        coordinates=_axes_from_domain(shape, domain_size) if domain_size_override else (
            _uniform_centers(x_lo, x_hi, nx),
            _uniform_centers(y_lo, y_hi, ny),
            _uniform_centers(z_lo, z_hi, nz),
        ),
    )


def load_3d_data(
    path: str,
    variables: Optional[Iterable[str]] = None,
    domain_size: Optional[Tuple[float, float, float]] = None,
    dx: Optional[float] = None,
    athplot_paths: Optional[Sequence[str]] = None,
    resolution: int = 0,
    crop: Optional[Tuple[float, float, float, float, float, float]] = None,
    interpolation: str = "linear",
) -> Data3D:
    if is_hdf5_file(path):
        return load_hdf5_data(
            path,
            variables=variables,
            domain_size=domain_size,
            dx=dx,
            resolution=resolution,
            crop=crop,
        )

    default_vars = ("dens", "temperature", "velx", "vely", "velz", "bcc1", "bcc2", "bcc3")
    return load_binary_data(
        path,
        variables=tuple(variables or default_vars),
        domain_size=domain_size,
        dx=dx,
        athplot_paths=athplot_paths,
        resolution=resolution,
        crop=crop,
        interpolation=interpolation,
    )
