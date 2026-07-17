#!/usr/bin/env python3
"""Render three orthogonal slices through a 3D AthenaK/BNS data cube."""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style  # central house style (repo root); see plot_style.py

from data3d import load_3d_data
from style3d import (
    apply_bns_matplotlib_style,
    convert_fields,
    default_scalar_style,
    finite_percentile_limits,
    get_scalar_field,
    make_coordinates,
    nearest_index,
    parse_center,
    parse_crop,
    parse_csv_floats,
    parse_domain_size,
)


def _needed_variables(name: str):
    key = name.lower()
    if key in {"density", "rho", "log_density", "log_dens", "log_rho"}:
        return ["dens"]
    if key in {"temperature", "temp"}:
        return ["temperature"]
    if key in {"bmag", "log_bmag", "magnetic", "divb",
               "current", "curl_b", "jmag", "log_current"}:
        return ["bcc1", "bcc2", "bcc3"]
    if key in {"vorticity", "omega", "qcriterion", "q"}:
        return ["velx", "vely", "velz"]
    return [name]


def _norm(values, norm_name, clim):
    vmin, vmax = clim
    if norm_name == "log":
        positive = values[np.isfinite(values) & (values > 0)]
        if positive.size == 0:
            raise ValueError("Log norm requested, but no positive values were found")
        return LogNorm(vmin=max(vmin, float(np.nanmin(positive))), vmax=vmax)
    if norm_name == "symlog":
        linthresh = max(abs(vmin), abs(vmax)) * 1e-3
        return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="Input .h5/.hdf5 or AthenaK .bin file")
    parser.add_argument("-o", "--output", default="orthoslices.png", help="Output PNG path")
    parser.add_argument("--scalar", default="density", help="Scalar or derived scalar to slice")
    parser.add_argument("--center", default=None, help="Slice center x,y,z; default is domain center")
    parser.add_argument("--domain-size", default=None, help="One value for cubic domain or Lx,Ly,Lz")
    parser.add_argument("--dx", type=float, default=None, help="Override uniform grid spacing")
    parser.add_argument("--resolution", type=int, default=256, help="Output grid cells per axis for AMR data (default 256)")
    parser.add_argument("--crop", default=None, help="Crop domain: half-width L or xmin,xmax,ymin,ymax,zmin,zmax")
    parser.add_argument("--cmap", default=None, help="Colormap name")
    parser.add_argument("--clim", default=None, help="Color limits as vmin,vmax")
    parser.add_argument("--norm", choices=["linear", "log", "symlog"], default="linear")
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--units", default="code", choices=["code", "cgs"], help="Unit system (default: code)")
    args = parser.parse_args()

    apply_bns_matplotlib_style()

    domain_size = parse_domain_size(args.domain_size)
    crop = parse_crop(args.crop)
    dataset = load_3d_data(args.input, variables=_needed_variables(args.scalar),
                           domain_size=domain_size, dx=args.dx,
                           resolution=args.resolution, crop=crop)
    convert_fields(dataset.fields, args.units)
    values = get_scalar_field(dataset.fields, args.scalar, dataset.spacing)

    coords = make_coordinates(dataset.shape, dataset.domain_size)
    center = parse_center(args.center) or (0.0, 0.0, 0.0)
    ix, iy, iz = nearest_index(coords, center)

    style = default_scalar_style(args.scalar, args.units)
    cmap = args.cmap or str(style["cmap"])
    clim = parse_csv_floats(args.clim) or finite_percentile_limits(values)
    if len(clim) != 2:
        raise ValueError("--clim must be vmin,vmax")

    norm = _norm(values, args.norm, clim)
    x = coords[0][:, 0, 0]
    y = coords[1][0, :, 0]
    z = coords[2][0, 0, :]

    panels = [
        (values[:, :, iz].T, (x[0], x[-1], y[0], y[-1]), r"$x~[\mathrm{M}_\odot]$", r"$y~[\mathrm{M}_\odot]$", f"xy, z={z[iz]:.3g}"),
        (values[:, iy, :].T, (x[0], x[-1], z[0], z[-1]), r"$x~[\mathrm{M}_\odot]$", r"$z~[\mathrm{M}_\odot]$", f"xz, y={y[iy]:.3g}"),
        (values[ix, :, :].T, (y[0], y[-1], z[0], z[-1]), r"$y~[\mathrm{M}_\odot]$", r"$z~[\mathrm{M}_\odot]$", f"yz, x={x[ix]:.3g}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=plot_style.figsize(1, 3, panel_w=3.0, panel_h=3.0),
                             constrained_layout=False)
    image = None
    for ax, (arr, extent, xlabel, ylabel, title) in zip(axes, panels):
        image = ax.imshow(
            arr,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    if args.title:
        fig.suptitle(args.title)
    else:
        fig.suptitle(f"t = {dataset.time:.3f}")

    fig.tight_layout()

    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    fig.colorbar(image, cax=cax, label=str(style["label"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
