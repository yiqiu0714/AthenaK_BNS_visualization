#!/usr/bin/env python3
"""Render AthenaK/BNS-style 3D isosurfaces from HDF5 or AthenaK binary data."""

import argparse
import os
from typing import Optional

import numpy as np
import pyvista as pv

import annotate3d
from data3d import load_3d_data
from style3d import (
    add_orthographic_scale_bar,
    auto_levels,
    camera_track_values,
    convert_fields,
    default_scalar_style,
    finite_percentile_limits,
    get_scalar_field,
    get_vector_field,
    apply_font_to_scalar_bars,
    font_kwargs,
    make_coordinate_axes,
    nice_scale_bar_length,
    plain_text_label,
    parse_crop,
    parse_csv_floats,
    parse_domain_size,
    relative_time_ms,
    resolve_font,
    time_label,
    set_pyvista_camera,
)


pv.global_theme.allow_empty_mesh = True


def smooth_contour(contour, iterations: int):
    """Volume-preserving smoothing for staircase-prone AMR contours."""
    if iterations <= 0:
        return contour
    return contour.smooth_taubin(
        n_iter=iterations,
        pass_band=0.08,
        boundary_smoothing=False,
        normalize_coordinates=True,
    )


def _needed_variables(*names: Optional[str]):
    raw = set()
    for name in names:
        if not name:
            continue
        key = name.lower()
        if key in {"density", "rho", "dens", "log_density", "log_dens", "log_rho"}:
            raw.add("dens")
        elif key in {"temperature", "temp"}:
            raw.add("temperature")
        elif key in {"ye", "electron_fraction", "s_00"}:
            raw.update({"ye", "s_00"})
        elif key in {"bmag", "log_bmag", "magnetic", "b", "bfield", "divb",
                     "current", "curl_b", "jmag", "log_current"}:
            raw.update({"bcc1", "bcc2", "bcc3"})
        elif key in {"vorticity", "omega", "qcriterion", "q"}:
            raw.update({"velx", "vely", "velz"})
        elif key in {"velocity", "vel", "v"}:
            raw.update({"velx", "vely", "velz"})
        elif key in {"vx", "velx"}:
            raw.add("velx")
        elif key in {"vy", "vely"}:
            raw.add("vely")
        elif key in {"vz", "velz"}:
            raw.add("velz")
        elif key in {"bx", "bcc1"}:
            raw.add("bcc1")
        elif key in {"by", "bcc2"}:
            raw.add("bcc2")
        elif key in {"bz", "bcc3"}:
            raw.add("bcc3")
        else:
            raw.add(name)
    return sorted(raw)


def build_grid(dataset, scalar_name: str, color_name: Optional[str]):
    axes = make_coordinate_axes(
        dataset.shape,
        dataset.domain_size,
        dataset.coordinates,
    )
    grid = pv.RectilinearGrid(*axes)

    scalar = get_scalar_field(dataset.fields, scalar_name, dataset.gradient_spacing)
    grid[scalar_name] = scalar.flatten(order="F")

    if color_name and color_name != scalar_name:
        color = get_scalar_field(dataset.fields, color_name, dataset.gradient_spacing)
        grid[color_name] = color.flatten(order="F")

    for vector_name in ("B", "velocity"):
        try:
            vx, vy, vz = get_vector_field(dataset.fields, vector_name)
        except KeyError:
            continue
        grid[vector_name] = np.stack(
            (vx.flatten(order="F"), vy.flatten(order="F"), vz.flatten(order="F")),
            axis=-1,
        )
    return grid, scalar


def add_streamlines(
    plotter,
    grid,
    dataset,
    vector_name: str,
    seed_mode: str,
    tube_radius: float,
    seed_file: Optional[str] = None,
) -> None:
    if vector_name not in grid.array_names:
        raise KeyError(f"Vector '{vector_name}' is not available in the grid")

    bounds = grid.bounds
    lx = bounds[1] - bounds[0]
    ly = bounds[3] - bounds[2]
    lz = bounds[5] - bounds[4]

    axes = make_coordinate_axes(dataset.shape, dataset.domain_size, dataset.coordinates)
    if seed_mode == "file":
        if not seed_file:
            raise ValueError("--streamline-seed-file is required for file seeds")
        points = np.loadtxt(seed_file, comments="#", usecols=(0, 1, 2), ndmin=2)
        if points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("Streamline seed file must contain finite x y z columns")
        source = pv.PolyData(points)
    elif seed_mode == "density-max" and "dens" in dataset.fields:
        idx = np.unravel_index(np.nanargmax(dataset.fields["dens"]), dataset.shape)
        center = tuple(float(axis[i]) for axis, i in zip(axes, idx))
        source = pv.Disc(center=center, normal=(1.0, 0.0, 1.0), outer=0.12 * max(lx, ly, lz), r_res=5, c_res=12)
    elif seed_mode == "dual-density-max" and "dens" in dataset.fields:
        idx = np.unravel_index(np.nanargmax(dataset.fields["dens"]), dataset.shape)
        center = np.asarray(tuple(float(axis[i]) for axis, i in zip(axes, idx)))
        sources = [
            pv.Disc(center=center, normal=(1.0, 0.0, 1.0), outer=0.12 * max(lx, ly, lz), r_res=5, c_res=12),
            pv.Disc(center=-center, normal=(0.0, 1.0, 1.0), outer=0.12 * max(lx, ly, lz), r_res=5, c_res=12),
        ]
        for source in sources:
            lines = grid.streamlines_from_source(
                source,
                vectors=vector_name,
                max_step_length=0.01 * max(lx, ly, lz),
                max_length=4.0 * max(lx, ly, lz),  # pyvista>=0.44 dropped max_time
                integration_direction="both",
            )
            plotter.add_mesh(lines.tube(radius=tube_radius), color="white", opacity=0.5, show_scalar_bar=False)
        return
    else:
        source = pv.Plane(
            center=grid.center,
            direction=(0.0, 0.0, 1.0),
            i_size=0.75 * lx,
            j_size=0.75 * ly,
            i_resolution=10,
            j_resolution=10,
        )

    lines = grid.streamlines_from_source(
        source,
        vectors=vector_name,
        max_step_length=0.01 * max(lx, ly, lz),
        max_length=4.0 * max(lx, ly, lz),  # pyvista>=0.44 dropped max_time
        integration_direction="both",
    )
    plotter.add_mesh(lines.tube(radius=tube_radius), color="white", opacity=0.5, show_scalar_bar=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="Input .h5/.hdf5 or AthenaK .bin file")
    parser.add_argument("-o", "--output", default="plot3d.png", help="Output PNG path")
    parser.add_argument("--html", default=None, help="Optional interactive HTML export")
    parser.add_argument("--scalar", default="density", help="Surface scalar or derived scalar")
    parser.add_argument("--color-by", default=None, help="Field used to color surfaces; defaults to --scalar")
    parser.add_argument("--levels", default=None, help="Comma-separated isosurface levels; default uses percentiles")
    parser.add_argument("--nlevels", type=int, default=4, help="Number of auto levels")
    parser.add_argument("--opacity", default=None, help="Comma-separated opacities; default ramps from 0.3 to 1")
    parser.add_argument("--cmap", default=None, help="Matplotlib/PyVista colormap name")
    parser.add_argument("--clim", default=None, help="Color limits as vmin,vmax")
    parser.add_argument("--domain-size", default=None, help="One value for cubic domain or Lx,Ly,Lz")
    parser.add_argument("--dx", type=float, default=None, help="Override uniform grid spacing")
    parser.add_argument("--resolution", type=int, default=256, help="Output grid cells per axis for AMR data (default 256)")
    parser.add_argument("--interpolation", choices=["linear", "nearest"],
                        default="linear", help="AMR reconstruction method "
                        "(default linear; nearest preserves native cells)")
    parser.add_argument("--crop", default=None, help="Crop domain: half-width L or xmin,xmax,ymin,ymax,zmin,zmax")
    parser.add_argument("--streamlines", default=None, choices=["B", "velocity"], help="Optional vector streamlines")
    parser.add_argument(
        "--streamline-seeds",
        default="dual-density-max",
        choices=["plane", "density-max", "dual-density-max", "file"],
        help="Streamline seed geometry",
    )
    parser.add_argument(
        "--streamline-seed-file",
        default=None,
        help="Text file with x y z seed columns; use with --streamline-seeds file",
    )
    parser.add_argument("--tube-radius", type=float, default=None, help="Streamline tube radius")
    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=20,
        help="Taubin smoothing iterations applied to each contour (default 20)",
    )
    parser.add_argument("--background", default="black", help="PyVista background color")
    parser.add_argument("--window-size", default="1024,1024", help="PNG size as width,height")
    parser.add_argument("--anti-aliasing", choices=["none", "ssaa", "msaa", "fxaa"],
                        default="ssaa", help="Edge anti-aliasing method "
                        "(default ssaa)")
    parser.add_argument("--camera-elevation", type=float, default=30.0)
    parser.add_argument("--camera-azimuth", type=float, default=0.0)
    parser.add_argument("--camera-distance", type=float, default=None)
    parser.add_argument("--view-angle", type=float, default=30.0)
    parser.add_argument(
        "--projection", choices=["perspective", "orthographic"], default="perspective"
    )
    parser.add_argument("--parallel-scale", type=float, default=None,
                        help="Orthographic half-height in code-length units")
    parser.add_argument("--camera-track", default=None,
                        help="Time-ms camera track with scale and optional ruler columns")
    parser.add_argument("--t0", type=float, default=None,
                        help="Merger time in code units; labels become merger-relative")
    parser.add_argument("--annotate", choices=["mathtext", "vtk"],
                        default="mathtext",
                        help="mathtext draws labels with Matplotlib (real maths); "
                             "vtk uses VTK's plain-string text")
    parser.add_argument("--font", default="sans",
                        help="Annotation font: sans, serif, cm, arial, times, courier")
    parser.add_argument("--axes", action="store_true", help="Show an orientation triad")
    parser.add_argument(
        "--scalebar",
        default=None,
        help="Orthographic ruler length in M, or 'auto'; omitted disables it",
    )
    parser.add_argument("--show-scalar-bar", action="store_true", help="Show a scalar bar")
    parser.add_argument("--off-screen", action="store_true", help="Use off-screen rendering")
    parser.add_argument("--units", default="code", choices=["code", "cgs"], help="Unit system (default: code)")
    args = parser.parse_args()

    color_name = args.color_by or args.scalar
    domain_size = parse_domain_size(args.domain_size)
    variables = _needed_variables(args.scalar, color_name, args.streamlines)
    # Density-based streamline seeding needs 'dens' even when the plotted scalar
    # doesn't (e.g. --scalar log_bmag --streamlines B).  Load it so the seeds
    # land on the stars instead of silently falling back to a plane.
    if args.streamlines and args.streamline_seeds in ("density-max", "dual-density-max"):
        variables = sorted(set(variables) | {"dens"})
    crop = parse_crop(args.crop)
    dataset = load_3d_data(args.input, variables=variables, domain_size=domain_size,
                           dx=args.dx, resolution=args.resolution, crop=crop,
                           interpolation=args.interpolation)
    convert_fields(dataset.fields, args.units)
    grid, scalar = build_grid(dataset, args.scalar, color_name)

    levels = np.asarray(parse_csv_floats(args.levels) or auto_levels(scalar, args.nlevels), dtype=float)
    opacities = np.asarray(parse_csv_floats(args.opacity) or np.linspace(0.3, 1.0, len(levels)), dtype=float)
    if len(opacities) != len(levels):
        raise ValueError("The number of opacity values must match the number of levels")

    style = default_scalar_style(color_name, args.units)
    cmap = args.cmap or str(style["cmap"])
    clim = parse_csv_floats(args.clim)
    if clim is None:
        color_values = grid[color_name]
        clim = finite_percentile_limits(color_values)
    if len(clim) != 2:
        raise ValueError("--clim must be vmin,vmax")
    window_size = tuple(int(x) for x in parse_csv_floats(args.window_size))
    if len(window_size) != 2 or min(window_size) <= 0:
        raise ValueError("--window-size must be positive width,height")

    time_ms = relative_time_ms(dataset.time, args.t0)
    parallel_scale = args.parallel_scale
    track_ruler = None
    if args.camera_track:
        parallel_scale, track_ruler = camera_track_values(args.camera_track, time_ms)
    orthographic = (
        args.projection == "orthographic"
        or args.camera_track is not None
        or args.scalebar is not None
    )


    plotter = pv.Plotter(off_screen=args.off_screen)
    if args.anti_aliasing != "none":
        plotter.enable_anti_aliasing(args.anti_aliasing)
    plotter.set_background(args.background)
    pv.global_theme.background = args.background

    font_family, font_file = resolve_font(args.font)
    text_font = font_kwargs(font_family, font_file)
    use_mathtext = args.annotate == "mathtext"
    if not use_mathtext:
        plotter.add_text(
            time_label(time_ms, args.t0 is not None),
            position="upper_left",
            color="white",
            font_size=12,
            **text_font,
        )
    if args.axes:
        plotter.add_axes(color="white", line_width=2)

    # "%.0f" collapses a 0.05-0.50 Ye ramp to a column of zeros, so pick the
    # precision from the span actually being shown.
    clim_span = abs(float(clim[1]) - float(clim[0]))
    bar_fmt = "%.2f" if clim_span < 1.0 else "%.1f" if clim_span < 10.0 else "%.0f"
    scalar_bar_args = {
        "title": plain_text_label(style["label"]) + "\n\n ",
        "color": "white",
        "title_font_size": 16,
        "label_font_size": 14,
        "n_labels": 5,
        "fmt": bar_fmt,
    }
    if font_family:
        scalar_bar_args["font_family"] = font_family
    if args.scalebar is not None:
        # The ruler owns the bottom strip; go vertical on the left instead.
        scalar_bar_args.update({
            # The title is centred over the bar, so keep the bar far enough
            # from the edge that a label like "log10 rho [g cm^-3]" still fits.
            "vertical": True,
            "width": 0.035,
            "height": 0.40,
            "position_x": 0.10,
            "position_y": 0.45,
        })
    else:
        scalar_bar_args.update({
            "width": 0.25,
            "height": 0.05,
            "position_x": 0.72,
            "position_y": 0.02,
        })

    added_any = False
    for level, opacity in zip(levels, opacities):
        contour = grid.contour([float(level)], scalars=args.scalar)
        if contour.n_points == 0:
            print(f"Skipping empty contour at level {level:g}")
            continue
        contour = smooth_contour(contour, args.smooth_iterations)
        contour = contour.sample(grid)
        contour.set_active_scalars(color_name)
        plotter.add_mesh(
            contour,
            scalars=color_name,
            cmap=cmap,
            clim=tuple(float(x) for x in clim),
            pbr=True,
            smooth_shading=True,
            split_sharp_edges=False,
            metallic=0.1,
            roughness=0.4,
            opacity=float(opacity),
            show_scalar_bar=False,
        )
        added_any = True

    if args.show_scalar_bar and added_any and not use_mathtext:
        plotter.add_scalar_bar(**scalar_bar_args)
        apply_font_to_scalar_bars(plotter, font_family, font_file)

    if args.streamlines:
        bounds = grid.bounds
        default_radius = 0.0025 * max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        add_streamlines(
            plotter,
            grid,
            dataset,
            args.streamlines,
            args.streamline_seeds,
            args.tube_radius or default_radius,
            seed_file=args.streamline_seed_file,
        )

    light = pv.Light(position=np.asarray(grid.center) + np.array([14.0, 20.0, 15.0]), color="white")
    plotter.add_light(light)
    set_pyvista_camera(
        plotter,
        grid,
        elevation=args.camera_elevation,
        azimuth=args.camera_azimuth,
        distance=args.camera_distance,
        view_angle=args.view_angle,
        orthographic=orthographic,
        parallel_scale=parallel_scale,
    )
    bar_length = None
    if args.scalebar is not None:
        bar_length = (
            track_ruler if args.scalebar.lower() == "auto" else float(args.scalebar)
        )
        if use_mathtext:
            # Matplotlib draws it later; resolve "auto" against the same view.
            if bar_length is None or bar_length <= 0:
                aspect = window_size[0] / float(window_size[1])
                bar_length = nice_scale_bar_length(
                    2.0 * float(plotter.camera.parallel_scale) * aspect
                )
        else:
            add_orthographic_scale_bar(
                plotter, bar_length, float(plotter.camera.parallel_scale),
                window_size,
                color="white",
                position="lower_right" if args.axes else "lower_left",
                text_font=text_font,
            )



    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.html:
        plotter.export_html(args.html)
    print(f"Saving {args.output}")
    if use_mathtext:
        # Read the camera before closing the plotter; the ruler needs its scale.
        view_scale = float(plotter.camera.parallel_scale)
        image = plotter.screenshot(return_img=True, window_size=list(window_size))
        plotter.close()
        show_bar = bool(args.show_scalar_bar and added_any)
        annotate3d.compose(
            image, args.output, font=args.font,
            time_text=annotate3d.math_time_label(time_ms, args.t0 is not None),
            cmap=cmap if show_bar else None,
            clim=clim if show_bar else None,
            label=str(style["label"]) if show_bar else None,
            scalebar_length=bar_length,
            parallel_scale=view_scale if bar_length else None,
            background=args.background,
        )
    else:
        plotter.show(screenshot=args.output, window_size=list(window_size),
                     auto_close=True)


if __name__ == "__main__":
    main()
