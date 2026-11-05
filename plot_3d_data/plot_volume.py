#!/usr/bin/env python3
"""Render a 3D AthenaK/BNS scalar as a volume (density-style transfer function).

Companion to plot_isosurfaces.py.  Volume rendering is the right tool for
density, whose colour ramp paints tenuous material dark; it is the wrong tool
for composition fields such as Ye, which saturate once the volume is optically
thick -- use plot_isosurfaces.py --color-by ye for those.
"""

import argparse
import os
from typing import Optional

import numpy as np
import pyvista as pv

import annotate3d
from data3d import load_3d_data
from style3d import (
    add_orthographic_scale_bar,
    camera_track_values,
    convert_fields,
    default_scalar_style,
    finite_percentile_limits,
    get_scalar_field,
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

# A 480^3 float32 field is ~422 MiB; VTK keeps its own copy on top of numpy's.
_CELL_WARN_THRESHOLD = 300**3


def _needed_variables(name: str):
    key = name.lower()
    if key in {"density", "rho", "dens", "log_density", "log_dens", "log_rho"}:
        return ["dens"]
    if key in {"temperature", "temp"}:
        return ["temperature"]
    if key in {"ye", "electron_fraction", "s_00"}:
        return ["ye", "s_00"]
    if key in {"bmag", "log_bmag", "magnetic", "divb",
               "current", "curl_b", "jmag", "log_current"}:
        return ["bcc1", "bcc2", "bcc3"]
    if key in {"vorticity", "omega", "qcriterion", "q"}:
        return ["velx", "vely", "velz"]
    return [name]


def resample_to_uniform(values, axes, samples: Optional[int] = None):
    """Resample a (possibly stretched) grid onto the uniform grid VTK needs.

    VTK's volume mapper resamples internally onto a uniform lattice, so a
    stretched mesh has its refined core averaged away and renders as an empty
    frame.  Doing the resampling here keeps that control in our hands.
    """
    from scipy.interpolate import RegularGridInterpolator

    counts = [len(axis) for axis in axes]
    if samples and samples > 0:
        counts = [int(samples)] * 3

    uniform_axes = tuple(
        np.linspace(float(axis[0]), float(axis[-1]), n, dtype=np.float64)
        for axis, n in zip(axes, counts)
    )
    interp = RegularGridInterpolator(
        tuple(np.asarray(a, dtype=np.float64) for a in axes),
        np.asarray(values, dtype=np.float32),
        method="linear",
        bounds_error=False,
        fill_value=float(np.nanmin(values)),
    )
    mesh = np.meshgrid(*uniform_axes, indexing="ij")
    points = np.stack([m.ravel() for m in mesh], axis=-1)
    resampled = interp(points).reshape(counts).astype(np.float32)
    return resampled, uniform_axes


def build_volume_grid(dataset, scalar_name: str, samples: Optional[int]):
    """Return a pv.ImageData carrying the scalar, plus the scalar array."""
    axes = make_coordinate_axes(dataset.shape, dataset.domain_size,
                                dataset.coordinates)
    values = get_scalar_field(dataset.fields, scalar_name,
                              dataset.gradient_spacing)

    needs_resample = (not dataset.is_uniform) or bool(samples)
    if needs_resample:
        values, axes = resample_to_uniform(values, axes, samples)

    spacing = tuple(
        float(axis[1] - axis[0]) if len(axis) > 1 else 1.0 for axis in axes
    )
    grid = pv.ImageData(
        dimensions=tuple(len(axis) for axis in axes),
        spacing=spacing,
        origin=tuple(float(axis[0]) for axis in axes),
    )
    grid[scalar_name] = values.flatten(order="F")
    grid.set_active_scalars(scalar_name)
    return grid, values


def volume_rendering_works() -> bool:
    """Probe whether this GL stack can actually composite a volume.

    Mesa/llvmpipe software rendering reports success but emits an empty frame
    for every volume mapper, so a rendered-pixel probe is the only reliable
    test.  Cheap enough (a 16^3 blob) to run once per invocation.
    """
    try:
        n = 16
        axis = np.linspace(-1.0, 1.0, n)
        gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
        blob = np.exp(-6.0 * (gx**2 + gy**2 + gz**2)).astype(np.float32)
        probe = pv.ImageData(dimensions=(n, n, n),
                             spacing=(2.0 / (n - 1),) * 3, origin=(-1, -1, -1))
        probe["f"] = blob.flatten(order="F")
        plotter = pv.Plotter(off_screen=True, window_size=[64, 64])
        plotter.set_background("black")
        plotter.add_volume(probe, scalars="f", cmap="inferno", clim=(0.0, 1.0),
                           opacity=np.linspace(0.0, 1.0, 256),
                           show_scalar_bar=False)
        plotter.camera_position = "xy"
        plotter.render()
        image = plotter.screenshot(return_img=True)
        plotter.close()
        return bool((image.sum(axis=2) > 10).sum() > 0)
    except Exception:
        return False


def smooth_contour(contour, iterations: int):
    """Volume-preserving smoothing for layered-volume shells."""
    if iterations <= 0:
        return contour
    return contour.smooth_taubin(
        n_iter=iterations,
        pass_band=0.08,
        boundary_smoothing=False,
        normalize_coordinates=True,
    )


def add_layered_volume(plotter, grid, scalar_name, clim, cmap, opacity_curve,
                       n_layers: int, smooth_iterations: int = 0,
                       scalar_bar_args=None):
    """Approximate a volume with a stack of translucent isosurfaces.

    Used where true volume rendering is unavailable.  Each shell takes its
    opacity from the same transfer function the volume mapper would have used,
    so the two modes read similarly.
    """
    lo, hi = float(clim[0]), float(clim[1])
    levels = np.linspace(lo, hi, n_layers + 2)[1:-1]
    added = False
    for level in levels:
        contour = grid.contour([float(level)], scalars=scalar_name)
        if contour.n_points == 0:
            continue
        contour = smooth_contour(contour, smooth_iterations)
        position = (level - lo) / (hi - lo) if hi > lo else 0.0
        index = int(np.clip(position * (len(opacity_curve) - 1), 0,
                            len(opacity_curve) - 1))
        # Many thin shells accumulate, so damp each one to keep the stack from
        # reading as a solid surface.
        shell_opacity = float(np.clip(
            opacity_curve[index] / max(n_layers / 4.0, 1.0), 0.003, 0.35
        ))
        plotter.add_mesh(
            contour, scalars=scalar_name, cmap=cmap, clim=(lo, hi),
            opacity=shell_opacity, show_scalar_bar=False, smooth_shading=True,
            split_sharp_edges=False,
        )
        added = True
    if added and scalar_bar_args is not None:
        plotter.add_scalar_bar(**scalar_bar_args)
    return added


def build_opacity(name: str, n_colors: int = 256) -> np.ndarray:
    """Opacity transfer functions tuned for log-density volume rendering."""
    ramp = np.linspace(0.0, 1.0, n_colors)
    key = name.lower()
    if key == "linear":
        return ramp
    if key == "sqrt":
        return np.sqrt(ramp)
    if key == "sigmoid":
        # Keeps the tenuous outflow nearly transparent, then opens up quickly
        # across the mid ramp so the remnant core still reads as solid.
        return 1.0 / (1.0 + np.exp(-12.0 * (ramp - 0.55)))
    if key == "geom":
        return ramp**3
    raise ValueError(f"Unknown opacity transfer function '{name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True,
                        help="Input .h5/.hdf5 or AthenaK .bin file")
    parser.add_argument("-o", "--output", default="volume.png",
                        help="Output PNG path")
    parser.add_argument("--scalar", default="log_rho",
                        help="Scalar or derived scalar to volume render")
    parser.add_argument("--cmap", default=None, help="Colormap name")
    parser.add_argument("--clim", default=None, help="Color limits as vmin,vmax")
    parser.add_argument("--opacity", default="sigmoid",
                        choices=["linear", "sqrt", "sigmoid", "geom"],
                        help="Opacity transfer function (default: sigmoid)")
    parser.add_argument("--opacity-scale", type=float, default=1.0,
                        help="Multiplier applied to the opacity curve")
    parser.add_argument("--opacity-unit-distance", type=float, default=None,
                        help="Distance over which the transfer function acts")
    parser.add_argument("--interpolation", choices=["linear", "nearest"],
                        default="linear", help="AMR reconstruction method "
                        "(default linear; nearest preserves native cells)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Resample onto this many cells per axis before rendering")
    parser.add_argument("--anti-aliasing", choices=["none", "ssaa", "msaa", "fxaa"],
                        default="ssaa", help="Edge anti-aliasing method "
                        "(default ssaa)")
    parser.add_argument("--shade", action="store_true",
                        help="Enable gradient shading in the volume mapper")
    parser.add_argument("--mode", choices=["auto", "volume", "layers"],
                        default="auto",
                        help="auto probes GL and falls back to layers (default)")
    parser.add_argument("--layers", type=int, default=24,
                        help="Shell count for --mode layers (default 24)")
    parser.add_argument("--smooth-iterations", type=int, default=12,
                        help="Taubin smoothing iterations for fallback shells "
                             "(default 12)")
    parser.add_argument("--domain-size", default=None,
                        help="One value for cubic domain or Lx,Ly,Lz")
    parser.add_argument("--dx", type=float, default=None,
                        help="Override uniform grid spacing")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Load-time grid cells per axis (default 256)")
    parser.add_argument("--crop", default=None,
                        help="Crop domain: half-width L or xmin,xmax,ymin,ymax,zmin,zmax")
    parser.add_argument("--background", default="black")
    parser.add_argument("--window-size", default="1024,1024",
                        help="PNG size as width,height")
    parser.add_argument("--camera-elevation", type=float, default=30.0)
    parser.add_argument("--camera-azimuth", type=float, default=0.0)
    parser.add_argument("--camera-distance", type=float, default=None)
    parser.add_argument("--view-angle", type=float, default=30.0)
    parser.add_argument("--projection", choices=["perspective", "orthographic"],
                        default="perspective")
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
    parser.add_argument("--axes", action="store_true",
                        help="Show an orientation triad")
    parser.add_argument("--scalebar", default=None,
                        help="Orthographic ruler length in M, or 'auto'")
    parser.add_argument("--show-scalar-bar", action="store_true")
    parser.add_argument("--off-screen", action="store_true")
    parser.add_argument("--units", default="code", choices=["code", "cgs"])
    args = parser.parse_args()

    domain_size = parse_domain_size(args.domain_size)
    crop = parse_crop(args.crop)
    dataset = load_3d_data(args.input, variables=_needed_variables(args.scalar),
                           domain_size=domain_size, dx=args.dx,
                           resolution=args.resolution, crop=crop,
                           interpolation=args.interpolation)
    convert_fields(dataset.fields, args.units)
    grid, values = build_volume_grid(dataset, args.scalar, args.samples)

    if grid.n_points > _CELL_WARN_THRESHOLD:
        print(
            f"Warning: volume has {grid.n_points:,} points "
            f"(~{grid.n_points * 4 / 1024**3:.2f} GiB per float32 copy). "
            "Use --samples/--resolution/--crop, and keep parallel jobs low."
        )

    window_size = tuple(int(x) for x in parse_csv_floats(args.window_size))
    if len(window_size) != 2 or min(window_size) <= 0:
        raise ValueError("--window-size must be positive width,height")

    style = default_scalar_style(args.scalar, args.units)
    cmap = args.cmap or str(style["cmap"])
    clim = parse_csv_floats(args.clim) or finite_percentile_limits(values)
    if len(clim) != 2:
        raise ValueError("--clim must be vmin,vmax")

    opacity = np.clip(build_opacity(args.opacity) * args.opacity_scale, 0.0, 1.0)

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
            position="upper_left", color="white", font_size=12, **text_font,
        )
    if args.axes:
        plotter.add_axes(color="white", line_width=2)

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
            "width": 0.25, "height": 0.05,
            "position_x": 0.72, "position_y": 0.02,
        })

    mode = args.mode
    if mode == "auto":
        mode = "volume" if volume_rendering_works() else "layers"
        if mode == "layers":
            print(
                "Note: this OpenGL stack cannot composite volumes "
                "(software llvmpipe renders them empty); "
                "falling back to --mode layers."
            )

    if mode == "volume":
        plotter.add_volume(
            grid,
            scalars=args.scalar,
            cmap=cmap,
            clim=tuple(float(x) for x in clim),
            opacity=opacity,
            opacity_unit_distance=args.opacity_unit_distance,
            shade=args.shade,
            show_scalar_bar=bool(args.show_scalar_bar and not use_mathtext),
            scalar_bar_args=(scalar_bar_args
                             if args.show_scalar_bar and not use_mathtext else None),
        )
    else:
        if not add_layered_volume(
            plotter, grid, args.scalar, clim, cmap, opacity, args.layers,
            args.smooth_iterations,
            (scalar_bar_args
             if args.show_scalar_bar and not use_mathtext else None),
        ):
            print(f"Warning: no shells fell inside clim {tuple(clim)}")
        light = pv.Light(position=np.asarray(grid.center) + np.array([14.0, 20.0, 15.0]),
                         color="white")
        plotter.add_light(light)

    apply_font_to_scalar_bars(plotter, font_family, font_file)

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
    print(f"Saving {args.output}")
    if use_mathtext:
        view_scale = float(plotter.camera.parallel_scale)
        image = plotter.screenshot(return_img=True, window_size=list(window_size))
        plotter.close()
        annotate3d.compose(
            image, args.output, font=args.font,
            time_text=annotate3d.math_time_label(time_ms, args.t0 is not None),
            cmap=cmap if args.show_scalar_bar else None,
            clim=clim if args.show_scalar_bar else None,
            label=str(style["label"]) if args.show_scalar_bar else None,
            scalebar_length=bar_length,
            parallel_scale=view_scale if bar_length else None,
            background=args.background,
        )
    else:
        plotter.show(screenshot=args.output, window_size=list(window_size),
                     auto_close=True)


if __name__ == "__main__":
    main()
