#!/usr/bin/env python3
"""Density shells + magnetic field lines for a BNS snapshot (publication figure).

Differs from ``plot_isosurfaces.py --streamlines B`` in three ways that matter
for a clean figure:

* B is zeroed below ``--b-mask-rho`` before tracing, so field lines end at the
  stellar surface instead of wandering through the noisy low-density atmosphere.
* Seeds sit on a small sphere inside *each* star (two density maxima found
  automatically), not on a disc, so both stars get an even bundle of lines.
* Lines are coloured by log10|B| with their own colour bar, on translucent
  depth-peeled density shells, and annotated with Matplotlib mathtext
  (clock, two colour bars, km ruler) on an orthographic view.

Example (inspiral, cgs):
    python plot_density_fieldlines.py -i bns.prim3D.00006.bin --crop 30 \
        --resolution 256 --t0 4247.5 --font serif -o dens_B.png
"""

import argparse
import os

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

import annotate3d
from data3d import load_3d_data
from style3d import (
    convert_fields,
    make_coordinate_axes,
    parse_crop,
    parse_csv_floats,
    relative_time_ms,
    set_pyvista_camera,
)

MSUN_KM = 1.476625  # G M_sun / c^2 [km]: code length -> km

# Cool ramp that stays visible on black and separates from the warm density map.
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "fieldlines", ["#2c5aa0", "#3fa7d6", "#9fe6ff", "#ffffff"])


def star_centers(dens, axes, min_sep):
    """Global density maximum plus the maximum at least `min_sep` away from it."""
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    i1 = np.unravel_index(np.nanargmax(dens), dens.shape)
    c1 = np.array([X[i1], Y[i1], Z[i1]])
    far = (X - c1[0]) ** 2 + (Y - c1[1]) ** 2 + (Z - c1[2]) ** 2 > min_sep ** 2
    masked = np.where(far, dens, -np.inf)
    i2 = np.unravel_index(np.nanargmax(masked), dens.shape)
    c2 = np.array([X[i2], Y[i2], Z[i2]])
    # A merged remnant has one maximum; only keep the second if it is a real star.
    if dens[i2] < 0.3 * dens[i1]:
        return [c1]
    return [c1, c2]


def _cut_at_return(pts, tol, min_arc):
    """Truncate a polyline where it first comes back within `tol` of its start."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    back = np.nonzero((arc > min_arc) & (np.linalg.norm(pts - pts[0], axis=1) < tol))[0]
    if len(back):
        return pts[: back[0] + 1], True
    return pts, False


def trace_once(grid, seeds, step, max_length, terminal_speed, close_tol):
    """Trace B from each seed so that every closed loop is drawn exactly once.

    With integration_direction="both" a closed field line is followed round and
    round until max_length, so a loop that closes after a few km is drawn ~20x on
    top of itself and renders as a thick bright ring.  Here forward and backward
    are traced separately, each is cut at its first return to the seed, and the
    backward half is dropped when the forward half already closed the loop.
    """
    out, closed_fwd = {}, set()
    for direction in ("forward", "backward"):
        sl = grid.streamlines_from_source(
            seeds, vectors="B", integration_direction=direction,
            max_step_length=step, max_length=max_length, terminal_speed=terminal_speed)
        for i in range(sl.n_cells):
            sid = int(sl.cell_data["SeedIds"][i])
            pts = sl.get_cell(i).points
            if len(pts) < 2 or (direction == "backward" and sid in closed_fwd):
                continue
            pts, closed = _cut_at_return(pts, close_tol, 4.0 * close_tol)
            if direction == "forward" and closed:
                closed_fwd.add(sid)
            out.setdefault(sid, []).append(pts if direction == "forward" else pts[::-1])
    polys = []
    for parts in out.values():
        pts = parts[0] if len(parts) == 1 else np.vstack([parts[1][:-1], parts[0]])
        polys.append(pv.PolyData(pts, lines=np.r_[len(pts), np.arange(len(pts))]))
    merged = polys[0].merge(polys[1:]) if len(polys) > 1 else polys[0]
    merged["B"] = merged.sample(grid)["B"]
    return merged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, help="AthenaK prim3D .bin or HDF5 box")
    p.add_argument("-o", "--output", default="dens_fieldlines.png")
    p.add_argument("--crop", default="30", help="Half-width L or xmin,xmax,... [code]")
    p.add_argument("--resolution", type=int, default=256)
    # Defaults follow incite-bns-zoom-2025/scripts/plots/plot3D_Bfield.py (E. Gutierrez):
    # log10 rho_code = -7,-6,-4,-3  ->  +17.79 in cgs
    p.add_argument("--levels", default="10.79,11.79,13.79,14.79",
                   help="Explicit log10 rho [g/cm^3] shells; empty string -> --shells glow")
    p.add_argument("--opacity", default="0.3,0.53,0.77,1.0", help="Opacities matching --levels")
    p.add_argument("--shells", type=int, default=16,
                   help="Number of unlit translucent shells between --shell-range; "
                        "many faint shells read as a glowing volume")
    p.add_argument("--shell-range", default="10,14.8")
    p.add_argument("--shell-opacity", type=float, default=0.07)
    p.add_argument("--clim", default="10.79,14.79", help="log10 rho colour range")
    p.add_argument("--cmap", default="jet")
    p.add_argument("--b-mask-rho", type=float, default=None,
                   help="Trace B only where log10 rho [g/cm^3] exceeds this "
                        "(default: everywhere, so lines leave the stars)")
    p.add_argument("--line-style", choices=["lines", "tubes"], default="tubes")
    p.add_argument("--line-color", default="white",
                   help='Colour name, or "B" to colour by log10|B| (adds a colour bar)')
    p.add_argument("--line-width", type=float, default=1.3, help="[px] for --line-style lines")
    p.add_argument("--line-opacity", type=float, default=0.25)
    p.add_argument("--seed-shape", choices=["disc", "sphere"], default="disc",
                   help="disc: tilted disc through each star (plot3D_Bfield.py style)")
    p.add_argument("--disc-radius", type=float, default=5.0, help="[code]")
    p.add_argument("--disc-res", default="5,8", help="radial,circumferential seeds")
    p.add_argument("--no-loop-cut", action="store_true",
                   help="Trace with integration_direction=both (closed loops retraced)")
    p.add_argument("--close-tol", type=float, default=0.8,
                   help="Cut a line when it returns this close to its seed [code]")
    p.add_argument("--max-length", type=float, default=3.0,
                   help="Max field-line length per direction, in units of the box width; "
                        "lower it once the field winds up near merger")
    p.add_argument("--depth-peeling", action="store_true",
                   help="Exact transparency sorting (hides lines behind opaque cores)")
    p.add_argument("--b-clim", default=None, help="log10|B| [G] colour range (default: auto)")
    p.add_argument("--seed-radius", type=float, default=4.0, help="Seed sphere radius [code]")
    p.add_argument("--seed-res", type=int, default=8, help="Seed sphere theta/phi resolution")
    p.add_argument("--min-sep", type=float, default=8.0,
                   help="Minimum separation between the two density maxima [code]")
    p.add_argument("--tube-radius", type=float, default=0.03, help="[code]")
    p.add_argument("--t0", type=float, default=None, help="Merger time [code] for the clock")
    p.add_argument("--font", default="serif", help="sans | serif | cm")
    p.add_argument("--text-scale", type=float, default=1.35,
                   help="Annotation size multiplier (1.5 reads at ~6-7 pt in a 1.8 in panel)")
    p.add_argument("--scalebar-km", type=float, default=20.0, help="0 = no ruler")
    p.add_argument("--no-clock", action="store_true", help="Omit the time label")
    p.add_argument("--no-colorbar", action="store_true",
                   help="Omit the colour bar(s), e.g. for frames combined into one strip")
    p.add_argument("--window-size", default="1600,1600")
    p.add_argument("--camera-elevation", type=float, default=20.0)
    p.add_argument("--camera-azimuth", type=float, default=None,
                   help="Default: perpendicular to the binary separation + --azimuth-offset")
    p.add_argument("--azimuth-offset", type=float, default=12.0)
    p.add_argument("--star-radius", type=float, default=8.5,
                   help="Approximate stellar radius [code], sets the default zoom")
    p.add_argument("--parallel-scale", type=float, default=None,
                   help="Half-height of the orthographic view [code]")
    p.add_argument("--background", default="black")
    args = p.parse_args()

    ds = load_3d_data(args.input, variables=("dens", "bcc1", "bcc2", "bcc3"),
                      resolution=args.resolution, crop=parse_crop(args.crop))
    convert_fields(ds.fields, "cgs")  # dens -> g/cm^3, bcc -> G
    axes = make_coordinate_axes(ds.shape, ds.domain_size, ds.coordinates)
    grid = pv.RectilinearGrid(*axes)

    log_rho = np.log10(np.maximum(ds.fields["dens"], 1e-30))
    grid["log_rho"] = log_rho.flatten(order="F")
    inside = (log_rho >= args.b_mask_rho) if args.b_mask_rho is not None else np.ones_like(log_rho, bool)
    B = np.stack([np.where(inside, ds.fields[c], 0.0).flatten(order="F")
                  for c in ("bcc1", "bcc2", "bcc3")], axis=-1)
    grid["B"] = B
    bmag = np.linalg.norm(B, axis=1)
    grid["log_B"] = np.log10(np.maximum(bmag, 1e-30))

    centers = star_centers(ds.fields["dens"], axes, args.min_sep)
    print("t =", ds.time, " star centres [code]:", [np.round(c, 2).tolist() for c in centers])

    L = max(grid.bounds[1] - grid.bounds[0], grid.bounds[3] - grid.bounds[2])
    lines = []
    r_res, c_res = (int(v) for v in parse_csv_floats(args.disc_res))
    for n, c in enumerate(centers):
        if args.seed_shape == "disc":
            normal = (1.0, 0.0, 1.0) if n == 0 else (0.0, 1.0, 1.0)
            seeds = pv.Disc(center=c, normal=normal, outer=args.disc_radius,
                            r_res=r_res, c_res=c_res)
        else:
            seeds = pv.Sphere(radius=args.seed_radius, center=c,
                              theta_resolution=args.seed_res, phi_resolution=args.seed_res)
        if args.no_loop_cut:   # plain "both" tracing, as plot3D_Bfield.py / v01-v11
            sl = grid.streamlines_from_source(
                seeds, vectors="B", integration_direction="both",
                max_step_length=0.005 * L, max_length=args.max_length * L,
                terminal_speed=1e-12 * bmag.max())
        else:
            sl = trace_once(grid, seeds, step=0.005 * L, max_length=args.max_length * L,
                            terminal_speed=1e-12 * bmag.max(), close_tol=args.close_tol)
        if sl.n_points:
            lines.append(sl)
    lines = lines[0].merge(lines[1:]) if len(lines) > 1 else lines[0]
    lines["log_B"] = np.log10(np.maximum(np.linalg.norm(lines["B"], axis=1), 1e-30))
    color_by_b = args.line_color == "B"
    b_clim = None
    if color_by_b:
        b_clim = parse_csv_floats(args.b_clim) or tuple(
            np.percentile(lines["log_B"][np.isfinite(lines["log_B"])], [5, 99.5]).round(1))
    print("field lines:", lines.n_cells, " log10|B| clim:", b_clim)

    clim = parse_csv_floats(args.clim)
    if args.levels:  # '' -> glow shells
        levels = parse_csv_floats(args.levels)
        opac = parse_csv_floats(args.opacity) or [0.25] * len(levels)
        lit = True
    else:
        lo, hi = parse_csv_floats(args.shell_range)
        levels = np.linspace(lo, hi, args.shells)
        opac = [args.shell_opacity] * len(levels)
        lit = False  # emissive look: colour comes from the map, not the lights
    win = tuple(int(v) for v in parse_csv_floats(args.window_size))

    pl = pv.Plotter(off_screen=True, window_size=list(win))
    pl.set_background(args.background)
    if args.depth_peeling:
        pl.enable_depth_peeling(number_of_peels=12)
    pl.enable_anti_aliasing("ssaa")

    # innermost first: without depth peeling VTK blends actors in add order, so the
    # outer (blue) shells must come last to tint the star as a translucent envelope
    for lev, op in sorted(zip(levels, opac), reverse=True):
        shell = grid.contour([lev], scalars="log_rho")
        if shell.n_points == 0:
            continue
        shell = shell.smooth_taubin(n_iter=20, pass_band=0.08, normalize_coordinates=True)
        shell["log_rho"] = np.full(shell.n_points, lev)
        pl.add_mesh(shell, scalars="log_rho", cmap=args.cmap, clim=clim,
                    opacity=op, smooth_shading=True, show_scalar_bar=False,
                    **(dict(pbr=True, specular_power=15) if lit else dict(lighting=False)))

    # lines after the shells (as plot3D_Bfield.py does): without depth peeling VTK
    # blends these translucent lines over the stars, so the interior dipole shows
    mesh = lines.tube(radius=args.tube_radius, n_sides=10) if args.line_style == "tubes" else lines
    style = dict(line_width=args.line_width, render_lines_as_tubes=False) \
        if args.line_style == "lines" else dict(smooth_shading=True, specular=0.4, specular_power=20)
    if color_by_b:
        pl.add_mesh(mesh, scalars="log_B", cmap=FIELD_CMAP, clim=b_clim,
                    opacity=args.line_opacity, show_scalar_bar=False, **style)
    else:
        pl.add_mesh(mesh, color=args.line_color, opacity=args.line_opacity,
                    show_scalar_bar=False, **style)
    pl.add_light(pv.Light(position=np.asarray(grid.center) + np.array([14.0, 20.0, 15.0]),
                          color="white"))
    azimuth = args.camera_azimuth
    if azimuth is None:
        if len(centers) == 2:
            # camera sits at (sin az, cos az); put it perpendicular to the separation
            sep = centers[1] - centers[0]
            azimuth = np.degrees(np.arctan2(sep[1], -sep[0])) + args.azimuth_offset
        else:
            azimuth = 20.0
    set_pyvista_camera(pl, grid, elevation=args.camera_elevation,
                       azimuth=azimuth, orthographic=True,
                       parallel_scale=args.parallel_scale or 1.15 * (
                           0.5 * np.linalg.norm(centers[-1] - centers[0]) + args.star_radius))
    scale = float(pl.camera.parallel_scale)
    img = pl.screenshot(return_img=True)
    pl.close()

    compose(img, args.output, font=args.font,
            time_ms=None if args.no_clock else relative_time_ms(ds.time, args.t0),
            relative=args.t0 is not None,
            rho_cmap=None if args.no_colorbar else args.cmap, rho_clim=clim,
            b_clim=None if args.no_colorbar else b_clim,
            scalebar_km=args.scalebar_km, parallel_scale=scale,
            background=args.background, text_scale=args.text_scale)
    print("wrote", args.output)


def compose(image, output, *, font, time_ms, relative, rho_cmap, rho_clim, b_clim,
            scalebar_km, parallel_scale, background, text_scale=1.0, color="white", dpi=200):
    h, w = image.shape[:2]

    def pt(frac):  # font size as a fraction of frame height
        return text_scale * frac * h * 72.0 / dpi

    with plt.rc_context(annotate3d.mathtext_rc(font)):
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        fig.patch.set_facecolor(background)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(image, interpolation="lanczos", aspect="auto")
        ax.set_axis_off()

        tlabel = None if time_ms is None else (
            rf"$t-t_\mathrm{{merger}} = {time_ms:.1f}\ \mathrm{{ms}}$" if relative
            else annotate3d.math_time_label(time_ms, relative))
        box = dict(boxstyle="round,pad=0.25", fc=background, ec="none", alpha=0.7)
        if time_ms is not None:
            fig.text(0.03, 0.965, tlabel, bbox=box,
                     color=color, ha="left", va="top", fontsize=pt(0.034))

        bars = [] if rho_cmap is None else [(rho_cmap, rho_clim, r"$\log_{10}\rho\ [\mathrm{g\,cm^{-3}}]$", 0.04)]
        if b_clim is not None:
            bars.append((FIELD_CMAP, b_clim, r"$\log_{10}|B|\ [\mathrm{G}]$", 0.56))
        elif bars:
            bars[0] = bars[0][:3] + (0.30,)  # single bar: centre it
        for cmap, clim, label, x0 in bars:
            cax = fig.add_axes([x0, 0.075, 0.40, 0.022])
            sm = ScalarMappable(norm=Normalize(*clim), cmap=cmap)
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cb.ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
            cb.ax.tick_params(colors=color, labelsize=pt(0.026), length=3, width=0.8)
            cb.outline.set_edgecolor(color)
            cb.outline.set_linewidth(0.8)
            fig.text(x0 + 0.20, 0.105, label, color=color, ha="center", va="bottom",
                     fontsize=pt(0.030))

        # orthographic: a bar of length L spans L / (2 * scale * aspect) of the width
        if scalebar_km <= 0:
            fig.savefig(output, dpi=dpi, facecolor=background)
            plt.close(fig)
            return
        frac = (scalebar_km / MSUN_KM) / (2.0 * parallel_scale * w / h)
        x1, y = 0.95, 0.915
        fig.add_artist(plt.Line2D([x1 - frac, x1], [y, y], color=color, lw=2.2,
                                  transform=fig.transFigure))
        fig.text(x1 - frac / 2, y + 0.012, rf"${scalebar_km:g}\ \mathrm{{km}}$", bbox=box,
                 color=color, ha="center", va="bottom", fontsize=pt(0.030))

        fig.savefig(output, dpi=dpi, facecolor=background)
        root, ext = os.path.splitext(output)
        if ext.lower() == ".png":
            fig.savefig(root + ".pdf", dpi=dpi, facecolor=background)
        plt.close(fig)


if __name__ == "__main__":
    main()
