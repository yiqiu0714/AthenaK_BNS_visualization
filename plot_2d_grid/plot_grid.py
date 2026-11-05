#!/usr/bin/env python3
"""Build a model / diagnostic / time comparison grid from AthenaK 2D slices.

Each of --rows and --cols is bound to one of model, diagnostic or time; the
remaining dimension is swept, writing one figure per value.  That covers the
three common figures with one mechanism:

    --rows model      --cols diagnostic   # models x diagnostics
    --rows diagnostic --cols time         # one model over time
    --rows model      --cols time         # one diagnostic, models over time

Examples
--------
python plot_grid.py \\
    --model /lus/.../92_mb32_n128"|"SFHo n128 \\
    --model /lus/.../46_mb32_n256"|"SFHo n256 \\
    --diagnostics density temperature s00 --frames 150 \\
    --rows model --cols diagnostic -o grid.png
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import BNS_all

import diagnostics as diags
import griddata
import panelgrid


def _parse_cmap(entries):
    """--cmap bnorm=magma -> {'bnorm': 'magma'}"""
    out = {}
    for entry in entries or []:
        name, sep, value = entry.partition("=")
        if not sep or not value:
            raise ValueError(f"--cmap needs NAME=CMAP (got '{entry}')")
        diags.get(name)
        out[name] = value
    return out


def _parse_clim(entries):
    """--clim density=1e8,1e15 -> {'density': (1e8, 1e15)}"""
    out = {}
    for entry in entries or []:
        name, sep, values = entry.partition("=")
        if not sep:
            raise ValueError(f"--clim needs NAME=lo,hi (got '{entry}')")
        parts = values.split(",")
        if len(parts) != 2:
            raise ValueError(f"--clim {name} needs exactly lo,hi (got '{values}')")
        diags.get(name)  # validate the name up front
        out[name] = (float(parts[0]), float(parts[1]))
    return out


def _parse_limit(text):
    if not text:
        return None
    parts = [float(v) for v in text.replace(":", ",").split(",")]
    if len(parts) == 1:
        return (-abs(parts[0]), abs(parts[0]))
    if len(parts) == 2:
        return (parts[0], parts[1])
    raise ValueError(f"limit must be L or lo,hi (got '{text}')")


def _slug(dim, value, select_by="index"):
    if dim == "model":
        return value.label.replace(" ", "_").replace("/", "-")
    if dim == "diagnostic":
        return str(value)
    return f"t{float(value):.0f}" if select_by == "time" else f"f{int(value):05d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, metavar="PATH|LABEL",
                        help="Simulation directory, optionally 'path|label'; repeatable")
    parser.add_argument("--diagnostics", nargs="+", required=True,
                        help=f"Any of: {', '.join(sorted(diags.DIAGNOSTICS))}")
    parser.add_argument("--frames", nargs="+", type=int, default=None,
                        help="Frame indices into the time-sorted list; -1 is the last")
    parser.add_argument("--times", nargs="+", type=float, default=None,
                        help="Simulation times in M_sun; each model uses its nearest "
                             "frame. Use this to compare runs at the same physical "
                             "time, since a frame index is run-relative.")
    parser.add_argument("--rows", default="model", choices=panelgrid.DIMENSIONS)
    parser.add_argument("--cols", default="diagnostic", choices=panelgrid.DIMENSIONS)
    parser.add_argument("--colorbar", default="auto",
                        choices=["auto", "right", "bottom", "left", "top", "none"],
                        help="Shared-colorbar side; 'auto' follows the layout")
    parser.add_argument("--sharex", default="true", choices=["true", "false"],
                        help="Interior panels drop BOTH the x tick labels and the "
                             "x axis label; only the bottom row keeps them")
    parser.add_argument("--sharey", default="true", choices=["true", "false"],
                        help="Interior panels drop BOTH the y tick labels and the "
                             "y axis label; only the left column keeps them")
    parser.add_argument("--clim", action="append", metavar="NAME=lo,hi",
                        help="Override a diagnostic's colour limits; repeatable")
    parser.add_argument("--cmap", action="append", metavar="NAME=CMAP",
                        help="Override a diagnostic's colormap; repeatable")
    parser.add_argument("--units", default="cgs", choices=["code", "cgs", "ngs"])
    parser.add_argument("--plane", default="xy", choices=["xy", "xz"])
    parser.add_argument("--jobname", default="bns")
    parser.add_argument("--crop", default=None,
                        help="Half-width L, or lo,hi, applied to both axes")
    parser.add_argument("--crop-units", default="display",
                        choices=["display", "code"],
                        help="Units for --crop/--xlim/--ylim: 'display' matches "
                             "the drawn axes (km when --units cgs, the default), "
                             "'code' is always M_sun as in BNS_all.py")
    parser.add_argument("--xlim", default=None)
    parser.add_argument("--ylim", default=None)
    parser.add_argument("--grid", default="true", choices=["true", "false"],
                        help="Overlay AMR block boundaries using BNS_all.py's "
                             "draw_refinement_grid (default true)")
    parser.add_argument("--spacing", type=float, default=0.02,
                        help="Vertical gap as a fraction of panel height; the "
                             "horizontal gap is twice that physical distance "
                             "(default 0.02)")
    parser.add_argument("--panel-size", default="3.6,3.6", metavar="W,H",
                        help="Per-panel size in inches")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("-o", "--output", default="grid.png")
    args = parser.parse_args()

    sweep_dim = panelgrid.resolve_axes(args.rows, args.cols)

    if args.frames is not None and args.times is not None:
        parser.error("use --frames or --times, not both")
    select_by = "time" if args.times is not None else "index"
    requests = args.times if select_by == "time" else (args.frames or [-1])

    models = [griddata.Model.parse(spec, args.jobname) for spec in args.model]
    for name in args.diagnostics:
        diags.get(name)

    crop = _parse_limit(args.crop)
    xlim = _parse_limit(args.xlim) or crop
    ylim = _parse_limit(args.ylim) or crop
    # Limits travel in code units; convert if the user typed what the axes show.
    if args.crop_units == "display":
        coord_scale, _ = BNS_all.spatial_units(args.units)
        xlim = tuple(v / coord_scale for v in xlim) if xlim else None
        ylim = tuple(v / coord_scale for v in ylim) if ylim else None
    panel_w, panel_h = (float(v) for v in args.panel_size.split(","))

    print(f"Models      : {', '.join(m.label for m in models)}")
    print(f"Diagnostics : {', '.join(args.diagnostics)}")
    print(f"{'Times' if select_by == 'time' else 'Frames'}      : {requests}"
          f"{'  (nearest frame per model)' if select_by == 'time' else ''}")
    print(f"Layout      : rows={args.rows}  cols={args.cols}  sweep={sweep_dim}")

    index, table = griddata.build_panels(models, args.diagnostics, requests,
                                         args.plane, select_by)

    # With time selection, say which frame each model actually landed on.
    if select_by == "time":
        for model in models:
            for want in requests:
                panel = table.get((model.path, args.diagnostics[0], want))
                if panel is not None and panel.time is not None:
                    print(f"  {model.label}: t={want:g} -> frame {panel.index} "
                          f"(t={panel.time:g})")

    sweep_values = {"model": models, "diagnostic": list(args.diagnostics),
                    "time": list(requests)}[sweep_dim]

    root, ext = os.path.splitext(args.output)
    ext = ext or ".png"
    written = []
    for value in sweep_values:
        # A swept dimension means several figures, so the value goes in the name.
        out = (f"{root}{ext}" if len(sweep_values) == 1
               else f"{root}_{sweep_dim}_{_slug(sweep_dim, value, select_by)}{ext}")
        path = panelgrid.render_figure(
            models=models, diagnostic_names=args.diagnostics, frames=requests,
            table=table, rows=args.rows, cols=args.cols,
            sweep_value=value, sweep_dim=sweep_dim,
            units=args.units, plane=args.plane, xlim=xlim, ylim=ylim,
            colorbar=args.colorbar,
            sharex=args.sharex == "true", sharey=args.sharey == "true",
            clim=_parse_clim(args.clim), cmap=_parse_cmap(args.cmap), select_by=select_by,
            grid=args.grid == "true",
            spacing=args.spacing,
            panel_w=panel_w, panel_h=panel_h, dpi=args.dpi, output=out,
        )
        written.append(path)
        print(f"Saved {path}")

    if len(written) > 1:
        print(f"{len(written)} figures written (one per {sweep_dim}).")


if __name__ == "__main__":
    main()
