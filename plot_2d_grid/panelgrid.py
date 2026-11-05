#!/usr/bin/env python3
"""Layout engine for model/diagnostic/time comparison grids.

The figure has two axes and there are three dimensions to spend on them --
model, diagnostic and frame -- so each of ``rows`` and ``cols`` is bound to one
dimension and the leftover dimension is swept, producing one figure per value.
That single rule covers models x diagnostics, one model over time, and one
diagnostic across models and times, without special cases.

Colour scales are fixed per diagnostic (see diagnostics.Diagnostic.norm), which
is what makes a single shared colorbar correct rather than merely convenient.
"""

import os
import sys
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.environ.get("ATHPLOT_PATH", os.path.join(_REPO, "plot-tools"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import BNS_all  # noqa: E402  (spatial_units + overlay helpers)
import plot_style  # noqa: E402
from athplot.load_ath_bin import BinaryData  # noqa: E402

import diagnostics as diags  # noqa: E402

DIMENSIONS = ("model", "diagnostic", "time")

# Which colorbar placements make geometric sense for each grouping.  A per-column
# group cannot sit on the right without wedging a bar between every pair of
# columns, so that combination is refused rather than rendered badly.
_VALID_LOCATIONS = {
    "column": ("bottom", "top"),
    "row": ("right", "left"),
    "figure": ("right", "bottom", "left", "top"),
}
# A horizontal colorbar sits ABOVE its column: below, it lands between the panels
# and the x tick labels and pushes them apart.
_DEFAULT_LOCATION = {"column": "top", "row": "right", "figure": "right"}


def resolve_axes(rows: str, cols: str) -> str:
    """Validate the row/col binding and return the swept dimension."""
    for name, value in (("rows", rows), ("cols", cols)):
        if value not in DIMENSIONS:
            raise ValueError(
                f"--{name} must be one of {', '.join(DIMENSIONS)} (got '{value}')"
            )
    if rows == cols:
        raise ValueError(f"rows and cols cannot both be '{rows}'")
    return next(d for d in DIMENSIONS if d not in (rows, cols))


def colorbar_grouping(rows: str, cols: str) -> str:
    """Where the diagnostic varies decides what a shared colorbar can span."""
    if cols == "diagnostic":
        return "column"
    if rows == "diagnostic":
        return "row"
    return "figure"


def resolve_location(grouping: str, requested: str) -> str:
    """Pick/validate the colorbar side, explaining any conflict in full."""
    if requested in ("auto", "", None):
        return _DEFAULT_LOCATION[grouping]
    valid = _VALID_LOCATIONS[grouping]
    if requested not in valid:
        spans = {"column": "a column", "row": "a row", "figure": "the figure"}[grouping]
        raise ValueError(
            f"colorbar '{requested}' conflicts with the layout: the diagnostic "
            f"varies along {'columns' if grouping == 'column' else 'rows' if grouping == 'row' else 'neither axis'}, "
            f"so each colorbar spans {spans} and must sit on "
            f"{' or '.join(valid)}. Use one of those, or 'auto'."
        )
    return requested


def _values(dim: str, models, diagnostic_names, frames):
    return {"model": list(models), "diagnostic": list(diagnostic_names),
            "time": list(frames)}[dim]


def _identity(dim: str, value, panel, units: str, time_is_shared: bool = True,
              select_by: str = "index") -> str:
    """Human label for one row/column, given which dimension it encodes.

    A time header states a single time for the whole row/column, which is only
    true when one model supplies it.  With several models in the figure their
    cadences differ, so the header falls back to the frame index -- the thing
    that actually is common -- and each panel carries its own time instead.
    """
    if dim == "model":
        return value.label
    if dim == "diagnostic":
        return diags.get(value).label(units)
    if select_by == "time":
        # The request itself is the shared quantity here, by construction.
        return rf"$t = {float(value):.0f}~\mathrm{{M}}_\odot$"
    if not time_is_shared:
        return f"frame {value}"
    time = panel.time if panel is not None else None
    return (rf"$t = {time:.1f}~\mathrm{{M}}_\odot$" if time is not None
            else f"frame {value}")


# Layout budget, in inches.  Everything is sized explicitly rather than handed
# to constrained_layout: the engine rescales panels to fit whatever labels and
# colorbars appear, which makes the requested panel size and the gap between
# panels advisory rather than actual.  BNS_all.py takes the same approach for
# its stacked xz/xy frame -- explicit gridspec, colorbar axes built from
# get_position() -- and that is what keeps a colorbar exactly as long as the
# panels it serves.
_TICKS_W = 0.95      # y tick labels + y axis label
_TICKS_H = 0.80      # x tick labels + x axis label
_ROW_ID_W = 0.45     # rotated row identity in the left margin
_COL_TITLE_H = 0.45  # column title above the top row
_EDGE = 0.18         # breathing room at the figure edge

_CB_GAP = 0.12       # panel -> colorbar
_CB_THICK = 0.20     # the bar itself
_CB_TEXT = 0.72      # bar -> tick labels -> colorbar label
_CB_BAND = _CB_GAP + _CB_THICK + _CB_TEXT


def render_figure(
    *,
    models,
    diagnostic_names: Sequence[str],
    frames: Sequence[int],
    table: Dict,
    rows: str,
    cols: str,
    sweep_value,
    sweep_dim: str,
    units: str = "cgs",
    plane: str = "xy",
    xlim=None,
    ylim=None,
    colorbar: str = "auto",
    sharex: bool = True,
    sharey: bool = True,
    clim: Optional[Dict[str, Sequence[float]]] = None,
    cmap: Optional[Dict[str, str]] = None,
    select_by: str = "index",
    grid: bool = True,
    spacing: float = 0.02,
    panel_w: float = 3.6,
    panel_h: float = 3.6,
    dpi: int = 200,
    output: str = "grid.png",
) -> str:
    """Draw and save one grid figure.  Returns the output path."""
    clim = clim or {}
    cmap = cmap or {}
    coord_scale, coord_label = BNS_all.spatial_units(units)

    row_values = _values(rows, models, diagnostic_names, frames)
    col_values = _values(cols, models, diagnostic_names, frames)
    nrows, ncols = len(row_values), len(col_values)

    grouping = colorbar_grouping(rows, cols)
    location = None if colorbar == "none" else resolve_location(grouping, colorbar)

    time_is_shared = (select_by == "time"
                      or not ("model" in (rows, cols) and len(models) > 1))
    show_panel_time = ("time" not in (rows, cols)) or not time_is_shared

    # When the colorbar already names the diagnostic, repeating it as a row or
    # column header is pure noise, so it is dropped.
    show_col_header = not (cols == "diagnostic" and location is not None)
    show_row_id = not (rows == "diagnostic" and location is not None)

    # --- figure geometry ----------------------------------------------------
    # ``spacing`` names the vertical physical gap.  Columns use twice that
    # distance: close enough to read as one grid, but wide enough that adjacent
    # panels do not visually merge.  Convert the inch gaps to GridSpec's
    # dimensionless fractions so this remains exact for non-square panel sizes.
    y_gap = spacing * panel_h
    x_gap = 2.0 * y_gap
    wspace = x_gap / panel_w + (0.0 if sharey else _TICKS_W / panel_w)
    hspace = y_gap / panel_h + (0.0 if sharex else _TICKS_H / panel_h)

    axes_w = panel_w * ncols + panel_w * wspace * (ncols - 1)
    axes_h = panel_h * nrows + panel_h * hspace * (nrows - 1)

    left_in = _EDGE + _TICKS_W + (_ROW_ID_W if show_row_id else 0.0)
    right_in = _EDGE
    bottom_in = _EDGE + _TICKS_H
    top_in = _EDGE + (_COL_TITLE_H if show_col_header else 0.0)

    if location == "left":
        left_in += _CB_BAND
    elif location == "right":
        right_in += _CB_BAND
    elif location == "bottom":
        bottom_in += _CB_BAND
    elif location == "top":
        top_in += _CB_BAND

    fig_w = left_in + axes_w + right_in
    fig_h = bottom_in + axes_h + top_in

    plot_style.use(panel_w=panel_w, panel_h=panel_h)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), squeeze=False,
        sharex="all" if sharex else "none",
        sharey="all" if sharey else "none",
        gridspec_kw=dict(
            left=left_in / fig_w, right=1.0 - right_in / fig_w,
            bottom=bottom_in / fig_h, top=1.0 - top_in / fig_h,
            wspace=wspace, hspace=hspace,
        ),
    )

    def coords(i, j):
        """Map a cell to (model, diagnostic name, frame)."""
        pick = {rows: row_values[i], cols: col_values[j], sweep_dim: sweep_value}
        return pick["model"], pick["diagnostic"], pick["time"]

    mappables: Dict[str, object] = {}
    drawn_any = False

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i][j]
            model, diag_name, frame = coords(i, j)
            diag = diags.get(diag_name)
            panel = table.get((model.path, diag_name, frame))

            if panel is None or panel.path is None:
                # Leave the ticks alone: with sharex/sharey these are the same
                # Axis objects as every other panel, so clearing them here blanks
                # the tick labels for the whole grid.
                ax.set_facecolor("0.92")
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize="small", color="0.35")
                continue

            data = BinaryData(panel.path)
            var = diags.apply_to(data, diag)
            scale, _, _, _, _ = diag.params(units)
            image = data.plot_slice(
                var, ax=ax, cmap=cmap.get(diag_name, diag.cmap),
                norm=diag.norm(units, clim.get(diag_name)),
                rescale=scale, unit_length=coord_scale,
            )
            mappables[diag_name] = image
            drawn_any = True

            if grid:
                # BNS_all's defaults (alpha 0.4, lw 0.6) are tuned for a full
                # domain; zoomed in, that many block edges hides the field, so
                # the mesh is drawn as a hint rather than a subject.
                BNS_all.draw_refinement_grid(ax, data, alpha=0.22, linewidth=0.35,
                                             coord_scale=coord_scale)
            if xlim is not None:
                ax.set_xlim(*(v * coord_scale for v in xlim))
            if ylim is not None:
                ax.set_ylim(*(v * coord_scale for v in ylim))

            if show_panel_time and panel.time is not None:
                ax.text(0.04, 0.94, rf"$t = {panel.time:.1f}~\mathrm{{M}}_\odot$",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize="small", color="white")

    if not drawn_any:
        plt.close(fig)
        raise RuntimeError(
            "No panel had data; check the model paths, diagnostics and frame indices."
        )

    # --- axis labels --------------------------------------------------------
    # Sharing means an interior panel repeats neither the tick labels nor the
    # axis label; the edge panels each carry their own, so a three-column figure
    # gets three x labels rather than one caption under the middle.
    x_label = rf"${plane[0]}~[{coord_label}]$"
    y_label = rf"${plane[1]}~[{coord_label}]$"
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i][j]
            bottom_edge = (i == nrows - 1) or not sharex
            left_edge = (j == 0) or not sharey
            ax.set_xlabel(x_label if bottom_edge else "")
            ax.set_ylabel(y_label if left_edge else "")
            if not bottom_edge:
                plt.setp(ax.get_xticklabels(), visible=False)
            if not left_edge:
                plt.setp(ax.get_yticklabels(), visible=False)
    fig.align_ylabels(axes[:, 0].tolist())

    # --- headers ------------------------------------------------------------
    for j, value in enumerate(col_values):
        if show_col_header:
            model, dname, frame = coords(0, j)
            axes[0][j].set_title(
                _identity(cols, value, table.get((model.path, dname, frame)),
                          units, time_is_shared, select_by)
            )

    # The row identity gets its own strip in the left margin so it never has to
    # share the ylabel slot with the spatial label.
    if show_row_id:
        for i, value in enumerate(row_values):
            model, dname, frame = coords(i, 0)
            pos = axes[i][0].get_position()
            fig.text(
                (_EDGE + _ROW_ID_W * 0.5 + (_CB_BAND if location == "left" else 0.0))
                / fig_w,
                (pos.y0 + pos.y1) * 0.5,
                _identity(rows, value, table.get((model.path, dname, frame)),
                          units, time_is_shared, select_by),
                rotation=90, va="center", ha="center",
            )

    # --- colorbars ----------------------------------------------------------
    # Built from the panels' own positions, so each bar starts and ends exactly
    # where its column/row does and every bar is the same length.
    if location is not None:
        gap_x, thick_x = _CB_GAP / fig_w, _CB_THICK / fig_w
        gap_y, thick_y = _CB_GAP / fig_h, _CB_THICK / fig_h
        title_y = (_COL_TITLE_H / fig_h) if show_col_header else 0.0

        def add_bar(name, x0, y0, width, height, orientation):
            if name not in mappables:
                return
            cax = fig.add_axes([x0, y0, width, height])
            cb = fig.colorbar(mappables[name], cax=cax, orientation=orientation)
            if orientation == "horizontal" and location == "top":
                # Ticks and label above the bar, clear of the panels below.
                cax.xaxis.set_ticks_position("top")
                cax.xaxis.set_label_position("top")
            cb.set_label(diags.get(name).label(units))

        if grouping == "column":
            for j, value in enumerate(col_values):
                name = value if cols == "diagnostic" else diagnostic_names[0]
                top = axes[0][j].get_position()
                bot = axes[nrows - 1][j].get_position()
                if location == "top":
                    add_bar(name, top.x0, top.y1 + gap_y + title_y,
                            top.width, thick_y, "horizontal")
                else:
                    add_bar(name, bot.x0, bot.y0 - gap_y - thick_y
                            - _TICKS_H / fig_h, bot.width, thick_y, "horizontal")
        elif grouping == "row":
            for i, value in enumerate(row_values):
                name = value if rows == "diagnostic" else diagnostic_names[0]
                left = axes[i][0].get_position()
                right = axes[i][ncols - 1].get_position()
                if location == "right":
                    add_bar(name, right.x1 + gap_x, left.y0,
                            thick_x, left.height, "vertical")
                else:
                    add_bar(name, left.x0 - gap_x - thick_x - _TICKS_W / fig_w,
                            left.y0, thick_x, left.height, "vertical")
        else:
            name = next(iter(mappables))
            first, last = axes[0][0].get_position(), axes[nrows - 1][ncols - 1].get_position()
            if location == "right":
                add_bar(name, last.x1 + gap_x, last.y0,
                        thick_x, first.y1 - last.y0, "vertical")
            elif location == "left":
                add_bar(name, first.x0 - gap_x - thick_x - _TICKS_W / fig_w,
                        last.y0, thick_x, first.y1 - last.y0, "vertical")
            elif location == "top":
                add_bar(name, first.x0, first.y1 + gap_y + title_y,
                        last.x1 - first.x0, thick_y, "horizontal")
            else:
                add_bar(name, first.x0,
                        last.y0 - gap_y - thick_y - _TICKS_H / fig_h,
                        last.x1 - first.x0, thick_y, "horizontal")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output
