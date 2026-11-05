# 2D Panel-Grid Figures

Builds **one** figure holding many 2D slice panels, so several models and
several diagnostics can be compared side by side instead of flipping between
PNGs from `BNS_all.sh`.

Driven by `BNS_grid.sh` in the repo root; the modules here do the work.

## The one idea

A figure has two axes, and there are three things worth spending them on —
**model**, **diagnostic** and **time**. So `--rows` and `--cols` each take one
dimension, and whatever is left over is *swept*: one figure per value.

| Goal | `--rows` | `--cols` | leftover |
|---|---|---|---|
| models × diagnostics | `model` | `diagnostic` | one frame |
| one model over time | `diagnostic` | `time` | one model |
| one diagnostic, models over time | `model` | `time` | one diagnostic |

A dimension of length 1 collapses to a single row or column — no special case.
If the leftover dimension holds several values you get several figures, named
`<out>_<dim>_<value>.png`, which `make_movies.sh` can stitch directly.

## Files

- `diagnostics.py` — registry: which dump prefix, variable, colormap, colour
  limits and derived-variable hook belong to each diagnostic. Colour limits and
  labels come from `athenak_units.plot_params`, so a panel here and a frame from
  `BNS_all.py` agree on the scale.
- `griddata.py` — finds `output-XXXX` segments and resolves frame indices to
  files. Reuses `BNS_all.find_output_dirs` / `_collect_frames_by_time`.
- `panelgrid.py` — the layout engine: axes grid, shared colorbars, headers.
- `plot_grid.py` — CLI entry point.

Nothing in `BNS_all.py`, `athenak_units.py` or `plot_style.py` is modified; this
module only imports from them.

## Diagnostics

`density`, `temperature`, `s00`, `bnorm`, `z4c`, `rad_E`, `rad_N`, `rad_N2`,
`rad_abs`, `rad_abs0`, `rad_eta0`, `rad_eta1` — the same names as the `SECTIONS`
entries in `BNS_all.sh`.

## Shared colour scales

Each diagnostic gets **one** norm, fixed from the registry rather than computed
per panel. That is what makes a single shared colorbar correct rather than
merely convenient: every panel of a diagnostic is directly comparable, and two
figures made on different days are comparable too.

Override per run:

```bash
--clim density=1e10,1e15 --clim temperature=0,50
```

## Where the colorbar goes

The colorbar belongs to whichever axis carries the diagnostic. This is geometry,
not taste:

| diagnostic varies along | one colorbar per | valid placement | default |
|---|---|---|---|
| columns | column | `bottom`, `top` | `top` |
| rows | row | `right`, `left` | `right` |
| neither (single diagnostic) | the figure | any | `right` |

A horizontal colorbar defaults to **`top`**: below the panels it lands between
them and the x tick labels and pushes the two apart.

Each bar's axes are built from its panels' own `get_position()`, so it starts
and ends exactly where the column (or row) does and every bar is the same
length — the same trick `BNS_all.py` uses for its stacked xz/xy frame.

`--colorbar auto` (the default) picks from that table. Asking for `right` when
the diagnostic varies along columns would wedge a bar between every pair of
columns, so it is refused with a message naming the conflict instead of
rendering something unreadable. `--colorbar none` omits them.

When the colorbar already names the diagnostic, the matching row/column header
is dropped rather than repeating it.

## Picking when: `--frames` vs `--times`

`--frames N` is a position in each model's time-sorted list. Exact, but
**run-relative**: runs with different output cadences put different times at the
same index.

`--times T` gives each model its nearest frame to simulation time `T`, which is
what makes a physical comparison honest. The resolved frame is printed per
model, so the match is auditable:

```
  n64:  t=4000 -> frame 799 (t=4000)
  n128: t=4000 -> frame 794 (t=4000)
  n256: t=4000 -> frame 796 (t=4000)
```

Use one or the other, not both.

## Examples

Models × diagnostics — the main comparison figure:

```bash
python plot_grid.py \
  --model /lus/.../92_mb32_n128"|"SFHo\ n128 \
  --model /lus/.../46_mb32_n256"|"SFHo\ n256 \
  --diagnostics density temperature s00 \
  --frames 1000 --rows model --cols diagnostic \
  --units cgs --crop 60 -o grid.png
```

One model through the merger — a single row:

```bash
python plot_grid.py \
  --model /lus/.../92_mb32_n128"|"SFHo\ n128 \
  --diagnostics density --frames 600 900 1200 1500 \
  --rows diagnostic --cols time --units cgs --crop 60 -o strip.png
```

Three runs at the same physical time — a resolution study:

```bash
python plot_grid.py \
  --model /lus/.../184_mb32_n64"|"n64 \
  --model /lus/.../92_mb32_n128"|"n128 \
  --model /lus/.../46_mb32_n256"|"n256 \
  --diagnostics bnorm --times 4000 \
  --rows diagnostic --cols model --plane xy \
  --units cgs --crop 60 -o bfield.png
```

One diagnostic, models over time:

```bash
python plot_grid.py \
  --model /lus/.../n128"|"n128 --model /lus/.../n256"|"n256 \
  --diagnostics s00 --frames 400 800 1200 \
  --rows model --cols time --units cgs --crop 60 -o ye.png
```

## Notes

- **Frame indices are per-run.** Index N is a position in each model's
  time-sorted, restart-deduplicated list, so it is *not* the same physical time
  in two runs with different output cadences — use `--times` when you want a
  physical match. With `--frames`, the panels print their own times so the
  difference stays visible, and when several models share a time column the
  header falls back to the frame index, the thing that really is common.
  Negative indices count from the end, so `-1` is the last frame of each run.
- A frame index past the end of a run leaves a grey "no data" panel and a note
  on stderr, rather than failing the whole figure.
- **The geometry is explicit, not `constrained_layout`.** The layout engine
  rescales panels to fit whatever labels and colorbars appear, which makes both
  the requested panel size and the gap between panels advisory rather than
  actual. Margins are budgeted in inches, the figure is sized to hold them, and
  the grid gets an explicit `wspace`/`hspace` — so `--panel-size` and
  `--spacing` mean exactly what they say.
- `--spacing` is the vertical gap as a fraction of panel height (default
  `0.02`, `0` for touching panels). The horizontal gap is exactly twice that
  physical distance. For the default `--panel-size 3.6,3.6`, this gives gaps
  of `0.072` inches between rows and `0.144` inches between columns.
- The default panels are square because the default crop has equal x/y ranges
  and `imshow` preserves the physical aspect ratio. A wider panel slot would
  become invisible side padding and make the column gap look larger than the
  requested GridSpec spacing.
- **`--sharex/--sharey` hide the axis label *and* the tick labels** on interior
  panels — that is where the space saving comes from. The bottom row carries x
  and the left column carries y, each panel with its own label, so a
  three-column figure gets three x labels rather than one caption under the
  middle. Set either to `false` to label every panel; the gutters then widen
  automatically to make room for the repeated tick labels.
- The row identity gets its own strip in the left margin, so it never competes
  with the y axis label for the same slot.
- `--grid true` (the default) overlays the AMR block boundaries with
  `BNS_all.draw_refinement_grid`; `--grid false` turns that overlay off.
- `--crop` is a half-width in **M_sun** regardless of `--units`; with
  `--units cgs` the axes are then drawn in km, matching `BNS_all.py`.
