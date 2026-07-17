"""Central Matplotlib house style for the BNS visualization tools.

Single source of truth for look-and-feel, modelled on the ``thc_plot`` style
(serif font, inward major+minor ticks on all four sides, frameless legend).
Edit the values here once instead of touching every plotting script.

Two kinds of settings:

* **Fixed style items** — constant everywhere (font family, tick direction,
  legend framing, math font, ...).
* **Size items** — fonts, tick lengths, line/marker widths. These are
  calibrated for a SINGLE panel and kept **constant per panel**: every subplot
  looks like a standalone 1x1 plot. Only the *figure size* grows with the
  subplot grid, via :func:`figsize` (``PANEL_W*ncols`` by ``PANEL_H*nrows``).

Typical use::

    import plot_style
    plot_style.use()                       # single panel, house style
    fig, ax = plt.subplots(figsize=plot_style.figsize())

    fig, axes = plot_style.subplots(2, 3)  # 2x3 grid, canvas scaled, style set

To switch to real LaTeX text, call ``plot_style.use(usetex=True)`` (needs a
working latex/dvipng toolchain; off by default because it is slow and fragile
when rendering many movie frames in parallel).
"""

import matplotlib as mpl

# Reference size of ONE panel, in inches.  Figure size for an nrows x ncols
# grid is (PANEL_W*ncols, PANEL_H*nrows) — see figsize().
PANEL_W = 6.0
PANEL_H = 4.0

# ---------------------------------------------------------------------------
# House style.  Size items (fonts, tick/line/marker sizes) are absolute and
# apply per panel; relative sizes ('large'/'medium') track font.size so they
# stay proportional if you change it.  Modelled on thc_plot.
# ---------------------------------------------------------------------------
_RCPARAMS = {
    # --- fixed style items ---
    "font.family": "serif",
    "text.usetex": False,          # use(usetex=True) to switch on real LaTeX
    "mathtext.fontset": "cm",      # serif math markup even with usetex off
    "axes.labelsize": "large",
    "axes.titlesize": "large",
    "xtick.minor.visible": True,
    "xtick.direction": "in",
    "xtick.top": True,
    "ytick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.right": True,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
    "legend.frameon": False,
    # --- size items (per panel, constant) ---
    "font.size": 16.0,
    "legend.fontsize": "medium",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "lines.linewidth": 1.6,
    "lines.markersize": 8,
    "xtick.major.size": 10,
    "xtick.minor.size": 6,
    "ytick.major.size": 10,
    "ytick.minor.size": 6,
    "figure.figsize": (PANEL_W, PANEL_H),
}


def figsize(nrows=1, ncols=1, panel_w=PANEL_W, panel_h=PANEL_H):
    """Figure size (inches) for an ``nrows`` x ``ncols`` grid.

    Per-panel size is held constant, so the canvas scales with the grid.
    Pass ``panel_w``/``panel_h`` to override the per-panel size (e.g. for
    wide time-series or tall stacked slices).
    """
    return (panel_w * ncols, panel_h * nrows)


def use(nrows=1, ncols=1, *, usetex=None, panel_w=PANEL_W, panel_h=PANEL_H,
        **overrides):
    """Apply the house style and set ``figure.figsize`` for the given grid.

    Size items are identical in every panel; only the figure size scales with
    ``nrows``/``ncols``.  ``usetex`` toggles real LaTeX; any extra keyword is
    forwarded to ``rcParams`` as an override (e.g. ``**{'font.size': 14}``).
    """
    rc = dict(_RCPARAMS)
    rc["figure.figsize"] = figsize(nrows, ncols, panel_w, panel_h)
    if usetex is not None:
        rc["text.usetex"] = usetex
    rc.update(overrides)
    mpl.rcParams.update(rc)


def subplots(nrows=1, ncols=1, *, usetex=None, panel_w=PANEL_W, panel_h=PANEL_H,
             **kwargs):
    """``plt.subplots`` with the house style applied and a grid-scaled figsize.

    Any ``plt.subplots`` keyword works; ``figsize`` defaults to
    :func:`figsize` but can be overridden.
    """
    import matplotlib.pyplot as plt
    use(nrows, ncols, usetex=usetex, panel_w=panel_w, panel_h=panel_h)
    kwargs.setdefault("figsize", figsize(nrows, ncols, panel_w, panel_h))
    return plt.subplots(nrows, ncols, **kwargs)
