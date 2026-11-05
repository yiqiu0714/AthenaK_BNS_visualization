#!/usr/bin/env python3
"""Draw 3D-render annotations with Matplotlib so the maths is set properly.

VTK draws plain strings: it has no maths layout, so a colour-bar title can only
ever read "log10 rho [g cm^-3]".  The renderers therefore hand their finished
frame here as an RGB array, and every label -- clock, colour bar, ruler -- is
composited on top with Matplotlib mathtext, giving the same
:math:`\\log_{10}\\rho` the 2D plots already use.
"""

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# Mathtext font sets paired with the --font choices, so a frame annotated here
# matches one annotated by VTK as closely as the two renderers allow.
_MATHTEXT_RC = {
    "sans":    {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                "mathtext.fontset": "dejavusans"},
    "serif":   {"font.family": "serif", "font.serif": ["STIXGeneral"],
                "mathtext.fontset": "stix"},
    "cm":      {"font.family": "serif", "font.serif": ["cmr10"],
                "mathtext.fontset": "cm"},
    "arial":   {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                "mathtext.fontset": "dejavusans"},
    "times":   {"font.family": "serif", "font.serif": ["STIXGeneral"],
                "mathtext.fontset": "stix"},
    "courier": {"font.family": "monospace", "mathtext.fontset": "dejavusans"},
}


def mathtext_rc(font: str) -> dict:
    """rcParams for a --font choice; unknown names fall back to DejaVu Sans."""
    return dict(_MATHTEXT_RC.get((font or "sans").lower(), _MATHTEXT_RC["sans"]))


def math_time_label(time_ms: float, relative: bool) -> str:
    """Compact clock as mathtext: t_merger +X.XXX ms."""
    if not relative:
        return rf"$t = {time_ms:.3f}$ ms"
    return rf"$t_{{\mathrm{{merger}}}}\ {time_ms:+.3f}$ ms"


def compose(
    image: np.ndarray,
    output: str,
    *,
    font: str = "sans",
    dpi: int = 100,
    time_text: Optional[str] = None,
    cmap=None,
    clim: Optional[Sequence[float]] = None,
    label: Optional[str] = None,
    scalebar_length: Optional[float] = None,
    parallel_scale: Optional[float] = None,
    background: str = "black",
    color: str = "white",
) -> None:
    """Write `image` to `output` with mathtext annotations drawn over it.

    `scalebar_length` is in code-length units and is placed using the
    orthographic identity: with perspective off, `parallel_scale` is the
    half-height of the view, so a bar of length L covers L / (2 * scale *
    aspect) of the frame width.  That makes the ruler exact, not approximate.
    """
    height, width = image.shape[0], image.shape[1]

    # Size type as a fraction of frame height so a 600 px test frame and a
    # 1200 px production frame carry visually identical annotations.
    def pt(fraction: float) -> float:
        return fraction * height * 72.0 / dpi

    with plt.rc_context(mathtext_rc(font)):
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor(background)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(image, interpolation="nearest", aspect="auto")
        ax.set_axis_off()

        if time_text:
            fig.text(0.018, 0.972, time_text, color=color, ha="left", va="top",
                     fontsize=pt(0.030))

        if label is not None and cmap is not None and clim is not None:
            bar_box = [0.045, 0.38, 0.020, 0.38]
            cax = fig.add_axes(bar_box)
            mappable = ScalarMappable(
                norm=Normalize(vmin=float(clim[0]), vmax=float(clim[1])), cmap=cmap
            )
            bar = fig.colorbar(mappable, cax=cax)
            span = abs(float(clim[1]) - float(clim[0]))
            decimals = 2 if span < 1.0 else 1 if span < 10.0 else 0
            bar.set_ticks(np.linspace(float(clim[0]), float(clim[1]), 5))
            bar.ax.set_yticklabels(
                [f"{v:.{decimals}f}"
                 for v in np.linspace(float(clim[0]), float(clim[1]), 5)]
            )
            # Neither default placement works for a bar this close to the edge:
            # to the right it lies over the render, centred on top it runs off
            # the left edge once the label is as wide as "log10 rho [g cm^-3]".
            # Anchor it to the same left margin as the clock instead.
            fig.text(0.018, bar_box[1] + bar_box[3] + 0.018, label, color=color,
                     ha="left", va="bottom", fontsize=pt(0.028))
            bar.ax.tick_params(colors=color, labelsize=pt(0.022), length=3,
                               width=0.8)
            bar.outline.set_edgecolor(color)
            bar.outline.set_linewidth(0.8)

        if scalebar_length and parallel_scale:
            aspect = width / float(height)
            view_width = 2.0 * float(parallel_scale) * aspect
            fraction = float(scalebar_length) / view_width
            if not 0 < fraction < 0.8:
                raise ValueError(
                    "Scale-bar length must be smaller than 80% of the view width"
                )
            x1, y0 = 0.95, 0.075
            x0 = x1 - fraction
            fig.add_artist(
                plt.Line2D([x0, x1], [y0, y0], color=color, lw=2.0,
                           transform=fig.transFigure)
            )
            for x in (x0, x1):
                fig.add_artist(
                    plt.Line2D([x, x], [y0 - 0.012, y0 + 0.012], color=color,
                               lw=2.0, transform=fig.transFigure)
                )
            fig.text((x0 + x1) / 2.0, y0 + 0.022,
                     rf"${scalebar_length:g}\ \mathrm{{M}}_\odot$",
                     color=color, ha="center", va="bottom", fontsize=pt(0.028))

        fig.savefig(output, dpi=dpi, facecolor=background)
        plt.close(fig)
