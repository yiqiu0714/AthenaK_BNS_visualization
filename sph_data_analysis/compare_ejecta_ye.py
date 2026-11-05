"""
Overlay the ejecta Ye distributions (Mej_Ye_{Ber,geo}.txt from analyze_ejecta.py) of
several models: Bernoulli criterion on the left panel, geodesic on the right, log y.

Every model must have been reduced with the SAME window (t_merger + T_POST_MS), or the
histograms integrate different time spans and are not comparable.

    python3 compare_ejecta_ye.py --t-post-ms 15 --out fig.png \
        --model DIR LABEL COLOR LINESTYLE [--model ...]
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style

plot_style.use(1, 2)


def _fmt_mass(m):
    return f'{m:.2e}'.replace('e-0', 'e-').replace('e+0', 'e')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', nargs=4, action='append', required=True,
                   metavar=('DIR', 'LABEL', 'COLOR', 'LS'),
                   help='analysis dir holding Mej_Ye_{Ber,geo}.txt, legend label, '
                        'matplotlib color, linestyle')
    p.add_argument('--t-post-ms', type=float, required=True,
                   help='histogram window after merger [ms], for the title')
    p.add_argument('--out', required=True)
    p.add_argument('--ylim', nargs=2, type=float, default=(1e-6, 1e-3))
    a = p.parse_args()

    hist = {(d, crit): np.loadtxt(os.path.join(d, f'Mej_Ye_{crit}.txt'))
            for d, *_ in a.model for crit in ('Ber', 'geo')}

    fig, axes = plt.subplots(1, 2, sharey=True,
                             figsize=plot_style.figsize(1, 2, panel_h=5.0))
    for ax, crit, title in ((axes[0], 'Ber', 'Bernoulli criterion'),
                            (axes[1], 'geo', 'Geodesic criterion')):
        for d, label, color, ls in a.model:
            h = hist[(d, crit)]
            ax.step(h[:, 0], h[:, 1], where='mid', color=color, ls=ls, lw=2.0)
        ax.set_yscale('log')
        ax.set_ylim(*a.ylim)
        ax.set_xlim(0.0, 0.6)
        ax.set_xlabel(r'$Y_e$')
        ax.set_title(title)
    axes[0].set_ylabel(r'$M_\mathrm{ej}$  [$M_\odot$]')

    # One shared legend below both panels; the per-model window totals (Bernoulli,
    # geodesic) go in the labels so neither panel's curves are covered.
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=2.0) for _, _, c, ls in a.model]
    labels = [f'{label}  (Ber {_fmt_mass(hist[(d, "Ber")][:, 1].sum())}, '
              f'geo {_fmt_mass(hist[(d, "geo")][:, 1].sum())} $M_\\odot$)'
              for d, label, _, _ in a.model]
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize='small')
    fig.suptitle(rf'Ejecta $Y_e$ distribution at $t_\mathrm{{merger}}$ + '
                 rf'{a.t_post_ms:g} ms')
    nrow = (len(a.model) + 1) // 2
    fig.tight_layout(rect=(0, 0.05 + 0.055 * nrow, 1, 1))
    fig.savefig(a.out, dpi=200)
    print(f'Saved {a.out}')


if __name__ == '__main__':
    main()
