"""
Plot ejecta and neutrino diagnostics produced by analyze_ejecta.py and
analyze_neutrinos.py.

Reads the .txt outputs from --output-dir and writes:
    fig_ejecta.pdf    (2x3 panel summary of the ejecta properties)
    fig_neutrino.pdf  (2 panels: neutrino luminosity + mean energy)
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style  # central house style (repo root); see plot_style.py

plot_style.use()

# Unit conversions
G       = 6.67430e-11
M_SUN   = 1.98892e30
C       = 2.99792458e8
MSUN_TO_MS = G * M_SUN / C**3 * 1e3


def cumulative(arr):
    t = arr[:, 0]; r = arr[:, 1]; dt = np.diff(t)
    return t[1:], np.cumsum(0.5 * (r[:-1] + r[1:]) * dt)


# Ejecta panels with a log y-axis show this many decades below their peak;
# the near-empty pre-merger / high-v tails would otherwise set the range.
LOG_DECADES = 4


def _logy(ax, *ys):
    """Log y-axis spanning LOG_DECADES below the largest positive value."""
    ax.set_yscale('log')
    top = max((np.max(y[y > 0]) for y in ys if np.any(y > 0)), default=None)
    if top is not None:
        ax.set_ylim(top * 10**-LOG_DECADES, top * 2)


def _fmt_mass(m):
    return f'{m:.2e}'.replace('e-0', 'e-').replace('e+0', 'e')


def plot_ejecta(adir, t_merger, radius):
    """2x3 ejecta summary. Masses in M_sun; times relative to the merger."""
    def load(f): return np.loadtxt(os.path.join(adir, f))
    def t_rel(t): return (t - t_merger) * MSUN_TO_MS
    xlab = r'$t - t_\mathrm{merger}$ [ms]'

    Mej_rate_geo = load('Mej_rate_geo.txt')
    Mej_rate_Ber = load('Mej_rate_Ber.txt')
    Mej_vinf_geo = load('Mej_vinf_geo.txt')
    Mej_vinf_Ber = load('Mej_vinf_Ber.txt')
    rhoej_geo    = load('rhoej_theta_geo.txt')
    rhoej_Ber    = load('rhoej_theta_Ber.txt')
    Ye_avg_geo   = load('Ye_avg_geo.txt')
    Ye_avg_Ber   = load('Ye_avg_Ber.txt')
    Mej_Ye_geo   = load('Mej_Ye_geo.txt')
    Mej_Ye_Ber   = load('Mej_Ye_Ber.txt')

    t_geo_cum, Mej_geo_cum = cumulative(Mej_rate_geo)
    t_Ber_cum, Mej_Ber_cum = cumulative(Mej_rate_Ber)
    lab_geo = f'geodesic ({_fmt_mass(Mej_geo_cum[-1])} $M_\\odot$)'
    lab_Ber = f'Bernoulli ({_fmt_mass(Mej_Ber_cum[-1])} $M_\\odot$)'
    # The distributions only cover the post-merger histogram window, so their
    # totals can be smaller than the time-integrated ones above.
    lab_geo_h = f'geodesic ({_fmt_mass(Mej_Ye_geo[:, 1].sum())} $M_\\odot$)'
    lab_Ber_h = f'Bernoulli ({_fmt_mass(Mej_Ye_Ber[:, 1].sum())} $M_\\odot$)'
    xlim_t = dict(left=-2.0)

    fig, axes = plot_style.subplots(2, 3, panel_w=5.0, panel_h=4.0)
    fig.suptitle(f'BNS SPH — Ejecta properties  (r = {radius:g} M$_\\odot$, '
                 f'$t_\\mathrm{{merger}}$ = {t_merger:g} M$_\\odot$)', fontsize=13)

    ax = axes[0, 0]
    rg = Mej_rate_geo[:, 1] / MSUN_TO_MS; rb = Mej_rate_Ber[:, 1] / MSUN_TO_MS
    ax.plot(t_rel(Mej_rate_geo[:, 0]), rg, color='tab:blue', label='geodesic', lw=1.2)
    ax.plot(t_rel(Mej_rate_Ber[:, 0]), rb, color='tab:orange', ls='--', label='Bernoulli', lw=1.2)
    _logy(ax, rg, rb)
    ax.set_xlabel(xlab); ax.set_ylabel(r'$\dot{M}_\mathrm{ej}$  [$M_\odot\,\mathrm{ms}^{-1}$]')
    ax.legend(fontsize=9); ax.set_xlim(**xlim_t); ax.set_title('Mass ejection rate')

    ax = axes[0, 1]
    ax.plot(t_rel(t_geo_cum), Mej_geo_cum, color='tab:blue', label=lab_geo, lw=1.2)
    ax.plot(t_rel(t_Ber_cum), Mej_Ber_cum, color='tab:orange', ls='--', label=lab_Ber, lw=1.2)
    _logy(ax, Mej_geo_cum, Mej_Ber_cum)
    ax.set_xlabel(xlab); ax.set_ylabel(r'$M_\mathrm{ej}$  [$M_\odot$]')
    ax.legend(fontsize=9); ax.set_xlim(**xlim_t); ax.set_title('Cumulative ejecta mass')

    ax = axes[0, 2]
    # Thin: instantaneous flux-weighted <Ye> crossing the sphere. Thick: <Ye>
    # of everything ejected so far -- the number to compare between models,
    # and only at equal t - t_merger, since it keeps rising as the wind arrives.
    for Yt, R, col, ls, lab in ((Ye_avg_geo, Mej_rate_geo, 'tab:blue', '-', 'geodesic'),
                                (Ye_avg_Ber, Mej_rate_Ber, 'tab:orange', '--', 'Bernoulli')):
        n = min(len(Yt), len(R))
        tt, yy, ww = Yt[:n, 0], Yt[:n, 1], R[:n, 1]
        dm = np.concatenate([[0.0], 0.5 * (ww[1:] + ww[:-1]) * np.diff(tt)])
        dmy = np.concatenate([[0.0], 0.5 * (ww[1:] * yy[1:] + ww[:-1] * yy[:-1]) * np.diff(tt)])
        M = np.cumsum(dm)
        ycum = np.where(M > 0, np.cumsum(dmy) / np.where(M > 0, M, 1.0), np.nan)
        ax.plot(t_rel(tt), yy, color=col, ls=ls, lw=0.7, alpha=0.6)
        ax.plot(t_rel(tt), ycum, color=col, ls=ls, lw=1.8,
                label=f'{lab} cumulative ({ycum[-1]:.3f})')
    ax.plot([], [], color='gray', lw=0.7, alpha=0.6, label='instantaneous')
    ax.axhline(0.5, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(xlab); ax.set_ylabel(r'$\langle Y_e \rangle_\mathrm{ej}$')
    ax.legend(fontsize=9); ax.set_xlim(**xlim_t); ax.set_ylim(0, 0.6); ax.set_title(r'Mass-weighted $\langle Y_e \rangle$ of the ejecta')

    ax = axes[1, 0]
    dv = np.diff(Mej_vinf_geo[:, 0]); dv = np.append(dv, dv[-1])
    vg = Mej_vinf_geo[:, 1] / dv; vb = Mej_vinf_Ber[:, 1] / dv
    ax.step(Mej_vinf_geo[:, 0], vg, color='tab:blue', label=lab_geo_h, lw=1.2, where='mid')
    ax.step(Mej_vinf_Ber[:, 0], vb, color='tab:orange', ls='--', label=lab_Ber_h, lw=1.2, where='mid')
    _logy(ax, vg, vb)
    ax.set_xlabel(r'$v_\infty$ [$c$]'); ax.set_ylabel(r'$dM_\mathrm{ej}/dv_\infty$  [$M_\odot\,c^{-1}$]')
    ax.legend(fontsize=9); ax.set_xlim(0, 0.8); ax.set_title('Velocity distribution')

    ax = axes[1, 1]
    ax.plot(rhoej_geo[:, 0] * 180 / np.pi, rhoej_geo[:, 1], color='tab:blue', label=lab_geo_h, lw=1.2)
    ax.plot(rhoej_Ber[:, 0] * 180 / np.pi, rhoej_Ber[:, 1], color='tab:orange', ls='--', label=lab_Ber_h, lw=1.2)
    _logy(ax, rhoej_geo[:, 1], rhoej_Ber[:, 1])
    ax.set_xlabel(r'$\theta$ [deg]'); ax.set_ylabel(r'$dM_\mathrm{ej}/d\theta$  [$M_\odot\,\mathrm{rad}^{-1}$]')
    ax.set_xlim(0, 180); ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.legend(fontsize=9); ax.set_title('Angular distribution')

    ax = axes[1, 2]
    dye = np.diff(Mej_Ye_geo[:, 0]); dye = np.append(dye, dye[-1])
    yg = Mej_Ye_geo[:, 1] / dye; yb = Mej_Ye_Ber[:, 1] / dye
    ax.step(Mej_Ye_geo[:, 0], yg, color='tab:blue', label=lab_geo_h, lw=1.2, where='mid')
    ax.step(Mej_Ye_Ber[:, 0], yb, color='tab:orange', ls='--', label=lab_Ber_h, lw=1.2, where='mid')
    _logy(ax, yg, yb)
    ax.axvline(0.5, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$Y_e$'); ax.set_ylabel(r'$dM_\mathrm{ej}/dY_e$  [$M_\odot$]')
    ax.set_xlim(0, 0.6); ax.legend(fontsize=9); ax.set_title(r'$Y_e$ distribution')

    fig.tight_layout()
    out = os.path.join(adir, 'fig_ejecta.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


def plot_ye_theta(adir, radius):
    """2-D angular distribution of ejecta composition (Y_e vs theta).

    Reads the Mej_Ye_theta_{geo,Ber}.npy arrays; skips quietly if they are
    absent (analysis produced before this diagnostic was added).
    """
    try:
        theta = np.degrees(np.loadtxt(os.path.join(adir, 'theta_centers_2d.txt')))
        ye    = np.loadtxt(os.path.join(adir, 'Ye_centers.txt'))
        H_geo = np.load(os.path.join(adir, 'Mej_Ye_theta_geo.npy'))
        H_Ber = np.load(os.path.join(adir, 'Mej_Ye_theta_Ber.npy'))
    except (OSError, FileNotFoundError):
        print('  [ye_theta] Mej_Ye_theta_*.npy not found; skipping '
              '(re-run the ejecta analysis to generate it).')
        return

    fig, axes = plot_style.subplots(1, 2, panel_w=5.5, panel_h=4.2)
    fig.suptitle(f'BNS SPH — Ejecta composition vs angle  (r = {radius:g} M$_\\odot$)',
                 fontsize=13)
    vmax = max(H_geo.max(), H_Ber.max())
    norm = (LogNorm(vmin=vmax * 10**-LOG_DECADES, vmax=vmax) if vmax > 0 else None)
    for ax, H, title in ((axes[0], H_geo, 'geodesic'), (axes[1], H_Ber, 'Bernoulli')):
        pcm = ax.pcolormesh(theta, ye, np.ma.masked_less_equal(H, 0), cmap='viridis',
                            norm=norm, shading='auto')
        ax.set_xlabel(r'$\theta$ [deg]'); ax.set_ylabel(r'$Y_e$')
        ax.set_xlim(0, 180); ax.xaxis.set_major_locator(MultipleLocator(30))
        ax.set_ylim(0, 0.6); ax.set_title(f'{title}  ({_fmt_mass(H.sum())} $M_\\odot$)')
        fig.colorbar(pcm, ax=ax, label=r'$M_\mathrm{ej}$ per bin [$M_\odot$]')

    fig.tight_layout()
    out = os.path.join(adir, 'fig_ejecta_ye_theta.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


def plot_neutrino(adir, t_ms, t_merger_ms, nu_xlabel, nu_xlim, radius):
    def load(f): return np.loadtxt(os.path.join(adir, f))

    SPECIES  = ['nue', 'nua', 'nux', 'anux']
    SP_LABEL = [r'$\nu_e$', r'$\bar\nu_e$', r'$\nu_x$', r'$\bar\nu_x$']
    SP_COLOR = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    Lnu  = {sp: load(f'Lnu_E_{sp}.txt') for sp in SPECIES}
    Ltot = load('Lnu_E_total.txt')
    Eav  = {sp: load(f'Eav_{sp}.txt')   for sp in SPECIES}

    # Two panels: (1) luminosity, log scale 1e51–3e54;  (2) mean energy, 0–60.
    fig, axes = plot_style.subplots(1, 2, panel_w=5.5, panel_h=4.2)
    fig.suptitle(f'BNS SPH — Neutrino emission  (r = {radius:g} M$_\\odot$)', fontsize=13)

    ax = axes[0]
    ax.semilogy(t_ms(Ltot[:, 0]), Ltot[:, 1], color='black', lw=1.5, label='total')
    for sp, lab, col in zip(SPECIES, SP_LABEL, SP_COLOR):
        d = Lnu[sp]; ax.semilogy(t_ms(d[:, 0]), d[:, 1], color=col, label=lab, lw=1.0)
    ax.axvline(t_merger_ms, color='gray', ls='--', lw=0.8)
    ax.set_xlabel(nu_xlabel); ax.set_ylabel(r'$L_{\nu,E}$  [erg s$^{-1}$]')
    ax.set_ylim(1e51, 3e54)
    ax.legend(fontsize=9, ncol=2); ax.set_xlim(**nu_xlim)
    ax.set_title('Energy luminosity')

    ax = axes[1]
    for sp, lab, col in zip(SPECIES, SP_LABEL, SP_COLOR):
        d = Eav[sp]; ax.plot(t_ms(d[:, 0]), d[:, 1], color=col, label=lab, lw=1.2)
    ax.axvline(t_merger_ms, color='gray', ls='--', lw=0.8)
    ax.set_xlabel(nu_xlabel); ax.set_ylabel(r'$\langle\varepsilon_\nu\rangle$  [MeV]')
    ax.set_ylim(0, 60)
    ax.legend(fontsize=9, ncol=2); ax.set_xlim(**nu_xlim)
    ax.set_title('Mean neutrino energy')

    fig.tight_layout()
    out = os.path.join(adir, 'fig_neutrino.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True,
                        help="Directory holding the analysis .txt outputs.")
    parser.add_argument("--t-merger", type=float, required=True,
                        help="Merger time [M_sun] (from merger_time.txt).")
    parser.add_argument("--radius", type=float, default=300.0,
                        help="Neutrino SPH extraction radius (for plot titles). Default: 300.")
    parser.add_argument("--ej-radius", type=float, default=200.0,
                        help="Ejecta SPH extraction radius (for plot titles). Default: 200.")
    parser.add_argument("--from-merger", action="store_true",
                        help="Use t - t_merger [ms] as the neutrino x-axis (xlim left=0) "
                             "instead of absolute time.")
    parser.add_argument("--no-ejecta", action="store_true", help="Skip the ejecta figure.")
    parser.add_argument("--no-neutrino", action="store_true", help="Skip the neutrino figure.")
    args = parser.parse_args()

    adir = args.output_dir
    t_merger = args.t_merger

    if args.from_merger:
        def t_ms(t): return (t - t_merger) * MSUN_TO_MS
        nu_xlabel = r'$t - t_\mathrm{merger}$ [ms]'
        nu_xlim = {'left': 0}
        t_merger_ms = 0.0
    else:
        def t_ms(t): return t * MSUN_TO_MS
        nu_xlabel = r'$t$ [ms]'
        nu_xlim = {}
        t_merger_ms = t_merger * MSUN_TO_MS

    if not args.no_ejecta:
        plot_ejecta(adir, t_merger, args.ej_radius)
        plot_ye_theta(adir, args.ej_radius)
    if not args.no_neutrino:
        plot_neutrino(adir, t_ms, t_merger_ms, nu_xlabel, nu_xlim, args.radius)


if __name__ == "__main__":
    main()
