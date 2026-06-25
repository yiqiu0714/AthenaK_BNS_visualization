"""
Plot ejecta and neutrino diagnostics produced by analyze_ejecta.py and
analyze_neutrinos.py.

Reads the .txt outputs from --output-dir and writes:
    fig_ejecta.pdf    (2x3 panel summary of the ejecta properties)
    fig_neutrino.pdf  (2 panels: neutrino luminosity + mean energy)
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Unit conversions
G       = 6.67430e-11
M_SUN   = 1.98892e30
C       = 2.99792458e8
MSUN_TO_MS = G * M_SUN / C**3 * 1e3


def cumulative(arr):
    t = arr[:, 0]; r = arr[:, 1]; dt = np.diff(t)
    return t[1:], np.cumsum(0.5 * (r[:-1] + r[1:]) * dt)


def plot_ejecta(adir, t_ms, radius):
    def load(f): return np.loadtxt(os.path.join(adir, f))

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

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'BNS SPH — Ejecta properties  (r = {radius:g} M$_\\odot$)', fontsize=13)

    ax = axes[0, 0]
    ax.plot(t_ms(Mej_rate_geo[:, 0]), Mej_rate_geo[:, 1] / MSUN_TO_MS * 1e3, color='tab:blue', label='geodesic', lw=1.2)
    ax.plot(t_ms(Mej_rate_Ber[:, 0]), Mej_rate_Ber[:, 1] / MSUN_TO_MS * 1e3, color='tab:orange', ls='--', label='Bernoulli', lw=1.2)
    ax.set_xlabel(r'$t - t_\mathrm{merger}$ [ms]'); ax.set_ylabel(r'$\dot{M}_\mathrm{ej}$  [$10^{-3}M_\odot\,\mathrm{ms}^{-1}$]')
    ax.legend(fontsize=9); ax.set_xlim(left=0); ax.set_title('Mass ejection rate')

    ax = axes[0, 1]
    ax.plot(t_ms(t_geo_cum), Mej_geo_cum * 1e3, color='tab:blue', label='geodesic', lw=1.2)
    ax.plot(t_ms(t_Ber_cum), Mej_Ber_cum * 1e3, color='tab:orange', ls='--', label='Bernoulli', lw=1.2)
    ax.set_xlabel(r'$t - t_\mathrm{merger}$ [ms]'); ax.set_ylabel(r'$M_\mathrm{ej}$  [$10^{-3}M_\odot$]')
    ax.legend(fontsize=9); ax.set_xlim(left=0); ax.set_title('Cumulative ejecta mass')

    ax = axes[0, 2]
    ax.plot(t_ms(Ye_avg_geo[:, 0]), Ye_avg_geo[:, 1], color='tab:blue', label='geodesic', lw=1.2)
    ax.plot(t_ms(Ye_avg_Ber[:, 0]), Ye_avg_Ber[:, 1], color='tab:orange', ls='--', label='Bernoulli', lw=1.2)
    ax.axhline(0.5, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$t - t_\mathrm{merger}$ [ms]'); ax.set_ylabel(r'$\langle Y_e \rangle_\mathrm{ej}$')
    ax.legend(fontsize=9); ax.set_xlim(left=0); ax.set_ylim(0, 0.6); ax.set_title(r'Mass-weighted $\langle Y_e \rangle$')

    ax = axes[1, 0]
    dv = np.diff(Mej_vinf_geo[:, 0]); dv = np.append(dv, dv[-1])
    ax.step(Mej_vinf_geo[:, 0], Mej_vinf_geo[:, 1] / dv * 1e3, color='tab:blue', label='geodesic', lw=1.2, where='mid')
    ax.step(Mej_vinf_Ber[:, 0], Mej_vinf_Ber[:, 1] / dv * 1e3, color='tab:orange', ls='--', label='Bernoulli', lw=1.2, where='mid')
    ax.set_xlabel(r'$v_\infty$ [$c$]'); ax.set_ylabel(r'$dM_\mathrm{ej}/dv_\infty$  [$10^{-3}M_\odot\,c^{-1}$]')
    ax.legend(fontsize=9); ax.set_xlim(0, 0.8); ax.set_title('Velocity distribution')

    ax = axes[1, 1]
    ax.plot(rhoej_geo[:, 0] * 180 / np.pi, rhoej_geo[:, 1] * 1e3, color='tab:blue', label='geodesic', lw=1.2)
    ax.plot(rhoej_Ber[:, 0] * 180 / np.pi, rhoej_Ber[:, 1] * 1e3, color='tab:orange', ls='--', label='Bernoulli', lw=1.2)
    ax.set_xlabel(r'$\theta$ [deg]'); ax.set_ylabel(r'$dM_\mathrm{ej}/d\theta$  [$10^{-3}M_\odot\,\mathrm{rad}^{-1}$]')
    ax.set_xlim(0, 180); ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.legend(fontsize=9); ax.set_title('Angular distribution')

    ax = axes[1, 2]
    dye = np.diff(Mej_Ye_geo[:, 0]); dye = np.append(dye, dye[-1])
    ax.step(Mej_Ye_geo[:, 0], Mej_Ye_geo[:, 1] / dye * 1e3, color='tab:blue', label='geodesic', lw=1.2, where='mid')
    ax.step(Mej_Ye_Ber[:, 0], Mej_Ye_Ber[:, 1] / dye * 1e3, color='tab:orange', ls='--', label='Bernoulli', lw=1.2, where='mid')
    ax.axvline(0.5, color='gray', ls=':', lw=0.8)
    ax.set_xlabel(r'$Y_e$'); ax.set_ylabel(r'$dM_\mathrm{ej}/dY_e$  [$10^{-3}M_\odot$]')
    ax.set_xlim(0, 0.6); ax.legend(fontsize=9); ax.set_title(r'$Y_e$ distribution')

    fig.tight_layout()
    out = os.path.join(adir, 'fig_ejecta.pdf')
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
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
                        help="SPH extraction radius (for plot titles). Default: 300.")
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
        plot_ejecta(adir, t_ms, args.radius)
    if not args.no_neutrino:
        plot_neutrino(adir, t_ms, t_merger_ms, nu_xlabel, nu_xlim, args.radius)


if __name__ == "__main__":
    main()
