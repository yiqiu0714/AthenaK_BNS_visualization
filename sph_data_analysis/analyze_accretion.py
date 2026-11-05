"""
Approximate post-merger accretion rate from baryon-number bookkeeping.

Baryon conservation for the post-merger disk reads
    (disk drain) = (accreted onto BH) + (ejected),
so the accretion rate is (the reference constant cancels under d/dt):

    Mdot_accr(t) = -dM_b/dt  +  Mdot_ej(t - r_ej/<v_ej>)

    M_b(t)   domain baryon (rest) mass       -> <jobname>.mhd.hst, col [3]=mass
    Mdot_ej  ejecta mass flux                -> Mej_rate_<crit>.txt
    <v_ej>   characteristic ejecta speed [c] -> flux-weighted vinf_avg_<crit>.txt
    r_ej     matter extraction radius        -> --r-ej (set to your sph radius!)

The domain baryon mass is read from the per-segment <jobname>.mhd.hst files under
<simpath>/output-*/ and concatenated across restarts (duplicate times keep the
later segment), then resampled onto a uniform grid so sample-based smoothing /
differencing is well-posed (restarts otherwise leave irregular dt that spikes the
derivative).

Command line:
    python3 analyze_accretion.py --simpath <sim> --analysis-dir <sim>/analysis \\
        --r-ej 200 [--jobname bns] [--crit Ber]
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

MSUN_TO_MS = 6.67430e-11 * 1.98892e30 / 2.99792458e8**3 * 1e3  # 4.9268e-3


def load_domain_mass(simpath, jobname):
    """Concatenated (time, baryon-mass) from every output-*/<jobname>.mhd.hst.

    Rows are sorted by time; duplicate times keep the later segment's value.
    """
    files = sorted(glob.glob(os.path.join(simpath, "output-*", f"{jobname}.mhd.hst")))
    if not files:
        raise FileNotFoundError(
            f"no output-*/{jobname}.mhd.hst under {simpath}")
    arrays = []
    for f in files:
        a = np.loadtxt(f, skiprows=2)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.size:
            arrays.append(a[:, [0, 2]])       # [1]=time, [3]=mass
    combined = np.vstack(arrays)
    combined = combined[np.argsort(combined[:, 0], kind="stable")]
    # keep last occurrence for each time (later segment wins)
    t = combined[:, 0]
    _, first_in_rev = np.unique(t[::-1], return_index=True)
    last = np.sort(len(t) - 1 - first_in_rev)
    combined = combined[last]
    return combined[:, 0], combined[:, 1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simpath", required=True,
                   help="Simulation directory containing the output-*/ segments.")
    p.add_argument("--analysis-dir", required=True,
                   help="Directory with Mej_rate_*.txt / vinf_avg_*.txt; the "
                        "accretion figure is written here too.")
    p.add_argument("--jobname", default="bns",
                   help="AthenaK job name prefixing the .hst files. Default: bns.")
    p.add_argument("--r-ej", type=float, default=200.0,
                   help="Matter extraction radius [M_sun], must match the sph radius.")
    p.add_argument("--crit", choices=["geo", "Ber"], default="Ber",
                   help="Ejecta criterion. Default: Ber.")
    p.add_argument("--smooth", type=int, default=9,
                   help="Boxcar width (samples) for smoothing dM_b/dt. Default: 9.")
    p.add_argument("--t-settle", type=float, default=3.0,
                   help="Skip this many ms after merger to exclude the plunge. Default: 3.")
    args = p.parse_args(argv)

    adir = args.analysis_dir

    # domain baryon mass M_b(t): concatenate segments, drop duplicate/restart
    # times, then resample onto a uniform grid.
    t_hst, mb = load_domain_mass(args.simpath, args.jobname)
    keep = np.concatenate([[True], np.diff(t_hst) > 0])
    t_hst, mb = t_hst[keep], mb[keep]
    dt = np.median(np.diff(t_hst))
    t_uni = np.arange(t_hst[0], t_hst[-1], dt)
    mb = np.interp(t_uni, t_hst, mb)
    t_hst = t_uni

    # ejecta flux and characteristic speed (flux-weighted mean vinf)
    rate = np.loadtxt(os.path.join(adir, f"Mej_rate_{args.crit}.txt"))
    t_ej, mdot_ej = rate[:, 0], rate[:, 1]
    vv = np.loadtxt(os.path.join(adir, f"vinf_avg_{args.crit}.txt"))
    w = np.interp(vv[:, 0], t_ej, mdot_ej)
    m = (vv[:, 1] > 0) & (w > 0)
    vej = np.sum(vv[m, 1] * w[m]) / np.sum(w[m])
    delay = args.r_ej / vej

    # accretion rate [Msun/ms]
    mbdot = np.gradient(uniform_filter1d(mb, size=max(1, args.smooth), mode="nearest"), t_hst)
    mdot_ej_shift = np.interp(t_hst - delay, t_ej, mdot_ej, left=0.0)
    mdot_accr = (-mbdot + mdot_ej_shift) / MSUN_TO_MS

    # merger = steepest domain-mass drop; focus after the plunge settles
    t_merger = t_hst[np.argmin(mbdot)]
    t0 = t_merger + args.t_settle / MSUN_TO_MS
    post = t_hst >= t0

    print(f"crit={args.crit}  <v_ej>={vej:.4f} c  r_ej={args.r_ej:g}  "
          f"delay={delay*MSUN_TO_MS:.2f} ms  t_merger={t_merger:.1f} Msun")

    # save the curve so it can be re-plotted / compared later
    np.savetxt(os.path.join(adir, f"Mdot_accr_{args.crit}.txt"),
               np.column_stack(((t_hst - t_merger) * MSUN_TO_MS, mdot_accr)),
               header=f"t - t_merger[ms]    Mdot_accr[Msun/ms]  (crit={args.crit}, "
                      f"r_ej={args.r_ej:g}, <v_ej>={vej:.4f}c)", fmt="%.6e")

    # plot
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot((t_hst[post] - t_merger) * MSUN_TO_MS, mdot_accr[post],
            color="tab:red", lw=1.4)
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xlim(left=(t0 - t_merger) * MSUN_TO_MS)
    ax.set_xlabel(r"$t - t_\mathrm{merger}$ [ms]")
    ax.set_ylabel(r"$\dot M_\mathrm{accr}$  [$M_\odot\,\mathrm{ms}^{-1}$]")
    fig.tight_layout()
    out = os.path.join(adir, "fig_accretion.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
