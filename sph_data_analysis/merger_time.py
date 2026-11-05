"""
Estimate the BNS merger time from the gravitational-wave (2,2) mode.

The merger time is defined as the time of the peak strain amplitude |h_22|
(the usual convention, e.g. the CoRe database). AthenaK only writes psi4, so
h_22 is reconstructed by integrating psi4_22 twice in time with the
fixed-frequency integration (FFI) of Reisswig & Pollney (2011):

    h~(w) = - psi4~(w) / max(|w|, w0)^2

where w0 is a low-frequency cutoff a bit below the initial GW frequency; it
suppresses the secular drifts that a plain time-domain double integral picks
up from noise and the constant psi4 offset AthenaK carries at early times.
The peak of |psi4_22|, which the older version used, comes later than the strain
peak by an amount that depends on resolution and EOS. It is still printed
for comparison.

Input, from all output-XXXX restart segments merged by time (later segments
override earlier ones for overlapping times):

    <simpath>/output-XXXX/waveforms/rpsi4_real_RRRR.txt
    <simpath>/output-XXXX/waveforms/rpsi4_imag_RRRR.txt

where RRRR is the (zero-padded) extraction radius.  Column 1 is the time and
column 6 the (2,2) mode (see the file header: '6:22').

If the run's own waveform does not contain the merger (e.g. a short parameter
study restarted from an earlier segment and stopped before merger), the
restart lineage is followed: output-0000/parent points at the segment the
run was restarted from, and that run's waveform (whose inspiral the child
shares) is used instead.  If no merger is found anywhere it exits with an
error -- pass --t-merger (or t_merger in config.ini) to set it by hand.

The peak is seen at the extraction radius R, so the merger time is shifted to
retarded time, as the THC analysis does (scripts/modules/utils.py
retarded_time):

    t_merger = t_peak(R) - r*,   r* = R_a + 2 M ln(R_a / 2M - 1),
    R_a = R (1 + M / 2R)^2       (isotropic -> areal radius)

with M the ADM mass of the system (--m-adm; default 2.7, THC's M_Inf). For
R = 300 that is r* ~ 324 M ~ 1.6 ms. Matter/neutrino quantities measured on
a sphere at time t are then plotted against t - t_merger: time since the
merger in the source frame, including the travel time to that sphere.

Writes the merger time (in M_sun) to the output file, plus a few comment
lines, so it can be read back by run_analysis.sh.
"""

import argparse
import glob
import os
import sys
import numpy as np

# Column index (0-based) of the (2,2) mode: header is "1:time 2:2-2 ... 6:22"
COL_22 = 5

# FFI cutoff as a fraction of the initial GW angular frequency.
FFI_FRAC = 0.75

# The |h22| peak only counts as a merger if at least this much waveform
# [M_sun] follows it; a maximum at the very end of a pre-merger run is just
# the still-growing inspiral amplitude.
MIN_POST_PEAK = 100.0

# Length [M_sun] of the cosine ramp applied to each end of psi4 before the FFT.
# FFI treats the series as periodic, so an untapered jump between the two ends
# rings into |h| there -- enough to put a spurious global maximum at the last
# sample. The peak search also ignores this much at each end.
TAPER = 100.0

# Default ADM mass [M_sun] for the tortoise radius (THC's M_Inf default).
# Off by a few % only moves t_merger by ~0.1 M.
M_ADM_DEFAULT = 2.7


def tortoise_radius(R, M):
    """r* for an isotropic extraction radius R around mass M (THC convention)."""
    Ra = R * (1.0 + M / (2.0 * R))**2
    return Ra + 2.0 * M * np.log(Ra / (2.0 * M) - 1.0)


def _load_segment(real_path, imag_path):
    """Return (time, real_22, imag_22) arrays for one waveform file pair.

    Returns None if the pair carries no usable rows. A segment that died
    almost immediately can leave a waveform file holding a SINGLE row, which
    np.loadtxt hands back as a 1-D array -- indexing that with [:, 0] used to
    raise IndexError and abort the whole analysis over one bad segment out of
    hundreds. atleast_2d makes the single-row case a normal (1, ncol) array.
    """
    real = np.atleast_2d(np.loadtxt(real_path))
    imag = np.atleast_2d(np.loadtxt(imag_path))
    ncol = min(real.shape[1], imag.shape[1])
    if real.shape[0] == 0 or ncol <= COL_22:
        return None
    n = min(real.shape[0], imag.shape[0])
    return real[:n, 0], real[:n, COL_22], imag[:n, COL_22]


def load_psi4_22(simpath, radius):
    """Merged (time, complex r*psi4_22) over all output-* segments of simpath."""
    rstr = f"{int(round(radius)):04d}"
    real_files = sorted(glob.glob(
        os.path.join(simpath, "output-*", "waveforms", f"rpsi4_real_{rstr}.txt")))
    if not real_files:
        raise FileNotFoundError(
            f"No rpsi4_real_{rstr}.txt files found under "
            f"{simpath}/output-*/waveforms . "
            f"Check the simpath and the waveform extraction radius.")

    merged = {}  # time -> complex psi4_22
    for real_path in real_files:
        imag_path = real_path.replace("rpsi4_real_", "rpsi4_imag_")
        if not os.path.exists(imag_path):
            print(f"  WARNING: missing {os.path.basename(imag_path)}, skipping")
            continue
        seg = _load_segment(real_path, imag_path)
        if seg is None:
            print(f"  WARNING: {os.path.basename(os.path.dirname(os.path.dirname(real_path)))}"
                  f"/{os.path.basename(real_path)} has no usable rows, skipping")
            continue
        for ti, ri, ii in zip(*seg):
            merged[ti] = ri + 1j * ii

    times = np.array(sorted(merged.keys()))
    psi4 = np.array([merged[ti] for ti in times])
    print(f"  {simpath}: {len(real_files)} segment(s), {len(times)} unique times "
          f"(t = {times[0]:.1f} .. {times[-1]:.1f} M_sun)")
    return times, psi4


def initial_gw_omega(t, psi4, radius):
    """Initial GW angular frequency from the psi4 phase, after the junk burst.

    Skips the first `radius` + 150 M (light travel to the detector plus the
    junk radiation), then takes the median |d phase/dt| over the next 300 M.
    """
    t0 = t[0] + radius + 150.0
    sel = (t >= t0) & (t <= t0 + 300.0)
    if sel.sum() < 20:                      # short series: use what there is
        sel = slice(len(t) // 4, None)
    phase = np.unwrap(np.angle(psi4[sel]))
    return float(np.median(np.abs(np.gradient(phase, t[sel]))))


def strain_ffi(t, psi4, omega0):
    """h_22(t) from r*psi4_22 by fixed-frequency integration.

    The series is resampled to a uniform grid (restart segments can leave
    small gaps or jitter).
    """
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    p = np.interp(tu, t, psi4.real) + 1j * np.interp(tu, t, psi4.imag)
    nr = min(int(TAPER / dt), len(tu) // 4)
    if nr > 1:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(nr) / nr))
        p[:nr] *= ramp
        p[-nr:] *= ramp[::-1]
    w = 2.0 * np.pi * np.fft.fftfreq(len(tu), d=dt)
    weff = np.where(np.abs(w) < omega0, np.sign(w) * omega0, w)
    weff[weff == 0] = omega0
    h = np.fft.ifft(-np.fft.fft(p) / weff**2)
    return tu, h


def merger_in_run(simpath, radius, omega0=None):
    """(t_merger, info) from simpath's own waveform, or (None, info)."""
    t, psi4 = load_psi4_22(simpath, radius)
    if omega0 is None:
        omega0 = FFI_FRAC * initial_gw_omega(t, psi4, radius)
    tu, h = strain_ffi(t, psi4, omega0)
    amp = np.abs(h)
    # Search window: drop the tapered ends and, for a run that starts at t=0,
    # the junk-radiation burst (it reaches the detector at t ~ radius).
    lo = t[0] + TAPER
    if t[0] < 1.0:
        lo = max(lo, radius + 150.0)
    hi = t[-1] - TAPER
    win = (tu >= lo) & (tu <= hi)
    if not win.any():
        print(f"  waveform too short ({t[-1]-t[0]:.0f} M) to search for a peak")
        return None, dict(t_end=float(t[-1]))
    i = int(np.flatnonzero(win)[np.argmax(amp[win])])
    t_h = float(tu[i])
    pw = (t >= lo) & (t <= hi)
    t_psi4 = float(t[pw][np.argmax(np.abs(psi4[pw]))]) if pw.any() else float('nan')
    info = dict(t_end=float(t[-1]), omega0=omega0, t_psi4=t_psi4, amp=float(amp[i]))
    print(f"  FFI cutoff w0 = {omega0:.4e} /M_sun  (f0 = {omega0/2/np.pi:.4e})")
    print(f"  peak |h22|    at t = {t_h:.2f} M_sun  (|h22| = {amp[i]:.4e})")
    print(f"  peak |psi4_22| at t = {t_psi4:.2f} M_sun  (old definition, for reference)")
    if hi - t_h < MIN_POST_PEAK:
        print(f"  -> |h22| peaks within {MIN_POST_PEAK:g} M of the end of the usable data "
              f"(t_end = {t[-1]:.1f}): merger is NOT in this run's waveform.")
        return None, info
    if t[0] >= 1.0 and t_h - lo < MIN_POST_PEAK:
        print(f"  -> |h22| peaks at the start of the data (t_start = {t[0]:.1f}): "
              f"merger is before this run's waveform.")
        return None, info
    return t_h, info


def parent_simpath(simpath):
    """Simulation directory this run was restarted from, via output-0000/parent."""
    first = sorted(glob.glob(os.path.join(simpath, "output-*")))
    if not first:
        return None
    link = os.path.join(first[0], "parent")
    if not os.path.exists(link):
        return None
    seg = os.path.realpath(link)            # .../<parent sim>/output-NNNN
    parent = os.path.dirname(seg)
    if os.path.realpath(parent) == os.path.realpath(simpath):
        return None
    return parent


def merger_time(simpath, radius, max_depth=5):
    """Merger time [M_sun] and the simpath whose waveform provided it."""
    path = simpath
    for _ in range(max_depth + 1):
        t_m, _ = merger_in_run(path, radius)
        if t_m is not None:
            return t_m, path
        nxt = parent_simpath(path)
        if nxt is None:
            break
        print(f"  following restart lineage -> {nxt}")
        path = nxt
    raise RuntimeError(
        f"No merger found in the waveform of {simpath} or its restart parents. "
        f"Set it by hand: --t-merger <M_sun> (or t_merger = ... in config.ini).")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simpath", required=True,
                        help="Simulation directory containing output-XXXX/waveforms.")
    parser.add_argument("--radius", type=float, default=300.0,
                        help="GW extraction radius (must match rpsi4_*_RRRR.txt). "
                             "Default: 300.")
    parser.add_argument("--t-merger", type=float, default=None,
                        help="Use this merger time [M_sun] (already retarded) "
                             "instead of computing it.")
    parser.add_argument("--m-adm", type=float, default=M_ADM_DEFAULT,
                        help=f"ADM mass [M_sun] for the retarded-time shift. "
                             f"Default: {M_ADM_DEFAULT} (THC's M_Inf).")
    parser.add_argument("--out", required=True,
                        help="Output file for the merger time (merger_time.txt).")
    args = parser.parse_args()

    rstar = tortoise_radius(args.radius, args.m_adm)
    t_peak = None
    if args.t_merger is not None:
        t_merger, source = args.t_merger, "set by hand (--t-merger)"
        print(f"Merger time set by hand: {t_merger:g} M_sun")
    else:
        try:
            t_peak, src = merger_time(args.simpath, args.radius)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)
        source = ("own waveform" if os.path.realpath(src) == os.path.realpath(args.simpath)
                  else f"waveform of restart parent {src}")
        t_merger = t_peak - rstar
        print(f"  retarded: t_merger = t_peak - r* = {t_peak:.2f} - {rstar:.2f} "
              f"= {t_merger:.2f} M_sun  (R = {args.radius:g}, M_ADM = {args.m_adm:g})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Merger time = peak of the (2,2) strain amplitude |h22| "
                "(h from psi4 by FFI)\n")
        f.write(f"# GW extraction radius = {args.radius:g}\n")
        f.write(f"# source: {source}\n")
        if t_peak is not None:
            f.write(f"# retarded time: t_merger = t_peak(R) - r*, t_peak = {t_peak:.6f}, "
                    f"r* = {rstar:.6f} (M_ADM = {args.m_adm:g})\n")
        f.write("# t_merger [M_sun]\n")
        f.write(f"{t_merger:.6f}\n")
    print(f"Wrote {args.out}  (t_merger = {t_merger:.2f} M_sun, {source})")


if __name__ == "__main__":
    main()
