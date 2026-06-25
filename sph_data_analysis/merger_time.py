"""
Estimate the BNS merger time from the gravitational-wave (2,2) mode.

The merger time is defined as the time of the maximum |psi4_22|^2 = real^2 + imag^2
of the (l=2, m=2) mode, read from the AthenaK waveform output:

    <simpath>/output-XXXX/waveforms/rpsi4_real_RRRR.txt
    <simpath>/output-XXXX/waveforms/rpsi4_imag_RRRR.txt

where RRRR is the (zero-padded) extraction radius.  Column 1 of each file is the
time and column 6 is the (2,2) mode (see the file header: '6:22').  Data from all
output-XXXX restart segments are merged by time (later segments override earlier
ones for overlapping times).

Writes a single number (the merger time in M_sun) to the output file, plus a few
comment lines, so it can be read back by run_analysis.sh.
"""

import argparse
import glob
import os
import numpy as np

# Column index (0-based) of the (2,2) mode: header is "1:time 2:2-2 ... 6:22"
COL_22 = 5


def _load_segment(real_path, imag_path):
    """Return (time, real_22, imag_22) arrays for one waveform file pair."""
    real = np.loadtxt(real_path)
    imag = np.loadtxt(imag_path)
    t = real[:, 0]
    return t, real[:, COL_22], imag[:, COL_22]


def merger_time(simpath, radius):
    """Compute the merger time [M_sun] from the (2,2) GW mode at `radius`."""
    rstr = f"{int(round(radius)):04d}"
    real_files = sorted(glob.glob(
        os.path.join(simpath, "output-*", "waveforms", f"rpsi4_real_{rstr}.txt")))
    if not real_files:
        raise FileNotFoundError(
            f"No rpsi4_real_{rstr}.txt files found under "
            f"{simpath}/output-*/waveforms . "
            f"Check the simpath and the waveform extraction radius.")

    # Merge segments by time; later segments override earlier overlaps.
    merged = {}  # time -> amp2
    for real_path in real_files:
        imag_path = real_path.replace("rpsi4_real_", "rpsi4_imag_")
        if not os.path.exists(imag_path):
            print(f"  WARNING: missing {os.path.basename(imag_path)}, skipping")
            continue
        t, re22, im22 = _load_segment(real_path, imag_path)
        amp2 = re22**2 + im22**2
        for ti, ai in zip(t, amp2):
            merged[ti] = ai

    times = np.array(sorted(merged.keys()))
    amp2 = np.array([merged[ti] for ti in times])

    i_peak = int(np.argmax(amp2))
    t_merger = float(times[i_peak])
    print(f"Read {len(real_files)} segment(s), {len(times)} unique times "
          f"(t = {times[0]:.1f} .. {times[-1]:.1f} M_sun)")
    print(f"Peak |psi4_22|^2 = {amp2[i_peak]:.6e} at t = {t_merger:.4f} M_sun")
    return t_merger, radius


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simpath", required=True,
                        help="Simulation directory containing output-XXXX/waveforms.")
    parser.add_argument("--radius", type=float, default=300.0,
                        help="GW extraction radius (must match rpsi4_*_RRRR.txt). "
                             "Default: 300.")
    parser.add_argument("--out", required=True,
                        help="Output file for the merger time (merger_time.txt).")
    args = parser.parse_args()

    t_merger, radius = merger_time(args.simpath, args.radius)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Merger time from max |psi4_22|^2 (real^2 + imag^2) of the (2,2) mode\n")
        f.write(f"# GW extraction radius = {radius:g}\n")
        f.write("# t_merger [M_sun]\n")
        f.write(f"{t_merger:.6f}\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
