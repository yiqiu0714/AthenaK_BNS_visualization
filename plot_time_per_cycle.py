"""
Plot distribution of wall-clock time per cycle from simulation .out files.

Handles multiple output-xxxx directories with overlapping restart data
by keeping only the latest run's cycles for overlapping time ranges.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style  # central house style (see plot_style.py)

plot_style.use()

# -------------------------------------------------------------------------
# Configuration (override on the command line; see --help)
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Plot wall-clock time-per-cycle distribution from AthenaK .out files."
)
parser.add_argument("--sim-dir", required=True,
                    help="Simulation directory containing the output-XXXX subdirectories.")
parser.add_argument("--sim-name", default="sim",
                    help="Label used in plot titles and the output subfolder (default: sim).")
parser.add_argument("--output-dirs", nargs="+", default=["output-0000"],
                    help="output-XXXX subdirectory names to read, relative to --sim-dir "
                         "(later entries override earlier ones for overlapping cycles).")
parser.add_argument("--out", default=None,
                    help="Directory for the saved figure (default: ./output/<sim-name>).")
parser.add_argument("--no-show", action="store_true",
                    help="Do not open an interactive window; just save the figure.")
args = parser.parse_args()

sim_name = args.sim_name
output_dir_basename_list = args.output_dirs
sim_dir = args.sim_dir
output_dir = args.out or os.path.join('output', sim_name)
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------------------
# Parse .out files from all output-xxxx directories
# -------------------------------------------------------------------------
line_re = re.compile(
    r'elapsed=([\d.eE+\-]+)\s+cycle=(\d+)\s+time=([\d.eE+\-]+)\s+dt=([\d.eE+\-]+)'
)

output_dirs = [os.path.join(sim_dir, b) for b in output_dir_basename_list]
print(f"Found {len(output_dirs)} output directories")

# Each output dir produces a segment: list of (elapsed, cycle, time, dt)
segments = []
for d in output_dirs:
    out_files = glob.glob(os.path.join(d, '*.out'))
    if not out_files:
        continue
    out_file = out_files[0]
    rows = []
    with open(out_file) as f:
        for line in f:
            m = line_re.match(line.strip())
            if m:
                rows.append((float(m.group(1)), int(m.group(2)),
                             float(m.group(3)), float(m.group(4))))
    if rows:
        arr = np.array(rows, dtype=[('elapsed', float), ('cycle', int),
                                     ('time', float), ('dt', float)])
        segments.append(arr)
        print(f"  {os.path.basename(d)}: {len(arr)} cycles, "
              f"cycle {arr['cycle'][0]}-{arr['cycle'][-1]}, "
              f"time {arr['time'][0]:.1f}-{arr['time'][-1]:.1f}")

# -------------------------------------------------------------------------
# Merge segments: later output-xxxx overrides earlier for overlapping cycles
# -------------------------------------------------------------------------
# Build a dict keyed by cycle number; later segments overwrite earlier ones
cycle_data = {}  # cycle -> (elapsed, time, dt)
for seg in segments:
    for row in seg:
        cycle_data[row['cycle']] = (row['elapsed'], row['time'], row['dt'])

cycles_sorted = np.array(sorted(cycle_data.keys()))
elapsed = np.array([cycle_data[c][0] for c in cycles_sorted])
sim_time = np.array([cycle_data[c][1] for c in cycles_sorted])

print(f"\nAfter merging: {len(cycles_sorted)} unique cycles "
      f"(cycle {cycles_sorted[0]}-{cycles_sorted[-1]})")

# -------------------------------------------------------------------------
# Compute time per cycle (wall-clock seconds between consecutive cycles)
# Only compute within contiguous runs (detect jumps from restarts via elapsed)
# -------------------------------------------------------------------------
d_elapsed = np.diff(elapsed)
d_cycle = np.diff(cycles_sorted)

# A restart resets elapsed to ~0; detect as large negative jump or cycle gap
contiguous = (d_elapsed > 0) & (d_cycle == 1)
time_per_cycle = d_elapsed[contiguous]
cycle_mid = cycles_sorted[:-1][contiguous]
sim_time_mid = sim_time[:-1][contiguous]

print(f"Contiguous cycle pairs: {len(time_per_cycle)}")
print(f"Time per cycle: median={np.median(time_per_cycle):.3f}s, "
      f"mean={np.mean(time_per_cycle):.3f}s, "
      f"std={np.std(time_per_cycle):.3f}s")

# -------------------------------------------------------------------------
# Plot 1: Histogram of time per cycle
# -------------------------------------------------------------------------
fig, axes = plot_style.subplots(2, 2, panel_w=7, panel_h=5)

# Percentiles weighted by total time (cumulative time fraction)
sorted_tpc = np.sort(time_per_cycle)
cum_frac = np.cumsum(sorted_tpc) / np.sum(sorted_tpc)
p_low = sorted_tpc[np.searchsorted(cum_frac, 0.05)]
p_high = sorted_tpc[np.searchsorted(cum_frac, 0.99)]
# x_max = min(p_high, time_per_cycle.max())
x_max = time_per_cycle.max() * 1.01
bins = np.linspace(0, x_max, 80)

# Top-left: histogram (count, log scale)
ax = axes[0, 0]
ax.hist(time_per_cycle, bins=bins, edgecolor='black', linewidth=0.3, alpha=0.8)
ax.axvline(np.median(time_per_cycle), color='red', ls='--', lw=1.5,
           label=f'median = {np.median(time_per_cycle):.3f} s')
ax.axvline(np.mean(time_per_cycle), color='orange', ls='--', lw=1.5,
           label=f'mean = {np.mean(time_per_cycle):.3f} s')
ax.set_yscale('log')
ax.set_xlabel('Wall-clock time per cycle [s]')
ax.set_ylabel('Count')
ax.set_title(f'{sim_name}\nTime per cycle distribution')
ax.legend()

# Top-right: time per cycle vs simulation time (log y-scale)
ax = axes[0, 1]
ax.scatter(sim_time_mid, time_per_cycle, s=0.3, alpha=0.5, rasterized=True)
# ax.set_yscale('log')
ax.set_ylim(0, 10.)
ax.set_xlabel('Simulation time [M$_{\\odot}$]')
ax.set_ylabel('Wall-clock time per cycle [s]')
ax.set_title('Time per cycle vs simulation time')

# Bottom-left: histogram weighted by time (shows where wall-clock time is spent)
ax = axes[1, 0]
ax.hist(time_per_cycle, bins=bins, weights=time_per_cycle,
        edgecolor='black', linewidth=0.3, alpha=0.8, color='C1')
weighted_mean = np.sum(time_per_cycle**2) / np.sum(time_per_cycle)
ax.axvline(weighted_mean, color='red', ls='--', lw=1.5,
           label=f'weighted mean = {weighted_mean:.3f} s')
ax.set_xlabel('Wall-clock time per cycle [s]')
ax.set_ylabel('Total wall-clock time [s]')
ax.set_title('Time spent per bin (weighted by duration)')
ax.set_xlim(0, x_max)
ax.legend()

# Bottom-right: cumulative fraction of wall-clock time
ax = axes[1, 1]
ax.plot(sorted_tpc, cum_frac, lw=1.5)
ax.set_xlabel('Wall-clock time per cycle [s]')
ax.set_ylabel('Cumulative fraction of total time')
ax.set_title('Cumulative wall-clock time distribution')
ax.set_xlim(0, x_max)
ax.axhline(0.5, color='gray', ls=':', lw=1)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'time_per_cycle.png'), dpi=200)
print(f"\nSaved to {os.path.join(output_dir, 'time_per_cycle.png')}")
if not args.no_show:
    plt.show()
