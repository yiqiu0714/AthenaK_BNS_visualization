"""
BNS_all visualization script
Combines all output-XXXX restart segments of a BNS simulation and
produces the same diagnostics as BNS.py, but spanning the full run.

Usage:
    python3 BNS_all.py --simpath PATH [--figpath PATH] [--no-trackers]
                       [--sections SECTION [SECTION ...]]

output-XXXX subdirectories are found automatically under SIMPATH.
Frames that appear in multiple segments (at restart boundaries) are
deduplicated by simulation time; the later segment's version is kept.
"""

import argparse
import glob
import multiprocessing
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Make the bundled plot-tools submodule and this directory importable,
# regardless of the current working directory.  Override the plot-tools
# location with the ATHPLOT_PATH environment variable if needed.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PLOT_TOOLS = os.environ.get("ATHPLOT_PATH", os.path.join(_HERE, "plot-tools"))
sys.path.append(_PLOT_TOOLS)
from athplot.load_ath_bin import BinaryData
from athplot.group_data import GroupData
from athplot.tracker import Tracker

sys.path.append(_HERE)
import athenak_units as au

# ---------------------------------------------------------------------------
# Global plot style  (identical to BNS.py)
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    'lines.markersize': 6,
    'font.family': 'serif',
    'font.size': 10.0,
    'text.usetex': False,
    'axes.labelsize': 'large',
    'xtick.major.size': 10,
    'xtick.minor.size': 6,
    'xtick.minor.visible': True,
    'xtick.labelsize': 'large',
    'xtick.direction': 'in',
    'xtick.top': True,
    'ytick.major.size': 10,
    'ytick.minor.size': 6,
    'ytick.minor.visible': True,
    'ytick.labelsize': 'large',
    'ytick.direction': 'in',
    'ytick.right': True,
    'legend.numpoints': 1,
    'legend.fontsize': 'medium',
    'legend.frameon': False,
    'legend.scatterpoints': 1,
    'figure.figsize': (5, 3),
})

FIGSIZE = (5, 3)

# Spatial coordinate grid is in geometrized M_sun units (G=c=M_sun=1),
# 1 code length = G*M_sun/c^2 = 1.476955 km.  Field values are rescaled by
# --units, but the coordinate axes are only converted to km for non-code units.
MSUN_TO_KM = au._unit_len_cgs / 1.0e5


def spatial_units(units):
    """Return (coord_scale, axis_label) for the spatial axes given a unit system.

    'code' keeps native M_sun coordinates; 'cgs'/'ngs' display km.
    """
    if units == "code":
        return 1.0, r"\mathrm{M}_\odot"
    return MSUN_TO_KM, r"\mathrm{km}"


# ---------------------------------------------------------------------------
# Helpers shared with BNS.py
# ---------------------------------------------------------------------------

def glob_bin_files(directory, prefix):
    """Return sorted list of existing binary files matching prefix.NNNNN.bin."""
    pattern = os.path.join(directory, "bin", f"{prefix}.?????.bin")
    return sorted(glob.glob(pattern))


def _detect_prefix_all(output_dirs, new_prefix, old_prefix):
    """Return new_prefix if files exist in any segment, else old_prefix."""
    for d in output_dirs:
        if glob_bin_files(d, new_prefix):
            return new_prefix
    return old_prefix


def print_stats(data, var_name):
    """Print min/max of var_name across all blocks for this frame."""
    max_val, min_val = -np.inf, np.inf
    for _, arr in data.get_block_data(var_name):
        max_val = max(max_val, arr.max())
        min_val = min(min_val, arr.min())
    print(f"    t={data.time:.3f} Msun  |  max={max_val:.3e}  min={min_val:.3e}")


def draw_trackers(ax, trackers, time,
                  colors=("cyan", "orange"), marker="x", markersize=80,
                  coord_scale=1.0):
    """Overlay NS positions on ax at the given simulation time.

    coord_scale multiplies the stored (M_sun) tracker coordinates so they match
    the displayed spatial units (e.g. km when coord_scale = MSUN_TO_KM).
    """
    labels = ["NS 1", "NS 2"]
    for tracker, color, label in zip(trackers, colors, labels):
        if tracker is None:
            continue
        idx = np.argmin(np.abs(tracker.time - time))
        ax.scatter(tracker.x[idx] * coord_scale, tracker.y[idx] * coord_scale,
                   marker=marker, c=color, s=markersize,
                   linewidths=1.5, zorder=10, label=label)


def rescale_image_extents(ax, coord_scale):
    """Multiply the extent of every image already drawn on ax by coord_scale.

    Used to convert the native M_sun grid coordinates of plot_slice() output to
    the displayed spatial units (km).  No-op when coord_scale == 1.0.
    """
    if coord_scale == 1.0:
        return
    for im in ax.get_images():
        x0, x1, y0, y1 = im.get_extent()
        im.set_extent([x0 * coord_scale, x1 * coord_scale,
                       y0 * coord_scale, y1 * coord_scale])


def draw_refinement_grid(ax, data, alpha=0.4, linewidth=0.6, coord_scale=1.0):
    """Draw block boundaries colored by AMR refinement level."""
    levels = sorted(set(b.level for b in data.blocks))
    if len(levels) <= 1:
        return

    cmap = mpl.colormaps.get_cmap("cool")
    level_min, level_max = levels[0], levels[-1]

    def level_color(lv):
        if level_max == level_min:
            return cmap(0.5)
        return cmap((lv - level_min) / (level_max - level_min))

    for block in data.blocks:
        ext = block.get_extent()
        if len(ext) < 4:
            continue
        xmin, xmax, ymin, ymax = (ext[0] * coord_scale, ext[1] * coord_scale,
                                  ext[2] * coord_scale, ext[3] * coord_scale)
        rect = mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=linewidth,
            edgecolor=level_color(block.level),
            facecolor="none",
            alpha=alpha,
            zorder=5,
        )
        ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Combined-tracker class
# ---------------------------------------------------------------------------

class _CombinedTracker:
    """Lightweight tracker holding concatenated time/x/y arrays from all segments."""

    def __init__(self, time, x, y):
        self.time = time
        self.x = x
        self.y = y


# ---------------------------------------------------------------------------
# Module-level worker for reading frame times (must be picklable for Pool)
# ---------------------------------------------------------------------------

def _read_frame_time(fpath):
    """Read simulation time from a binary data file. Returns (fpath, time_or_None)."""
    try:
        return (fpath, BinaryData(fpath).time)
    except Exception:
        return (fpath, None)


# ---------------------------------------------------------------------------
# Multi-segment data collection helpers
# ---------------------------------------------------------------------------

def find_output_dirs(simpath):
    """Return sorted list of output-XXXX subdirectories under simpath."""
    pattern = os.path.join(simpath, "output-[0-9][0-9][0-9][0-9]")
    return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))


def _collect_frames_by_time(output_dirs, file_prefix):
    """
    Gather all files for file_prefix across output_dirs.
    Returns dict: sim_time -> xy_fpath.
    Later segments overwrite earlier ones for duplicate times.
    """
    all_files = []
    for d in output_dirs:
        all_files.extend(glob_bin_files(d, file_prefix))

    if not all_files:
        return {}

    n_workers = min(multiprocessing.cpu_count(), len(all_files))
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(_read_frame_time, all_files)

    # Iterate in order so later segments overwrite earlier ones
    time_to_fpath = {}
    for fpath, t in results:
        if t is not None:
            time_to_fpath[t] = fpath
    return time_to_fpath


def collect_combined_frames(output_dirs, file_prefix, xz_prefix=None, step=1):
    """
    Collect all frames for file_prefix across all segments, deduplicated by
    simulation time (later segment wins).

    Returns list of (global_idx, xy_fpath, xz_fpath_or_None) sorted by sim time,
    with step applied after deduplication.
    """
    time_to_xy = _collect_frames_by_time(output_dirs, file_prefix)
    if not time_to_xy:
        return []

    # Build set of all existing XZ files for quick lookup
    xz_file_set = set()
    if xz_prefix is not None:
        for d in output_dirs:
            xz_file_set.update(glob_bin_files(d, xz_prefix))

    sorted_times = sorted(time_to_xy.keys())[::step]

    result = []
    for global_idx, t in enumerate(sorted_times):
        xy_fpath = time_to_xy[t]
        xz_fpath = None
        if xz_prefix is not None:
            m = re.search(r'\.(\d{5})\.bin$', os.path.basename(xy_fpath))
            if m:
                xz_candidate = os.path.join(
                    os.path.dirname(xy_fpath), f"{xz_prefix}.{m.group(1)}.bin"
                )
                if xz_candidate in xz_file_set:
                    xz_fpath = xz_candidate
        result.append((global_idx, xy_fpath, xz_fpath))
    return result


def combine_hst(output_dirs, filename):
    """
    Load and concatenate history (.hst) files across all segments.
    Rows are sorted by simulation time (col 0); duplicate times keep the
    later segment's row.  Returns numpy array or None if no files found.
    """
    arrays = []
    for d in output_dirs:
        fpath = os.path.join(d, filename)
        if os.path.exists(fpath):
            try:
                a = np.loadtxt(fpath, skiprows=2)
                if a.ndim == 1:
                    a = a.reshape(1, -1)
                if a.size > 0:
                    arrays.append(a)
            except Exception as e:
                print(f"  Warning: could not load {fpath}: {e}")

    if not arrays:
        return None

    combined = np.vstack(arrays)
    sort_idx = np.argsort(combined[:, 0], kind='stable')
    combined = combined[sort_idx]

    # Keep last occurrence for each time (later segment wins)
    times = combined[:, 0]
    _, first_in_rev = np.unique(times[::-1], return_index=True)
    last_in_orig = np.sort(len(times) - 1 - first_in_rev)
    return combined[last_in_orig]


def load_trackers_all(output_dirs):
    """
    Load and concatenate tracker files across all segments.
    Returns list of two _CombinedTracker objects (entries may be None).
    """
    trackers = []
    for k in (0, 1):
        times_list, x_list, y_list = [], [], []
        for d in output_dirs:
            fpath = os.path.join(d, f"bns.co_{k}.txt")
            if os.path.exists(fpath):
                try:
                    t = Tracker(fpath)
                    times_list.append(t.time)
                    x_list.append(t.x)
                    y_list.append(t.y)
                except Exception as e:
                    print(f"  Warning: could not load {fpath}: {e}")

        if not times_list:
            print(f"  No tracker files found for NS {k}")
            trackers.append(None)
            continue

        all_times = np.concatenate(times_list)
        all_x     = np.concatenate(x_list)
        all_y     = np.concatenate(y_list)

        sort_idx  = np.argsort(all_times, kind='stable')
        all_times = all_times[sort_idx]
        all_x     = all_x[sort_idx]
        all_y     = all_y[sort_idx]

        # Keep last occurrence for each time
        _, first_in_rev = np.unique(all_times[::-1], return_index=True)
        keep = np.sort(len(all_times) - 1 - first_in_rev)

        trackers.append(_CombinedTracker(all_times[keep], all_x[keep], all_y[keep]))
    return trackers


# ---------------------------------------------------------------------------
# Diagnostic history plots
# ---------------------------------------------------------------------------

def plot_history_all(output_dirs, figpath):
    """Baryonic mass and Hamiltonian constraint norm across all segments."""
    mhd = combine_hst(output_dirs, "bns.mhd.hst")
    if mhd is not None:
        fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
        ax.plot(mhd[:, 0], mhd[:, 2])
        ax.set_xlabel("t (Msun)")
        ax.set_ylabel("Baryonic mass")
        fig.savefig(os.path.join(figpath, "bns_history_mass.png"), dpi=150)
        plt.close(fig)
        print("  Saved bns_history_mass.png")
    else:
        print("  Skipping mass history: no bns.mhd.hst found")

    z4c = combine_hst(output_dirs, "bns.z4c.user.hst")
    if z4c is not None:
        fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
        ax.plot(z4c[:, 0], z4c[:, 3])
        ax.set_xlabel("t (Msun)")
        ax.set_ylabel("H-norm2")
        ax.set_yscale("log")
        fig.savefig(os.path.join(figpath, "bns_history_hnorm2.png"), dpi=150)
        plt.close(fig)
        print("  Saved bns_history_hnorm2.png")
    else:
        print("  Skipping H-norm2: no bns.z4c.user.hst found")


def plot_user_history_all(output_dirs, figpath):
    """Max density and min lapse from bns.user.hst across all segments."""
    d = combine_hst(output_dirs, "bns.user.hst")
    if d is None:
        print("  Skipping user history: no bns.user.hst found")
        return

    time_u   = d[:, 0]
    rho_max  = d[:, 2]
    alph_min = d[:, 3]

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.plot(time_u, rho_max)
    ax.set_xlabel("t (Msun)")
    ax.set_ylabel(r"$\rho_\mathrm{max}$")
    fig.savefig(os.path.join(figpath, "bns_user_rho_max.png"), dpi=150)
    plt.close(fig)
    print("  Saved bns_user_rho_max.png")

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.plot(time_u, alph_min)
    ax.set_xlabel("t (Msun)")
    ax.set_ylabel(r"$\alpha_\mathrm{min}$")
    fig.savefig(os.path.join(figpath, "bns_user_alpha_min.png"), dpi=150)
    plt.close(fig)
    print("  Saved bns_user_alpha_min.png")


def plot_tracker_trajectories_all(output_dirs, figpath,
                                  coord_scale=1.0, coord_label=r"\mathrm{M}_\odot"):
    """Plot NS trajectories (raw) from concatenated tracker data."""
    trackers = load_trackers_all(output_dirs)
    if all(t is None for t in trackers):
        print("  No tracker files found, skipping trajectory plot")
        return

    colors  = ["tab:red",  "tab:blue"]
    colors2 = ["salmon",   "skyblue"]
    labels  = ["NS 1", "NS 2"]

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    for tracker, color, color2, label in zip(trackers, colors, colors2, labels):
        if tracker is None:
            continue
        ax.plot(tracker.x * coord_scale, tracker.y * coord_scale,
                ".", color=color2, ms=2, label=f"{label} raw")
        try:
            xs, ys = tracker.smooth_walk()
            ax.plot(xs * coord_scale, ys * coord_scale,
                    "-", color=color, lw=1, label=f"{label} smoothed")
        except Exception:
            pass

    ax.set_xlabel(rf"$x~[{coord_label}]$")
    ax.set_ylabel(rf"$y~[{coord_label}]$")
    ax.set_aspect("equal")
    ax.legend()
    fig.savefig(os.path.join(figpath, "bns_tracker_trajectories.png"), dpi=150)
    plt.close(fig)
    print("  Saved bns_tracker_trajectories.png")


def _parse_run_speed_log(logfile):
    """
    Parse one AthenaK log file.
    Returns (sim_time_mid_array, speed_array) or (None, None).
    Elapsed time in each log resets to 0 at the start of that segment,
    so each log is processed independently.
    """
    if not os.path.exists(logfile):
        return None, None

    pattern = re.compile(
        r"elapsed=([0-9.eE+-]+)\s+cycle=([0-9]+)\s+time=([0-9.eE+-]+)"
    )
    elapsed, cycle, sim_time = [], [], []
    with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                elapsed.append(float(m.group(1)))
                cycle.append(int(m.group(2)))
                sim_time.append(float(m.group(3)))

    if len(elapsed) < 2:
        return None, None

    speed, sim_time_mid = [], []
    for i in range(1, len(elapsed)):
        dt_wall = elapsed[i] - elapsed[i - 1]
        dcycle  = cycle[i] - cycle[i - 1]
        if dt_wall > 0:
            speed.append(dcycle / dt_wall)
            sim_time_mid.append(0.5 * (sim_time[i] + sim_time[i - 1]))

    if not speed:
        return None, None
    return np.array(sim_time_mid), np.array(speed)


def plot_run_speed_all(output_dirs, figpath):
    """Plot cycles/second vs simulation time, stitched across all segments."""
    all_times, all_speeds = [], []
    for d in output_dirs:
        log_files = sorted(glob.glob(os.path.join(d, "*.out")))
        if not log_files:
            print(f"  Warning: no *.out files found in {d}")
            continue
        for log_file in log_files:
            t, s = _parse_run_speed_log(log_file)
            if t is not None:
                all_times.append(t)
                all_speeds.append(s)

    if not all_times:
        print("  No log data found for run speed")
        return

    sim_time = np.concatenate(all_times)
    speed    = np.concatenate(all_speeds)
    sort_idx = np.argsort(sim_time)
    sim_time = sim_time[sort_idx]
    speed    = speed[sort_idx]

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.plot(sim_time, speed)
    ax.set_xlabel("Simulation time (Msun)")
    ax.set_ylabel("Cycles / wall-second")
    fig.savefig(os.path.join(figpath, "bns_run_speed.png"), dpi=150)
    plt.close(fig)
    print("  Saved bns_run_speed.png")


# ---------------------------------------------------------------------------
# Per-frame workers (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _bmag_from_components(bx, by, bz):
    """Flat (Euclidean) magnetic field magnitude |B| = sqrt(bx^2 + by^2 + bz^2)."""
    return np.sqrt(bx * bx + by * by + bz * bz)


def _render_frame_worker_all(kw):
    """Per-frame plot worker for combined multi-segment data."""
    global_idx      = kw['global_idx']
    xy_fpath        = kw['xy_fpath']
    xz_fpath        = kw['xz_fpath']   # None if not available
    var_name        = kw['var_name']
    plot_kwargs     = kw['plot_kwargs']
    xlim            = kw['xlim']
    ylim            = kw['ylim']
    zlim            = kw['zlim']
    label           = kw['label']
    out_prefix      = kw['out_prefix']
    figpath         = kw['figpath']
    print_minmax    = kw['print_minmax']
    show_refinement = kw['show_refinement']
    show_trackers   = kw['show_trackers']
    trackers        = kw['trackers']
    use_abs         = kw.get('use_abs', False)
    derived         = kw.get('derived', None)  # {'name', 'func', 'inputs'} or None
    coord_scale     = kw.get('coord_scale', 1.0)
    coord_label     = kw.get('coord_label', r"\mathrm{M}_\odot")

    # Convert the (M_sun) plot limits to the displayed spatial units.
    if xlim is not None: xlim = tuple(v * coord_scale for v in xlim)
    if ylim is not None: ylim = tuple(v * coord_scale for v in ylim)
    if zlim is not None: zlim = tuple(v * coord_scale for v in zlim)

    try:
        data_xy = BinaryData(xy_fpath)
    except FileNotFoundError:
        print(f"  Missing {xy_fpath}, skipping")
        return

    data_xz = None
    if xz_fpath is not None:
        try:
            data_xz = BinaryData(xz_fpath)
        except FileNotFoundError:
            pass

    if derived is not None:
        data_xy.register_derived_variable(derived['name'], derived['func'],
                                          *derived['inputs'])
        if data_xz is not None:
            data_xz.register_derived_variable(derived['name'], derived['func'],
                                              *derived['inputs'])
        var_name = derived['name']

    if use_abs:
        abs_var = f"derived:abs_{var_name}"
        data_xy.register_derived_variable(abs_var, np.abs, var_name)
        if data_xz is not None:
            data_xz.register_derived_variable(abs_var, np.abs, var_name)
        var_name = abs_var

    if print_minmax:
        print_stats(data_xy, var_name)

    if data_xz is not None:
        fig = plt.figure(figsize=(4.5, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.03, right=0.83)
        ax_xz = fig.add_subplot(gs[0])
        ax_xy = fig.add_subplot(gs[1], sharex=ax_xz)

        pcm = data_xz.plot_slice(var_name, ax=ax_xz, **plot_kwargs)
        data_xy.plot_slice(var_name, ax=ax_xy, **plot_kwargs)
        rescale_image_extents(ax_xz, coord_scale)
        rescale_image_extents(ax_xy, coord_scale)

        if show_refinement:
            draw_refinement_grid(ax_xz, data_xz, coord_scale=coord_scale)
            draw_refinement_grid(ax_xy, data_xy, coord_scale=coord_scale)
        if show_trackers and trackers is not None:
            draw_trackers(ax_xy, trackers, data_xy.time, coord_scale=coord_scale)

        if xlim is not None: ax_xz.set_xlim(*xlim); ax_xy.set_xlim(*xlim)
        ax_xz.set_ylim(*zlim)
        if ylim is not None: ax_xy.set_ylim(*ylim)
        plt.setp(ax_xz.get_xticklabels(), visible=False)
        ax_xz.set_xlabel(""); ax_xz.set_ylabel(rf"$z~[{coord_label}]$")
        ax_xy.set_xlabel(rf"$x~[{coord_label}]$")
        ax_xy.set_ylabel(rf"$y~[{coord_label}]$")
        ax_xz.set_title(f"$t = {data_xy.time:.2f}~\\mathrm{{M}}_\\odot$")
        if ylim is not None and coord_scale == 1.0:
            ax_xz.yaxis.set_major_locator(mpl.ticker.FixedLocator([0, 20, 40, 60]))
            ax_xy.yaxis.set_major_locator(mpl.ticker.FixedLocator([-60, -40, -20, 0, 20, 40]))
        else:
            ax_xy.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune='upper'))
        fig.align_ylabels([ax_xz, ax_xy])
        p_top = ax_xz.get_position(); p_bot = ax_xy.get_position()
        cax = fig.add_axes([p_top.x1 + 0.015, p_bot.y0, 0.04, p_top.y1 - p_bot.y0])
        cb = fig.colorbar(pcm, cax=cax)
        cb.ax.yaxis.set_label_position('right')
        cb.set_label(label or var_name)
    else:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        pcm = data_xy.plot_slice(var_name, ax=ax, **plot_kwargs)
        rescale_image_extents(ax, coord_scale)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        fig.colorbar(pcm, cax=cax, label=label or var_name)

        if show_refinement:
            draw_refinement_grid(ax, data_xy, coord_scale=coord_scale)
        if show_trackers and trackers is not None:
            draw_trackers(ax, trackers, data_xy.time, coord_scale=coord_scale)

        if xlim is not None: ax.set_xlim(*xlim)
        if ylim is not None: ax.set_ylim(*ylim)
        ax.set_xlabel(rf"$x~[{coord_label}]$")
        ax.set_ylabel(rf"$y~[{coord_label}]$")
        ax.set_title(f"$t = {data_xy.time:.2f}~\\mathrm{{M}}_\\odot$")
        fig.tight_layout()

    fig.savefig(os.path.join(figpath, f"{out_prefix}_{global_idx:05d}.png"), dpi=200,
                bbox_inches='tight')
    plt.close(fig)


def render_frames_all(output_dirs, figpath, file_prefix, var_name,
                      cmap, norm=None, vmin=None, vmax=None,
                      label=None, xlim=(-60, 60), ylim=(-60, 60),
                      step=1, out_prefix=None,
                      print_minmax=False,
                      show_trackers=True, show_refinement=True,
                      trackers=None,
                      scale_factor=1.0,
                      skip_existing=False,
                      use_abs=False,
                      derived=None,
                      coord_scale=1.0,
                      coord_label=r"\mathrm{M}_\odot"):
    """
    Render one PNG per deduplicated frame across all segments, in parallel.
    XZ prefix is auto-detected via '_xy' -> '_xz' substitution in file_prefix.

    'derived', if given, is {'name', 'func', 'inputs'}: a derived variable
    (name must start with 'derived:') registered per frame from the listed
    input variables before plotting (e.g. |B| from bcc1/bcc2/bcc3).
    """
    xz_prefix = file_prefix.replace("_xy", "_xz") if "_xy" in file_prefix else None
    frames = collect_combined_frames(output_dirs, file_prefix, xz_prefix, step)

    if not frames:
        print(f"  No files found for prefix '{file_prefix}'")
        return

    if out_prefix is None:
        out_prefix = f"bns_frame_{var_name.replace(':', '_')}"

    if skip_existing:
        frames = [(gi, xy, xz) for gi, xy, xz in frames
                  if not os.path.exists(os.path.join(figpath, f"{out_prefix}_{gi:05d}.png"))]
        if not frames:
            print(f"  All frames already exist, skipping '{var_name}'")
            return
        print(f"  Skipping existing, rendering {len(frames)} new frame(s)")

    zlim = (0, ylim[1]) if ylim is not None else (0, None)
    plot_kwargs = dict(cmap=cmap, interpolation="nearest", rescale=scale_factor)
    if norm is not None: plot_kwargs["norm"] = norm
    if vmin is not None: plot_kwargs["vmin"] = vmin
    if vmax is not None: plot_kwargs["vmax"] = vmax

    worker_args = [
        dict(global_idx=global_idx, xy_fpath=xy_fpath, xz_fpath=xz_fpath,
             var_name=var_name, plot_kwargs=plot_kwargs,
             xlim=xlim, ylim=ylim, zlim=zlim,
             label=label, out_prefix=out_prefix, figpath=figpath,
             print_minmax=print_minmax,
             show_refinement=show_refinement,
             show_trackers=show_trackers, trackers=trackers,
             use_abs=use_abs, derived=derived,
             coord_scale=coord_scale, coord_label=coord_label)
        for global_idx, xy_fpath, xz_fpath in frames
    ]

    n_workers = min(multiprocessing.cpu_count(), len(worker_args))
    with multiprocessing.Pool(n_workers) as pool:
        pool.map(_render_frame_worker_all, worker_args)

    print(f"  Rendered {len(frames)} frames for '{var_name}'")


def _calc_average_energy(rE, rN):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rN > 0, rE / rN, 0.0)


def _calc_average_energy_scaled(scale):
    """Return a version of _calc_average_energy that applies a unit scale."""
    def _fn(rE, rN):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(rN > 0, rE / rN * scale, 0.0)
    return _fn


def _render_avg_energy_worker_all(kw):
    """Per-frame avg_energy worker for combined multi-segment data."""
    global_idx      = kw['global_idx']
    xy_n_fpath      = kw['xy_n_fpath']
    xy_e_fpath      = kw['xy_e_fpath']
    xz_n_fpath      = kw['xz_n_fpath']   # None if not available
    xz_e_fpath      = kw['xz_e_fpath']
    xlim            = kw['xlim']
    ylim            = kw['ylim']
    zlim            = kw['zlim']
    figpath         = kw['figpath']
    print_minmax    = kw['print_minmax']
    show_refinement = kw['show_refinement']
    show_trackers   = kw['show_trackers']
    trackers        = kw['trackers']

    avg_energy_scale = kw.get('avg_energy_scale', 1.0)
    avg_energy_label = kw.get('avg_energy_label', "Average energy (code units)")
    avg_energy_norm  = kw.get('avg_energy_norm', LogNorm(vmin=1e-4, vmax=1e-3))
    species          = kw.get('species', 0)
    out_prefix       = kw.get('out_prefix', 'bns_frame_avg_energy')
    coord_scale      = kw.get('coord_scale', 1.0)
    coord_label      = kw.get('coord_label', r"\mathrm{M}_\odot")

    if xlim is not None: xlim = tuple(v * coord_scale for v in xlim)
    if ylim is not None: ylim = tuple(v * coord_scale for v in ylim)
    if zlim is not None: zlim = tuple(v * coord_scale for v in zlim)

    try:
        data_xy = GroupData([xy_n_fpath, xy_e_fpath])
    except FileNotFoundError as err:
        print(f"  Missing file: {err}, skipping frame {global_idx}")
        return

    e_var = f"E:{species}"
    n_var = f"N:{species}"
    _avg_fn = _calc_average_energy_scaled(avg_energy_scale)
    data_xy.register_derived_variable("derived:avg_energy", _avg_fn, e_var, n_var)
    if print_minmax and hasattr(data_xy, 'get_block_data'):
        print_stats(data_xy, "derived:avg_energy")

    plot_kwargs = dict(norm=avg_energy_norm, interpolation="nearest")

    data_xz = None
    if xz_n_fpath is not None and xz_e_fpath is not None:
        try:
            data_xz = GroupData([xz_n_fpath, xz_e_fpath])
            data_xz.register_derived_variable("derived:avg_energy",
                                              _avg_fn, e_var, n_var)
        except FileNotFoundError:
            data_xz = None

    if data_xz is not None:
        fig = plt.figure(figsize=(4.5, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.03, right=0.83)
        ax_xz = fig.add_subplot(gs[0])
        ax_xy = fig.add_subplot(gs[1], sharex=ax_xz)

        pcm = data_xz.plot_slice("derived:avg_energy", ax=ax_xz, **plot_kwargs)
        data_xy.plot_slice("derived:avg_energy", ax=ax_xy, **plot_kwargs)
        rescale_image_extents(ax_xz, coord_scale)
        rescale_image_extents(ax_xy, coord_scale)

        if show_refinement:
            draw_refinement_grid(ax_xz, data_xz.datasets[0], coord_scale=coord_scale)
            draw_refinement_grid(ax_xy, data_xy.datasets[0], coord_scale=coord_scale)
        if show_trackers and trackers is not None:
            draw_trackers(ax_xy, trackers, data_xy.datasets[0].time, coord_scale=coord_scale)

        if xlim is not None: ax_xz.set_xlim(*xlim); ax_xy.set_xlim(*xlim)
        ax_xz.set_ylim(*zlim)
        if ylim is not None: ax_xy.set_ylim(*ylim)
        plt.setp(ax_xz.get_xticklabels(), visible=False)
        ax_xz.set_xlabel(""); ax_xz.set_ylabel(rf"$z~[{coord_label}]$")
        ax_xy.set_xlabel(rf"$x~[{coord_label}]$")
        ax_xy.set_ylabel(rf"$y~[{coord_label}]$")
        ax_xz.set_title(f"$t = {data_xy.datasets[0].time:.2f}~\\mathrm{{M}}_\\odot$")
        if ylim is not None and coord_scale == 1.0:
            ax_xz.yaxis.set_major_locator(mpl.ticker.FixedLocator([0, 20, 40, 60]))
            ax_xy.yaxis.set_major_locator(mpl.ticker.FixedLocator([-60, -40, -20, 0, 20, 40]))
        else:
            ax_xy.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune='upper'))
        fig.align_ylabels([ax_xz, ax_xy])
        p_top = ax_xz.get_position(); p_bot = ax_xy.get_position()
        cax = fig.add_axes([p_top.x1 + 0.015, p_bot.y0, 0.04, p_top.y1 - p_bot.y0])
        cb = fig.colorbar(pcm, cax=cax)
        cb.ax.yaxis.set_label_position('right')
        cb.set_label(avg_energy_label)
    else:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        pcm = data_xy.plot_slice("derived:avg_energy", ax=ax, **plot_kwargs)
        rescale_image_extents(ax, coord_scale)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        fig.colorbar(pcm, cax=cax, label=avg_energy_label)

        if show_refinement:
            draw_refinement_grid(ax, data_xy.datasets[0], coord_scale=coord_scale)
        if show_trackers and trackers is not None:
            draw_trackers(ax, trackers, data_xy.datasets[0].time, coord_scale=coord_scale)

        if xlim is not None: ax.set_xlim(*xlim)
        if ylim is not None: ax.set_ylim(*ylim)
        ax.set_xlabel(rf"$x~[{coord_label}]$")
        ax.set_ylabel(rf"$y~[{coord_label}]$")
        ax.set_title(f"$t = {data_xy.datasets[0].time:.2f}~\\mathrm{{M}}_\\odot$")
        fig.tight_layout()

    fig.savefig(os.path.join(figpath, f"{out_prefix}_{global_idx:05d}.png"), dpi=200,
                bbox_inches='tight')
    plt.close(fig)


def render_avg_energy_all(output_dirs, figpath, print_minmax=False,
                          show_trackers=True, show_refinement=True,
                          trackers=None, xlim=(-60, 60), ylim=(-60, 60),
                          units='code', species=0,
                          out_prefix='bns_frame_avg_energy',
                          var_key='avg_energy',
                          coord_scale=1.0,
                          coord_label=r"\mathrm{M}_\odot"):
    """
    Derived average neutrino energy = E:s / N:s, rendered in parallel across
    all segments.  Frames are deduplicated by simulation time.
    """
    _sc, _lb, _log, _lo, _hi = au.plot_params(var_key, units)
    _norm = LogNorm(vmin=_lo, vmax=_hi) if _log else Normalize(vmin=_lo, vmax=_hi)
    n_prefix = _detect_prefix_all(output_dirs, "bns.rad_N_xy", "bns.rad_m1_N")
    e_prefix = _detect_prefix_all(output_dirs, "bns.rad_E_xy", "bns.rad_m1_E")

    t_to_n = _collect_frames_by_time(output_dirs, n_prefix)
    t_to_e = _collect_frames_by_time(output_dirs, e_prefix)

    if not t_to_n or not t_to_e:
        print("  No N/E files found for avg_energy")
        return

    common_times = sorted(set(t_to_n.keys()) & set(t_to_e.keys()))
    if not common_times:
        print("  No matching N/E frame pairs found for avg_energy")
        return

    zlim = (0, ylim[1]) if ylim is not None else (0, None)
    worker_args = [
        dict(global_idx=idx,
             xy_n_fpath=t_to_n[t], xy_e_fpath=t_to_e[t],
             xz_n_fpath=None, xz_e_fpath=None,
             xlim=xlim, ylim=ylim, zlim=zlim, figpath=figpath,
             print_minmax=print_minmax,
             show_refinement=show_refinement,
             show_trackers=show_trackers, trackers=trackers,
             avg_energy_scale=_sc,
             avg_energy_label=_lb,
             avg_energy_norm=_norm,
             species=species,
             out_prefix=out_prefix,
             coord_scale=coord_scale, coord_label=coord_label)
        for idx, t in enumerate(common_times)
    ]

    n_workers = min(multiprocessing.cpu_count(), len(worker_args))
    with multiprocessing.Pool(n_workers) as pool:
        pool.map(_render_avg_energy_worker_all, worker_args)

    print(f"  Rendered {len(common_times)} {out_prefix} frames")


def _compute_s00_avg_worker_fpath(fpath):
    """Worker: load one frame by path and return (time, area-weighted mean of s_00)."""
    try:
        data = BinaryData(fpath)
    except FileNotFoundError:
        print(f"  Missing {fpath}, skipping")
        return None

    total_sum  = 0.0
    total_area = 0.0
    for block, arr in data.get_block_data("s_00"):
        ext = block.get_extent()
        if len(ext) < 4:
            continue
        xmin, xmax, ymin, ymax = ext[0], ext[1], ext[2], ext[3]
        block_area  = (xmax - xmin) * (ymax - ymin)
        total_sum  += np.mean(arr) * block_area
        total_area += block_area

    if total_area == 0:
        return None
    return (data.time, total_sum / total_area)


def plot_s00_timeseries_all(output_dirs, figpath, file_prefix="bns.prim_xy"):
    """Compute and plot the area-weighted average Y_e (s_00) vs time across all segments.

    Collects all bns.prim_xy frames across output-XXXX subdirs (deduplicated by
    simulation time), computes the 2-D slice area-weighted mean of s_00 for each
    frame in parallel, then saves a 1-D time-series plot.
    """
    time_to_fpath = _collect_frames_by_time(output_dirs, file_prefix)
    if not time_to_fpath:
        print(f"  No files found for prefix '{file_prefix}' across segments")
        return

    sorted_fpaths = [time_to_fpath[t] for t in sorted(time_to_fpath.keys())]

    n_workers = min(multiprocessing.cpu_count(), len(sorted_fpaths))
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(_compute_s00_avg_worker_fpath, sorted_fpaths)

    results = sorted((r for r in results if r is not None), key=lambda x: x[0])
    if not results:
        print("  No s_00 data found")
        return

    times  = np.array([r[0] for r in results])
    avg_ye = np.array([r[1] for r in results])

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.plot(times, avg_ye)
    ax.set_xlabel("t (Msun)")
    ax.set_ylabel(r"$\langle Y_e \rangle$")
    fig.savefig(os.path.join(figpath, "bns_avg_s00_timeseries.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved bns_avg_s00_timeseries.png ({len(times)} frames)")


def _compute_bmax_worker_fpath(fpath):
    """Worker: load one 3D frame and return (time, bmax, x, y, z) in code units.

    Scans every block for the global maximum |B| = sqrt(bcc1^2+bcc2^2+bcc3^2)
    and records the cell-center coordinates where that maximum occurs.
    """
    try:
        data = BinaryData(fpath)
    except FileNotFoundError:
        print(f"  Missing {fpath}, skipping")
        return None

    best_b = -np.inf
    best_xyz = (np.nan, np.nan, np.nan)
    for block, (bx, by, bz) in data.get_block_data(("bcc1", "bcc2", "bcc3")):
        bmag = _bmag_from_components(bx, by, bz)
        idx = np.unravel_index(int(np.argmax(bmag)), bmag.shape)  # (kz, ky, kx)
        bval = float(bmag[idx])
        if bval > best_b:
            best_b = bval
            coords = block.get_coords()
            if len(coords) == 3:
                xc, yc, zc = coords
                kz, ky, kx = idx
                best_xyz = (float(xc[kx]), float(yc[ky]), float(zc[kz]))
            else:
                best_xyz = (np.nan, np.nan, np.nan)

    if not np.isfinite(best_b):
        return None
    return (data.time, best_b, best_xyz[0], best_xyz[1], best_xyz[2])


def plot_bmax_timeseries_all(output_dirs, figpath, units='cgs',
                             file_prefix="bns.prim3D"):
    """Track the global maximum |B| and its location vs time across all segments.

    Reads the 3-D primitive files (which carry bcc1/bcc2/bcc3), finds the global
    max |B| per frame and the cell where it occurs, writes bns_bmax_vs_time.dat,
    and saves a Bmax(t) plot.  Honors --units: 'code' keeps M_sun/code units,
    'cgs'/'ngs' report t [ms], |B| [G], x/y/z [km].
    """
    time_to_fpath = _collect_frames_by_time(output_dirs, file_prefix)
    if not time_to_fpath:
        print(f"  No files found for prefix '{file_prefix}' across segments")
        return

    sorted_fpaths = [time_to_fpath[t] for t in sorted(time_to_fpath.keys())]

    n_workers = min(multiprocessing.cpu_count(), len(sorted_fpaths))
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(_compute_bmax_worker_fpath, sorted_fpaths)

    results = sorted((r for r in results if r is not None), key=lambda x: x[0])
    if not results:
        print("  No B-field data found")
        return

    t = np.array([r[0] for r in results])
    b = np.array([r[1] for r in results])
    x = np.array([r[2] for r in results])
    y = np.array([r[3] for r in results])
    z = np.array([r[4] for r in results])

    # Unit conversions (spatial axes follow the same code/km convention as slices)
    t_out = au.conv_time(t, units)
    b_out = au.conv_bfield(b, units)
    coord_scale, _ = spatial_units(units)
    x_out, y_out, z_out = x * coord_scale, y * coord_scale, z * coord_scale

    is_code = (units == "code")
    t_unit  = "Msun" if is_code else "ms"
    b_unit  = "code" if is_code else "G"
    xyz_unit = "Msun" if is_code else "km"

    dat_path = os.path.join(figpath, "bns_bmax_vs_time.dat")
    header = (f"t [{t_unit}]    Bmax [{b_unit}]    "
              f"x [{xyz_unit}]    y [{xyz_unit}]    z [{xyz_unit}]")
    np.savetxt(dat_path, np.column_stack([t_out, b_out, x_out, y_out, z_out]),
               header=header)
    print(f"  Saved bns_bmax_vs_time.dat ({len(t)} frames)")

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.plot(t_out, b_out)
    ax.set_xlabel(f"t ({t_unit})")
    ax.set_ylabel(rf"$|B|_\mathrm{{max}}$ ({b_unit})")
    ax.set_yscale("log")
    fig.savefig(os.path.join(figpath, "bns_bmax.png"), dpi=150)
    plt.close(fig)
    print("  Saved bns_bmax.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--simpath", required=True,
                        help="Parent simulation directory containing output-XXXX subdirs")
    parser.add_argument("--figpath", default=None,
                        help="Output figure directory (default: <simpath>/Figs)")
    parser.add_argument("--no-trackers", dest="no_trackers", action="store_true",
                        help="Disable tracker overlays on all 2-D slice plots")
    parser.add_argument("--full-domain", dest="full_domain", action="store_true",
                        help="Show full simulation domain instead of the default ±60 Msun window")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                        help="Skip frames whose output PNG already exists in figpath")
    parser.add_argument("--sections", nargs="+", default=["all"],
                        choices=[
                            "history", "rho_max", "trackers", "run_speed",
                            "density", "temperature", "s00", "s00_avg",
                            "bnorm", "bmax",
                            "rad_E", "rad_N", "rad_N2", "rad_abs", "rad_abs0",
                            "rad_eta0", "rad_eta1", "avg_energy",
                            "nu_energy", "nu_energy1", "nu_energy2",
                            "z4c",
                            "all",
                        ],
                        help="Which sections to run (default: all)")
    parser.add_argument("--units", default="cgs",
                        choices=["code", "cgs", "ngs"],
                        help="Unit system for physical quantities: "
                             "'code' = raw AthenaK GeometricKilometer units, "
                             "'cgs' = Gaussian CGS (g, cm, s, erg) [default], "
                             "'ngs' = bns_nurates units (nm, g, s, MeV)")
    return parser.parse_args()


def main():
    args    = parse_args()
    simpath = args.simpath
    figpath = args.figpath or os.path.join(simpath, "Figs")
    sections = set(args.sections)
    run_all  = "all" in sections
    skip_existing = args.skip_existing
    show_trackers = not args.no_trackers
    full_domain   = args.full_domain
    units         = args.units
    # Spatial axes: native M_sun for 'code', km for 'cgs'/'ngs'.
    coord_scale, coord_label = spatial_units(units)

    output_dirs = find_output_dirs(simpath)
    if not output_dirs:
        print(f"ERROR: No output-XXXX directories found under {simpath}")
        sys.exit(1)

    os.makedirs(figpath, exist_ok=True)
    print(f"SIMPATH  : {simpath}")
    print(f"FIGPATH  : {figpath}")
    print(f"Segments : {len(output_dirs)}")
    for d in output_dirs:
        print(f"  {d}")
    print(f"Sections : {', '.join(sections)}")
    print(f"Trackers : {'on' if show_trackers else 'off'}")
    print(f"Domain   : {'full' if full_domain else '±60 Msun'}")
    print(f"Units    : {units}  (spatial axes in {'M_sun' if coord_scale == 1.0 else 'km'})\n")

    def enabled(name):
        return run_all or name in sections

    # Pre-load combined trackers once (reused by all frame renders)
    trackers = load_trackers_all(output_dirs)

    # Auto-detect radiation file prefix scheme (new: bns.rad_E_xy, old: bns.rad_m1_E)
    _RAD_E = _detect_prefix_all(output_dirs, "bns.rad_E_xy",   "bns.rad_m1_E")
    _RAD_N = _detect_prefix_all(output_dirs, "bns.rad_N_xy",   "bns.rad_m1_N")
    _ABS_0 = _detect_prefix_all(output_dirs, "bns.abs_0_xy",   "bns.rad_m1_abs_0")
    _ABS_1 = _detect_prefix_all(output_dirs, "bns.abs_1_xy",   "bns.rad_m1_abs_1")
    _ETA_0 = _detect_prefix_all(output_dirs, "bns.eta_0_xy",   "bns.rad_m1_eta_0")
    _ETA_1 = _detect_prefix_all(output_dirs, "bns.eta_1_xy",   "bns.rad_m1_eta_1")

    XY_LIM = None if full_domain else (-60, 60)

    if enabled("history"):
        print("[history] Plotting mass & H-norm2...")
        plot_history_all(output_dirs, figpath)

    if enabled("rho_max"):
        print("[rho_max] Plotting rho_max & alpha_min...")
        plot_user_history_all(output_dirs, figpath)

    if enabled("trackers"):
        print("[trackers] Plotting NS trajectories...")
        plot_tracker_trajectories_all(output_dirs, figpath,
                                      coord_scale=coord_scale, coord_label=coord_label)

    if enabled("run_speed"):
        print("[run_speed] Plotting run speed from logs...")
        plot_run_speed_all(output_dirs, figpath)

    if enabled("density"):
        print("[density] Rendering density frames...")
        _sc, _lb, _log, _lo, _hi = au.plot_params('dens', units)
        render_frames_all(output_dirs, figpath,
                          file_prefix="bns.prim_xy", var_name="dens",
                          cmap="inferno",
                          norm=LogNorm(vmin=_lo, vmax=_hi) if _log else Normalize(vmin=_lo, vmax=_hi),
                          label=_lb,
                          out_prefix="bns_frame_dens",
                          xlim=XY_LIM, ylim=XY_LIM,
                          print_minmax=False,
                          show_trackers=show_trackers, show_refinement=True,
                          trackers=trackers, scale_factor=_sc,
                          skip_existing=skip_existing,
                          coord_scale=coord_scale, coord_label=coord_label)

    if enabled("temperature"):
        print("[temperature] Rendering temperature frames...")
        _sc, _lb, _log, _lo, _hi = au.plot_params('temperature', units)
        render_frames_all(output_dirs, figpath,
                          file_prefix="bns.prim_xy", var_name="temperature",
                          cmap="viridis", vmin=0.0, vmax=70.0,
                          label=_lb,
                          out_prefix="bns_frame_temperature",
                          xlim=XY_LIM, ylim=XY_LIM,
                          print_minmax=True,
                          show_trackers=show_trackers, show_refinement=True,
                          trackers=trackers, skip_existing=skip_existing,
                          coord_scale=coord_scale, coord_label=coord_label)

    if enabled("s00"):
        print("[s00] Rendering s_00 (Ye proxy) frames...")
        _sc, _lb, _log, _lo, _hi = au.plot_params('s_00', units)
        render_frames_all(output_dirs, figpath,
                          file_prefix="bns.prim_xy", var_name="s_00",
                          cmap="plasma", vmin=0.0, vmax=0.2,
                          label=_lb, out_prefix="bns_frame_s00",
                          xlim=XY_LIM, ylim=XY_LIM,
                          print_minmax=True,
                          show_trackers=show_trackers, show_refinement=True,
                          trackers=trackers, skip_existing=skip_existing,
                          coord_scale=coord_scale, coord_label=coord_label)

    if enabled("bnorm"):
        print("[bnorm] Rendering magnetic field magnitude |B| frames...")
        _sc, _lb, _log, _lo, _hi = au.plot_params('bmag', units)
        render_frames_all(output_dirs, figpath,
                          file_prefix="bns.prim_xy", var_name="derived:bmag",
                          cmap="plasma",
                          norm=LogNorm(vmin=_lo, vmax=_hi) if _log else Normalize(vmin=_lo, vmax=_hi),
                          label=_lb, out_prefix="bns_frame_bmag",
                          xlim=XY_LIM, ylim=XY_LIM,
                          print_minmax=True,
                          show_trackers=show_trackers, show_refinement=True,
                          trackers=trackers, scale_factor=_sc,
                          skip_existing=skip_existing,
                          derived=dict(name="derived:bmag",
                                       func=_bmag_from_components,
                                       inputs=("bcc1", "bcc2", "bcc3")),
                          coord_scale=coord_scale, coord_label=coord_label)

    if enabled("bmax"):
        print("[bmax] Computing global max |B| time series...")
        plot_bmax_timeseries_all(output_dirs, figpath, units=units,
                                 file_prefix="bns.prim3D")

    if enabled("rad_E"):
        print("[rad_E] Rendering radiation energy E:0 frames...")
        _sc, _lb, _log, _lo, _hi = au.plot_params('E:0', units)
        render_frames_all(output_dirs, figpath,
                          file_prefix=_RAD_E, var_name="E:0",
                          cmap="inferno",
                          norm=LogNorm(vmin=_lo, vmax=_hi) if _log else Normalize(vmin=_lo, vmax=_hi),
                          label=_lb, out_prefix="bns_frame_rad_E0",
                          xlim=XY_LIM, ylim=XY_LIM,
                          print_minmax=True,
                          show_trackers=show_trackers, show_refinement=True,
                          trackers=trackers, scale_factor=_sc, skip_existing=skip_existing,
                          coord_scale=coord_scale, coord_label=coord_label)

    if enabled("rad_N"):
        print("[rad_N] Rendering radiation number density N:0 frames...")
        _sc, _lb, _log, _lo, _hi = au.plot_params('N:0', units)
        render_frames_all(output_dirs, figpath,
                          file_prefix=_RAD_N, var_name="N:0",
                          cmap="inferno",
                          norm=LogNorm(vmin=_lo, vmax=_hi) if _log else Normalize(vmin=_lo, vmax=_hi),
                          label=_lb, out_prefix="bns_frame_rad_N0",
                          xlim=XY_LIM, ylim=XY_LIM,
                          print_minmax=True,
                          show_trackers=show_trackers, show_refinement=True,
                          trackers=trackers, scale_factor=_sc, skip_existing=skip_existing,
                          coord_scale=coord_scale, coord_label=coord_label)

    if enabled("rad_N2"):
        print("[rad_N2] Rendering radiation number density N:2 (heavy lepton) frames...")
        _sc, _lb, _log, _lo, _hi = au.plot_params('N:2', units)
        render_frames_all(output_dirs, figpath,
                          file_prefix=_RAD_N, var_name="N:2",
                          cmap="inferno",
                          norm=LogNorm(vmin=_lo, vmax=_hi) if _log else Normalize(vmin=_lo, vmax=_hi),
                          label=_lb, out_prefix="bns_frame_rad_N2",
                          xlim=XY_LIM, ylim=XY_LIM,
                          print_minmax=True,
                          show_trackers=show_trackers, show_refinement=True,
                          trackers=trackers, scale_factor=_sc, skip_existing=skip_existing,
                          coord_scale=coord_scale, coord_label=coord_label)

    if enabled("rad_abs"):
        print("[rad_abs] Rendering absorption opacity abs_1:0 frames...")
        _sc, _lb, _log, _lo, _hi = au.plot_params('abs_1:0', units)
        render_frames_all(output_dirs, figpath,
                          file_prefix=_ABS_1, var_name="abs_1:0",
                          cmap="inferno",
                          norm=LogNorm(vmin=_lo, vmax=_hi) if _log else Normalize(vmin=_lo, vmax=_hi),
                          label=_lb,
                          out_prefix="bns_frame_abs1_0",
                          xlim=XY_LIM, ylim=XY_LIM,
                          print_minmax=True,
                          show_trackers=show_trackers, show_refinement=True,
                          trackers=trackers, scale_factor=_sc, skip_existing=skip_existing,
                          coord_scale=coord_scale, coord_label=coord_label)

    if enabled("rad_abs0"):
        print("[rad_abs0] Rendering absorption opacity abs_0:0 frames...")
        _sc, _lb, _log, _lo, _hi = au.plot_params('abs_0:0', units)
        render_frames_all(output_dirs, figpath,
                          file_prefix=_ABS_0, var_name="abs_0:0",
                          cmap="inferno",
                          norm=LogNorm(vmin=_lo, vmax=_hi) if _log else Normalize(vmin=_lo, vmax=_hi),
                          label=_lb,
                          out_prefix="bns_frame_abs0_0",
                          xlim=XY_LIM, ylim=XY_LIM,
                          print_minmax=True,
                          show_trackers=show_trackers, show_refinement=True,
                          trackers=trackers, scale_factor=_sc, skip_existing=skip_existing,
                          coord_scale=coord_scale, coord_label=coord_label)

    if enabled("rad_eta0"):
        print("[rad_eta0] Rendering emissivity eta_0:0 frames...")
        _sc, _lb, _log, _lo, _hi = au.plot_params('eta_0:0', units)
        render_frames_all(output_dirs, figpath,
                          file_prefix=_ETA_0, var_name="eta_0:0",
                          cmap="inferno",
                          norm=LogNorm(vmin=_lo, vmax=_hi) if _log else Normalize(vmin=_lo, vmax=_hi),
                          label=_lb,
                          out_prefix="bns_frame_eta0_0",
                          xlim=XY_LIM, ylim=XY_LIM,
                          print_minmax=True,
                          show_trackers=show_trackers, show_refinement=True,
                          trackers=trackers, scale_factor=_sc, skip_existing=skip_existing,
                          coord_scale=coord_scale, coord_label=coord_label)

    if enabled("rad_eta1"):
        print("[rad_eta1] Rendering emissivity eta_1:0 frames...")
        _sc, _lb, _log, _lo, _hi = au.plot_params('eta_1:0', units)
        render_frames_all(output_dirs, figpath,
                          file_prefix=_ETA_1, var_name="eta_1:0",
                          cmap="inferno",
                          norm=LogNorm(vmin=_lo, vmax=_hi) if _log else Normalize(vmin=_lo, vmax=_hi),
                          label=_lb,
                          out_prefix="bns_frame_eta1_0",
                          xlim=XY_LIM, ylim=XY_LIM,
                          print_minmax=True,
                          show_trackers=show_trackers, show_refinement=True,
                          trackers=trackers, scale_factor=_sc, skip_existing=skip_existing,
                          coord_scale=coord_scale, coord_label=coord_label)

    if enabled("z4c"):
        print("[z4c] Rendering con_H (Hamiltonian constraint) frames...")
        _sc, _lb, _log, _lo, _hi = au.plot_params('con_H', units)
        render_frames_all(output_dirs, figpath,
                          file_prefix="bns.con_xy", var_name="con_H",
                          cmap="inferno",
                          norm=LogNorm(vmin=_lo, vmax=_hi) if _log else Normalize(vmin=_lo, vmax=_hi),
                          label=_lb, out_prefix="bns_frame_con_H",
                          xlim=XY_LIM, ylim=XY_LIM,
                          print_minmax=True,
                          show_trackers=show_trackers, show_refinement=True,
                          trackers=trackers, skip_existing=skip_existing,
                          use_abs=True,
                          coord_scale=coord_scale, coord_label=coord_label)

    if enabled("s00_avg"):
        print("[s00_avg] Computing average Y_e (s_00) time series...")
        plot_s00_timeseries_all(output_dirs, figpath, file_prefix="bns.prim_xy")

    if enabled("avg_energy"):
        print("[avg_energy] Rendering average neutrino energy frames...")
        render_avg_energy_all(output_dirs, figpath, print_minmax=True,
                              show_trackers=show_trackers, show_refinement=True,
                              trackers=trackers, xlim=XY_LIM, ylim=XY_LIM,
                              units=units,
                              coord_scale=coord_scale, coord_label=coord_label)

    _NU_ENERGY_SECTIONS = [
        ("nu_energy",  0, r"$\nu_e$"),
        ("nu_energy1", 1, r"$\bar{\nu}_e$"),
        ("nu_energy2", 2, r"$\nu_x$"),
    ]
    for _sec, _sp, _sp_label in _NU_ENERGY_SECTIONS:
        if enabled(_sec):
            print(f"[{_sec}] Rendering neutrino energy E/{_sp_label} (species {_sp}) frames...")
            render_avg_energy_all(output_dirs, figpath, print_minmax=True,
                                  show_trackers=show_trackers, show_refinement=True,
                                  trackers=trackers, xlim=XY_LIM, ylim=XY_LIM,
                                  units=units, species=_sp,
                                  out_prefix=f"bns_frame_{_sec}",
                                  var_key='nu_energy',
                                  coord_scale=coord_scale, coord_label=coord_label)

    print("\nDone.")


if __name__ == "__main__":
    main()
