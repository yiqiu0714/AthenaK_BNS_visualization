#!/usr/bin/env bash
# BNS_grid — model / diagnostic / time comparison grids  (2-D slices)
# -------------------------------------------------------------------
# Builds ONE figure holding many 2-D slice panels, so several models and
# several diagnostics can be compared side by side instead of flipping
# between PNGs from BNS_all.sh.
#
# The figure has two axes and there are three things to spend them on —
# model, diagnostic and time — so ROWS and COLS each take one, and whatever
# is left over is swept, writing one figure per value.  That covers:
#
#   ROWS=model      COLS=diagnostic   # models x diagnostics   (FRAMES=one)
#   ROWS=diagnostic COLS=time         # one model over time    (MODELS=one)
#   ROWS=model      COLS=time         # one diagnostic, models over time
#
# Machine-specific paths come from config.ini, same as the other drivers,
# but MODELS below is explicit because a comparison spans several runs.
#
# Usage:
#   bash BNS_grid.sh                          # use the settings below
#   bash BNS_grid.sh DIR1 LABEL1 DIR2 LABEL2  # override MODELS on the CLI
#   DRY_RUN=true bash BNS_grid.sh             # print the command, render nothing

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRID_DIR="${SCRIPT_DIR}/plot_2d_grid"
source "${SCRIPT_DIR}/load_config.sh"

PYTHONPATH_EXTRA="$(_cfg_get pythonpath_extra)"
export PYTHONPATH="${PYTHONPATH_EXTRA:+${PYTHONPATH_EXTRA}:}${SCRIPT_DIR}/plot-tools:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python3.12}"
command -v "${PYTHON}" &>/dev/null || PYTHON=python3

# ======================== models to compare ============
# One "simpath|label" per model.  The label is what appears on the figure.
# A single entry is fine — that is the "one model over time" case.
MODELS=(
    "/lus/flare/projects/CompactBinaryMerger/yiqiu/Large_sim/SFHo_highres/184_mb32_n64|\$dx=184\$ m"
    "/lus/flare/projects/CompactBinaryMerger/yiqiu/Large_sim/SFHo_highres/92_mb32_n128|\$dx=92\$ m"
    "/lus/flare/projects/CompactBinaryMerger/yiqiu/Large_sim/SFHo_highres/46_mb32_n256|\$dx=46\$ m"
)

# ======================== what to show =================
# Any of: density temperature s00 bnorm z4c
#         rad_E rad_N rad_N2 rad_abs rad_abs0 rad_eta0 rad_eta1
DIAGNOSTICS=(bnorm)

# WHEN to sample.  Set TIMES to compare runs at the same physical time: each
# model uses its nearest frame, which is the only honest way to compare runs
# whose output cadences differ (at t=4000 the three runs here sit at frames
# 799 / 794 / 796).  Leave TIMES empty and set FRAMES to pick by index instead;
# index N is exact but run-relative.  Set one or the other, not both.
TIMES=(4000)           # simulation times in M_sun
FRAMES=()              # frame indices; -1 is the last frame

# ======================== layout =======================
ROWS=diagnostic        # model | diagnostic | time
COLS=model             # model | diagnostic | time   (must differ from ROWS)

# Where the shared colorbars go.  "auto" follows the layout, which is the only
# placement that cannot collide: the diagnostic decides what a colorbar spans,
# so it is per-column (-> bottom) when COLS=diagnostic, per-row (-> right) when
# ROWS=diagnostic, and one for the figure otherwise.
COLORBAR=auto          # auto | right | bottom | left | top | none

# Sharing hides BOTH the tick labels and the axis label on interior panels, so
# only the bottom row carries x and only the left column carries y.  That is
# where the space saving comes from; set false to label every panel.
SHAREX=true
SHAREY=true

GRID=true              # overlay AMR block boundaries (draw_refinement_grid)
# Vertical gap as a fraction of panel height.  The horizontal gap is twice
# this physical distance so adjacent columns stay distinct without looking
# disconnected.
SPACING=0.02

# ======================== data / units =================
UNITS=cgs              # code | cgs | ngs   (cgs/ngs put the axes in km)
PLANE=xy               # xy | xz
JOBNAME=bns
# Half-width of the view.  CROP_UNITS=display means CROP is in the units the
# axes are drawn in (km when UNITS=cgs), so CROP=50 gives x,y from -50 to 50 km.
# Set CROP_UNITS=code for M_sun, matching BNS_all.py.
CROP=50
CROP_UNITS=display     # display | code

# ======================== colour limits ================
# Defaults come from athenak_units.plot_params, so two figures made on
# different days stay comparable.  Override per diagnostic here.
declare -A CLIM
# CLIM[density]="1e8,1e15"
# CLIM[temperature]="0,50"
# |B| defaults to 1e12-1e15 G (athenak_units.plot_params).  Peak |B| in the
# SFHo_highres runs is only 2e12-5e13 G at t=4000, so that range leaves the
# panels nearly empty; uncomment to fit the data instead.
# CLIM[bnorm]="1e10,5e13"

# Per-diagnostic colormap override.  magma/inferno are perceptually uniform and
# print well; turbo shows the most mid-range structure but is not uniform.
declare -A CMAP
CMAP[bnorm]=magma

# ======================== output =======================
# Square spatial limits need square panels: imshow preserves physical aspect,
# so a wider slot only becomes invisible whitespace between columns.
PANEL_SIZE="3.6,3.6"   # per-panel inches, W,H
DPI=200
OUTDIR="$(_cfg_get figpath)"
: "${OUTDIR:=${SCRIPT_DIR}/Figs_grid}"
OUTNAME="grid_${DIAGNOSTICS[0]}_${ROWS}_x_${COLS}.png"

DRY_RUN=${DRY_RUN:-false}

# ---------------------------------------------------------------------------
# Allow overriding MODELS from the command line (DIR LABEL pairs), matching
# sph_data_analysis/run_comparison.sh.
# ---------------------------------------------------------------------------
if [[ $# -gt 0 ]]; then
    if [[ $(( $# % 2 )) -ne 0 ]]; then
        echo "ERROR: command-line args must be DIR LABEL pairs (got $# args)."
        echo "Usage: $0 [DIR1 LABEL1 DIR2 LABEL2 ...]"
        exit 1
    fi
    MODELS=()
    while [[ $# -gt 0 ]]; do
        MODELS+=("$1|$2")
        shift 2
    done
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: MODELS is empty — add at least one 'simpath|label' entry."
    exit 1
fi

# ---------------------------------------------------------------------------
# Assemble the command
# ---------------------------------------------------------------------------
CMD=("${PYTHON}" "${GRID_DIR}/plot_grid.py")
for spec in "${MODELS[@]}"; do
    CMD+=(--model "${spec}")
done
CMD+=(--diagnostics "${DIAGNOSTICS[@]}")
if [[ ${#TIMES[@]} -gt 0 && ${#FRAMES[@]} -gt 0 ]]; then
    echo "ERROR: set TIMES or FRAMES, not both."
    exit 1
elif [[ ${#TIMES[@]} -gt 0 ]]; then
    CMD+=(--times "${TIMES[@]}")
elif [[ ${#FRAMES[@]} -gt 0 ]]; then
    CMD+=(--frames "${FRAMES[@]}")
else
    echo "ERROR: set at least one of TIMES or FRAMES."
    exit 1
fi
CMD+=(--rows "${ROWS}" --cols "${COLS}")
CMD+=(--colorbar "${COLORBAR}")
CMD+=(--sharex "${SHAREX}" --sharey "${SHAREY}")
CMD+=(--units "${UNITS}" --plane "${PLANE}" --jobname "${JOBNAME}")
CMD+=(--panel-size "${PANEL_SIZE}" --dpi "${DPI}")
CMD+=(--grid "${GRID}" --spacing "${SPACING}")
[[ -n "${CROP}" ]]                && CMD+=(--crop "${CROP}" --crop-units "${CROP_UNITS}")
for name in "${!CLIM[@]}"; do
    CMD+=(--clim "${name}=${CLIM[$name]}")
done
for name in "${!CMAP[@]}"; do
    CMD+=(--cmap "${name}=${CMAP[$name]}")
done
CMD+=(-o "${OUTDIR}/${OUTNAME}")

echo "Models      : ${#MODELS[@]}"
echo "Diagnostics : ${DIAGNOSTICS[*]}"
if [[ ${#TIMES[@]} -gt 0 ]]; then
    echo "Times       : ${TIMES[*]} (nearest frame per model)"
else
    echo "Frames      : ${FRAMES[*]}"
fi
echo "Layout      : rows=${ROWS} cols=${COLS} colorbar=${COLORBAR}"
echo "Output      : ${OUTDIR}/${OUTNAME}"
echo ""

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "=== DRY RUN — command that would run ==="
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

mkdir -p "${OUTDIR}"
"${CMD[@]}"
