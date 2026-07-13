#!/usr/bin/env bash
# =============================================================================
# run_comparison.sh  —  overlay neutrino diagnostics from 2+ models
#
# Wraps comparison_plot.py.  Reuses the .txt outputs already produced by
# run_analysis.sh in each model's analysis directory (no data regeneration).
#
# Usage:
#   bash run_comparison.sh                 # uses the MODELS list below
#   bash run_comparison.sh DIR1 LABEL1 DIR2 LABEL2 [...]   # override on CLI
#
# Edit the MODELS list below (one "analysis_dir|label" per model), then run.
# First model is drawn solid, second dashed, third dotted, fourth dash-dot.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS  ← edit per run
# ─────────────────────────────────────────────────────────────────────────────

# Models to overlay: one "analysis_dir|legend label" per line (2 to 4 models).
# Each analysis_dir must contain the merger_time.txt / Lnu_E_*.txt / Eav_*.txt
# produced by run_analysis.sh.
MODELS=(
    "${SCRATCH}/athenak/nurates_base_noeq_VVLR/analysis|base"
    "${SCRATCH}/athenak/nurates_nucl_noeq_VVLR/analysis|nucl"
    "${SCRATCH}/athenak/nurates_dU_noeq_VVLR/analysis|dU"
    "${SCRATCH}/athenak/nurates_WM_noeq_VVLR/analysis|WM"
)

# Output figure filename.
OUT="fig_neutrino_comparison.png"

# Where to write the figure (empty = <first analysis dir>/comparison_plot/).
OUTDIR="${SCRATCH}/athenak/comparisons"

# Override x-range [ms] about merger, e.g. XLIM=(-2 20).  Empty = model overlap.
XLIM=()

# ─────────────────────────────────────────────────────────────────────────────
# Allow overriding the MODELS list from the command line (DIR LABEL pairs).
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# SETUP  (same Lmod handling as run_analysis.sh)
# ─────────────────────────────────────────────────────────────────────────────
#if ! command -v module &>/dev/null; then
#    for _init in "${MODULESHOME:-/usr/share/lmod/lmod}/init/bash" \
#                 /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh; do
#        [[ -r "${_init}" ]] && source "${_init}" && break
#    done
#fi
#if command -v module &>/dev/null; then
#    module load python py-numpy py-matplotlib
#fi

source /home/mu76rom/envs/conda.start

# ─────────────────────────────────────────────────────────────────────────────
# Build comparison_plot.py args from MODELS
# ─────────────────────────────────────────────────────────────────────────────
ARGS=()
for entry in "${MODELS[@]}"; do
    dir="${entry%%|*}"
    label="${entry#*|}"
    if [[ ! -d "${dir}" ]]; then
        echo "ERROR: analysis directory not found: ${dir}"; exit 1
    fi
    ARGS+=(--dir "${dir}" --label "${label}")
done

ARGS+=(--out "${OUT}")
[[ -n "${OUTDIR}" ]] && ARGS+=(--outdir "${OUTDIR}")
[[ ${#XLIM[@]} -eq 2 ]] && ARGS+=(--xlim-ms "${XLIM[0]}" "${XLIM[1]}")

echo "============================================================"
echo "  BNS neutrino comparison plot"
echo "  Models : ${#MODELS[@]}"
for entry in "${MODELS[@]}"; do echo "    - ${entry%%|*}  (${entry#*|})"; done
echo "============================================================"

python3 "${SCRIPT_DIR}/comparison_plot.py" "${ARGS[@]}"
