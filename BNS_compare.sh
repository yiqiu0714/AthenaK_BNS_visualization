#!/usr/bin/env bash
# =============================================================================
# BNS_compare.sh  —  compare opacity / radiation slices between TWO runs
#
# Wraps BNS_compare.py.  For each variable and matched output time it renders a
# 3-panel figure  [run A] [run B] [ratio A/B]  on the xz slice, reusing the same
# units / colormaps as BNS_all.py.  Mirrors the two-folder convention of
# sph_data_analysis/run_comparison.sh.
#
# Usage:
#   bash BNS_compare.sh                          # uses RUNS / VARS below
#   bash BNS_compare.sh DIRA LABELA DIRB LABELB  # override the two runs on CLI
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS  ← edit per comparison
# ─────────────────────────────────────────────────────────────────────────────
# Exactly two runs: "run_dir|label".  Each run_dir contains output-*/bin/.
RUNS=(
    "${SCRATCH}/athenak/nurates_base_noeq_VVLR|base"
    #"${SCRATCH}/athenak/nurates_nucl_noeq_VVLR|nucl"
    #"${SCRATCH}/athenak/nurates_dU_noeq_VVLR|dU"
    "${SCRATCH}/athenak/nurates_WM_noeq_VVLR|WM"
)

# Variables to compare, as base:species  (nue=0, anue=1, nux=2, anux=3).
# bases: eta_0 eta_1 abs_0 abs_1 E N
VARS=(temperature s_00 abs_1:0 abs_1:1 abs_1:2 eta_1:0 eta_1:1 eta_1:2)

UNITS=cgs              # code | cgs | ngs   (cgs/ngs -> axes in km, opacities scaled)
XLIM=(-60 60)          # x-range [M_sun]
ZLIM=(-60 60)          # z-range [M_sun]
PLANE=xz               # xy not available for neutrino quantities
NX=400; NZ=400         # rasterization resolution of the common grid
RMAX=3.0               # ratio panel spans 1/RMAX .. RMAX  (white = equal)
EVERY=1                # use every Nth matched time
OUTDIR="${SCRATCH}/athenak/comparisons/compare_${PLANE}"    # empty = <run A>/compare/

# ─────────────────────────────────────────────────────────────────────────────
# CLI override:  DIRA LABELA DIRB LABELB
# ─────────────────────────────────────────────────────────────────────────────
if [[ $# -eq 4 ]]; then
    RUNS=("$1|$2" "$3|$4")
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [DIRA LABELA DIRB LABELB]"; exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# SETUP  (same Lmod handling as run_comparison.sh / BNS_all.sh)
# ─────────────────────────────────────────────────────────────────────────────
#if ! command -v module &>/dev/null; then
#    for _init in "${MODULESHOME:-/usr/share/lmod/lmod}/init/bash" \
#                 /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh; do
#        [[ -r "${_init}" ]] && source "${_init}" && break
#    done
#fi
#if command -v module &>/dev/null; then
#    module load python py-matplotlib py-numpy py-scipy 2>/dev/null || \
#        module load frameworks 2>/dev/null || true
#fi

source /home/mu76rom/envs/conda.start

# ─────────────────────────────────────────────────────────────────────────────
ARGS=()
for entry in "${RUNS[@]}"; do
    dir="${entry%%|*}"; label="${entry#*|}"
    [[ -d "${dir}" ]] || { echo "ERROR: run directory not found: ${dir}"; exit 1; }
    ARGS+=(--run "${dir}" --label "${label}")
done
ARGS+=(--vars "${VARS[@]}")
ARGS+=(--units "${UNITS}")
ARGS+=(--xlim "${XLIM[0]}" "${XLIM[1]}")
ARGS+=(--zlim "${ZLIM[0]}" "${ZLIM[1]}")
ARGS+=(--nx "${NX}" --nz "${NZ}" --rmax "${RMAX}" --every "${EVERY}")
ARGS+=(--plane "${PLANE}")
[[ -n "${OUTDIR}" ]] && ARGS+=(--outdir "${OUTDIR}")

echo "============================================================"
echo "  BNS opacity slice comparison (A vs B, + ratio)"
for entry in "${RUNS[@]}"; do echo "    - ${entry%%|*}  (${entry#*|})"; done
echo "  vars : ${VARS[*]}"
echo "============================================================"

python3 "${SCRIPT_DIR}/BNS_compare.py" "${ARGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# Make movies
# ─────────────────────────────────────────────────────────────────────────────

VARS_STRS=(temperature s_00 abs_1_0 abs_1_1 abs_1_2 eta_1_0 eta_1_1 eta_1_2)
FRAMERATE=30

cd ${OUTDIR}
for var in "${VARS_STRS[@]}"; do
    ffmpeg  -framerate ${FRAMERATE} \
            -pattern_type glob -i "cmp_${var}_*.png"
            -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libopenh264
            -pix_fmt yuv420p -movflags +faststart
            ${var}.mp4
done
cd ${SCRIPT_DIR}
