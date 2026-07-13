#!/usr/bin/env bash
# =============================================================================
# run_analysis.sh  —  BNS SPH ejecta + neutrino analysis wrapper
#
# Usage:
#   bash run_analysis.sh [--ejecta] [--neutrino] [--plot] [--all]
#
# With no flags all steps (merger time, ejecta, neutrino, plot) are run.
#
# Machine-specific PATHS (simpath, eos_table, optional pythonpath_extra) are read
# from ../config.ini.  The run settings below stay in this script — edit per run.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${REPO_DIR}/config.ini"
PLOTTOOLS="${REPO_DIR}/plot-tools"

# ─────────────────────────────────────────────────────────────────────────────
# Read paths from config.ini  (plain INI: parsed, never executed)
# ─────────────────────────────────────────────────────────────────────────────
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: ${CONFIG_FILE} not found."
    echo "       Copy config.example.ini to config.ini and set 'simpath' + 'eos_table'."
    exit 1
fi

_cfg_get() {
    awk -v key="$1" '
        /^[[:space:]]*[#;]/ { next }
        /^[[:space:]]*\[/   { next }
        {
            k = $0; sub(/=.*/, "", k)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", k)
            if (k == key && index($0, "=")) {
                v = substr($0, index($0, "=") + 1)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
                print v; exit
            }
        }
    ' "${CONFIG_FILE}"
}

SIMPATH="$(_cfg_get simpath)"
EOS_TABLE="$(_cfg_get eos_table)"
PYTHONPATH_EXTRA="$(_cfg_get pythonpath_extra)"   # optional (e.g. a vtk install)

if [[ -z "${SIMPATH}" ]]; then
    echo "ERROR: 'simpath' is not set in ${CONFIG_FILE}."; exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# RUN SETTINGS  ← edit per run
# ─────────────────────────────────────────────────────────────────────────────

# Directory where all output txt/npy/pdf files will be written.
OUTPUT_DIR="${SIMPATH}/analysis"

# SPH extraction radius [M_sun] used for BOTH ejecta and neutrino analysis.
# Must have mhd/adm/z4c and rad_m1_E/F/N sph output at this radius.
RADIUS=300.0

# GW extraction radius for the merger-time estimate (rpsi4_*_RRRR.txt).
WF_RADIUS=300

# Neutrino plot x-axis: 0 = absolute time [ms]; 1 = t − t_merger [ms] (xlim left=0).
PLOT_FROM_MERGER=0

# How many ms post-merger to include in the 2D ejecta histogram.
T_POST_MS=25.0

# Density floor used in the simulation [code units].
DFLOOR=3e-15

# Number of parallel workers for the snapshot loop.
# Login node: keep ≤ 4.  Compute node (salloc/srun): set to the core count.
N_WORKERS=8

# ── auto-discover all output-*/sph directories (sorted) ──────────────────────
SPH_DIRS=()
for d in "${SIMPATH}"/output-*/sph; do
    [[ -d "$d" ]] && SPH_DIRS+=("$d")
done
if [[ ${#SPH_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: no output-*/sph directories found under ${SIMPATH}"; exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# ARG PARSING
# ─────────────────────────────────────────────────────────────────────────────
RUN_EJECTA=0; RUN_NEUTRINO=0; RUN_PLOT=0
if [[ $# -eq 0 ]]; then
    RUN_EJECTA=1; RUN_NEUTRINO=1; RUN_PLOT=1
fi
for arg in "$@"; do
    case "$arg" in
        --ejecta)   RUN_EJECTA=1   ;;
        --neutrino) RUN_NEUTRINO=1 ;;
        --plot)     RUN_PLOT=1     ;;
        --all)      RUN_EJECTA=1; RUN_NEUTRINO=1; RUN_PLOT=1 ;;
        *) echo "Unknown flag: $arg"; echo "Usage: $0 [--ejecta] [--neutrino] [--plot] [--all]"; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
# Aurora uses environment modules. Lmod's `module` is usually only a shell
# function for interactive shells, so source its init first if needed.
#if ! command -v module &>/dev/null; then
#    for _init in "${MODULESHOME:-/usr/share/lmod/lmod}/init/bash" \
#                 /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh; do
#        [[ -r "${_init}" ]] && source "${_init}" && break
#    done
#fi
#if command -v module &>/dev/null; then
#    module load python py-numpy py-scipy py-matplotlib py-h5py
#fi
## The module Python may have no pip/user-site; provide extra packages (e.g. vtk
## needed by athplot.load_sph_vtk) via pythonpath_extra in config.ini.
#if [[ -n "${PYTHONPATH_EXTRA}" ]]; then
#    export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
#fi

source /home/mu76rom/envs/conda.start

mkdir -p "${OUTPUT_DIR}"

# Build a Python list literal from SPH_DIRS, e.g. ['/a', '/b']
SPH_LIST_PY="["
for d in "${SPH_DIRS[@]}"; do SPH_LIST_PY+="'$d',"; done
SPH_LIST_PY="${SPH_LIST_PY%,}]"

echo "============================================================"
echo "  BNS SPH analysis"
echo "  SPH dirs   : ${#SPH_DIRS[@]} segment(s)"
echo "  Radius     : ${RADIUS} M_sun"
echo "  Output     : ${OUTPUT_DIR}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: MERGER TIME  (from max |psi4_22|^2 of the (2,2) GW mode)
# ─────────────────────────────────────────────────────────────────────────────
MERGER_FILE="${OUTPUT_DIR}/merger_time.txt"
echo ""
echo "── Merger time (GW (2,2) mode) ──────────────────────────────"
python3 "${SCRIPT_DIR}/merger_time.py" \
    --simpath "${SIMPATH}" --radius "${WF_RADIUS}" --out "${MERGER_FILE}"
# Read the merger time (first non-comment numeric line) from the file.
T_MERGER_MSUN="$(awk '!/^#/ && NF {print $1; exit}' "${MERGER_FILE}")"
if [[ -z "${T_MERGER_MSUN}" ]]; then
    echo "ERROR: could not read merger time from ${MERGER_FILE}"; exit 1
fi
echo "  T_merger = ${T_MERGER_MSUN} M_sun"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 + 2: EJECTA and NEUTRINO  (run concurrently — they share no state)
# ─────────────────────────────────────────────────────────────────────────────
EJECTA_PID=""; NEUTRINO_PID=""

if [[ $RUN_EJECTA -eq 1 ]]; then
    echo ""
    echo "── Ejecta analysis (background) ─────────────────────────────"
    if [[ -z "${EOS_TABLE}" ]]; then
        echo "ERROR: 'eos_table' is not set in ${CONFIG_FILE} (required for ejecta)."; exit 1
    fi
    python3 - << PYEOF &
import sys
sys.path.insert(0, '${PLOTTOOLS}')
sys.path.insert(0, '${SCRIPT_DIR}')
import analyze_ejecta as ae

ae.SPH_DIR_LIST   = ${SPH_LIST_PY}
ae.EOS_TABLE_PATH = '${EOS_TABLE}'
ae.RADIUS         = ${RADIUS}
ae.DFLOOR         = ${DFLOOR}
ae.T_MERGER_MSUN  = ${T_MERGER_MSUN}
ae.T_POST_MS      = ${T_POST_MS}
ae.N_WORKERS      = ${N_WORKERS}
MSUN_TO_S = 6.67430e-11 * 1.98892e30 / 2.99792458e8**3
ae.T_STOP  = ae.T_MERGER_MSUN + ae.T_POST_MS * 1e-3 / MSUN_TO_S
ae.OUTPUT_DIR = '${OUTPUT_DIR}'
ae.main()
PYEOF
    EJECTA_PID=$!
fi

if [[ $RUN_NEUTRINO -eq 1 ]]; then
    echo "── Neutrino analysis (background) ───────────────────────────"
    python3 - << PYEOF &
import sys
sys.path.insert(0, '${PLOTTOOLS}')
sys.path.insert(0, '${SCRIPT_DIR}')
import analyze_neutrinos as an

an.SPH_DIR_LIST = ${SPH_LIST_PY}
an.RADIUS       = ${RADIUS}
an.N_WORKERS    = ${N_WORKERS}
an.OUTPUT_DIR   = '${OUTPUT_DIR}'
an.main()
PYEOF
    NEUTRINO_PID=$!
fi

if [[ -n "${EJECTA_PID}" ]]; then
    wait "${EJECTA_PID}" && echo "── Ejecta done ──" || { echo "ERROR: ejecta failed"; exit 1; }
fi
if [[ -n "${NEUTRINO_PID}" ]]; then
    wait "${NEUTRINO_PID}" && echo "── Neutrino done ──" || { echo "ERROR: neutrino failed"; exit 1; }
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: PLOT
# ─────────────────────────────────────────────────────────────────────────────
if [[ $RUN_PLOT -eq 1 ]]; then
    echo ""
    echo "── Plotting ─────────────────────────────────────────────────"
    PLOT_ARGS=(--output-dir "${OUTPUT_DIR}" --t-merger "${T_MERGER_MSUN}" --radius "${RADIUS}")
    [[ "${PLOT_FROM_MERGER}" == "1" ]] && PLOT_ARGS+=(--from-merger)
    python3 "${SCRIPT_DIR}/plot_analysis.py" "${PLOT_ARGS[@]}"
    echo "── Plotting done ────────────────────────────────────────────"
fi

echo ""
echo "All done. Results in: ${OUTPUT_DIR}"
