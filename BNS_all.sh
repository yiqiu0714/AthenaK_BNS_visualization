#!/usr/bin/env bash
# BNS_all visualization driver
# -----------------------------
# Machine-specific PATHS live in config.ini (copy it from config.example.ini):
#   cp config.example.ini config.ini   # then edit config.ini for your machine
#
# config.ini is a plain INI file (read, not executed) holding simpath and an
# optional figpath. It is gitignored, so those paths never get committed. All
# the run settings below (which diagnostics to plot, units, movie options) stay
# here in the script — edit them per run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Paths — read from config.ini
# simpath should be the parent directory that contains the output-XXXX
# restart-segment subdirectories, e.g.:
#   DD2_m1_VVLR/
#     output-0000/
#     output-0001/
#     ...
# ---------------------------------------------------------------------------
CONFIG_FILE="${SCRIPT_DIR}/config.ini"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: ${CONFIG_FILE} not found."
    echo "       Copy config.example.ini to config.ini and set 'simpath'."
    exit 1
fi

# Read one key from the [paths] section (or anywhere) of the INI file.
# Lines are parsed as data; comments (#/;) and section headers are ignored.
_cfg_get() {
    awk -v key="$1" '
        /^[[:space:]]*[#;]/ { next }              # comment line
        /^[[:space:]]*\[/   { next }              # section header
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
FIGPATH="$(_cfg_get figpath)"
if [[ -z "${SIMPATH}" ]]; then
    echo "ERROR: 'simpath' is not set in ${CONFIG_FILE}."
    exit 1
fi
: "${FIGPATH:=${SIMPATH}/Figs}"      # default: <simpath>/Figs

# ---------------------------------------------------------------------------
# Run settings — edit per run
# ---------------------------------------------------------------------------
# Set to "true" to show NS tracker markers on 2-D plots
SHOW_TRACKERS=false

# Set to "true" to skip frames whose PNG already exists in the Figs directory
SKIP_EXISTING=false

# Set to "true" to show the full simulation domain instead of the ±60 Msun window
FULL_DOMAIN=false

# Unit system for physical quantities: 'code', 'cgs', or 'ngs'  (default: cgs)
UNITS=cgs

# Movie encoding (passed to make_movies.sh)
MOVIE_FPS=10        # frames per second
MOVIE_STRIDE=10     # use every Nth frame (1 = all)

# ---------------------------------------------------------------------------
# SECTIONS — comment out any line to disable that diagnostic
# ---------------------------------------------------------------------------
SECTIONS=(
    history       # baryonic mass & Hamiltonian constraint norm vs time
    rho_max       # central density & min lapse vs time
    # trackers      # NS trajectory plot (both stars, smoothed)
    run_speed     # AthenaK cycles/sec from log file

    density       # density (dens) slice frames  [+ trackers + AMR grid]
    temperature   # temperature slice frames      [+ trackers + AMR grid]
    s00           # electron fraction (s_00) frames
    # s00_avg       # area-weighted average Y_e vs time (1-D timeseries)
    z4c
    rad_E         # radiation energy E:0 frames          [bns.rad_m1_E]
    rad_N         # radiation number density N:0 frames  [bns.rad_m1_N]
    # rad_N2        # radiation number density N:2 (heavy lepton) frames
    # rad_abs       # absorption opacity abs_1:0 frames    [bns.rad_m1_abs_1]
    # rad_abs0    # absorption opacity abs_0:0 frames    [bns.rad_m1_abs_0]
    # rad_eta0    # emissivity eta_0:0 frames            [bns.rad_m1_eta_0]
    # rad_eta1    # emissivity eta_1:0 frames            [bns.rad_m1_eta_1]
    # avg_energy  # derived average neutrino energy frames
    # nu_energy   # neutrino energy E/N species 0 (ν_e)    [MeV in cgs]
    # nu_energy1  # neutrino energy E/N species 1 (ν̄_e)    [MeV in cgs]
    # nu_energy2  # neutrino energy E/N species 2 (ν_x)    [MeV in cgs]
)
# ---------------------------------------------------------------------------

# Load required modules on HPC systems. Comment out / replace with your own
# environment activation (e.g. `conda activate myenv`) if not using `module`.
#
# Lmod's `module` is usually just a shell function defined for interactive/login
# shells, so it is NOT available in a plain `bash BNS_all.sh` (non-interactive)
# shell. Source Lmod's init script first so `module` works here too.
if ! command -v module &>/dev/null; then
    for _init in "${MODULESHOME:-/usr/share/lmod/lmod}/init/bash" \
                 /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh \
                 /etc/profile.d/00-modulepath.sh; do
        [[ -r "${_init}" ]] && source "${_init}" && break
    done
fi
if command -v module &>/dev/null; then
    module load python py-matplotlib py-numpy py-scipy
    module load ffmpeg          # provides ffmpeg for make_movies.sh
else
    echo "WARNING: 'module' command not found; relying on the current environment."
    echo "         Make sure python/matplotlib/numpy/scipy and ffmpeg are available."
fi

ARGS=(--simpath "${SIMPATH}")
[[ -n "${FIGPATH:-}"  ]] && ARGS+=(--figpath  "${FIGPATH}")
[[ "${SHOW_TRACKERS}" == "false" ]] && ARGS+=(--no-trackers)
[[ "${SKIP_EXISTING}" == "true"  ]] && ARGS+=(--skip-existing)
[[ "${FULL_DOMAIN}"   == "true"  ]] && ARGS+=(--full-domain)
ARGS+=(--units "${UNITS}")
ARGS+=(--sections "${SECTIONS[@]}")

python3 "${SCRIPT_DIR}/BNS_all.py" "${ARGS[@]}"

# ---------------------------------------------------------------------------
# Movie generation — automatically create MP4s from rendered frames
# ---------------------------------------------------------------------------
echo ""
echo "=== Making movies from ${FIGPATH} ==="
bash "${SCRIPT_DIR}/make_movies.sh" "${MOVIE_FPS}" "${MOVIE_STRIDE}" "${FIGPATH}"
