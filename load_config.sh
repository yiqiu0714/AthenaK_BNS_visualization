#!/usr/bin/env bash
# load_config.sh — shared machine-config loader for the BNS visualization scripts.
# -----------------------------------------------------------------------------
# SOURCE this file (don't execute it). After sourcing you get:
#   * _cfg_get <key>   — read a value from config.ini
#   * the machine environment set up (Lmod made available + the config's
#     `env_setup` command run, e.g. `module load python`)
#
# The point: ALL machine-specific settings live in ONE file, config.ini —
# both the paths (simpath/figpath/eos_table/pythonpath_extra) AND the
# environment activation (env_setup). Move to a new machine → edit config.ini
# only; none of the *.sh scripts need touching.
#
# config.ini itself stays a plain INI file (parsed, not executed); only the
# single `env_setup` value is run, as your explicit environment command.

# Directory of THIS file = repo root, where config.ini lives beside it.
_LC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${_LC_DIR}/config.ini}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: ${CONFIG_FILE} not found."
    echo "       Copy config.example.ini to config.ini and edit it for your machine."
    exit 1
fi

# Read one key from config.ini. Comments (#/;) and [section] headers are
# ignored; the first "key = value" match anywhere in the file wins.
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

# --- Machine environment setup ------------------------------------------------
# Lmod's `module` is usually only a shell function in interactive/login shells,
# so it is NOT available in a plain `bash script.sh`. Source Lmod's init first
# so `module` (and thus env_setup) works in non-interactive shells too.
if ! command -v module &>/dev/null; then
    for _init in "${MODULESHOME:-/usr/share/lmod/lmod}/init/bash" \
                 /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh \
                 /etc/profile.d/00-modulepath.sh; do
        [[ -r "${_init}" ]] && source "${_init}" && break
    done
fi

# Run the machine-specific environment command(s) from config.ini's env_setup.
# (`|| ...` keeps a harmless module warning from aborting `set -e` scripts.)
_ENV_SETUP="$(_cfg_get env_setup)"
if [[ -n "${_ENV_SETUP}" ]]; then
    eval "${_ENV_SETUP}" \
        || echo "WARNING: env_setup returned non-zero: ${_ENV_SETUP}" >&2
else
    echo "WARNING: 'env_setup' is empty in ${CONFIG_FILE};" \
         "relying on the current environment." >&2
fi
