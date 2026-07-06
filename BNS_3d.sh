#!/usr/bin/env bash
# BNS_3d — 3D isosurface and orthoslice driver  (Aurora edition)
# --------------------------------------------------------------
# Renders the 3D prim snapshots (bns.prim3D.NNNNN.bin) found in every
# output-XXXX/ segment as Matplotlib orthoslices and/or PyVista isosurfaces,
# then stitches them into movies.
#
# Machine-specific PATHS live in config.ini (same file as BNS_all.sh):
#   cp config.example.ini config.ini   # then edit simpath / pythonpath_extra
#
# config.ini is a plain INI file (read, not executed). It is gitignored, so
# those paths never get committed. All run settings below stay in this script.
#
# Timing (256^3, crop 80, per snapshot, single core):
#   orthoslice  ~7 s       isosurface  ~17 s

set -uo pipefail

# ======================== internals (paths/env) ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT3D_DIR="${SCRIPT_DIR}/plot_3d_data"
CONFIG_FILE="${SCRIPT_DIR}/config.ini"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: ${CONFIG_FILE} not found."
    echo "       Copy config.example.ini to config.ini and set 'simpath'."
    exit 1
fi

# Read one key from the [paths] section (or anywhere) of the INI file.
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
PYTHONPATH_EXTRA="$(_cfg_get pythonpath_extra)"
if [[ -z "${SIMPATH}" ]]; then
    echo "ERROR: 'simpath' is not set in ${CONFIG_FILE}."
    exit 1
fi

# 3D frames go to a separate directory so they don't collide with the 2D Figs.
FIGPATH="${SIMPATH}/Figs_3d"

# ---------------------------------------------------------------------------
# Python environment (Aurora).  Source Lmod so `module` works in this
# non-interactive shell, then load the toolchain.  pyvista / h5py / vtk /
# imageio-ffmpeg are not provided as modules — they live in pythonpath_extra
# (installed with: python3.12 -m pip install --target <pythonpath_extra> ...).
# ---------------------------------------------------------------------------
if ! command -v module &>/dev/null; then
    for _init in "${MODULESHOME:-/usr/share/lmod/lmod}/init/bash" \
                 /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh \
                 /etc/profile.d/00-modulepath.sh; do
        [[ -r "${_init}" ]] && source "${_init}" && break
    done
fi
if command -v module &>/dev/null; then
    module load python/3.12.12 py-numpy py-matplotlib py-scipy 2>/dev/null
else
    echo "WARNING: 'module' command not found; relying on the current environment."
fi

PYTHON="${PYTHON:-python3.12}"
command -v "${PYTHON}" &>/dev/null || PYTHON=python3

# pythonpath_extra (vtk/pyvista/h5py/imageio-ffmpeg) + repo plot-tools (athplot),
# prepended so they take precedence over any module-provided copies.
export PYTHONPATH="${PYTHONPATH_EXTRA:+${PYTHONPATH_EXTRA}:}${SCRIPT_DIR}/plot-tools:${PYTHONPATH:-}"

# ======================== what to plot =================
PLOTS=(
    orthoslice          # xy / xz / yz Matplotlib slices
    isosurface          # PyVista 3D nested isosurfaces
    # isosurface_colorT   # density isosurface colored by temperature
)

SCALARS=(
    log_dens            # log10(density)
    temperature
    log_bmag            # log10(|B|)  — Gauss in cgs, code-log10 in code units
    # current           # |curl B| (B-field "current" proxy)
    # vorticity         # |curl v|
    # qcriterion        # Q-criterion
)

# Overlay magnetic field lines on every isosurface: set to "B" (or "velocity"),
# empty to disable.  Adds noticeable render time.
STREAMLINES=""

# ======================== frame selection ==============
STRIDE=10              # render every Nth snapshot (1 = all)
# START=0              # uncomment to override first snapshot index
# END=80               # uncomment to override last  snapshot index

# ======================== parallelism ==================
JOBS=4                 # parallel jobs (1 = serial, 4-8 recommended on login node)

# ======================== units ========================
UNITS=cgs              # 'code' or 'cgs'  (cgs: density g/cm^3, |B| Gauss)
# UNITS=code

# ======================== grid / crop ==================
RESOLUTION=256         # output grid cells per axis
CROP=80                # half-width crop: domain -> [-CROP, CROP]^3

# ======================== isosurface settings ==========
WINDOW_SIZE="1200,1200"
CAMERA_ELEVATION=30
CAMERA_AZIMUTH=25
VIEW_ANGLE=35
BACKGROUND=black

# Per-scalar isosurface levels and color limits.
#   cgs log_dens : log10(6.178e17 * rho_code), so ~8-14 in g/cm^3.
#   cgs log_bmag : log10(|B| in Gauss), so ~9-15 (peak ~1e15 G).
#   temperature  : MeV in both code and cgs.
# NOTE: these levels assume UNITS=cgs.  For UNITS=code use the code-unit ranges
# noted in comments (e.g. log_bmag -13..-7, log_dens via 6e8..2e15 / 6.178e17).
declare -A ISO_LEVELS ISO_OPACITY ISO_CLIM ISO_CMAP
ISO_LEVELS[log_dens]="8,10,12,14"
ISO_OPACITY[log_dens]="0.15,0.25,0.4,0.8"
ISO_CLIM[log_dens]="6,15"
ISO_CMAP[log_dens]="inferno"

ISO_LEVELS[temperature]="5,15,30,45"
ISO_OPACITY[temperature]="0.15,0.3,0.5,0.8"
ISO_CLIM[temperature]="0,55"
ISO_CMAP[temperature]="viridis"

# cgs (Gauss).  For UNITS=code use ISO_LEVELS="-12,-10.5,-9,-7.5" / ISO_CLIM="-13,-7".
ISO_LEVELS[log_bmag]="12,13,14"
ISO_OPACITY[log_bmag]="0.2,0.35,0.8"
ISO_CLIM[log_bmag]="9,15"
ISO_CMAP[log_bmag]="plasma"

# current (|curl B|): levels/clim left empty -> auto percentiles per frame.
ISO_CMAP[current]="inferno"

ISO_CMAP[vorticity]="magma"
ISO_CMAP[qcriterion]="magma"

# ======================== movie settings ================
MOVIE_FPS=10               # frames per second
MOVIE_CRF=18               # H.264 quality: 0 (lossless) - 51 (worst)
# MAKE_MOVIES=false        # uncomment to skip movie generation

# ======================== skip existing ================
SKIP_EXISTING=false
# SKIP_EXISTING=true

# ======================== run banner ===================
mkdir -p "${FIGPATH}"
echo "SIMPATH : ${SIMPATH}"
echo "FIGPATH : ${FIGPATH}"
echo "PYTHON  : ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "UNITS   : ${UNITS}   SCALARS: ${SCALARS[*]}"
echo ""

contains() { local e; for e in "${@:2}"; do [[ "$e" == "$1" ]] && return 0; done; return 1; }

# ---------------------------------------------------------------------------
# Discover snapshot files across all output-XXXX segments
# ---------------------------------------------------------------------------
mapfile -t ALL_FILES < <(
    find "${SIMPATH}"/output-*/bin -name 'bns.prim3D.*.bin' 2>/dev/null | sort
)

if [[ ${#ALL_FILES[@]} -eq 0 ]]; then
    echo "ERROR: no bns.prim3D.*.bin files found under ${SIMPATH}/output-*/bin/"
    exit 1
fi
echo "Found ${#ALL_FILES[@]} prim3D snapshots under ${SIMPATH}"

# ---------------------------------------------------------------------------
# Apply stride / start / end filters
# ---------------------------------------------------------------------------
SELECTED=()
for f in "${ALL_FILES[@]}"; do
    idx=$(basename "$f" .bin | grep -oP '\d+$' | sed 's/^0*//' )
    idx=${idx:-0}
    [[ -n "${START:-}" && $idx -lt $START ]] && continue
    [[ -n "${END:-}"   && $idx -gt $END   ]] && continue
    (( idx % STRIDE == 0 )) || continue
    SELECTED+=("$f")
done
echo "Selected ${#SELECTED[@]} snapshots (stride=${STRIDE}), JOBS=${JOBS}"

# ---------------------------------------------------------------------------
# Build command list — one line per plot
# ---------------------------------------------------------------------------
CMDFILE=$(mktemp "${TMPDIR:-/tmp}/bns3d_cmds.XXXXXX")
TOTAL_QUEUED=0
SKIPPED=0

for FILE in "${SELECTED[@]}"; do
    TAG=$(basename "$FILE" .bin | grep -oP '\d+$')

    for SCALAR in "${SCALARS[@]}"; do
        CMAP="${ISO_CMAP[$SCALAR]:-inferno}"

        # --- orthoslice ---
        if contains orthoslice "${PLOTS[@]}"; then
            OUT="${FIGPATH}/ortho_${SCALAR}_${TAG}.png"
            if [[ "${SKIP_EXISTING}" == "true" && -f "$OUT" ]]; then
                (( SKIPPED++ ))
            else
                echo "${PYTHON} ${PLOT3D_DIR}/plot_orthoslices.py \
-i ${FILE} -o ${OUT} \
--scalar ${SCALAR} --resolution ${RESOLUTION} --crop ${CROP} \
--cmap ${CMAP} --dpi 200 --units ${UNITS}" >> "$CMDFILE"
                (( TOTAL_QUEUED++ ))
            fi
        fi

        # --- isosurface ---
        if contains isosurface "${PLOTS[@]}"; then
            OUT="${FIGPATH}/iso_${SCALAR}_${TAG}.png"
            if [[ "${SKIP_EXISTING}" == "true" && -f "$OUT" ]]; then
                (( SKIPPED++ ))
            else
                CMD="${PYTHON} ${PLOT3D_DIR}/plot_isosurfaces.py \
-i ${FILE} -o ${OUT} \
--scalar ${SCALAR} --resolution ${RESOLUTION} --crop ${CROP} \
--cmap ${CMAP} --off-screen \
--window-size ${WINDOW_SIZE} \
--camera-elevation ${CAMERA_ELEVATION} --camera-azimuth ${CAMERA_AZIMUTH} \
--view-angle ${VIEW_ANGLE} --background ${BACKGROUND} --show-scalar-bar \
--units ${UNITS}"
                [[ -n "${ISO_LEVELS[$SCALAR]:-}" ]]  && CMD+=" --levels=${ISO_LEVELS[$SCALAR]}"
                [[ -n "${ISO_OPACITY[$SCALAR]:-}" ]] && CMD+=" --opacity ${ISO_OPACITY[$SCALAR]}"
                [[ -n "${ISO_CLIM[$SCALAR]:-}" ]]    && CMD+=" --clim=${ISO_CLIM[$SCALAR]}"
                [[ -n "${STREAMLINES}" ]]            && CMD+=" --streamlines ${STREAMLINES} --streamline-seeds dual-density-max"
                echo "$CMD" >> "$CMDFILE"
                (( TOTAL_QUEUED++ ))
            fi
        fi

        # --- isosurface colored by temperature ---
        if contains isosurface_colorT "${PLOTS[@]}" && [[ "$SCALAR" != "temperature" ]]; then
            OUT="${FIGPATH}/iso_${SCALAR}_colorT_${TAG}.png"
            if [[ "${SKIP_EXISTING}" == "true" && -f "$OUT" ]]; then
                (( SKIPPED++ ))
            else
                CMD="${PYTHON} ${PLOT3D_DIR}/plot_isosurfaces.py \
-i ${FILE} -o ${OUT} \
--scalar ${SCALAR} --color-by temperature \
--resolution ${RESOLUTION} --crop ${CROP} \
--cmap viridis --clim=0,50 --off-screen \
--window-size ${WINDOW_SIZE} \
--camera-elevation ${CAMERA_ELEVATION} --camera-azimuth ${CAMERA_AZIMUTH} \
--view-angle ${VIEW_ANGLE} --background ${BACKGROUND} --show-scalar-bar \
--units ${UNITS}"
                [[ -n "${ISO_LEVELS[$SCALAR]:-}" ]]  && CMD+=" --levels=${ISO_LEVELS[$SCALAR]}"
                [[ -n "${ISO_OPACITY[$SCALAR]:-}" ]] && CMD+=" --opacity ${ISO_OPACITY[$SCALAR]}"
                [[ -n "${STREAMLINES}" ]]            && CMD+=" --streamlines ${STREAMLINES} --streamline-seeds dual-density-max"
                echo "$CMD" >> "$CMDFILE"
                (( TOTAL_QUEUED++ ))
            fi
        fi

    done
done

echo "Queued ${TOTAL_QUEUED} plots, skipped ${SKIPPED}"

# ---------------------------------------------------------------------------
# Execute — serial or parallel.  Uses xargs -P (GNU parallel not required).
# ---------------------------------------------------------------------------
if [[ $TOTAL_QUEUED -eq 0 ]]; then
    echo "Nothing to render."
    rm -f "$CMDFILE"
    exit 0
fi

START_TIME=$SECONDS

if [[ $JOBS -le 1 ]]; then
    DONE=0
    while IFS= read -r cmd; do
        (( DONE++ ))
        echo "[${DONE}/${TOTAL_QUEUED}] $(echo "$cmd" | grep -oP '(ortho|iso)\S*\.png')"
        eval "$cmd" 2>&1 | grep -vaE "WARN\|"
    done < "$CMDFILE"
else
    echo "Running ${TOTAL_QUEUED} plots, ${JOBS} at a time (xargs)..."
    # Each line of CMDFILE is one full command; run it with bash -c, JOBS in flight.
    xargs -P "$JOBS" -I CMD bash -c CMD < "$CMDFILE" 2>&1 | grep -vaE "WARN\|"
fi

ELAPSED=$(( SECONDS - START_TIME ))
rm -f "$CMDFILE"

echo ""
echo "=== Done: ${TOTAL_QUEUED} plots in ${ELAPSED}s (${JOBS} jobs), ${SKIPPED} skipped ==="
echo "Output in ${FIGPATH}"

# ---------------------------------------------------------------------------
# Movie generation — create one MP4 per frame sequence
# ---------------------------------------------------------------------------
if [[ "${MAKE_MOVIES:-true}" != "true" ]]; then
    echo "Movie generation skipped (MAKE_MOVIES=false)"
    exit 0
fi

# Prefer the bundled ffmpeg (has libx264) from imageio-ffmpeg over the system one
_FFMPEG="$("${PYTHON}" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)" \
    || _FFMPEG=""
if [[ -z "${_FFMPEG}" ]] && command -v ffmpeg &>/dev/null; then
    _FFMPEG="ffmpeg"
fi
if [[ -z "${_FFMPEG}" ]]; then
    echo "WARNING: ffmpeg not found — skipping movie generation."
    echo "  pip install --target <pythonpath_extra> imageio-ffmpeg  to enable."
    exit 0
fi

# Auto-detect encoder
_VCODEC=""
_encoders=$("${_FFMPEG}" -hide_banner -encoders 2>/dev/null)
for _c in libx264 h264_nvenc libx265 mpeg4; do
    if echo "${_encoders}" | grep -q " ${_c} "; then
        _VCODEC="${_c}"
        break
    fi
done
[[ -z "${_VCODEC}" ]] && _VCODEC="gif"

case "${_VCODEC}" in
    libx264|libx265|h264_nvenc)  _QFLAGS=(-crf "${MOVIE_CRF}") ;;
    gif)                         _QFLAGS=()                     ;;
    *)                           _QFLAGS=(-q:v 5)               ;;
esac

echo ""
echo "=== Making movies (encoder: ${_VCODEC}, fps: ${MOVIE_FPS}) ==="

cd "${FIGPATH}"

# Discover unique prefixes from iso_<name>_NNNNN.png and ortho_<name>_NNNNN.png
shopt -s nullglob
_ALL_FRAMES=( iso_*_?????.png ortho_*_?????.png )
shopt -u nullglob

if [[ ${#_ALL_FRAMES[@]} -eq 0 ]]; then
    echo "No frame sequences found in ${FIGPATH}"
    exit 0
fi

unset _seen; declare -A _seen
_PREFIXES=()
for f in "${_ALL_FRAMES[@]}"; do
    p="${f%_?????.png}"
    if [[ -z "${_seen[$p]+x}" ]]; then
        _seen[$p]=1
        _PREFIXES+=("$p")
    fi
done

MOVIEDIR="${FIGPATH}/movies"
mkdir -p "${MOVIEDIR}"

_MOVIE_COUNT=0
for prefix in "${_PREFIXES[@]}"; do
    shopt -s nullglob
    FRAMES=( $(ls "${prefix}"_?????.png 2>/dev/null | sort) )
    shopt -u nullglob

    [[ ${#FRAMES[@]} -lt 2 ]] && continue

    [[ "${_VCODEC}" == "gif" ]] && _ext="gif" || _ext="mp4"
    outfile="${MOVIEDIR}/${prefix}.${_ext}"
    listfile=$(mktemp "${TMPDIR:-/tmp}/bns3d_fflist.XXXXXX")

    for f in "${FRAMES[@]}"; do
        printf "file '%s/%s'\n" "${FIGPATH}" "${f}" >> "${listfile}"
    done

    echo "  ${prefix}: ${#FRAMES[@]} frames -> $(basename "${outfile}")"

    if [[ "${_VCODEC}" == "gif" ]]; then
        "${_FFMPEG}" -y \
            -r "${MOVIE_FPS}" \
            -f concat -safe 0 -i "${listfile}" \
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse=dither=bayer" \
            -loop 0 \
            "${outfile}" \
            -loglevel warning -stats
    else
        "${_FFMPEG}" -y \
            -r "${MOVIE_FPS}" \
            -f concat -safe 0 -i "${listfile}" \
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
            -c:v "${_VCODEC}" -pix_fmt yuv420p \
            "${_QFLAGS[@]}" \
            "${outfile}" \
            -loglevel warning -stats
    fi

    rm -f "${listfile}"
    (( _MOVIE_COUNT++ ))
done

echo ""
echo "=== ${_MOVIE_COUNT} movie(s) saved to ${MOVIEDIR} ==="
