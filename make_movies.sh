#!/usr/bin/env bash
# make_movies.sh
#
# Creates one MP4 per frame sequence found in FIGPATH.  Handles both layouts:
#   * per-diagnostic subfolders  <name>_xy/<...>_NNNNN.png   (2-D BNS_all driver)
#   * flat sequences             <prefix>_NNNNN.png          (3-D BNS_3d: iso_*,
#                                                             ortho_*; legacy 2-D)
# STRIDE controls sub-sampling: 1 = every frame, 10 = every 10th frame, etc.
#
# Usage:
#   bash make_movies.sh               # use settings below
#   bash make_movies.sh 25 1          # FPS=25, every frame
#   bash make_movies.sh 25 10         # FPS=25, every 10th frame
#   bash make_movies.sh 25 10 /path   # also override FIGPATH

# Log everything to a file so errors are readable even if the terminal closes.
_LOGFILE="${HOME}/make_movies.log"
exec > >(tee -a "${_LOGFILE}") 2>&1
echo "===== $(date) ====="
echo "Log: ${_LOGFILE}"

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------
# Directory containing the rendered frames (subfolders or flat *_NNNNN.png).
# Usually passed as the 3rd argument by BNS_all.sh / BNS_3d.sh; edit here only
# if you run make_movies.sh standalone.
FIGPATH="/path/to/your/simulation/run/Figs"

# Frames per second in the output movie.
# 10–15  → slow, good for inspecting individual frames
# 24–30  → cinematic / standard
FPS=10

# Use every STRIDEth frame from the available PNGs.
# 1  → all frames (00000, 00001, 00002, ...)
# 10 → every 10th (00000, 00010, 00020, ...)
STRIDE=1

# H.264 quality: 0 (lossless) – 51 (worst).  18 ≈ visually lossless.
# Only used by encoders that support -crf (libx264, libx265, h264_nvenc).
CRF=18

# Video encoder.  Leave empty for auto-detection (recommended).
# Override if needed, e.g.:  VCODEC="mpeg4"
VCODEC=""
# ---------------------------------------------------------------------------

# Command-line overrides
FPS="${1:-${FPS}}"
STRIDE="${2:-${STRIDE}}"
FIGPATH="${3:-${FIGPATH}}"

# ---------------------------------------------------------------------------
# Locate an ffmpeg binary. Prefer one already in PATH; otherwise try the Lmod
# `ffmpeg` module (Aurora ships ffmpeg/7.1), then the pip-installed
# imageio-ffmpeg binary (the same one BNS_3d.sh uses). This lets make_movies.sh
# run standalone, not only when BNS_all.sh has already `module load ffmpeg`-ed.
# ---------------------------------------------------------------------------
FFMPEG=""
if command -v ffmpeg &>/dev/null; then
    FFMPEG="ffmpeg"
else
    if ! command -v module &>/dev/null; then
        for _init in "${MODULESHOME:-/usr/share/lmod/lmod}/init/bash" \
                     /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh; do
            [[ -r "${_init}" ]] && source "${_init}" && break
        done
    fi
    command -v module &>/dev/null && module load ffmpeg &>/dev/null
    if command -v ffmpeg &>/dev/null; then
        FFMPEG="ffmpeg"
    else
        # Fall back to imageio-ffmpeg's bundled binary. Read pythonpath_extra
        # from config.ini (next to this script) so the import can be found.
        _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        _PPX="$(awk -F= '/^[[:space:]]*#/{next}
            /^[[:space:]]*pythonpath_extra[[:space:]]*=/{
                gsub(/^[[:space:]]+|[[:space:]]+$/,"",$2); print $2; exit}' \
            "${_SCRIPT_DIR}/config.ini" 2>/dev/null)"
        command -v module &>/dev/null && module load python/3.12.12 &>/dev/null
        for _py in python3.12 python3 python; do
            command -v "${_py}" &>/dev/null || continue
            FFMPEG="$(PYTHONPATH="${_PPX}:${PYTHONPATH:-}" "${_py}" -c \
                "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)"
            [[ -n "${FFMPEG}" ]] && break
            FFMPEG=""
        done
    fi
fi

if [[ -z "${FFMPEG}" ]]; then
    echo "ERROR: ffmpeg not found."
    echo "  On Aurora:   module load ffmpeg"
    echo "  or install:  python3.12 -m pip install --target <pythonpath_extra> imageio-ffmpeg"
    echo "  or conda:    conda install -c conda-forge ffmpeg"
    return 1 2>/dev/null || exit 1
fi
echo "ffmpeg  : ${FFMPEG}"

# Auto-detect the best available encoder (libx264 > h264_nvenc > libx265 > mpeg4).
# Falls back to GIF, which is always supported by ffmpeg.
if [[ -z "${VCODEC}" ]]; then
    _encoders=$("${FFMPEG}" -hide_banner -encoders 2>/dev/null)
    for _c in libx264 h264_nvenc libx265 mpeg4; do
        if echo "${_encoders}" | grep -q " ${_c} "; then
            VCODEC="${_c}"
            break
        fi
    done
    [[ -z "${VCODEC}" ]] && VCODEC="gif"
fi
echo "Encoder : ${VCODEC}"

# Quality flags (not used for GIF — palette filter handles quality instead)
case "${VCODEC}" in
    libx264|libx265|h264_nvenc)  QUALITY_FLAGS=(-crf "${CRF}") ;;
    gif)                         QUALITY_FLAGS=()               ;;
    *)                           QUALITY_FLAGS=(-q:v 5)         ;;
esac

if [[ ! -d "${FIGPATH}" ]]; then
    echo "ERROR: FIGPATH not found: ${FIGPATH}"
    echo "  Edit FIGPATH at the top of make_movies.sh to point to your Figs/ directory."
    return 1 2>/dev/null || exit 1
fi

# Absolutize so the ffmpeg concat lists are unambiguous regardless of CWD.
FIGPATH="$(cd "${FIGPATH}" && pwd)"

MOVIEDIR="${FIGPATH}/all_movies"
mkdir -p "${MOVIEDIR}"

echo "FIGPATH : ${FIGPATH}"
echo "FPS     : ${FPS}"
echo "STRIDE  : ${STRIDE}"

[[ "${VCODEC}" == "gif" ]] && _EXT="gif" || _EXT="mp4"

# ---------------------------------------------------------------------------
# Encode one movie:  _encode_movie <outfile> <frame> [<frame> ...]
# ---------------------------------------------------------------------------
_encode_movie() {
    local outfile="$1"; shift
    local listfile; listfile="$(mktemp "${TMPDIR:-/tmp}/mm_fflist.XXXXXX")"
    local f
    for f in "$@"; do printf "file '%s'\n" "${f}" >> "${listfile}"; done

    if [[ "${VCODEC}" == "gif" ]]; then
        # High-quality GIF via palette generation (single-pass filtergraph).
        "${FFMPEG}" -y \
            -r "${FPS}" \
            -f concat -safe 0 -i "${listfile}" \
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse=dither=bayer" \
            -loop 0 \
            "${outfile}" \
            -loglevel warning -stats
    else
        "${FFMPEG}" -y \
            -r "${FPS}" \
            -f concat -safe 0 -i "${listfile}" \
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
            -c:v "${VCODEC}" -pix_fmt yuv420p \
            "${QUALITY_FLAGS[@]}" \
            "${outfile}" \
            -loglevel warning -stats
    fi
    rm -f "${listfile}"
}

# Return every STRIDEth element of the (already sorted) arguments, one per line.
_subsample() {
    local -a all=( "$@" ) out=()
    local j
    for (( j=0; j<${#all[@]}; j+=STRIDE )); do out+=("${all[j]}"); done
    printf '%s\n' "${out[@]}"
}

_n_made=0

# ---------------------------------------------------------------------------
# 1) Per-diagnostic subfolders (2-D driver layout):  <name>_xy/*_NNNNN.png
#    One movie per subfolder, written straight into all_movies/.
# ---------------------------------------------------------------------------
shopt -s nullglob
SUBDIRS=( "${FIGPATH}"/*/ )
shopt -u nullglob

for d in "${SUBDIRS[@]}"; do
    d="${d%/}"
    [[ "$(basename "${d}")" == "all_movies" ]] && continue

    shopt -s nullglob
    ALL_FRAMES=( "${d}"/*_?????.png )
    shopt -u nullglob
    [[ ${#ALL_FRAMES[@]} -eq 0 ]] && continue

    IFS=$'\n' ALL_FRAMES=( $(printf '%s\n' "${ALL_FRAMES[@]}" | sort) ); unset IFS
    IFS=$'\n' FRAMES=( $(_subsample "${ALL_FRAMES[@]}") ); unset IFS

    name="$(basename "${d}")"                     # e.g. temperature_xy
    echo "[${name}]  ${#ALL_FRAMES[@]} available, ${#FRAMES[@]} used  ->  ${name}.${_EXT}"
    _encode_movie "${MOVIEDIR}/${name}.${_EXT}" "${FRAMES[@]}"
    _n_made=$((_n_made + 1))
    echo ""
done

# ---------------------------------------------------------------------------
# 2) Flat frame sequences (3-D BNS_3d Figs_3d, or legacy flat 2-D output):
#    <prefix>_NNNNN.png directly in FIGPATH.  Only used when no subfolder
#    movies were made, so the two layouts never double-count.
# ---------------------------------------------------------------------------
if [[ ${_n_made} -eq 0 ]]; then
    shopt -s nullglob
    PNGS=( "${FIGPATH}"/bns_frame_*_?????.png \
           "${FIGPATH}"/iso_*_?????.png \
           "${FIGPATH}"/ortho_*_?????.png )
    shopt -u nullglob

    unset _seen; declare -A _seen
    PREFIXES=()
    for f in "${PNGS[@]}"; do
        b="$(basename "${f}")"; p="${b%_?????.png}"
        if [[ -z "${_seen[$p]+x}" ]]; then
            _seen[$p]=1
            PREFIXES+=("$p")
        fi
    done

    for p in "${PREFIXES[@]}"; do
        shopt -s nullglob
        ALL_FRAMES=( "${FIGPATH}/${p}"_?????.png )
        shopt -u nullglob
        [[ ${#ALL_FRAMES[@]} -eq 0 ]] && continue

        IFS=$'\n' ALL_FRAMES=( $(printf '%s\n' "${ALL_FRAMES[@]}" | sort) ); unset IFS
        IFS=$'\n' FRAMES=( $(_subsample "${ALL_FRAMES[@]}") ); unset IFS

        echo "[${p}]  ${#ALL_FRAMES[@]} available, ${#FRAMES[@]} used  ->  ${p}.${_EXT}"
        _encode_movie "${MOVIEDIR}/${p}.${_EXT}" "${FRAMES[@]}"
        _n_made=$((_n_made + 1))
        echo ""
    done
fi

if [[ ${_n_made} -eq 0 ]]; then
    echo "No frame sequences found under ${FIGPATH}"
    echo "  (looked for <name>/*_NNNNN.png subfolders and flat"
    echo "   bns_frame_*/iso_*/ortho_*_NNNNN.png)"
else
    echo "Made ${_n_made} movie(s) in ${MOVIEDIR}/"
fi
echo "Done."
