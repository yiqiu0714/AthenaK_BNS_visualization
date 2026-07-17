#!/usr/bin/env bash
# make_movies.sh
#
# Creates one MP4 per frame sequence found in FIGPATH.
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
# Directory containing the rendered bns_frame_*.png frames.
# Usually passed as the 3rd argument by BNS_all.sh; edit here only if you run
# make_movies.sh standalone.
FIGPATH="/path/to/your/simulation/run/Figs"

# Frames per second in the output movie.
# 10–15  → slow, good for inspecting individual frames
# 24–30  → cinematic / standard
FPS=10

# Use every STRIDEth frame from the available PNGs.
# 1  → all frames (00000, 00001, 00002, ...)
# 10 → every 10th (00000, 00010, 00020, ...)
STRIDE=10

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

if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg not found in PATH."
    echo "  Install via conda:  conda install -c conda-forge ffmpeg"
    echo "  Then activate the environment and re-run this script."
    return 1 2>/dev/null || exit 1
fi

# Auto-detect the best available encoder (libx264 > h264_nvenc > libx265 > mpeg4).
# Falls back to GIF, which is always supported by ffmpeg.
if [[ -z "${VCODEC}" ]]; then
    _encoders=$(ffmpeg -hide_banner -encoders 2>/dev/null)
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

# ---------------------------------------------------------------------------
# Discover frame sequences.  Each diagnostic renders into its own subfolder
# <name>_xy/ holding one sequence (<job>_frame_<name>_NNNNN.png), so there is
# exactly one movie per subfolder.  Single-image diagnostics (history,
# run_speed, ...) stay flat in FIGPATH and are ignored.
# ---------------------------------------------------------------------------
MOVIEDIR="${FIGPATH}/all_movies"
mkdir -p "${MOVIEDIR}"

shopt -s nullglob
SUBDIRS=( "${FIGPATH}"/*/ )
shopt -u nullglob

echo "FIGPATH : ${FIGPATH}"
echo "FPS     : ${FPS}"
echo "STRIDE  : ${STRIDE}"

# ---------------------------------------------------------------------------
# Build one movie per frame subfolder (sequential)
# ---------------------------------------------------------------------------
_n_made=0
for d in "${SUBDIRS[@]}"; do
    d="${d%/}"
    [[ "$(basename "${d}")" == "all_movies" ]] && continue

    shopt -s nullglob
    ALL_FRAMES=( "${d}"/*_?????.png )
    shopt -u nullglob
    [[ ${#ALL_FRAMES[@]} -eq 0 ]] && continue

    IFS=$'\n' ALL_FRAMES=( $(printf '%s\n' "${ALL_FRAMES[@]}" | sort) ); unset IFS

    # Sub-sample: keep every STRIDEth frame
    FRAMES=()
    for (( j=0; j<${#ALL_FRAMES[@]}; j+=STRIDE )); do
        FRAMES+=("${ALL_FRAMES[j]}")
    done

    name="$(basename "${d}")"                     # e.g. temperature_xy
    [[ "${VCODEC}" == "gif" ]] && _ext="gif" || _ext="mp4"
    outfile="${MOVIEDIR}/${name}.${_ext}"
    listfile="${d}/.fflist"

    echo "[${name}]  ${#ALL_FRAMES[@]} available, ${#FRAMES[@]} used  ->  $(basename "${outfile}")"

    # Write ffmpeg concat file list (absolute frame paths)
    : > "${listfile}"
    for f in "${FRAMES[@]}"; do
        printf "file '%s'\n" "${f}" >> "${listfile}"
    done

    if [[ "${VCODEC}" == "gif" ]]; then
        # High-quality GIF via palette generation (single-pass filtergraph)
        # palettegen builds a per-stream optimal 256-colour palette;
        # paletteuse+bayer dithering gives clean colour gradients.
        ffmpeg -y \
            -r "${FPS}" \
            -f concat -safe 0 -i "${listfile}" \
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse=dither=bayer" \
            -loop 0 \
            "${outfile}" \
            -loglevel warning -stats
    else
        # -r before -i  : input framerate (1 frame = 1/FPS seconds)
        # scale filter  : round width and height to even numbers (required for yuv420p)
        ffmpeg -y \
            -r "${FPS}" \
            -f concat -safe 0 -i "${listfile}" \
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
            -c:v "${VCODEC}" -pix_fmt yuv420p \
            "${QUALITY_FLAGS[@]}" \
            "${outfile}" \
            -loglevel warning -stats
    fi

    rm -f "${listfile}"
    _n_made=$((_n_made + 1))
    echo ""
done

if [[ ${_n_made} -eq 0 ]]; then
    echo "No frame subfolders (*/*_NNNNN.png) found under ${FIGPATH}"
else
    echo "Made ${_n_made} movie(s) in ${MOVIEDIR}/"
fi
echo "Done."
