# AthenaK BNS Visualization

Plotting and movie-generation tools for **binary neutron star (BNS)** merger
simulations run with [AthenaK](https://github.com/IAS-Astrophysics/athenak).

The main driver renders time-history plots and 2-D slice frames (density,
temperature, electron fraction, neutrino/radiation fields, …) from a
simulation's `output-XXXX` restart segments, then stitches the frames into
movies with `ffmpeg`.

## Contents

| File | Purpose |
|------|---------|
| `BNS_all.py` | Main visualization driver. Fully configurable via command-line flags (`--help`). |
| `BNS_all.sh` | Convenience wrapper: reads paths from `config.ini`, builds the argument list, runs `BNS_all.py`, then `make_movies.sh`. |
| `make_movies.sh` | Builds one MP4/GIF per frame sequence found in a `Figs/` directory. |
| `athenak_units.py` | Unit conversions (code / CGS / NGS) and per-quantity plot parameters. |
| `plot_time_per_cycle.py` | Standalone tool: wall-clock time-per-cycle distribution from `.out` logs. |
| `sph_data_analysis/` | SPH ejecta + neutrino analysis (spherical-surface VTK output). See its section below. |
| `config.example.ini` | Template for machine-specific paths → copy to `config.ini`. |
| `plot-tools/` | Git submodule: the [`athplot`](https://github.com/jfields7/plot-tools) package used to read AthenaK binary/VTK output. |

## Setup

```bash
# 1. Clone with the submodule
git clone --recurse-submodules <this-repo-url>
cd AthenaK_BNS_visualization
# (or, if already cloned:  git submodule update --init --recursive)

# 2. Install Python dependencies
pip install -r requirements.txt
pip install -e plot-tools          # installs athplot + its deps (vtk, PyCompOSE)

# 3. Create your local config (paths only)
cp config.example.ini config.ini
$EDITOR config.ini                 # set simpath (figpath is optional)
```

`ffmpeg` is required for movie generation. On HPC systems the wrappers call
`module load …`; edit those lines in `BNS_all.sh` if you use conda/venv instead.

## Usage

### Full pipeline (frames + movies)

```bash
bash BNS_all.sh
```

This reads `simpath`/`figpath` from `config.ini`, renders the `SECTIONS` selected
inside `BNS_all.sh`, writes frames to `figpath` (default `<simpath>/Figs`), and
assembles movies into `<figpath>/all_movies/`.

### Driver directly

`BNS_all.py` takes all configuration on the command line — no editing required:

```bash
python3 BNS_all.py --simpath /path/to/run \
                   --figpath /path/to/run/Figs \
                   --units cgs \
                   --sections history rho_max nu_energy \
                   --no-trackers --skip-existing
```

Run `python3 BNS_all.py --help` for the full list of sections and flags.

### Movies only

```bash
bash make_movies.sh <fps> <stride> <FIGPATH>
# e.g. every 10th frame at 15 fps:
bash make_movies.sh 15 10 /path/to/run/Figs
```

### Time-per-cycle diagnostic

```bash
python3 plot_time_per_cycle.py --sim-dir /path/to/run \
                               --sim-name my_run \
                               --output-dirs output-0000 output-0001 \
                               --no-show
```

## Configuration

Machine-specific **paths** (`simpath`, optional `figpath`) live in `config.ini`
— a plain INI file that the scripts read (never execute). It is copied from
`config.example.ini` and **gitignored**, so your local paths never get
committed. All **run settings** — which `SECTIONS` to plot, `UNITS`,
tracker/domain flags, and movie `FPS`/`STRIDE` — stay in `BNS_all.sh`, where you
edit them per run. `BNS_all.sh` exits with an error if `config.ini` is missing.

The plot-tools location is resolved automatically (the bundled submodule), or
can be overridden with the `ATHPLOT_PATH` environment variable.

## SPH data analysis (`sph_data_analysis/`)

Post-processing of the spherical-surface (`sph`) VTK output: dynamical **ejecta**
properties and **neutrino** emission, plus a **merger-time** estimate from the
gravitational waveform.

| File | Purpose |
|------|---------|
| `run_analysis.sh` | Orchestrator: reads paths from `../config.ini`, then runs the merger-time, ejecta, neutrino, and plotting steps. |
| `merger_time.py` | Merger time from the max of \|psi4₂₂\|² (real²+imag²) of the (2,2) GW mode, merged across all `output-*/waveforms`. Writes `merger_time.txt`. |
| `analyze_ejecta.py` | Per-snapshot ejecta diagnostics (mass-ejection rate, vinf, Yₑ, angular distributions). Needs the EOS table. |
| `analyze_neutrinos.py` | Neutrino energy luminosity, mean energy, and angular distribution from `rad_m1_E/F/N`. |
| `plot_analysis.py` | Builds `fig_ejecta.pdf` (2×3) and `fig_neutrino.pdf` (2 panels: log luminosity 1e51–3e54, linear mean energy 0–60). |
| `utils_ejecta.py` | Shared spherical-integration helpers. |

```bash
cd sph_data_analysis
bash run_analysis.sh              # merger time + ejecta + neutrino + plots
bash run_analysis.sh --neutrino --plot   # selected steps only
```

Defaults: a **single extraction radius `r = 300`** for both ejecta and neutrino
analysis, and **all** `output-*/sph` segments are processed (sorted, like
`BNS_all.sh`). Paths (`simpath`, `eos_table`, optional `pythonpath_extra`) come
from `config.ini`; run settings (radius, `N_WORKERS`, post-merger window, etc.)
are edited at the top of `run_analysis.sh`. Outputs are written to
`<simpath>/analysis/`.

Requires `h5py` (for the EOS table) and `vtk` (via `athplot.load_sph_vtk`) in
addition to the base dependencies; on Aurora these come from `module load
py-h5py` and `pythonpath_extra`.

## Notes

- `core.*` dumps, `__pycache__/`, and generated output (`Figs/`, `*.png`,
  `*.mp4`, …) are gitignored.
- This repo is focused on the visualization scripts. The separate
  [`LargeSim`](https://github.com/computationalrelativity/LargeSim) workflow
  repo and the older Perlmutter script variants are intentionally excluded.
