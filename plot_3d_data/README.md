# 3D Data Plotting Scripts

Reusable plotting scripts for AthenaK/BNS-style 3D data.

These scripts were patterned after:

- `incite-bns-zoom-2025/scripts/plots/plot3D*.py`: PyVista isosurfaces on a black background, optional magnetic streamlines, fixed camera, square high-resolution renders.
- `BNS_all.py`: compact publication-style Matplotlib defaults, serif labels, inward ticks, explicit colorbar axes, and conservative colormap/norm choices.

## Files

- `data3d.py`: loads coarse HDF5 files and AthenaK `.bin` files into consistent `(x, y, z)` arrays.
- `style3d.py`: shared Matplotlib/PyVista style helpers, camera setup, scalar defaults, and derived fields.
- `plot_isosurfaces.py`: PyVista 3D isosurface renderer, with optional streamlines.
- `plot_orthoslices.py`: Matplotlib three-plane slice plot for quick inspection and publication-style diagnostics.
- `requirements.txt`: Python packages used by the scripts.

## Input Data

Supported inputs:

- Coarse-grained HDF5: datasets such as `dens`, `temperature`, `velx`, `vely`, `velz`, `bcc1`, `bcc2`, `bcc3`.
- AthenaK `.bin`: read through `athplot.load_ath_bin.BinaryData`.

HDF5 domain metadata is detected from:

- `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax`
- `domain_size`
- `Lx`, `Ly`, `Lz`
- `nx`, `ny`, `nz`, `dx`

If no domain metadata is present, the scripts default to a cubic domain with side length `8.0`.

## Examples

Density isosurfaces, colored by density:

```bash
python plot_isosurfaces.py -i frame_coarse_M8.h5 \
  --scalar density --levels -7,-6,-5,-4 \
  -o density_3d.png
```

Temperature isosurfaces:

```bash
python plot_isosurfaces.py -i frame_coarse_M8.h5 \
  --scalar temperature --levels 10,20,35,50 \
  --cmap viridis -o temperature_3d.png
```

Magnetic-field magnitude isosurfaces with field-line streamlines. With
`--units cgs` the field components are converted to **Gauss**, so `log_bmag`
levels are `log10(|B|/G)` (~9–15); with `--units code` use code-unit log10
levels (~ −13…−7):

```bash
python plot_isosurfaces.py -i bns.prim3D.00000.bin --units cgs \
  --scalar log_bmag --levels 12,13,14 --clim 9,15 \
  --streamlines B --streamline-seeds dual-density-max \
  -o bfield_3d.png
```

`--scalar current` renders ‖∇×B‖ (a magnetic "current" proxy); `divb`, `bmag`,
`vorticity`, and `qcriterion` are also available as derived scalars.

Q-criterion vortical structures colored by magnetic strength:

```bash
python plot_isosurfaces.py -i frame_coarse_M8.h5 \
  --scalar qcriterion --levels 0.001,0.01,0.05 \
  --color-by log_bmag --streamlines B \
  -o qcriterion_3d.png
```

Three orthogonal slices through the grid center:

```bash
python plot_orthoslices.py -i frame_coarse_M8.h5 \
  --scalar density --center 0,0,0 \
  -o density_slices.png
```

## What To Plot For Different 3D Data

Scalar fields:

- Density, temperature, lapse, pressure, entropy, electron fraction: use isosurfaces for 3D morphology, or orthogonal slices for quantitative inspection.
- Wide-dynamic-range positive fields such as density or magnetic energy: use `log10` or log-normalized slices.
- Signed fields such as divergence, curl components, or velocity components: use diverging colormaps and symmetric limits.

Vector fields:

- Magnetic field or velocity: use streamlines or tube traces seeded from a plane, a disc, or density maxima.
- Combine vectors with scalar surfaces: for example density isosurfaces plus magnetic streamlines, or Q-criterion surfaces colored by `log_bmag`.

Tensor/gradient-derived fields:

- Vorticity magnitude: reveals shear layers, Kelvin-Helmholtz structures, and rotational funnels.
- Q-criterion: highlights vortical regions where rotation dominates strain.
- Divergence of B: use slices or signed isosurfaces as a numerical-error diagnostic.

Time series of 3D outputs:

- Render one frame per file with fixed `--camera-elevation`, `--camera-azimuth`, `--camera-distance`, and fixed color limits/levels.
- Turn the PNG sequence into a movie outside these scripts with `ffmpeg`.

## Notes

- The default PyVista render uses a black background, PBR lighting, translucent nested isosurfaces, and a fixed inclined camera.
- The Matplotlib slice plot uses the same compact style as `BNS_all.py`.
- Large native 3D outputs should be coarse-grained first; full `2048^3` files are too large for interactive rendering.
