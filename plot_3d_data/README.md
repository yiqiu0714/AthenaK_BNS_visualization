# 3D Data Plotting Scripts

Reusable plotting scripts for AthenaK/BNS-style 3D data.

These scripts were patterned after:

- `incite-bns-zoom-2025/scripts/plots/plot3D*.py`: PyVista isosurfaces on a black background, optional magnetic streamlines, fixed camera, square high-resolution renders.
- `BNS_all.py`: compact publication-style Matplotlib defaults, serif labels, inward ticks, explicit colorbar axes, and conservative colormap/norm choices.

## Files

- `data3d.py`: loads coarse HDF5 files and AthenaK `.bin` files into consistent `(x, y, z)` arrays.
- `style3d.py`: shared Matplotlib/PyVista style helpers, camera setup, scalar defaults, and derived fields.
- `plot_isosurfaces.py`: PyVista 3D isosurface renderer, with optional streamlines.
- `plot_volume.py`: PyVista volume renderer for density-like fields.
- `annotate3d.py`: draws the clock, colour bar and ruler with Matplotlib mathtext
  on top of a finished render, since VTK cannot typeset maths.
- `plot_orthoslices.py`: Matplotlib three-plane slice plot for quick inspection and publication-style diagnostics.
- `requirements.txt`: Python packages used by the scripts.

## Input Data

Supported inputs:

- Coarse-grained HDF5: datasets such as `dens`, `temperature`, `velx`, `vely`, `velz`, `bcc1`, `bcc2`, `bcc3`.
- AthenaK `.bin`: read through `athplot.load_ath_bin.BinaryData`.

Coordinates come from explicit `x1f`/`x2f`/`x3f` face arrays (or `x1v`/`x2v`/`x3v`
centers) when the file has them, so stretched meshes keep their real spacing
instead of being flattened onto a uniform grid.  Failing that, domain metadata
is detected from:

- `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax`
- `domain_size`
- `Lx`, `Ly`, `Lz`
- `extent` (half-width; the box spans `[-extent, +extent]` on each axis)
- `nx`, `ny`, `nz`, `dx`

If no domain metadata is present, the scripts default to a cubic domain with side length `8.0`.

Snapshot time is read from either `time` or `Time`.  `--crop` and `--resolution`
are applied while reading the HDF5 file, so a 480^3 box can be cut down before
it is ever held in memory at full size.

### Electron fraction

Y_e is named `ye` in coarse-grained HDF5 boxes and `s_00` (the advected passive
scalar) in AthenaK `prim3D` output.  `--scalar ye` / `--color-by ye` resolves
whichever one the file actually has.

### Smooth-field quality

AthenaK AMR data must be reconstructed on one grid before VTK can contour it.
The default `--interpolation linear` performs trilinear reconstruction between
native cell centres, which removes the staircase duplication produced by
nearest-neighbour painting for smooth fields such as density, temperature and
`Y_e`. Use `--interpolation nearest` when inspecting exact native cell values.

Isosurfaces default to 20 volume-preserving Taubin smoothing passes and SSAA.
Use `--smooth-iterations 0` to retain the unsmoothed contour or
`--anti-aliasing none` to disable supersample anti-aliasing. These improve the
rendered mesh; they do not create additional physical resolution. Increase
`--resolution` only up to the useful native AMR scale, because memory grows as
N^3.

On machines where true volume compositing is unavailable, `--mode auto` uses
24 lighter, Taubin-smoothed shells instead of 14 relatively opaque shells.
This reduces visible banding while keeping the total optical depth comparable.
The outer density shells used for temperature/`Y_e` colouring should remain
more transparent than the inner shell (the driver uses `0.18,0.65`).

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

Composition on density isosurfaces -- Y_e painted on `log10(rho) = 11, 13`
surfaces.  This is the right tool for composition: a volume rendering cannot
show Y_e once the ejecta is optically thick, because every ray then leaves
through tens of M of `Y_e ~ 0.4` outflow and the frame saturates to flat colour
whatever the opacity.  Density escapes that fate only because its colour ramp
paints tenuous material dark.

```bash
python plot_isosurfaces.py -i box.h5 --units cgs \
  --scalar log_rho --color-by ye --levels 11,13 --opacity 0.45,0.45 \
  --clim 0.05,0.45 --cmap coolwarm \
  -o ye_on_density.png
```

Density volume rendering, with a merger-relative clock, an orthographic camera
and a self-labelling ruler:

```bash
python plot_volume.py -i box.h5 --units cgs \
  --scalar log_rho --clim 7,15 --opacity sigmoid \
  --t0 5310 --projection orthographic --parallel-scale 90 \
  --scalebar auto --axes --off-screen \
  -o density_volume.png
```

`--mode auto` (the default) probes the OpenGL stack first: software renderers
such as Mesa/llvmpipe accept a volume and then emit an empty frame, so on those
it falls back to `--mode layers`, a stack of translucent isosurfaces driven by
the same transfer function.  Force either with `--mode volume` / `--mode layers`.

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
- Pass `--t0 <merger time in code units>` so frames are labelled
  `t_merger +X.XXX ms` rather than absolute simulation time.
- Keep `--clim` fixed across the sequence, otherwise a colour means something
  different in every frame.
- `--camera-track <file>` interpolates a zoom over the sequence.  Rows are
  `time_ms parallel_scale` (or `time_ms r99 half_height ruler`); comment lines
  start with `#`.
- Turn the PNG sequence into a movie outside these scripts with `ffmpeg`.

## Notes

- The default PyVista render uses a black background, PBR lighting, translucent nested isosurfaces, and a fixed inclined camera.
- `--annotate mathtext` (the default) renders the frame in PyVista, then
  composites the clock, colour bar and ruler with Matplotlib, so labels are
  typeset maths: `$\log_{10}\rho$ [g cm$^{-3}$]`, `$Y_e$`, `$M_\odot$`.
  VTK draws plain strings only, so `--annotate vtk` (the older path) can print
  no better than `log10 rho [g cm^-3]`; it stays available because it needs no
  Matplotlib pass.
- `--font` sets the annotation typeface: `sans` (DejaVu Sans, the default and
  matplotlib's own sans), `serif` (STIX, a Times-alike), `cm` (Computer Modern,
  as LaTeX papers are set), or VTK's built-in `arial`/`times`/`courier`. The
  first three ship inside matplotlib, so no system font packages are needed, and
  they apply to the clock, the ruler and the scalar bar alike.
- The Matplotlib slice plot uses the same compact style as `BNS_all.py`.
- Large native 3D outputs should be coarse-grained first; full `2048^3` files are too large for interactive rendering.
- Volume rendering is the memory-hungry mode: one 480^3 float32 field is about
  422 MiB and VTK keeps its own copy on top of numpy's, so keep parallel jobs
  low and use `--crop` / `--resolution` / `--samples`.
- With `--projection orthographic`, `--parallel-scale` is the half-height of the
  view in code units, so a ruler of length L spans exactly `L / (2 * scale *
  aspect)` of the window width.  That identity is what makes `--scalebar` exact,
  and it holds only because the projection is orthographic.
