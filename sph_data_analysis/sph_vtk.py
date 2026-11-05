"""Pure-numpy reader for AthenaK spherical-surface VTK dumps.

Replaces ``athplot.load_sph_vtk`` for the analysis scripts in this directory.
Two reasons it exists:

1. ``athplot.load_sph_vtk`` needs the ``vtk`` python package, which is not
   importable in any python available here (the module environments have no
   vtk, and the copy under ``pythonpath_extra`` is built against a different
   CPython ABI: ``undefined symbol: PyObject_CheckBuffer``).  The legacy VTK
   binary format AthenaK writes is simple enough to parse with numpy alone.

2. AthenaK now writes *multi-radius* surface dumps.  The old output had one
   radius per file::

       bns.r=200.00.adm.12073.vtk        DIMENSIONS 1 256 512

   the new output packs a whole range of radii into a single file::

       bns.r=100.00-500.00.adm.00184.vtk DIMENSIONS 64 128 256

   ``athplot``'s reader rejects the latter outright
   (``RuntimeError: More than one radial point detected.``).

``SphericalData`` here reads both and always presents the *same* 2-D API that
the old single-radius reader did -- ``self.r`` is a scalar, ``self.theta``,
``self.phi`` and every entry of ``self.fields`` have shape ``(nphi, ntheta)``.
For a multi-radius file the shell nearest the requested radius is extracted, so
downstream analysis code needs no changes.

The requested radius defaults to the module-level default (300 M_sun, see
``DEFAULT_RADIUS``), which callers can change once with ``set_default_radius``
rather than threading a radius through every call site.

File layout (legacy VTK, big-endian binary)::

    # vtk DataFile Version 3.0
    # AthenaK data at time=... cycle=... nradii=... rmin=... rmax=... xc=...
    BINARY
    DATASET STRUCTURED_GRID
    DIMENSIONS nr ntheta nphi
    POINTS N float                 <- N*3 big-endian float32: (r, theta, phi)
    FIELD FieldData 3
    TIME 1 1 float / CYCLE 1 1 int / RADII 1 nr float
    POINT_DATA N
    SCALARS <name> float 1
    LOOKUP_TABLE default
    <N big-endian float32>         <- repeated per field

Point ordering is VTK's usual first-axis-fastest, i.e. the flat index is
``ir + nr*(itheta + ntheta*iphi)``, so a C-order reshape to
``(nphi, ntheta, nr)`` followed by ``[:, :, ir]`` selects one shell.
"""

import os
import re

import numpy as np

# Default surface radius (in M_sun) used when a caller does not name one.
DEFAULT_RADIUS = 300.0

_default_radius = DEFAULT_RADIUS

# Radii closer than this (in M_sun) are treated as the same shell.
_R_TOL = 1e-6


def set_default_radius(radius):
    """Set the radius used by ``SphericalData`` when none is passed.

    Call this once at the top of an analysis run.  Because the analysis
    scripts fan out with ``multiprocessing`` in fork mode, workers inherit the
    value; the pool initializers set it again so spawn mode also works.
    """
    global _default_radius
    _default_radius = float(radius)


def get_default_radius():
    return _default_radius


# --------------------------------------------------------------------------
# Low-level legacy-VTK parsing
# --------------------------------------------------------------------------

# Legacy VTK binary payloads are big-endian regardless of the host.
_VTK_DTYPES = {
    'bit': np.dtype('u1'),
    'unsigned_char': np.dtype('u1'),
    'char': np.dtype('i1'),
    'unsigned_short': np.dtype('>u2'),
    'short': np.dtype('>i2'),
    'unsigned_int': np.dtype('>u4'),
    'int': np.dtype('>i4'),
    'unsigned_long': np.dtype('>u8'),
    'long': np.dtype('>i8'),
    'float': np.dtype('>f4'),
    'double': np.dtype('>f8'),
}


def _dtype_for(name):
    try:
        return _VTK_DTYPES[name]
    except KeyError:
        raise RuntimeError(f"Unsupported VTK data type '{name}'")


def _read_block(fh, dtype, count):
    """Read ``count`` values, then swallow the newline VTK writes after them."""
    nbytes = int(count) * dtype.itemsize
    raw = fh.read(nbytes)
    if len(raw) != nbytes:
        raise RuntimeError(
            f"Truncated VTK binary block: wanted {nbytes} bytes, got {len(raw)}")
    arr = np.frombuffer(raw, dtype=dtype)
    _eat_newline(fh)
    return arr


def _skip_block(fh, dtype, count):
    fh.seek(int(count) * dtype.itemsize, os.SEEK_CUR)
    _eat_newline(fh)


def _eat_newline(fh):
    pos = fh.tell()
    if fh.read(1) != b'\n':
        fh.seek(pos)


def parse_meta_line(line):
    """Parse the AthenaK metadata comment (line 2 of every sph VTK file)."""
    meta = {}
    for tok in line.split():
        if '=' not in tok:
            continue
        key, _, val = tok.partition('=')
        meta[key] = val
    out = {}
    for key, cast in (('time', float), ('cycle', int), ('nradii', int),
                      ('rmin', float), ('rmax', float),
                      ('xc', float), ('yc', float), ('zc', float)):
        if key in meta:
            try:
                out[key] = cast(meta[key])
            except ValueError:
                pass
    for key in ('theta_spacing', 'theta_centering'):
        if key in meta:
            out[key] = meta[key]
    return out


def read_header(filename, encoding='latin_1'):
    """Return the metadata dict from a sph VTK file without reading any data.

    Much cheaper than building a ``SphericalData``, which matters now that a
    multi-radius dump is >100 MB.
    """
    with open(filename, 'rb') as fh:
        fh.readline()                       # '# vtk DataFile Version 3.0'
        line = fh.readline().decode(encoding, 'replace')
    return parse_meta_line(line)


def read_time(filename):
    """Return just the simulation time of a sph VTK file."""
    meta = read_header(filename)
    if 'time' not in meta:
        raise RuntimeError(f"No time= found in the header of {filename}")
    return meta['time']


# --------------------------------------------------------------------------
# SphericalData
# --------------------------------------------------------------------------

class SphericalData:
    """An AthenaK spherical-surface VTK dump, presented as a 2-D shell.

    Parameters
    ----------
    filename : str
    radius : float, optional
        Requested shell radius in M_sun.  Ignored for single-radius files.
        Defaults to :func:`get_default_radius`.
    fields : sequence of str, optional
        Only read these point-data arrays.  Pass an empty sequence to read the
        geometry alone.  Skipped arrays are seeked over, not decoded.

    Attributes
    ----------
    r : float
        The radius of the extracted shell (a scalar, as in the old reader).
    theta, phi : ndarray, shape (nphi, ntheta)
    fields : dict of str -> ndarray, shape (nphi, ntheta)
    radii : ndarray
        Every radius stored in the file (length 1 for the old format).
    nradii, rmin, rmax : file-level radial metadata.
    multi_radius : bool
        True when the file holds a range of radii.
    """

    def __init__(self, filename, radius=None, fields=None, encoding='latin_1'):
        self.filename = filename
        want = None if fields is None else set(fields)

        self.time = None
        self.cycle = None
        self.xc = self.yc = self.zc = None
        self.theta_spacing = None
        self.theta_centering = None

        dims = None
        npoints = None
        points = None
        radii_field = None
        self.field_names = []
        self.fields = {}

        with open(filename, 'rb') as fh:
            fh.readline()                                   # version banner
            meta = parse_meta_line(fh.readline().decode(encoding, 'replace'))
            for key, val in meta.items():
                setattr(self, key, val)

            fmt = fh.readline().strip()
            if fmt != b'BINARY':
                raise RuntimeError(
                    f"{filename}: expected a BINARY legacy VTK file, got "
                    f"{fmt.decode(encoding, 'replace')!r}")

            while True:
                line = fh.readline()
                if not line:
                    break
                tok = line.split()
                if not tok:
                    continue
                key = tok[0].decode(encoding, 'replace')

                if key == 'DATASET':
                    kind = tok[1].decode(encoding, 'replace')
                    if kind != 'STRUCTURED_GRID':
                        raise RuntimeError(
                            f"{filename}: expected DATASET STRUCTURED_GRID, "
                            f"got {kind}")

                elif key == 'DIMENSIONS':
                    dims = [int(t) for t in tok[1:4]]

                elif key == 'POINTS':
                    npoints = int(tok[1])
                    dtype = _dtype_for(tok[2].decode(encoding, 'replace'))
                    points = _read_block(fh, dtype, npoints * 3)
                    points = points.reshape(npoints, 3)

                elif key == 'FIELD':
                    for _ in range(int(tok[2])):
                        ftok = fh.readline().split()
                        fname = ftok[0].decode(encoding, 'replace')
                        ncomp, ntup = int(ftok[1]), int(ftok[2])
                        dtype = _dtype_for(ftok[3].decode(encoding, 'replace'))
                        vals = _read_block(fh, dtype, ncomp * ntup)
                        if fname == 'RADII':
                            radii_field = np.asarray(vals, dtype=np.float64)
                        elif fname == 'TIME' and self.time is None:
                            self.time = float(vals[0])
                        elif fname == 'CYCLE' and self.cycle is None:
                            self.cycle = int(vals[0])

                elif key == 'POINT_DATA':
                    npoints = int(tok[1])

                elif key in ('SCALARS', 'VECTORS'):
                    fname = tok[1].decode(encoding, 'replace')
                    dtype = _dtype_for(tok[2].decode(encoding, 'replace'))
                    if key == 'SCALARS':
                        ncomp = int(tok[3]) if len(tok) > 3 else 1
                        nxt = fh.tell()
                        if fh.readline().split()[:1] != [b'LOOKUP_TABLE']:
                            fh.seek(nxt)
                    else:
                        ncomp = 3
                    count = npoints * ncomp
                    if want is not None and fname not in want:
                        _skip_block(fh, dtype, count)
                        continue
                    self.field_names.append(fname)
                    self.fields[fname] = (_read_block(fh, dtype, count), ncomp)

                elif key in ('CELL_DATA', 'LOOKUP_TABLE'):
                    continue

        if dims is None or points is None:
            raise RuntimeError(f"{filename}: missing DIMENSIONS or POINTS")

        nr, ntheta, nphi = dims
        self.dims = tuple(dims)
        self.nradii = getattr(self, 'nradii', None) or nr
        self.multi_radius = nr > 1

        # Radii: prefer the RADII field, fall back to the coordinate array.
        if radii_field is not None and radii_field.size == nr:
            self.radii = radii_field
        else:
            self.radii = np.asarray(points[:nr, 0], dtype=np.float64)
        if not hasattr(self, 'rmin'):
            self.rmin = float(self.radii.min())
        if not hasattr(self, 'rmax'):
            self.rmax = float(self.radii.max())

        # Pick the shell.
        self.radius_requested = (
            float(radius) if radius is not None else _default_radius)
        if nr == 1:
            self.ir = 0
        else:
            self.ir = int(np.argmin(np.abs(self.radii - self.radius_requested)))
        self.r = float(self.radii[self.ir])

        # Present a 2-D shell, exactly as the single-radius reader did.
        self.shape = (nphi, ntheta)
        grid = points.reshape(nphi, ntheta, nr, 3)
        self.theta = np.ascontiguousarray(grid[:, :, self.ir, 1], dtype=np.float64)
        self.phi = np.ascontiguousarray(grid[:, :, self.ir, 2], dtype=np.float64)

        for fname, (flat, ncomp) in list(self.fields.items()):
            if ncomp == 1:
                shell = flat.reshape(nphi, ntheta, nr)[:, :, self.ir]
            else:
                shell = flat.reshape(nphi, ntheta, nr, ncomp)[:, :, self.ir, :]
            self.fields[fname] = np.ascontiguousarray(shell, dtype=np.float64)

    def radius_error(self):
        """How far the extracted shell sits from the one that was asked for."""
        return abs(self.r - self.radius_requested)

    def __repr__(self):
        return (f"SphericalData({os.path.basename(self.filename)}, "
                f"r={self.r:g} of {self.nradii} radii, t={self.time})")


# --------------------------------------------------------------------------
# Output-style discovery
# --------------------------------------------------------------------------

# bns.r=200.00.adm.12073.vtk  /  bns.r=100.00-500.00.adm.00184.vtk
_SPH_NAME_RE = re.compile(
    r'^(?P<job>.+?)\.r=(?P<rtag>[0-9]+(?:\.[0-9]+)?(?:-[0-9]+(?:\.[0-9]+)?)?)'
    r'\.(?P<var>.+)\.(?P<idx>[0-9]+)\.vtk$')


# AthenaK also changed how variables are grouped into files.  The field names
# *inside* the files are unchanged; only the grouping differs, so a logical
# field is served by whichever file is present:
#
#   old: z4c_alpha, z4c_betax, z4c_betay, z4c_betaz   (one file each)
#   new: z4c_gauge                                    (all four together)
#   old: rad_m1_E, rad_m1_F, rad_m1_N                 (one file each)
#   new: rad_m1_moments (a.k.a. m1_mom)               (E, N and F together)
#
# Candidates are listed most-specific first, so a directory holding the old
# per-variable files keeps using them.
VAR_GROUPS = {
    'z4c_alpha': ('z4c_alpha', 'z4c_gauge'),
    'z4c_betax': ('z4c_betax', 'z4c_gauge'),
    'z4c_betay': ('z4c_betay', 'z4c_gauge'),
    'z4c_betaz': ('z4c_betaz', 'z4c_gauge'),
    'rad_m1_E':  ('rad_m1_E', 'm1_mom', 'rad_m1_moments'),
    'rad_m1_N':  ('rad_m1_N', 'm1_mom', 'rad_m1_moments'),
    'rad_m1_F':  ('rad_m1_F', 'm1_mom', 'rad_m1_moments'),
}

# File variables that pack every M1 moment into one file.
M1_GROUPED_VARS = ('m1_mom', 'rad_m1_moments')


class SphStyle:
    """Which on-disk radius tag an analysis should use, and what it covers."""

    def __init__(self, r_tag, rmin, rmax, multi, radius, variables=()):
        self.r_tag = r_tag              # e.g. 'r=300.00' or 'r=100.00-500.00'
        self.rmin = rmin
        self.rmax = rmax
        self.multi = multi
        self.radius = radius            # the radius that was requested
        self.variables = set(variables)  # file variables present on disk

    def var_for(self, field):
        """Which file variable provides ``field`` in this directory.

        Returns None if nothing on disk provides it.
        """
        for cand in VAR_GROUPS.get(field, (field,)):
            if cand in self.variables:
                return cand
        return None

    def path_for(self, sph_dir, field, index, job='bns'):
        """Path of the file providing ``field``, or None if there is none."""
        var = self.var_for(field)
        if var is None:
            return None
        return self.path(sph_dir, var, index, job=job)

    def prefix(self, sph_dir, job='bns'):
        return os.path.join(sph_dir, f'{job}.{self.r_tag}.')

    def path(self, sph_dir, var, index, job='bns'):
        return self.prefix(sph_dir, job) + f'{var}.{index:05d}.vtk'

    def covers(self, radius):
        return self.rmin - _R_TOL <= radius <= self.rmax + _R_TOL

    def describe(self):
        if self.multi:
            return (f"multi-radius dumps ({self.r_tag}); extracting the shell "
                    f"nearest r={self.radius:g} M_sun")
        if abs(self.rmin - self.radius) > _R_TOL:
            return (f"single-radius dumps ({self.r_tag}); note these sit at "
                    f"r={self.rmin:g}, not the requested r={self.radius:g}")
        return f"single-radius dumps ({self.r_tag}) at r={self.rmin:g} M_sun"

    def __repr__(self):
        return f"SphStyle({self.r_tag}, multi={self.multi})"


def open_set(paths, radius=None, fields=None):
    """Open each *distinct* path once; return ``{path: SphericalData}``.

    With the grouped output format several logical fields live in the same
    file (e.g. the lapse and all three shift components in ``z4c_gauge``), so
    a worker that naively opened one file per field would read the same
    >100 MB dump four times.
    """
    out = {}
    for path in paths:
        if path is not None and path not in out:
            out[path] = SphericalData(path, radius=radius, fields=fields)
    return out


def _parse_r_tag(r_tag):
    body = r_tag[2:] if r_tag.startswith('r=') else r_tag
    if '-' in body:
        lo, _, hi = body.partition('-')
        return float(lo), float(hi), True
    val = float(body)
    return val, val, False


def scan_sph_dirs(sph_dirs, job='bns'):
    """Return the radius tags present on disk, as ``{r_tag: set(vars)}``."""
    tags = {}
    for d in sph_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            m = _SPH_NAME_RE.match(fname)
            if not m or m.group('job') != job:
                continue
            tags.setdefault('r=' + m.group('rtag'), set()).add(m.group('var'))
    return tags


def resolve_style(sph_dirs, radius=None, var=None, job='bns'):
    """Pick the on-disk output style to analyse at ``radius``.

    Handles both the old one-radius-per-file dumps and the new multi-radius
    dumps, and copes with a run that switched format mid-way (each segment
    directory is scanned, and the tag that best matches wins).

    Raises RuntimeError with a readable message if nothing usable is found.
    """
    if radius is None:
        radius = _default_radius
    radius = float(radius)

    all_tags = scan_sph_dirs(sph_dirs, job=job)
    # `var` narrows which tags are eligible (e.g. only consider a radius that
    # actually has adm output), but the recorded variable set stays complete
    # so var_for() can still resolve every other field.
    if var is None:
        tags = all_tags
    else:
        tags = {t: v for t, v in all_tags.items()
                if any(c in v for c in VAR_GROUPS.get(var, (var,)))}
    if not tags:
        what = f" for variable '{var}'" if var else ""
        raise RuntimeError(
            f"No sph VTK files{what} found in: {list(sph_dirs)}\n"
            "Expected names like 'bns.r=300.00.adm.00123.vtk' (single radius) "
            "or 'bns.r=100.00-500.00.adm.00123.vtk' (multi-radius).")

    cands = []
    for r_tag in tags:
        try:
            rmin, rmax, multi = _parse_r_tag(r_tag)
        except ValueError:
            continue
        cands.append(SphStyle(r_tag, rmin, rmax, multi, radius,
                              variables=all_tags[r_tag]))
    if not cands:
        raise RuntimeError(f"Could not parse any radius tag from: {sorted(tags)}")

    def rank(s):
        exact = s.covers(radius) and not s.multi and abs(s.rmin - radius) <= _R_TOL
        # Prefer an exact single-radius match, then any range covering the
        # radius, then whatever lands closest to it.
        if exact:
            return (0, 0.0, s.rmax - s.rmin)
        if s.covers(radius):
            return (1, 0.0, s.rmax - s.rmin)
        dist = min(abs(s.rmin - radius), abs(s.rmax - radius))
        return (2, dist, s.rmax - s.rmin)

    cands.sort(key=rank)
    return cands[0]


def resolve_styles(sph_dirs, radius=None, var=None, job='bns'):
    """Resolve the output style *per directory*.

    A long run can switch output format between segments, so each segment
    directory gets its own :class:`SphStyle`.  Directories with no usable
    files are left out.  Raises RuntimeError if no directory has any.
    """
    if radius is None:
        radius = _default_radius
    styles = {}
    for d in sph_dirs:
        try:
            styles[d] = resolve_style([d], radius=radius, var=var, job=job)
        except RuntimeError:
            continue
    if not styles:
        raise RuntimeError(
            resolve_style.__name__ + ": no sph VTK files"
            + (f" for variable '{var}'" if var else "")
            + f" found in: {list(sph_dirs)}")
    return styles


def describe_styles(styles, radius=None):
    """One-line human summary of a ``resolve_styles`` result."""
    if radius is None:
        radius = _default_radius
    tags = {}
    for d, s in styles.items():
        tags.setdefault(s.r_tag, []).append(d)
    parts = []
    for r_tag, dirs in sorted(tags.items()):
        s = styles[dirs[0]]
        kind = 'multi-radius' if s.multi else 'single-radius'
        parts.append(f"{r_tag} [{kind}, {len(dirs)} dir(s)]")
    out = f"sph output style at r={radius:g} M_sun: " + "; ".join(parts)
    shells = sorted({s.r_tag for s in styles.values()})
    if len(shells) > 1:
        out += "  (format changes between segments)"
    return out
