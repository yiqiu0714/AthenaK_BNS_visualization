"""
snapshot_cache.py — incremental per-snapshot result cache for the SPH analysis.

WHY
    analyze_ejecta.py and analyze_neutrinos.py both do the same thing: loop over
    N SPH snapshots with a Pool, get one independent result dict per snapshot,
    then reduce those into time series / summed histograms. The loop is the
    entire cost, and it was redone from scratch on every invocation — so a run
    that got killed (e.g. by a 1 h queue wall) lost everything, and adding new
    output-* segments meant re-reading every old snapshot again.

    This module persists the per-snapshot results, so a re-run only processes
    snapshots it has never seen. It is the SPH analogue of BNS_all.sh's
    --skip-existing, except the unit of work is a snapshot instead of a PNG.

WHAT IS CACHED
    Only the raw per-snapshot worker output, verbatim. The reduction that turns
    those into Mej_rate_*.txt / Lnu_E_*.txt still runs in full on every
    invocation, over cached + new results together. So the outputs are identical
    whether the cache was cold or warm; the cache can only change HOW LONG the
    run takes, never WHAT it produces.

CORRECTNESS / INVALIDATION
    The cache directory name embeds a hash of everything that can change a
    per-snapshot number: the physics parameters (radius, density floor, EOS
    table, bin edges, ...) AND the source text of the worker function itself.
    Edit the worker, change the radius, or point at a different EOS table and
    you get a DIFFERENT cache directory — stale results can never be picked up.
    Old directories are left on disk (they are not deleted); remove them by hand
    if you want the space back.

LAYOUT   <OUTPUT_DIR>/cache/<name>-<hash>/
    meta.json          the parameters this hash stands for, in plain text
    part-00000.npz     up to CHUNK snapshot records, written once, never edited
    part-00001.npz     ...

    Parts are append-only and each is written to a temp file then os.replace()d,
    which is atomic. A job killed mid-write therefore loses at most the one part
    being written (<= CHUNK snapshots), never the parts already on disk, and can
    never leave a half-read record behind.

CONCURRENCY
    Part filenames are claimed with O_CREAT|O_EXCL and carry the writer's pid,
    so two analyses sharing a cache (e.g. ejecta and neutrinos of the same sim,
    or two jobs racing) cannot clobber each other's parts.
"""

import hashlib
import json
import os
import numpy as np

CHUNK = 40          # snapshots per part file; also the max lost to a kill


def stamp(path):
    """Cheap identity of a source file: (size, mtime_ns). None if missing.

    Stored with every record and re-checked on reuse, so a snapshot whose VTK
    was rewritten -- a rewound/re-run segment reusing the same index, say after
    nan_cleanup.pbs archived it -- is recomputed instead of served stale. This
    is what lets the SPH directory list stay OUT of the cache key, so adding new
    output-* segments reuses everything already computed.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [int(st.st_size), int(st.st_mtime_ns)]


def _fingerprint(params, sources=()):
    """Stable short hash of the parameters + worker source text."""
    h = hashlib.sha1()
    h.update(json.dumps(params, sort_keys=True, default=repr).encode())
    for src in sources:
        h.update(src.encode())
    return h.hexdigest()[:12]


class SnapshotCache:
    """Append-only store of per-snapshot result dicts, keyed by snapshot index."""

    def __init__(self, output_dir, name, params, sources=(), enabled=True):
        self.enabled = bool(enabled)
        self.name    = name
        self.params  = params
        self.key     = _fingerprint(params, sources)
        self.dir     = os.path.join(output_dir, 'cache', f'{name}-{self.key}')
        self._pending = []
        self._parts_written = 0

    # -- reading -----------------------------------------------------------
    def load(self):
        """Return {snapshot_index: record}. Empty dict if disabled/cold."""
        out = {}
        if not self.enabled or not os.path.isdir(self.dir):
            return out
        for fname in sorted(os.listdir(self.dir)):
            if not fname.startswith('part-') or not fname.endswith('.npz'):
                continue
            path = os.path.join(self.dir, fname)
            try:
                with np.load(path, allow_pickle=True) as z:
                    for idx, rec in zip(z['idx'], z['rec']):
                        # object-array elements come back as the dict itself,
                        # but a 0-d ndarray wrapper is possible -> unwrap both
                        if isinstance(rec, np.ndarray) and rec.ndim == 0:
                            rec = rec.item()
                        out[int(idx)] = rec
            except Exception as exc:                     # corrupt/partial part
                print(f'  [cache] ignoring unreadable {fname}: {exc}')
        return out

    # -- writing -----------------------------------------------------------
    def add(self, idx, record):
        """Buffer one record; a part is written every CHUNK additions."""
        if not self.enabled:
            return
        self._pending.append((int(idx), record))
        if len(self._pending) >= CHUNK:
            self.flush()

    def flush(self):
        """Write buffered records as a new part file (atomic)."""
        if not self.enabled or not self._pending:
            return
        os.makedirs(self.dir, exist_ok=True)
        meta = os.path.join(self.dir, 'meta.json')
        if not os.path.exists(meta):
            with open(meta, 'w') as fh:
                json.dump({'name': self.name, 'key': self.key,
                           'params': {k: repr(v) for k, v in self.params.items()}},
                          fh, indent=2, sort_keys=True)

        # claim a part name nobody else holds
        n = self._parts_written
        while True:
            path = os.path.join(self.dir, f'part-{n:05d}.npz')
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.close(fd)
                break
            except FileExistsError:
                n += 1
        self._parts_written = n + 1

        idx = np.array([i for i, _ in self._pending], dtype=np.int64)
        rec = np.empty(len(self._pending), dtype=object)
        for k, (_, r) in enumerate(self._pending):
            rec[k] = r
        # NB the temp name must already end in .npz, else savez appends it
        tmp = f'{path}.{os.getpid()}.tmp.npz'
        np.savez_compressed(tmp, idx=idx, rec=rec)
        os.replace(tmp, path)
        self._pending = []

    # -- reporting ---------------------------------------------------------
    def describe(self, n_cached, n_missing):
        if not self.enabled:
            return f'  [cache] disabled — processing all {n_missing} snapshots'
        return (f'  [cache] {self.dir}\n'
                f'  [cache] {n_cached} snapshot(s) reused, {n_missing} to process')


class TimeIndexCache:
    """Cache of VTK header times, keyed by file path + (size, mtime).

    WHY
        analyze_neutrinos builds a time index over the adm and four z4c gauge
        outputs before it can match each flux snapshot to its metric files.
        That is ~5 header reads per snapshot, done SERIALLY and BEFORE the
        per-snapshot cache is consulted -- for a 200-segment run that is ~60k
        file opens, which on its own exceeds a 1 h queue wall. The work was
        then thrown away and redone from scratch by the next job, so the
        analysis could never finish no matter how many times it was resumed.

        A file's header time never changes unless the file itself does, so it
        is cacheable on exactly the same terms as a snapshot result.

    The store is a single npz rewritten in place (atomically), flushed every
    FLUSH_EVERY new entries so a killed job keeps most of its scan.
    """

    FLUSH_EVERY = 5000

    def __init__(self, output_dir, name, enabled=True):
        self.enabled = bool(enabled)
        self.path    = os.path.join(output_dir, 'cache', f'{name}.npz')
        self._map    = {}      # path -> (time, size, mtime_ns)
        self._new    = 0
        self.hits = self.misses = 0
        self._load()

    def _load(self):
        if not self.enabled or not os.path.exists(self.path):
            return
        try:
            with np.load(self.path, allow_pickle=False) as z:
                for pth, t, sz, mt in zip(z['path'], z['time'], z['size'], z['mtime']):
                    self._map[str(pth)] = (float(t), int(sz), int(mt))
        except Exception as exc:
            print(f'  [timeindex] ignoring unreadable cache: {exc}')
            self._map = {}

    def lookup(self, path, size, mtime_ns):
        """Cached header time for this exact file, or None."""
        rec = self._map.get(path)
        if rec is not None and rec[1] == int(size) and rec[2] == int(mtime_ns):
            self.hits += 1
            return rec[0]
        self.misses += 1
        return None

    def put(self, path, size, mtime_ns, t):
        if not self.enabled:
            return
        self._map[path] = (float(t), int(size), int(mtime_ns))
        self._new += 1
        if self._new >= self.FLUSH_EVERY:
            self.save()

    def save(self):
        if not self.enabled or not self._new:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        items = sorted(self._map.items())
        tmp = f'{self.path}.{os.getpid()}.tmp.npz'
        np.savez_compressed(
            tmp,
            path=np.array([k for k, _ in items]),
            time=np.array([v[0] for _, v in items], dtype=np.float64),
            size=np.array([v[1] for _, v in items], dtype=np.int64),
            mtime=np.array([v[2] for _, v in items], dtype=np.int64))
        os.replace(tmp, self.path)
        self._new = 0
