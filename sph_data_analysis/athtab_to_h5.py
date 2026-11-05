#!/usr/bin/env python3
"""athtab_to_h5.py — convert an AthenaK .athtab EOS table to PyCompOSE HDF5.

The ejecta step (analyze_ejecta.load_eos) needs a PyCompOSE HDF5 table with the
datasets nb / yq / t / mn / Q1 / Q7.  AthenaK itself is fed the *same* table in
the binary .athtab format (parfile <mhd>/<dyn_eos> `table = ...`), which
PyCompOSE writes alongside the HDF5.  When only the .athtab is on disk (e.g.
SFHo), this script reconstructs the HDF5 from it — no re-download, and the
result is guaranteed to be the table the simulation actually ran with.

Usage:
    python3 athtab_to_h5.py SFHo.athtab SFHo.h5

.athtab layout (see PyCompOSE compose/eos.py:write_athtab): an ASCII header
(<metadatabegin>/<scalarsbegin>/<pointsbegin>/<fieldsbegin> blocks) followed by
the raw binary arrays nb[nn], yq[ny], t[nt], then every listed field flattened
in (nb, yq, t) C order.
"""

import argparse
import sys

import h5py
import numpy as np


def read_athtab(path):
    """Parse an .athtab file into (scalars, axes, fields)."""
    meta, scalars, points, fields = {}, {}, {}, []
    section = None
    header_bytes = 0

    with open(path, 'rb') as f:
        for raw in f:
            header_bytes += len(raw)
            line = raw.decode('ascii').strip()
            if line.endswith('begin>'):
                section = line[1:-6]
                continue
            if line.endswith('end>'):
                if section == 'fields':
                    break
                section = None
                continue
            if section == 'fields':
                fields.append(line)
            elif '=' in line:
                key, val = (s.strip() for s in line.split('=', 1))
                {'metadata': meta, 'scalars': scalars, 'points': points}[section][key] = val

    if meta.get('precision', 'double') != 'double':
        raise RuntimeError(f"unsupported precision {meta['precision']!r} (only double)")
    dtype = np.dtype('<f8' if meta.get('endianness', sys.byteorder) == 'little' else '>f8')

    nn, ny, nt = (int(points[k]) for k in ('nb', 'yq', 't'))
    flat = np.fromfile(path, dtype=dtype, offset=header_bytes)

    expected = nn + ny + nt + len(fields) * nn * ny * nt
    if flat.size != expected:
        raise RuntimeError(f'{path}: got {flat.size} doubles, expected {expected}')

    axes = {'nb': flat[:nn], 'yq': flat[nn:nn + ny], 't': flat[nn + ny:nn + ny + nt]}
    body = flat[nn + ny + nt:].reshape(len(fields), nn, ny, nt)
    return ({k: float(v) for k, v in scalars.items()}, axes,
            {name: body[i] for i, name in enumerate(fields)})


def write_h5(path, scalars, axes, fields):
    with h5py.File(path, 'w') as f:
        for name, val in scalars.items():
            f.create_dataset(name, data=val)
        for name, arr in axes.items():
            f.create_dataset(name, data=arr)
        for name, arr in fields.items():
            f.create_dataset(name, data=arr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('athtab', help='input .athtab table')
    ap.add_argument('h5', help='output PyCompOSE .h5 table')
    args = ap.parse_args()

    scalars, axes, fields = read_athtab(args.athtab)
    print(f'{args.athtab}: nb={axes["nb"].size} yq={axes["yq"].size} t={axes["t"].size}, '
          f'{len(fields)} fields {sorted(fields)}')
    missing = [q for q in ('Q1', 'Q7') if q not in fields]
    if missing:
        raise RuntimeError(f'table lacks {missing}; analyze_ejecta needs Q1 and Q7')
    write_h5(args.h5, scalars, axes, fields)
    print(f'wrote {args.h5}')


if __name__ == '__main__':
    main()
