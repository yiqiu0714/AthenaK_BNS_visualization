#!/usr/bin/env python3
"""eos_from_run.py — find the EOS table a run used, from the run's own output.

Every AthenaK .bin output embeds a PAR_DUMP of the parfile in its ASCII header,
including the dynamical EOS table:

    <mhd>
    dyn_eos = compose
    table   = /home/.../SFHo.athtab      <- the table this run actually used

so the analysis never has to be told which EOS a simulation is -- it can read it
back out of the data. (The <problem> block holds a second `table`, the cold one
the Lorene pgen reads; that is NOT the one we want.)

The ejecta step needs the PyCompOSE HDF5 twin of that .athtab; this script maps
<stem>.athtab -> <stem>.h5 and checks it exists. Build a missing one with
athtab_to_h5.py.

Usage:
    python3 eos_from_run.py --simpath /path/to/sim            # print the .h5
    python3 eos_from_run.py --simpath /path/to/sim --athtab   # print the .athtab
"""

import argparse
import glob
import os
import sys


def read_par_dump(binfile):
    """Parse the PAR_DUMP header of an AthenaK .bin file into {block: {k: v}}."""
    par = {}
    with open(binfile, 'rb') as f:
        if f.readline() != b'Athena binary output version=1.1\n':
            raise RuntimeError(f'{binfile}: unrecognized data file format')
        offset = None
        for _ in range(16):                      # preheader is short; bail out safely
            line = f.readline().decode('ascii')
            if line.startswith('  header offset='):
                offset = int(line[16:])
                break
        if offset is None:
            raise RuntimeError(f'{binfile}: could not read header offset')

        end = f.tell() + offset
        block = None
        while f.tell() < end:
            line = f.readline().decode('ascii')
            if not line or line[0] == '#':
                continue
            if line[0] == '<':
                block = line[1:-2]
                par[block] = {}
            elif '=' in line and block is not None:
                key, val = line.split('=', 1)
                par[block][key.strip()] = val.split('#', 1)[0].strip()
    return par


def find_bin(simpath, jobname='bns'):
    """Return any .bin output under simpath (they all carry the same PAR_DUMP)."""
    for pattern in (f'output-*/bin/{jobname}.*.bin', 'output-*/bin/*.bin', '*.bin'):
        hits = sorted(glob.glob(os.path.join(simpath, pattern)))
        if hits:
            return hits[0]
    raise RuntimeError(f'no .bin outputs found under {simpath}')


def eos_table_for_run(simpath, jobname='bns'):
    """Return (athtab_path, h5_path, binfile) for the run's dynamical EOS table."""
    binfile = find_bin(simpath, jobname)
    par = read_par_dump(binfile)

    for block in ('mhd', 'hydro'):
        athtab = par.get(block, {}).get('table')
        if athtab:
            break
    else:
        raise RuntimeError(
            f'{binfile}: no <mhd>/<hydro> `table` in the PAR_DUMP -- this run '
            'does not use a tabulated EOS (dyn_eos = '
            f'{par.get("mhd", {}).get("dyn_eos", "?")}).')

    stem, ext = os.path.splitext(athtab)
    return athtab, stem + '.h5', binfile


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--simpath', required=True, help='run directory (holds output-*/)')
    ap.add_argument('--jobname', default='bns', help='AthenaK job basename (default: bns)')
    ap.add_argument('--athtab', action='store_true',
                    help='print the .athtab path instead of the .h5 twin')
    args = ap.parse_args()

    athtab, h5, binfile = eos_table_for_run(args.simpath, args.jobname)
    if args.athtab:
        print(athtab)
        return

    if not os.path.isfile(h5):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'athtab_to_h5.py')
        sys.exit(
            f"ERROR: the run used '{athtab}' (per {binfile}),\n"
            f"       but its PyCompOSE HDF5 twin '{h5}' does not exist.\n"
            f"       Build it with:\n"
            f"         python3 {script} \\\n"
            f"             {athtab} \\\n"
            f"             {h5}\n"
            f"       (or set eos_table explicitly in config.ini)")
    print(h5)


if __name__ == '__main__':
    main()
