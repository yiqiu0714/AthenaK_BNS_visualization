#!/usr/bin/env python3
"""Reconstruct a cropped uniform box from an AthenaK prim3D .bin once and save it as HDF5.

Reading a 22 GB prim3D dump takes ~7 min; the cached box (xyz order, x1v/x2v/x3v axes)
is read back by data3d.load_3d_data in seconds, so plot scripts can be iterated on it.

    python cache_box.py <prim3D.bin> <out.h5> <half-width [code]> <resolution>
"""
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data3d import load_3d_data  # noqa: E402


def main():
    src, dst, crop, res = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    d = load_3d_data(src, variables=("dens", "bcc1", "bcc2", "bcc3", "temperature", "s_00"),
                     resolution=res, crop=(-crop, crop, -crop, crop, -crop, crop))
    with h5py.File(dst, "w") as f:
        for k, v in d.fields.items():
            f.create_dataset(k, data=v.astype(np.float32))
        for i, ax in enumerate(d.coordinates, start=1):
            f.create_dataset(f"x{i}v", data=np.asarray(ax))
        f.attrs["axis_order"] = "xyz"
        f.attrs["time"] = d.time
        f.attrs["source"] = src
    print("time", d.time, "shape", d.shape)


if __name__ == "__main__":
    main()
