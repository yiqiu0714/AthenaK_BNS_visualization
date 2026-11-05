#!/usr/bin/env python3
"""Quantify how dipolar each star's magnetic field is in a cached BNS box.

For each star (the two density maxima):
  axis    unit vector of the volume-averaged B inside r < CORE (a poloidal dipole has a
          nearly uniform core field along its axis)
  f_tor   fraction of magnetic energy in the toroidal (phi) component about that axis,
          inside the star (rho > RHO_IN g/cm^3): ~0 for a pure poloidal dipole, ~1/3 for
          an isotropic tangle, -> 1 for a wound-up toroidal field
  coh     |<B>| / <|B|> in the core: 1 for a uniform core field, ~0 for a tangle
  E_out/E_in  magnetic energy in the low-density surroundings (1e8 < rho < 1e11, r < 2R)
          relative to the star interior

    python field_geometry.py box1.h5 [box2.h5 ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data3d import load_3d_data                      # noqa: E402
from style3d import convert_fields, make_coordinate_axes   # noqa: E402

RHO_IN = 1e12      # g/cm^3, "inside the star"
CORE = 4.0         # code length, core radius for the axis
R_STAR = 8.5       # code length, approximate stellar radius
T_MERGER = 4247.5  # code time, 46_mb32_n256
MSUN_MS = 4.925491e-3


def star_centres(dens, X, Y, Z, min_sep=8.0):
    i1 = np.unravel_index(np.nanargmax(dens), dens.shape)
    c1 = np.array([X[i1], Y[i1], Z[i1]])
    far = (X - c1[0]) ** 2 + (Y - c1[1]) ** 2 + (Z - c1[2]) ** 2 > min_sep ** 2
    i2 = np.unravel_index(np.nanargmax(np.where(far, dens, -np.inf)), dens.shape)
    return [c1, np.array([X[i2], Y[i2], Z[i2]])]


def analyse(path):
    ds = load_3d_data(path, variables=("dens", "bcc1", "bcc2", "bcc3"))
    convert_fields(ds.fields, "cgs")
    X, Y, Z = np.meshgrid(*make_coordinate_axes(ds.shape, ds.domain_size, ds.coordinates),
                          indexing="ij")
    rho = ds.fields["dens"]
    B = np.stack([ds.fields[c] for c in ("bcc1", "bcc2", "bcc3")], axis=-1).astype(np.float64)
    B2 = (B ** 2).sum(-1)
    tms = (ds.time - T_MERGER) * MSUN_MS
    print(f"\n{os.path.basename(path)}  t = {ds.time:.1f} M  (t - t_merger = {tms:.1f} ms)")
    for k, c in enumerate(star_centres(rho, X, Y, Z)):
        d = np.stack([X - c[0], Y - c[1], Z - c[2]], axis=-1)
        r = np.linalg.norm(d, axis=-1)
        core = r < CORE
        mean_b = B[core].mean(0)
        axis = mean_b / np.linalg.norm(mean_b)
        coh = np.linalg.norm(mean_b) / np.sqrt(B2[core]).mean()
        inside = (rho > RHO_IN) & (r < 2 * R_STAR)
        # toroidal unit vector about `axis` through the star centre
        rperp = d - (d @ axis)[..., None] * axis
        phi = np.cross(axis, rperp)
        nphi = np.linalg.norm(phi, axis=-1, keepdims=True)
        phi = np.divide(phi, nphi, out=np.zeros_like(phi), where=nphi > 1e-6)
        bphi = (B * phi).sum(-1)
        f_tor = (bphi[inside] ** 2).sum() / B2[inside].sum()
        out = (rho > 1e8) & (rho < 1e11) & (r < 2 * R_STAR)
        e_ratio = B2[out].sum() / B2[inside].sum()
        bmax_in = np.sqrt(B2[inside].max())
        bmax_out = np.sqrt(B2[out].max()) if out.any() else 0.0
        print(f"  star {k}: axis=({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f})  "
              f"f_tor={f_tor:.2f}  coh={coh:.2f}  E_out/E_in={e_ratio:.2e}  "
              f"Bmax_in={bmax_in:.1e} G  Bmax_out={bmax_out:.1e} G")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyse(p)
