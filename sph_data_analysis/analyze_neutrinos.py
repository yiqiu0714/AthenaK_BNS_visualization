"""
Neutrino luminosity analysis from rad_m1_F / rad_m1_E / rad_m1_N spherical-surface
VTK outputs of AthenaK BNS simulations.

Per snapshot, loads 8 VTK files:
  bns.r=R.rad_m1_F.NNNNN.vtk  — Fx:s, Fy:s, Fz:s  (Eulerian energy-flux per species)
  bns.r=R.rad_m1_E.NNNNN.vtk  — E:s                (Eulerian energy density per species)
  bns.r=R.rad_m1_N.NNNNN.vtk  — N:s                (number density per species)
  bns.r=R.adm.NNNNN.vtk       — adm_gxx..gyz       (ADM 3-metric)
  bns.r=R.z4c_alpha.NNNNN.vtk — z4c_alpha           (lapse)
  bns.r=R.z4c_beta*.NNNNN.vtk — z4c_beta*           (shift)

Physics
-------
Energy luminosity per species s:
    L_E,s(t) = ∫ (α F^r_s - β^r E_s) · r² dΩ
             [code units → erg/s]
where F^i_s = γ^ij F_j,s, using the stored M1 flux covector components.
The radial projection uses the metric-normalized coordinate radial covector
used by THC: rhat_i = r_i / sqrt(g^ij_4 r_i r_j), where g^ij_4 is the
spatial block of the inverse 4-metric.
E_s and F_i,s are already densitized M1 evolved variables, so no extra √γ
factor is applied in the surface integral.
This matches the THC/AthenaK M1 calc_E_flux expression.
The angular area integral uses AthenaK's native SPH `weights` field:
    weights = r² dΩ  (Gauss-Legendre in μ = cos θ, uniform in φ)

Flux-weighted mean energy per species:
    <ε_s>(t) = ∫ F^r_E,s · r² dΩ / ∫ F^r_N,s · r² dΩ  → MeV
THC constructs the denominator from the M1 number flux on the detector sphere.
The AthenaK SPH files used here store E_s, F_i,s, and N_s but not the reconstructed
number-current direction fnu^a, so the post-process uses the same radial transport
velocity as the energy luminosity:
    F^r_N,s ≈ F^r_E,s · N_s / E_s.
This is exact when the energy and number currents are advected with the same
radial velocity, and is the appropriate free-streaming detector-sphere limit.

Species ordering (M1_TOTAL_NUM_SPECIES = 4):
    s = 0  νe   (electron neutrino)
    s = 1  ν̄e   (electron antineutrino)
    s = 2  νx   (heavy-flavour neutrino)
    s = 3  ν̄x   (heavy-flavour antineutrino)
Note: in bns_nurates, species 2 and 3 are symmetric (nux = anux).  The code
stores each separately; both are output here.

Required parfile output blocks (add for each desired radius):
    <outputXX>
    file_type  = sph
    variable   = rad_m1_F
    dt         = 5.0
    radius     = 400.0
    ntheta     = 256

    <outputXX+1>
    file_type  = sph
    variable   = rad_m1_E
    dt         = 5.0
    radius     = 400.0
    ntheta     = 256

    <outputXX+2>
    file_type  = sph
    variable   = rad_m1_N
    dt         = 5.0
    radius     = 400.0
    ntheta     = 256

Output files (written to OUTPUT_DIR):
    Lnu_E_{sp}.txt        time, energy luminosity per species [M_sun], [erg/s]
    Lnu_E_total.txt       time, sum over all 4 species         [M_sun], [erg/s]
    Eav_{sp}.txt          time, flux-weighted mean energy       [M_sun], [MeV]
    dEnu_dtheta_{sp}.txt  θ, time-integrated dE_ν/dθ           [rad],   [erg/rad]
"""

import os
import sys
import numpy as np
from multiprocessing import Pool, cpu_count

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
# plot-tools submodule lives at the repository root (one level above this folder).
_PLOTTOOLS = os.environ.get(
    "ATHPLOT_PATH", os.path.abspath(os.path.join(SCRIPTDIR, os.pardir, "plot-tools")))
sys.path.append(_PLOTTOOLS)
sys.path.append(SCRIPTDIR)  # utils_ejecta.py lives alongside this script

from athplot.load_sph_vtk import SphericalData
from athplot.utils.units import conv_luminosity, conv_energy, cactus, cgs
from utils_ejecta import sum_over_time


# ===================================================================
# Configuration  — edit this block before running
# ===================================================================

# --- Simulation ---
SIM_NAME = 'M135_M135_D45_DD2_VVLR_AMR_rererun'
SPH_DIR_BASENAMES = [
    'output_scratch/output-0000/sph',
    'output_scratch/output-0001/sph',
    'output_scratch/output-0002/sph',
]

# Extraction radius [code units, M_sun] — must match a radius in the parfile
RADIUS = 300.0

# --- Data directories ---
SPH_DIR_LIST = [os.path.join('../sim', SIM_NAME, b) for b in SPH_DIR_BASENAMES]

# --- Output ---
OUTPUT_DIR = os.path.join('output', SIM_NAME)

# --- Parallelism ---
# Number of worker processes for the snapshot loop.
# Set to 1 to disable multiprocessing (easier debugging).
N_WORKERS = min(8, cpu_count())


# ===================================================================
# Constants
# ===================================================================

# Species labels matching AthenaK's M1 ordering
SPECIES  = ['nue', 'nua', 'nux', 'anux']
NSPECIES = len(SPECIES)

# AthenaK stores N:s (neutrino number density) in EOS units (fm^-3) while
# E:s is in code energy-density units (M_sun^-2).  To form a consistent ratio,
# convert N from fm^-3 to code units (M_sun^-3):
#   1 code_length = G M_sun / c^2 = 1.477e18 fm
#   1 M_sun^3     = (1.477e18)^3 fm^3 = 3.222e54 fm^3
#   ∴ 1 fm^-3     = 3.222e54 M_sun^-3
_G_SI        = 6.67430e-11
_M_SUN_KG    = 1.98892e30
_C_SI        = 2.99792458e8
_FM_PER_M    = 1e15
N_CODE_PER_FM3 = (_G_SI * _M_SUN_KG / _C_SI**2 * _FM_PER_M)**3  # ≈ 3.222e54


# ===================================================================
# Main analysis
# ===================================================================

def _build_time_list(sph_dir, r_str, prefix):
    """Return sorted list of (time_float, path) for all matching VTK files."""
    result = []
    pat = f'bns.{r_str}.{prefix}.'
    for fname in os.listdir(sph_dir):
        if fname.startswith(pat) and fname.endswith('.vtk'):
            path = os.path.join(sph_dir, fname)
            # Read only the metadata line (avoids the full VTK parse)
            with open(path, 'r', encoding='latin_1') as f:
                f.readline()
                meta = f.readline()
            for tok in meta.split():
                if tok.startswith('time='):
                    result.append((float(tok[5:]), path))
                    break
    result.sort(key=lambda x: x[0])
    return result


def _nearest_path(time_list, t, tol=4.0):
    """Return path of snapshot with time nearest to t, or None if outside tol."""
    if not time_list:
        return None
    times = np.array([x[0] for x in time_list])
    idx   = np.searchsorted(times, t)
    # check both neighbours
    best_i, best_dt = None, np.inf
    for i in (idx - 1, idx):
        if 0 <= i < len(times):
            dt = abs(times[i] - t)
            if dt < best_dt:
                best_dt, best_i = dt, i
    if best_dt <= tol:
        return time_list[best_i][1]
    return None


def _inverse_spatial_metric(gxx, gyy, gzz, gxy, gxz, gyz):
    gamma_det = (gxx * gyy * gzz +
                 2 * gxy * gxz * gyz -
                 gxx * gyz**2 -
                 gyy * gxz**2 -
                 gzz * gxy**2)

    guxx = (gyy * gzz - gyz**2) / gamma_det
    guyy = (gxx * gzz - gxz**2) / gamma_det
    guzz = (gxx * gyy - gxy**2) / gamma_det
    guxy = (gxz * gyz - gxy * gzz) / gamma_det
    guxz = (gxy * gyz - gxz * gyy) / gamma_det
    guyz = (gxy * gxz - gxx * gyz) / gamma_det
    return guxx, guyy, guzz, guxy, guxz, guyz


def _raise_spatial_covector(Fx, Fy, Fz, gxx, gyy, gzz, gxy, gxz, gyz):
    guxx, guyy, guzz, guxy, guxz, guyz = _inverse_spatial_metric(
        gxx, gyy, gzz, gxy, gxz, gyz)

    Fux = guxx * Fx + guxy * Fy + guxz * Fz
    Fuy = guxy * Fx + guyy * Fy + guyz * Fz
    Fuz = guxz * Fx + guyz * Fy + guzz * Fz
    return Fux, Fuy, Fuz


def _metric_normalized_radial_projection(Vx, Vy, Vz,
                                         gxx, gyy, gzz, gxy, gxz, gyz,
                                         alpha, betax, betay, betaz,
                                         theta, phi, radius):
    """Project V^i along r_i / sqrt(g^ij_4 r_i r_j), matching THC."""
    guxx, guyy, guzz, guxy, guxz, guyz = _inverse_spatial_metric(
        gxx, gyy, gzz, gxy, gxz, gyz)
    oo_alpha2 = 1.0 / alpha**2

    # THC normalizes r_i with the inverse 4-metric.  Its spatial block is
    # g^ij_4 = gamma^ij - beta^i beta^j / alpha^2.
    g4uxx = guxx - betax * betax * oo_alpha2
    g4uyy = guyy - betay * betay * oo_alpha2
    g4uzz = guzz - betaz * betaz * oo_alpha2
    g4uxy = guxy - betax * betay * oo_alpha2
    g4uxz = guxz - betax * betaz * oo_alpha2
    g4uyz = guyz - betay * betaz * oo_alpha2

    rx = radius * np.sin(theta) * np.cos(phi)
    ry = radius * np.sin(theta) * np.sin(phi)
    rz = radius * np.cos(theta)

    rr_metric = np.sqrt(g4uxx * rx**2 + g4uyy * ry**2 + g4uzz * rz**2 +
                        2.0 * g4uxy * rx * ry +
                        2.0 * g4uxz * rx * rz +
                        2.0 * g4uyz * ry * rz)
    return (rx * Vx + ry * Vy + rz * Vz) / rr_metric


def _process_snapshot(args):
    """Per-snapshot worker — runs in a separate process.

    Imports are done inside the function so the worker is self-contained and
    can be pickled by multiprocessing without carrying module-level state.
    """
    (flux_path, ener_path, nden_path, adm_path, alpha_path,
     betax_path, betay_path, betaz_path,
     theta_edges) = args

    import numpy as _np
    from athplot.load_sph_vtk import SphericalData as _SD
    from athplot.utils.units import conv_luminosity as _conv_lum, \
                                    conv_energy as _conv_e, \
                                    cactus as _cactus, cgs as _cgs

    flux_vtk  = _SD(flux_path)
    ener_vtk  = _SD(ener_path)
    nden_vtk  = _SD(nden_path)
    adm_vtk   = _SD(adm_path)
    alpha_vtk = _SD(alpha_path)
    betax_vtk = _SD(betax_path)
    betay_vtk = _SD(betay_path)
    betaz_vtk = _SD(betaz_path)

    time  = flux_vtk.time
    radius = flux_vtk.r
    theta = flux_vtk.theta
    phi   = flux_vtk.phi
    surface_weights = flux_vtk.fields['weights']

    # -- Duplication correction --
    z4c_alpha_raw = alpha_vtk.fields['z4c_alpha']
    phi_zero    = (phi == phi.min())
    theta_north = (theta == theta.min())
    dup_corr = _np.ones(z4c_alpha_raw.shape)
    if z4c_alpha_raw[phi_zero & ~theta_north].max() > 1.0:
        dup_corr[phi_zero] = 0.5
    if z4c_alpha_raw[theta_north].max() > 1.0:
        dup_corr[theta_north] = 0.25

    gxx = adm_vtk.fields['adm_gxx'] * dup_corr
    gyy = adm_vtk.fields['adm_gyy'] * dup_corr
    gzz = adm_vtk.fields['adm_gzz'] * dup_corr
    gxy = adm_vtk.fields['adm_gxy'] * dup_corr
    gxz = adm_vtk.fields['adm_gxz'] * dup_corr
    gyz = adm_vtk.fields['adm_gyz'] * dup_corr
    z4c_alpha  = z4c_alpha_raw * dup_corr
    z4c_betax  = betax_vtk.fields['z4c_betax'] * dup_corr
    z4c_betay  = betay_vtk.fields['z4c_betay'] * dup_corr
    z4c_betaz  = betaz_vtk.fields['z4c_betaz'] * dup_corr

    nspecies = 4
    # N unit conversion: fm^-3 → M_sun^-3
    _G, _M, _C, _FM = 6.67430e-11, 1.98892e30, 2.99792458e8, 1e-15
    n_conv = (_G * _M / _C**2 / _FM)**3

    Lnu_E_list   = []
    Eav_list     = []
    dLdtheta_list = []

    dtheta = _np.diff(theta_edges)

    for s in range(nspecies):
        Fx = flux_vtk.fields[f'Fx:{s}'] * dup_corr
        Fy = flux_vtk.fields[f'Fy:{s}'] * dup_corr
        Fz = flux_vtk.fields[f'Fz:{s}'] * dup_corr
        E  = ener_vtk.fields[f'E:{s}']  * dup_corr
        N  = nden_vtk.fields[f'N:{s}']  * dup_corr

        Fux, Fuy, Fuz = _raise_spatial_covector(
            Fx, Fy, Fz, gxx, gyy, gzz, gxy, gxz, gyz)
        Fr            = _metric_normalized_radial_projection(
            Fux, Fuy, Fuz, gxx, gyy, gzz, gxy, gxz, gyz,
            z4c_alpha, z4c_betax, z4c_betay, z4c_betaz,
            theta, phi, radius)
        beta_r        = _metric_normalized_radial_projection(
            z4c_betax, z4c_betay, z4c_betaz,
            gxx, gyy, gzz, gxy, gxz, gyz,
            z4c_alpha, z4c_betax, z4c_betay, z4c_betaz,
            theta, phi, radius)
        radial_flux   = z4c_alpha * Fr - beta_r * E
        L_code        = _np.sum(radial_flux * surface_weights)
        L_erg         = _conv_lum(_cactus, _cgs, L_code)
        Lnu_E_list.append(L_erg)

        # THC's detector mean energy is energy luminosity divided by number
        # luminosity.  The exact M1 number flux uses fnu^a, which is not stored
        # in the current SPH output, so use the same radial transport velocity
        # as the energy flux: F_N^r ~= F_E^r * N/E.
        N_code = N * n_conv
        number_flux = _np.zeros_like(radial_flux)
        _np.divide(radial_flux * N_code, E, out=number_flux, where=(E > 0.0))
        Ndot_code = _np.sum(number_flux * surface_weights)
        eps_MeV = (_conv_e(_cactus, _cgs, L_code / Ndot_code) / (_cgs.eV * 1e6)
                   if Ndot_code > 0.0 else 0.0)
        Eav_list.append(eps_MeV)

        outgoing_L  = _np.where(radial_flux > 0.0, radial_flux, 0.0)
        dL_theta, _ = _np.histogram(
            theta.flatten(), bins=theta_edges,
            weights=outgoing_L.flatten() * surface_weights.flatten(),
        )
        dLdtheta_list.append(dL_theta / dtheta)

    return {'time': time, 'Lnu_E': Lnu_E_list,
            'Eav': Eav_list, 'dLdtheta': dLdtheta_list}


def main():
    # ------------------------------------------------------------------
    # Build snapshot index: scan for rad_m1_F files (the new M1 output).
    # adm / z4c_alpha may use a different counter series (different restart
    # offset) but cover the same simulation times, so we match by time.
    # ------------------------------------------------------------------
    r_str = f'r={RADIUS:.2f}'

    m1_index_map = {}
    for d in SPH_DIR_LIST:
        if not os.path.exists(d):
            continue
        for fname in os.listdir(d):
            if fname.startswith(f'bns.{r_str}.rad_m1_F.') and fname.endswith('.vtk'):
                idx = int(fname.split('.')[-2])
                m1_index_map[idx] = d

    indices = sorted(m1_index_map.keys())
    print(f"Found {len(indices)} rad_m1_F snapshots at r={RADIUS} M_sun")
    if len(indices) == 0:
        raise RuntimeError(
            f"No rad_m1_F VTK files found in {SPH_DIR_LIST}.\n"
            "Add 'variable = rad_m1_F / rad_m1_E / rad_m1_N' sph output blocks to the parfile."
        )

    # Build sorted time lists for adm and z4c gauge outputs (counter may differ)
    print("Building time index for adm / z4c gauge files...")
    adm_times    = []
    alpha_times  = []
    betax_times  = []
    betay_times  = []
    betaz_times  = []
    for d in SPH_DIR_LIST:
        if not os.path.exists(d):
            continue
        adm_times.extend(_build_time_list(d, r_str, 'adm'))
        alpha_times.extend(_build_time_list(d, r_str, 'z4c_alpha'))
        betax_times.extend(_build_time_list(d, r_str, 'z4c_betax'))
        betay_times.extend(_build_time_list(d, r_str, 'z4c_betay'))
        betaz_times.extend(_build_time_list(d, r_str, 'z4c_betaz'))
    adm_times.sort(key=lambda x: x[0])
    alpha_times.sort(key=lambda x: x[0])
    betax_times.sort(key=lambda x: x[0])
    betay_times.sort(key=lambda x: x[0])
    betaz_times.sort(key=lambda x: x[0])
    print(f"  adm files:       {len(adm_times)}")
    print(f"  z4c_alpha files: {len(alpha_times)}")
    print(f"  z4c_beta  files: {len(betax_times)}")

    # ------------------------------------------------------------------
    # Build theta bins for dL/dtheta diagnostics.  Surface integrals use the
    # native AthenaK SPH weights stored in each VTK file.
    # ------------------------------------------------------------------
    first_flux_path = os.path.join(
        m1_index_map[indices[0]], f'bns.{r_str}.rad_m1_F.{indices[0]:05d}.vtk')
    _fv = SphericalData(first_flux_path)
    theta_1d        = _fv.theta[0, :]
    theta_1d_sorted = np.sort(theta_1d)
    theta_edges     = np.concatenate([[0.0],
                                      0.5 * (theta_1d_sorted[:-1] + theta_1d_sorted[1:]),
                                      [np.pi]])
    theta_centers   = theta_1d_sorted
    del _fv

    # dup check on first snapshot with alpha
    _t0 = SphericalData(first_flux_path).time
    _ap = _nearest_path(alpha_times, _t0)
    if _ap:
        _av = SphericalData(_ap)
        _a  = _av.fields['z4c_alpha']
        print(f"  [dup check] alpha range: min={_a.min():.4f}  max={_a.max():.4f}")

    # ------------------------------------------------------------------
    # Build worker argument list (pre-resolve adm/alpha paths by time)
    # ------------------------------------------------------------------
    worker_args = []
    skipped = 0
    for index in indices:
        sph_dir = m1_index_map[index]
        pfx     = os.path.join(sph_dir, f'bns.{r_str}.')
        flux_path = pfx + f'rad_m1_F.{index:05d}.vtk'
        t_flux = SphericalData(flux_path).time   # read header only via VTK
        adm_path   = _nearest_path(adm_times,   t_flux)
        alpha_path = _nearest_path(alpha_times, t_flux)
        bx_path    = _nearest_path(betax_times, t_flux)
        by_path    = _nearest_path(betay_times, t_flux)
        bz_path    = _nearest_path(betaz_times, t_flux)
        if (adm_path is None or alpha_path is None or bx_path is None or
                by_path is None or bz_path is None):
            skipped += 1
            continue
        worker_args.append((
            flux_path,
            pfx + f'rad_m1_E.{index:05d}.vtk',
            pfx + f'rad_m1_N.{index:05d}.vtk',
            adm_path, alpha_path,
            bx_path, by_path, bz_path,
            theta_edges,
        ))
    if skipped:
        print(f"  [skip] {skipped} snapshots with no adm/alpha/beta match")
    print(f"Processing {len(worker_args)} snapshots with N_WORKERS={N_WORKERS}...")

    # ------------------------------------------------------------------
    # Parallel snapshot loop
    # ------------------------------------------------------------------
    if N_WORKERS > 1:
        with Pool(N_WORKERS) as pool:
            results = list(pool.imap_unordered(_process_snapshot, worker_args,
                                               chunksize=max(1, len(worker_args) // (N_WORKERS * 4))))
    else:
        results = [_process_snapshot(a) for a in worker_args]

    # Progress summary
    results.sort(key=lambda x: x['time'])
    for i, res in enumerate(results):
        if i % 50 == 0:
            print(f"  snapshot {i:4d}/{len(results)}  t={res['time']:.2f} M_sun  "
                  f"L_nue={res['Lnu_E'][0]:.3e} erg/s")

    # ------------------------------------------------------------------
    # Unpack results into time-sorted accumulators
    # ------------------------------------------------------------------
    time_arr     = np.array([r['time']    for r in results])
    Lnu_E_snap   = [np.array([r['Lnu_E'][s]   for r in results]) for s in range(NSPECIES)]
    Eav_snap     = [np.array([r['Eav'][s]     for r in results]) for s in range(NSPECIES)]
    dLdtheta_snap = [np.array([r['dLdtheta'][s] for r in results]) for s in range(NSPECIES)]

    # Total luminosity across all species (already time-sorted)
    Lnu_total = sum(Lnu_E_snap[s] for s in range(NSPECIES))

    # Time-integrated dE_ν/dθ per species [erg/rad] via midpoint Riemann sum over time.
    # dLdtheta_snap is in code_lum/rad; time_arr is in M_sun (= code time);
    # their integral is in code_energy/rad → convert to erg/rad with conv_energy.
    _conv_e = conv_energy(cactus, cgs, 1.0)  # erg per code_energy ≈ 1.79e54
    dEnu_dtheta = [_conv_e * sum_over_time(dLdtheta_snap[s], time_arr)
                   for s in range(NSPECIES)]

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\nPeak energy luminosities:")
    for s, sp in enumerate(SPECIES):
        print(f"  {sp:4s}: {Lnu_E_snap[s].max():.3e} erg/s")
    print(f"  total: {Lnu_total.max():.3e} erg/s")

    print("\nTime-averaged mean energies (last snapshot):")
    for s, sp in enumerate(SPECIES):
        print(f"  {sp:4s}: {Eav_snap[s][-1]:.2f} MeV")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for s, sp in enumerate(SPECIES):
        # Energy luminosity time series
        np.savetxt(
            os.path.join(OUTPUT_DIR, f'Lnu_E_{sp}.txt'),
            np.column_stack((time_arr, Lnu_E_snap[s])),
            header=f'time[M_sun]    Lnu_E_{sp}[erg/s]',
            fmt='%.6e',
        )
        # Mean energy time series
        np.savetxt(
            os.path.join(OUTPUT_DIR, f'Eav_{sp}.txt'),
            np.column_stack((time_arr, Eav_snap[s])),
            header=(f'time[M_sun]    Eav_{sp}[MeV]  '
                    f'(flux-weighted mean energy at r={RADIUS})'),
            fmt='%.6e',
        )
        # Time-integrated angular distribution
        np.savetxt(
            os.path.join(OUTPUT_DIR, f'dEnu_dtheta_{sp}.txt'),
            np.column_stack((theta_centers, dEnu_dtheta[s])),
            header=f'theta[rad]    dEnu_dtheta_{sp}[erg/rad]  (time-integrated, outgoing only)',
            fmt='%.6e',
        )

    np.savetxt(
        os.path.join(OUTPUT_DIR, 'Lnu_E_total.txt'),
        np.column_stack((time_arr, Lnu_total)),
        header='time[M_sun]    Lnu_E_total[erg/s]  (sum over all 4 species)',
        fmt='%.6e',
    )

    print(f'\nAll outputs saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
