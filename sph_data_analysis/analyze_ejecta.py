"""
Combined analysis script: Mej_rate + 2D ejecta histogram (vinf x theta)

Each snapshot's VTK files are loaded once and all shared intermediate
quantities (nb, e, h, W, u_t, u^r, sqrt_det, masks) are computed once.

The 2D histogram dMej[i_v, i_theta] unifies the former separate vinf histogram
and rhoej_theta outputs: its marginals reproduce both.

Outputs:
  Mej_rate.txt, Mej_rate_Ber.txt, Mej_rate_geo.txt
  Mej_vinf_geo.txt, Mej_vinf_Ber.txt          (vinf marginal, dM per v-bin)
  rhoej_theta_Ber.txt, rhoej_theta_geo.txt     (theta marginal, dM/dtheta)
  Mej_vinf_theta_geo.npy, Mej_vinf_theta_Ber.npy   (full 2D arrays, shape Nv x Ntheta)
  vinf_centers.txt, theta_centers_2d.txt             (axis labels for 2D arrays)
"""

import os
import sys
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.integrate import simpson
import h5py
from scipy.interpolate import RegularGridInterpolator

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
# plot-tools submodule lives at the repository root (one level above this folder).
_PLOTTOOLS = os.environ.get(
    "ATHPLOT_PATH", os.path.abspath(os.path.join(SCRIPTDIR, os.pardir, "plot-tools")))
sys.path.append(_PLOTTOOLS)
sys.path.append(SCRIPTDIR)  # utils_ejecta.py lives alongside this script

from athplot.load_sph_vtk import SphericalData
from athplot.utils.units import conv_dens, cactus, cgs
from utils_ejecta import (calc_ur, sqrt_det_metric_adm, get_riemann_weights,
                          sum_over_time)


# ===================================================================
# Configuration
# ===================================================================

# --- Simulation ---
SIM_NAME = 'M135_M135_D45_DD2_VVLR_AMR_rererun'
SPH_DIR_BASENAMES = [
    'output_scratch/output-0000/sph',
    'output_scratch/output-0001/sph',
    'output_scratch/output-0002/sph',
]

# --- EOS table ---
EOS_TABLE_PATH = '../PyCompOSE/examples/DD2/DD2.h5'

# --- SPH extraction radius [code units, M_sun] ---
# Must match a radius that has mhd_w_bcc/adm/z4c sph output in the parfile.
RADIUS = 300.0

# --- Density floor [g/cm^3] ---
DFLOOR = 3e-15

# --- Post-merger truncation for 2D histogram ---
# Mej_rate accumulates all snapshots with no truncation.
T_MERGER_MSUN = 3.1225e3           # [Msun]
MSUN_TO_S     = 6.67430e-11 * 1.98892e30 / 2.99792458e8**3
T_POST_MS     = 21.0               # [ms] stop 2D histogram after merger
T_STOP        = T_MERGER_MSUN + T_POST_MS * 1e-3 / MSUN_TO_S

# --- Velocity bins for vinf [c] ---
DV       = 0.01
V_BINS   = np.arange(0.0, 1.0 + DV, DV)
V_CENTERS = 0.5 * (V_BINS[:-1] + V_BINS[1:])

# --- Parallelism ---
N_WORKERS = min(8, cpu_count())

# --- Electron fraction (Y_e) bins ---
DYE      = 0.01
YE_BINS  = np.arange(0.0, 0.6 + DYE, DYE)
YE_CENTERS = 0.5 * (YE_BINS[:-1] + YE_BINS[1:])

# --- Data directories ---
SPH_DIR_LIST = [os.path.join('../sim', SIM_NAME, b) for b in SPH_DIR_BASENAMES]

# --- Output ---
OUTPUT_DIR = os.path.join('output', SIM_NAME)


# ===================================================================
# EOS table
# ===================================================================

def load_eos(path):
    """Load a PyCompOSE HDF5 table and build enthalpy interpolators.

    Returns a dict with:
        interp_e, interp_p_rho  — interpolators for Q7 and p/(nb*m_n)
        mn_mev                  — neutron mass [MeV]
        h_min                   — minimum specific enthalpy over the table
        nb_min, nb_max, ye_min, ye_max, t_min, t_max  — table bounds
    """
    with h5py.File(path, 'r') as f:
        nb_axis = f['nb'][:]         # [fm^-3]
        yq_axis = f['yq'][:]         # [dimensionless]
        t_axis  = f['t'][:]          # [MeV]
        mn_mev  = float(f['mn'][()]) # neutron mass [MeV]
        Q1_grid = f['Q1'][:]         # p / nb  [MeV]
        Q7_grid = f['Q7'][:]         # e / (nb * m_n) - 1

    log_nb_axis = np.log(nb_axis)
    log_t_axis  = np.log(t_axis)

    nb_min, nb_max = nb_axis.min(), nb_axis.max()
    t_min,  t_max  = t_axis.min(),  t_axis.max()
    ye_min, ye_max = yq_axis.min(), yq_axis.max()

    # Minimum specific enthalpy h = 1 + e/(nb*m_n) + p/(nb*m_n)
    # over the full EOS table — used in the Bernoulli unbound criterion.
    h_min = np.min(1.0 + Q7_grid + Q1_grid / mn_mev)

    interp_e = RegularGridInterpolator(
        (log_nb_axis, yq_axis, log_t_axis), Q7_grid,
        method='linear', bounds_error=False, fill_value=None,
    )
    interp_p_rho = RegularGridInterpolator(
        (log_nb_axis, yq_axis, log_t_axis), Q1_grid / mn_mev,
        method='linear', bounds_error=False, fill_value=None,
    )

    return dict(
        interp_e=interp_e,
        interp_p_rho=interp_p_rho,
        mn_mev=mn_mev,
        h_min=h_min,
        nb_min=nb_min,
        nb_max=nb_max,
        ye_min=ye_min,
        ye_max=ye_max,
        t_min=t_min,
        t_max=t_max,
    )


# ===================================================================
# Main analysis
# ===================================================================

# Module-level state set by the Pool initializer (once per worker process)
_W_EOS     = None
_W_MN_GRAM = None


def _init_ejecta_worker(eos, mn_gram):
    """Pool initializer: load EOS into each worker process once."""
    global _W_EOS, _W_MN_GRAM
    _W_EOS     = eos
    _W_MN_GRAM = mn_gram


def _process_snapshot_ejecta(args):
    """Per-snapshot worker for ejecta analysis."""
    (mhd_path, adm_path, alpha_path, betax_path, betay_path, betaz_path,
     riemann_weights, theta_edges, dfloor, t_stop) = args

    import numpy as _np
    from athplot.load_sph_vtk import SphericalData as _SD
    from utils_ejecta import (calc_ur as _calc_ur,
                              sqrt_det_metric_adm as _sqrt_det)
    from scipy.interpolate import RegularGridInterpolator  # noqa: F401 (already imported)
    from athplot.utils.units import (conv_dens as _conv_dens,
                                     cactus as _cactus, cgs as _cgs)

    eos     = _W_EOS
    mn_gram = _W_MN_GRAM

    mhd   = _SD(mhd_path)
    adm   = _SD(adm_path)
    alpha = _SD(alpha_path)
    betax = _SD(betax_path)
    betay = _SD(betay_path)
    betaz = _SD(betaz_path)

    time  = mhd.time
    r     = mhd.r
    theta = mhd.theta
    phi   = mhd.phi

    z4c_alpha_raw = alpha.fields['z4c_alpha']
    phi_zero    = (phi == phi.min())
    theta_north = (theta == theta.min())
    dup_corr = _np.ones(z4c_alpha_raw.shape)
    if z4c_alpha_raw[phi_zero & ~theta_north].max() > 1.0:
        dup_corr[phi_zero] = 0.5
    if z4c_alpha_raw[theta_north].max() > 1.0:
        dup_corr[theta_north] = 0.25

    dens        = mhd.fields['dens']        * dup_corr
    ye          = mhd.fields['s_00']        * dup_corr
    temperature = mhd.fields['temperature'] * dup_corr
    velx = mhd.fields['velx'] * dup_corr
    vely = mhd.fields['vely'] * dup_corr
    velz = mhd.fields['velz'] * dup_corr

    gxx = adm.fields['adm_gxx'] * dup_corr
    gyy = adm.fields['adm_gyy'] * dup_corr
    gzz = adm.fields['adm_gzz'] * dup_corr
    gxy = adm.fields['adm_gxy'] * dup_corr
    gxz = adm.fields['adm_gxz'] * dup_corr
    gyz = adm.fields['adm_gyz'] * dup_corr
    z4c_alpha = z4c_alpha_raw * dup_corr
    z4c_betax = betax.fields['z4c_betax'] * dup_corr
    z4c_betay = betay.fields['z4c_betay'] * dup_corr
    z4c_betaz = betaz.fields['z4c_betaz'] * dup_corr

    input_shape = dens.shape
    nb = _conv_dens(_cactus, _cgs, dens) * 1e-39 / mn_gram

    pts = _np.column_stack((
        _np.log(_np.clip(nb.flatten(), eos['nb_min'], eos['nb_max'])),
        _np.clip(ye.flatten(), eos['ye_min'], eos['ye_max']),
        _np.log(_np.clip(temperature.flatten(), eos['t_min'], eos['t_max'])),
    ))
    e_interp = eos['interp_e'](pts).reshape(input_shape)
    p_rho    = eos['interp_p_rho'](pts).reshape(input_shape)
    h        = 1.0 + e_interp + p_rho

    W = _np.sqrt(1.0 + gxx*velx**2 + gyy*vely**2 + gzz*velz**2
                 + 2*gxy*velx*vely + 2*gxz*velx*velz + 2*gyz*vely*velz)
    ut_contrav = W / z4c_alpha
    ur = _calc_ur(velx - z4c_betax*ut_contrav,
                  vely - z4c_betay*ut_contrav,
                  velz - z4c_betaz*ut_contrav, theta, phi)
    ut_cov = (- z4c_alpha * W
              + gxx*z4c_betax*velx + gyy*z4c_betay*vely + gzz*z4c_betaz*velz
              + gxy*(z4c_betax*vely + z4c_betay*velx)
              + gxz*(z4c_betax*velz + z4c_betaz*velx)
              + gyz*(z4c_betay*velz + z4c_betaz*vely))

    mask_floor = dens > dfloor
    mask_out   = (ur > 0) & mask_floor
    hut_cov    = h * ut_cov
    mask_Ber   = (hut_cov < -eos['h_min']) & mask_out
    mask_geo   = (ut_cov  < -1)             & mask_out

    sqrt_det      = _sqrt_det(gxx, gyy, gzz, gxy, gxz, gyz, z4c_alpha)
    base_integrand = dens * ur * sqrt_det * r**2 * _np.sin(theta)

    cell_rate     = base_integrand * riemann_weights
    flux_rate_geo = _np.where(mask_geo, cell_rate, 0.0)
    flux_rate_Ber = _np.where(mask_Ber, cell_rate, 0.0)

    total_geo = flux_rate_geo.sum()
    total_Ber = flux_rate_Ber.sum()

    vinf_geo = _np.sqrt(_np.where(ut_cov  < -1, 1.0 - ut_cov **-2, 0.0))
    # Bernoulli v_inf uses the asymptotic enthalpy h_min (ejecta cools to the
    # cold floor): W_inf = -h*u_t / h_min, so v_inf = sqrt(1 - (h_min/h*u_t)^2).
    # Threshold matches mask_Ber so v_inf is defined for exactly the unbound cells.
    vinf_Ber = _np.sqrt(_np.where(hut_cov < -eos['h_min'],
                                  1.0 - (eos['h_min'] / hut_cov)**2, 0.0))

    result = {
        'time':         time,
        'Mej_rate':     _np.where(mask_out, base_integrand, 0.0).sum() * riemann_weights.sum(),
        'Mej_rate_geo': total_geo,
        'Mej_rate_Ber': total_Ber,
        'vinf_avg_geo': (_np.sum(vinf_geo * flux_rate_geo) / total_geo if total_geo > 0 else 0.0),
        'vinf_avg_Ber': (_np.sum(vinf_Ber * flux_rate_Ber) / total_Ber if total_Ber > 0 else 0.0),
        'ye_avg_geo':   (_np.sum(ye * flux_rate_geo) / total_geo if total_geo > 0 else 0.0),
        'ye_avg_Ber':   (_np.sum(ye * flux_rate_Ber) / total_Ber if total_Ber > 0 else 0.0),
        'in_2d':        time <= t_stop,
    }

    if result['in_2d']:
        sel_geo = (ut_cov  < -1).flatten()
        sel_Ber = (hut_cov < -eos['h_min']).flatten()
        result['hist2d_geo'], _, _ = _np.histogram2d(
            vinf_geo.flatten()[sel_geo], theta.flatten()[sel_geo],
            bins=[V_BINS, theta_edges],
            weights=flux_rate_geo.flatten()[sel_geo],
        )
        result['hist2d_Ber'], _, _ = _np.histogram2d(
            vinf_Ber.flatten()[sel_Ber], theta.flatten()[sel_Ber],
            bins=[V_BINS, theta_edges],
            weights=flux_rate_Ber.flatten()[sel_Ber],
        )
        result['hist_ye_geo'], _ = _np.histogram(
            ye.flatten()[sel_geo], bins=YE_BINS,
            weights=flux_rate_geo.flatten()[sel_geo],
        )
        result['hist_ye_Ber'], _ = _np.histogram(
            ye.flatten()[sel_Ber], bins=YE_BINS,
            weights=flux_rate_Ber.flatten()[sel_Ber],
        )
        # 2-D Y_e vs theta: angular distribution of ejecta composition
        result['hist2d_ye_theta_geo'], _, _ = _np.histogram2d(
            ye.flatten()[sel_geo], theta.flatten()[sel_geo],
            bins=[YE_BINS, theta_edges],
            weights=flux_rate_geo.flatten()[sel_geo],
        )
        result['hist2d_ye_theta_Ber'], _, _ = _np.histogram2d(
            ye.flatten()[sel_Ber], theta.flatten()[sel_Ber],
            bins=[YE_BINS, theta_edges],
            weights=flux_rate_Ber.flatten()[sel_Ber],
        )

    return result


def main():
    # ------------------------------------------------------------------
    # Build snapshot index and Riemann weights from first snapshot
    # ------------------------------------------------------------------
    r_str = f'r={RADIUS:.2f}'
    index_map = {}
    for d in SPH_DIR_LIST:
        if not os.path.exists(d):
            continue
        for fname in os.listdir(d):
            if fname.startswith(f'bns.{r_str}.adm.'):
                idx = int(fname.split('.')[-2])
                index_map[idx] = d

    indices = sorted(index_map.keys())
    print(f"Found {len(indices)} snapshots. Loading EOS and grid...")

    eos     = load_eos(EOS_TABLE_PATH)
    mn_gram = eos['mn_mev'] * cgs.eV * 1e6 / cgs.light_speed**2

    # Compute Riemann weights and theta grid from first snapshot's geometry
    _first_dir = index_map[indices[0]]
    _first_mhd = SphericalData(
        os.path.join(_first_dir, f'bns.{r_str}.mhd_w_bcc.{indices[0]:05d}.vtk'))
    theta_1d        = _first_mhd.theta[0, :]
    phi_1d          = _first_mhd.phi[:, 0]
    riemann_weights = get_riemann_weights(
        np.ones(_first_mhd.theta.shape), theta_1d, phi_1d)
    theta_1d_sorted = np.sort(theta_1d)
    theta_edges     = np.concatenate([[0.0],
                                      0.5*(theta_1d_sorted[:-1]+theta_1d_sorted[1:]),
                                      [np.pi]])
    theta_centers   = theta_1d_sorted
    del _first_mhd

    # ------------------------------------------------------------------
    # Build worker argument list
    # ------------------------------------------------------------------
    def _paths(d, idx):
        p = os.path.join(d, f'bns.{r_str}.')
        return (p+f'mhd_w_bcc.{idx:05d}.vtk',
                p+f'adm.{idx:05d}.vtk',
                p+f'z4c_alpha.{idx:05d}.vtk',
                p+f'z4c_betax.{idx:05d}.vtk',
                p+f'z4c_betay.{idx:05d}.vtk',
                p+f'z4c_betaz.{idx:05d}.vtk')

    worker_args = [
        _paths(index_map[i], i) + (riemann_weights, theta_edges, DFLOOR, T_STOP)
        for i in indices
    ]
    print(f"Processing {len(worker_args)} snapshots with N_WORKERS={N_WORKERS}...")

    # ------------------------------------------------------------------
    # Parallel snapshot loop
    # ------------------------------------------------------------------
    if N_WORKERS > 1:
        with Pool(N_WORKERS,
                  initializer=_init_ejecta_worker,
                  initargs=(eos, mn_gram)) as pool:
            results = list(pool.imap_unordered(
                _process_snapshot_ejecta, worker_args,
                chunksize=max(1, len(worker_args) // (N_WORKERS * 4))))
    else:
        _init_ejecta_worker(eos, mn_gram)
        results = [_process_snapshot_ejecta(a) for a in worker_args]

    results.sort(key=lambda r: r['time'])

    # ------------------------------------------------------------------
    # Unpack results
    # ------------------------------------------------------------------
    time_arr         = np.array([r['time']         for r in results])
    Mej_rate_arr     = np.array([r['Mej_rate']     for r in results])
    Mej_rate_geo_arr = np.array([r['Mej_rate_geo'] for r in results])
    Mej_rate_Ber_arr = np.array([r['Mej_rate_Ber'] for r in results])
    vinf_avg_geo_arr = np.array([r['vinf_avg_geo'] for r in results])
    vinf_avg_Ber_arr = np.array([r['vinf_avg_Ber'] for r in results])
    ye_avg_geo_arr   = np.array([r['ye_avg_geo']   for r in results])
    ye_avg_Ber_arr   = np.array([r['ye_avg_Ber']   for r in results])

    res_2d           = [r for r in results if r['in_2d']]
    time_2d_arr      = np.array([r['time']         for r in res_2d])
    hist2d_geo_arr   = np.array([r['hist2d_geo']   for r in res_2d])
    hist2d_Ber_arr   = np.array([r['hist2d_Ber']   for r in res_2d])
    hist_ye_geo_arr  = np.array([r['hist_ye_geo']  for r in res_2d])
    hist_ye_Ber_arr  = np.array([r['hist_ye_Ber']  for r in res_2d])
    hist2d_ye_theta_geo_arr = np.array([r['hist2d_ye_theta_geo'] for r in res_2d])
    hist2d_ye_theta_Ber_arr = np.array([r['hist2d_ye_theta_Ber'] for r in res_2d])

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    dMej_2d_geo = sum_over_time(hist2d_geo_arr, time_2d_arr)   # (Nv, Ntheta)
    dMej_2d_Ber = sum_over_time(hist2d_Ber_arr, time_2d_arr)

    dMej_Ye_geo = sum_over_time(hist_ye_geo_arr, time_2d_arr)  # (NYe,)
    dMej_Ye_Ber = sum_over_time(hist_ye_Ber_arr, time_2d_arr)

    dMej_ye_theta_geo = sum_over_time(hist2d_ye_theta_geo_arr, time_2d_arr)  # (NYe, Ntheta)
    dMej_ye_theta_Ber = sum_over_time(hist2d_ye_theta_Ber_arr, time_2d_arr)

    # Marginals
    dMej_geo = dMej_2d_geo.sum(axis=1)          # (Nv,)  — vinf histogram
    dMej_Ber = dMej_2d_Ber.sum(axis=1)

    dtheta = np.diff(theta_edges)                # (Ntheta,)
    rhoej_theta_geo = dMej_2d_geo.sum(axis=0) / dtheta   # dM/dtheta [Msun/rad]
    rhoej_theta_Ber = dMej_2d_Ber.sum(axis=0) / dtheta

    # Total ejecta mass via time-integrated Mej_rate
    Mej     = simpson(y=Mej_rate_arr,     x=time_arr)
    Mej_Ber = simpson(y=Mej_rate_Ber_arr, x=time_arr)
    Mej_geo = simpson(y=Mej_rate_geo_arr, x=time_arr)

    print(f'Total Mej:     {Mej:.6e} Msun')
    print(f'Total Mej_Ber: {Mej_Ber:.6e} Msun')
    print(f'Total Mej_geo: {Mej_geo:.6e} Msun')
    print(f'Total Mej (geodesic,  vinf): {np.sum(dMej_geo):.6e} Msun')
    print(f'Total Mej (Bernoulli, vinf): {np.sum(dMej_Ber):.6e} Msun')

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- Mej_rate --
    with open(os.path.join(OUTPUT_DIR, 'Mej_rate.txt'), 'w') as f:
        f.write('# time(Msun)    Mej_rate(Msun/Msun)\n')
        for t, rate in zip(time_arr, Mej_rate_arr):
            f.write(f'{t:.6e}    {rate:.6e}\n')

    with open(os.path.join(OUTPUT_DIR, 'Mej_rate_Ber.txt'), 'w') as f:
        f.write('# time(Msun)    Mej_rate_Ber(Msun/Msun)\n')
        for t, rate in zip(time_arr, Mej_rate_Ber_arr):
            f.write(f'{t:.6e}    {rate:.6e}\n')

    with open(os.path.join(OUTPUT_DIR, 'Mej_rate_geo.txt'), 'w') as f:
        f.write('# time(Msun)    Mej_rate_geo(Msun/Msun)\n')
        for t, rate in zip(time_arr, Mej_rate_geo_arr):
            f.write(f'{t:.6e}    {rate:.6e}\n')

    # -- Mass-averaged vinf --
    np.savetxt(os.path.join(OUTPUT_DIR, 'vinf_avg_geo.txt'),
               np.column_stack((time_arr, vinf_avg_geo_arr)),
               header='time(Msun)    vinf_avg_geo[c]  (geodesic: u_t < -1)',
               fmt='%.6e')
    np.savetxt(os.path.join(OUTPUT_DIR, 'vinf_avg_Ber.txt'),
               np.column_stack((time_arr, vinf_avg_Ber_arr)),
               header='time(Msun)    vinf_avg_Ber[c]  (Bernoulli: h*u_t < -h_min)',
               fmt='%.6e')

    # -- vinf (marginal over theta) --
    np.savetxt(os.path.join(OUTPUT_DIR, 'Mej_vinf_geo.txt'),
               np.column_stack((V_CENTERS, dMej_geo)),
               header='vinf[c]    Mej_per_bin[Msun]  (geodesic: u_t < -1)',
               fmt='%.6e')
    np.savetxt(os.path.join(OUTPUT_DIR, 'Mej_vinf_Ber.txt'),
               np.column_stack((V_CENTERS, dMej_Ber)),
               header='vinf[c]    Mej_per_bin[Msun]  (Bernoulli: h*u_t < -h_min)',
               fmt='%.6e')

    # -- rhoej_theta (marginal over vinf, divided by dtheta) --
    np.savetxt(os.path.join(OUTPUT_DIR, 'rhoej_theta_geo.txt'),
               np.column_stack((theta_centers, rhoej_theta_geo)),
               header='theta(rad)    rho_ej_geo(dM/dtheta)[Msun/rad]', fmt='%.6e')
    np.savetxt(os.path.join(OUTPUT_DIR, 'rhoej_theta_Ber.txt'),
               np.column_stack((theta_centers, rhoej_theta_Ber)),
               header='theta(rad)    rho_ej_Ber(dM/dtheta)[Msun/rad]', fmt='%.6e')

    # -- 2D arrays (Nv x Ntheta) --
    np.save(os.path.join(OUTPUT_DIR, 'Mej_vinf_theta_geo.npy'), dMej_2d_geo)
    np.save(os.path.join(OUTPUT_DIR, 'Mej_vinf_theta_Ber.npy'), dMej_2d_Ber)
    np.savetxt(os.path.join(OUTPUT_DIR, 'vinf_centers.txt'), V_CENTERS,
               header='vinf bin centers [c]', fmt='%.6e')
    np.savetxt(os.path.join(OUTPUT_DIR, 'theta_centers_2d.txt'), theta_centers,
               header='theta bin centers [rad]', fmt='%.6e')

    # -- 2D Y_e x theta arrays (NYe x Ntheta): angular composition of ejecta --
    np.save(os.path.join(OUTPUT_DIR, 'Mej_Ye_theta_geo.npy'), dMej_ye_theta_geo)
    np.save(os.path.join(OUTPUT_DIR, 'Mej_Ye_theta_Ber.npy'), dMej_ye_theta_Ber)

    # -- Y_e outputs --
    np.savetxt(os.path.join(OUTPUT_DIR, 'Ye_avg_geo.txt'),
               np.column_stack((time_arr, ye_avg_geo_arr)),
               header='time(Msun)    Ye_avg_geo  (mass-weighted, geodesic: u_t < -1)',
               fmt='%.6e')
    np.savetxt(os.path.join(OUTPUT_DIR, 'Ye_avg_Ber.txt'),
               np.column_stack((time_arr, ye_avg_Ber_arr)),
               header='time(Msun)    Ye_avg_Ber  (mass-weighted, Bernoulli: h*u_t < -h_min)',
               fmt='%.6e')
    np.savetxt(os.path.join(OUTPUT_DIR, 'Mej_Ye_geo.txt'),
               np.column_stack((YE_CENTERS, dMej_Ye_geo)),
               header='Ye    Mej_per_bin[Msun]  (geodesic: u_t < -1)',
               fmt='%.6e')
    np.savetxt(os.path.join(OUTPUT_DIR, 'Mej_Ye_Ber.txt'),
               np.column_stack((YE_CENTERS, dMej_Ye_Ber)),
               header='Ye    Mej_per_bin[Msun]  (Bernoulli: h*u_t < -h_min)',
               fmt='%.6e')
    np.savetxt(os.path.join(OUTPUT_DIR, 'Ye_centers.txt'), YE_CENTERS,
               header='Ye bin centers', fmt='%.6e')

    print(f'All outputs saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
