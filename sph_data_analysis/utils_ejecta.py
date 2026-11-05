import numpy as np
from scipy.integrate import simpson

import os
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))

import sys
# plot-tools submodule lives at the repository root (one level above this folder).
# Override with the ATHPLOT_PATH environment variable if needed.
_PLOTTOOLS = os.environ.get(
    "ATHPLOT_PATH", os.path.abspath(os.path.join(SCRIPTDIR, os.pardir, "plot-tools")))
sys.path.append(_PLOTTOOLS)
from sph_vtk import SphericalData  # noqa: F401 (re-exported for callers)

def integrate_over_phi(integrand, phi):
    integral_phi = simpson(y=integrand, x=phi, axis=0)
    return integral_phi

def integrate_over_theta(integrand, theta):
    integral_theta = simpson(y=integrand, x=theta, axis=1)
    return integral_theta

def integrate_2d_sph(integrand, theta, phi):
    integral_theta = simpson(y=integrand, x=theta, axis=1)
    integral_total = simpson(y=integral_theta, x=phi)

    return integral_total

def calc_ur(ux, uy, uz, theta, phi):
    ur = (ux * np.sin(theta) * np.cos(phi) +
          uy * np.sin(theta) * np.sin(phi) +
          uz * np.cos(theta))
    return ur

def sqrt_det_metric_adm(gxx, gyy, gzz, gxy, gxz, gyz, alpha):
    gamma_det = (gxx * gyy * gzz +
                 2 * gxy * gxz * gyz -
                 gxx * gyz**2 -
                 gyy * gxz**2 -
                 gzz * gxy**2)
    sqrt_det = np.sqrt(gamma_det) * alpha
    return sqrt_det

import numpy as np

def sum_over_time(arr, time):
    """
    Integrate arr over time using a midpoint Riemann sum.

    Each snapshot contributes arr[i] * dt[i], where dt[i] is the midpoint
    width of the time interval around snapshot i.  More robust than Simpson's
    rule when the signal changes rapidly between snapshots.

    Parameters
    ----------
    arr  : ndarray, shape (n_times, ...)
    time : ndarray, shape (n_times,)

    Returns
    -------
    ndarray, shape (...,)
    """
    dt = np.empty_like(time)
    dt[0]    = time[1] - time[0]
    dt[-1]   = time[-1] - time[-2]
    dt[1:-1] = 0.5 * (time[2:] - time[:-2])
    return np.sum(arr * dt[(slice(None),) + (None,) * (arr.ndim - 1)], axis=0)

def _get_widths(c):
    """Cell widths from center coordinates via midpoint rule."""
    e = np.zeros(len(c) + 1)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0]  = c[0]  - (e[1]  - c[0])
    e[-1] = c[-1] + (c[-1] - e[-2])
    return np.abs(np.diff(e))

def get_riemann_weights(integrand, theta_1d, phi_1d):
    """
    Per-cell Riemann integration weights for a 2D spherical surface integral.

    Returns an array of shape (Nphi, Ntheta) containing dtheta * dphi for each
    cell, with pole cells divided by Nphi to avoid double-counting the singular
    axis.
    """
    dt = _get_widths(theta_1d)   # (Ntheta,)
    dp = _get_widths(phi_1d)     # (Nphi,)

    weights = np.ones_like(integrand)
    is_pole = np.isclose(theta_1d, 0, atol=1e-7) | np.isclose(theta_1d, np.pi, atol=1e-7)
    weights[:, is_pole] /= len(phi_1d)

    return weights * dt[None, :] * dp[:, None]

def calculate_Mej_rate_riemann(integrand, theta_1d, phi_1d):
    return np.sum(integrand * get_riemann_weights(integrand, theta_1d, phi_1d))

def integrate_over_phi_riemann(integrand, theta_1d, phi_1d):
    """
    Calculates d(dM/dtheta)/dt by integrating the 2D integrand over phi
    using a Riemann sum with polar weighting.
    """
    dp = _get_widths(phi_1d)   # Shape (Nphi,)

    # Create 2D weight map for poles
    # integrand shape is typically (Nphi, Ntheta)
    weights = np.ones_like(integrand)
    is_pole = (np.isclose(theta_1d, 0, atol=1e-7)) | (np.isclose(theta_1d, np.pi, atol=1e-7))

    # Divide the columns corresponding to poles by Nphi
    # (Since we sum over phi, we only average the contribution at the singular poles)
    weights[:, is_pole] /= len(phi_1d)

    # Perform the sum over the phi axis (axis 0)
    # result shape: (Ntheta,)
    return np.sum(integrand * weights * dp[:, None], axis=0)