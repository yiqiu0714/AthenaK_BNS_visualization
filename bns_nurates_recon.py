"""
bns_nurates optically-thick distribution reconstruction, ported to NumPy.

Given the grey M1 moments n (number density) and J (energy density) per
neutrino species, bns_nurates reconstructs an assumed Fermi-Dirac spectrum in
the optically-thick regime (``CalculateThickParamsFromM1`` in
``distribution.hpp``).  The three reconstructed quantities are:

    y      = n^4 / J^3 / (4 pi/(hc)^3)          [dimensionless]
           = F2(eta)^4 / F3(eta)^3   for a true Fermi-Dirac gas
    eta_t  = degeneracy parameter mu/T,  eta_t = g(y)   (rational fits)
    temp_t = F2(eta_t)/F3(eta_t) * (J/n)         [MeV]  (spectral temperature)

`y` is a pure function of the degeneracy alone; its supremum is
64/81 = 0.7901... (the eta -> infinity limit of F2^4/F3^3).  When y reaches
that ceiling the inversion saturates and eta_t is hard-clamped at 20 (see the
``else`` branch + final min() in ``CalculateThickParamsFromM1``).  These
functions reproduce that behaviour exactly, so 2-D maps show where the clamp
fires and how far the grey moments push `y` toward (or past) 64/81.

This is a faithful port of:
  * ``FDI_p2`` / ``FDI_p3`` in ``src/bns_nurates/include/functions.hpp``
  * ``CalculateThickParamsFromM1`` in
    ``src/bns_nurates/include/distribution.hpp``
verified against 1.5*zeta(3), 6*(7/8)*pi^4/90, and eta round-trip (incl. clamp).

All three quantities depend on n and J ONLY (chi enters only the thick/thin
blend weight, not y/eta_t/temp_t, and there is no EOS dependence here), so the
sim inputs required for preprocessing are exactly the per-species n and J.
The code-unit -> bns_nurates(ngs)-unit conversion is applied here.

FRAME (important): bns_nurates uses the FLUID-frame (comoving) moments
    n = N / Gamma / volform      (rad_n output, variables 'n:0'.. )
    J = calc_J_from_rT(...)       (rad_J output, variables 'J:0'.. )
NOT the lab-frame conserved moments N, E (rad_N / rad_E, 'N:0' / 'E:0'), which
differ by Lorentz/closure factors (see radiation_m1_calc_opacities_nurates.cpp
~L197-240).  So feed these functions the rad_J / rad_n fields.  Their native
units are nuclear (MeV/fm^3 for J, fm^-3 for n) -- NOT the code km^-2 of rad_E
-- so both scale to ngs by the fm^3 -> nm^3 factor below (see S_J / S_N).

NOTE on heavy leptons: in M1 "nux" means mu+tau, so the athenak wrapper feeds
bns_nurates 0.5*n and 0.5*J for species 2/3.  These helpers do NOT apply that
factor -- they are exact for species 0/1 (nue/anue).  For species 2/3 pass
half N and half E if you need the exact bns_nurates value (y scales by 0.5,
temp_t is unchanged since it depends on J/n only).
"""

import numpy as np

import athenak_units as au

# --- physical constants (identical to src/bns_nurates/include/constants.hpp) --
kBS_Pi         = 3.14159265358979323846264338328
kBS_H          = 4.135667696e-21     # [MeV s]
kBS_Clight     = 2.99792458e+17      # [nm s^-1]
kBS_FourPi_hc3 = (4. * kBS_Pi) / (kBS_H * kBS_Clight) ** 3   # [MeV^-3 nm^-3]

# --- rad_J / rad_n native units -> bns_nurates (ngs) units --------------------
# rad_J is written in MeV/fm^3 and rad_n in fm^-3 (nuclear/EOS units): see
# derived_variables.cpp ~L1261-1269 -- the evolved N is already in fm^-3 and J
# is rescaled by ene_conv into MeV/fm^3 so that <e> = J/n is directly in MeV.
# These are NOT the code km^-2 units that rad_E uses.  Both convert to ngs
# (nm^-3, MeV nm^-3) by the SAME fm^3 -> nm^3 volume factor 1e18; MeV is common
# to both systems, so the J/n ratio (hence temp_t in MeV) is scale-invariant.
S_J = au.NUC2NGS_ENEDENS    # MeV/fm^3 -> MeV/nm^3  (= 1e18)
S_N = au.CODE2NGS_NUMDENS   # fm^-3    -> nm^-3      (= 1e18)

# Degenerate ceiling of y = F2^4/F3^3  (eta -> inf limit, = 64/81).
Y_DEGEN_LIMIT = 64.0 / 81.0

_FLOOR = 1e-100     # same guard as bns_nurates on n, J (in ngs units)
_ETA_CAP = 20.0     # hard clamp bns_nurates applies to eta_t (sim-faithful)
_ETA_UNCAPPED = 1e3 # finite sentinel for y >= 64/81 in the uncapped diagnostic

fdi_litconst = 7.38905609893065023   # e^2


def _horner(coeffs, x):
    """Evaluate polynomial with coefficients highest-power-first (like polyval)."""
    r = np.zeros_like(x, dtype=float)
    for c in coeffs:
        r = r * x + c
    return r


def fdi_p2(y):
    """Complete Fermi-Dirac integral of order 2, F2(y) = int x^2/(e^{x-y}+1) dx.

    Rational minimax approximation ported from FDI_p2() in functions.hpp.
    """
    y = np.asarray(y, dtype=float)
    x = -np.abs(y)
    A = [2, 1914.06748184935743, 273.085756700981399, 8.5861610217850095,
         0.0161890243763741414, 7656.2699273974454, 1399.35442210906621,
         72.929152915475392]
    B = [2711.49678259128843, 1299.85460914884154, 222.606134197895041,
         172.881855215582924, 112.951038040682055, 24.0376482128898634,
         -2.68393549333878715, -2.14077421411719935, 0.326188299771397236,
         2517.1726659917047, 3038.7689794575778, 2541.7823512372631,
         1428.0589853413436, 531.62378035996132, 122.54595216479181,
         8.395768655115050, -3.9142702096919080]
    C = [3.28986813369645287, 0.333333333333333333]  # pi^2/3, 1/3
    with np.errstate(over='ignore'):
        ex = np.exp(x)
        t = ex * fdi_litconst
        fd_lo = ex * (A[0] - ex * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))) /
                              (A[5] + t * (A[6] + t * (A[7] + t))))
        s = -0.5 * x
        t2 = 1.0 - s
        num = (B[0]+t2*(B[1]+t2*(B[2]+t2*(B[3]+t2*(B[4]+t2*(B[5]+t2*(B[6]+t2*(B[7]-t2*B[8]))))))))
        den = (B[9]+s*(B[10]+s*(B[11]+s*(B[12]+s*(B[13]+s*(B[14]+s*(B[15]+s*(B[16]-s))))))))
        fd = np.where(x < -2.0, fd_lo, num / den)
    return np.where(y > 0.0, fd + y * (C[0] + C[1] * y * y), fd)


def fdi_p3(y):
    """Complete Fermi-Dirac integral of order 3, F3(y) = int x^3/(e^{x-y}+1) dx.

    Rational minimax approximation ported from FDI_p3() in functions.hpp.
    """
    y = np.asarray(y, dtype=float)
    x = -np.abs(y)
    A = [6, 5121.6401850302408, 664.28706260743472, 19.0856927562699544,
         0.0410982603688952131, 13657.7071600806539, 2136.54222460571183,
         92.376788603062645]
    B = [7881.24597452900838, 4323.07526636309661, 1260.13125873282465,
         653.359212389160499, 354.630774329461644, 113.373708671587772,
         19.9559488532742796, 1.59407954898394322, 2570.7250703533430,
         2972.7443644211129, 2393.9995533270879, 1259.0724833462608,
         459.86413596901097, 112.60906419590854, 16.468882811659000]
    C = [11.3643939539669510, 4.93480220054467931, 0.25]  # 7pi^4/60, pi^2/2, 1/4
    with np.errstate(over='ignore'):
        ex = np.exp(x)
        t = ex * fdi_litconst
        fd_lo = ex * (A[0] - ex * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))) /
                              (A[5] + t * (A[6] + t * (A[7] + t))))
        s = -0.5 * x
        t2 = 1.0 - s
        num = (B[0]+t2*(B[1]+t2*(B[2]+t2*(B[3]+t2*(B[4]+t2*(B[5]+t2*(B[6]+t2*B[7])))))))
        den = (B[8]+s*(B[9]+s*(B[10]+s*(B[11]+s*(B[12]+s*(B[13]+s*(B[14]+s)))))))
        fd = np.where(x < -2.0, fd_lo, num / den)
    y2 = y * y
    return np.where(y > 0.0, -fd + C[0] + y2 * (C[1] + y2 * C[2]), fd)


def eta_t_from_y(y, apply_cap=True):
    """Invert y -> eta_t via the four-branch rational fits (distribution.hpp).

    apply_cap=True (bns_nurates behaviour): the exp-rational branch is used only
    for y2 < y < y3, everything else falls to eta_t = 20, and a final min(.,20)
    hard-clamps -- this is what the sim actually feeds the opacities.

    apply_cap=False (diagnostic): the exp-rational branch, which recovers the
    real eta_t well past 20 near y3, is NOT clamped; only y >= y3 (beyond the
    Fermi-Dirac ceiling 64/81, where eta_t formally diverges) is set to a large
    sentinel.  This shows the *actual* reconstructed degeneracy the grey moments
    imply, i.e. how hard the run is pushing against the guard."""
    y = np.asarray(y, dtype=float)
    y1, y2, y3 = 0.005, 0.7, 0.7901234567745267
    n1 = [0.19926987701997, 38865.0149220478, 14364.6331099737,
          5750.1878570758, 1120.71610972194, 1.60356108438235e-8]
    n2 = [41.3836568203438, 32.5515666786612, 157.774993512235,
          66.5726772426253, 14.4883415579211, 0.315360380575709,
          0.000660414331285249]
    n3 = [3852.81416018959, 5316.18895799799, 1102.91561586553,
          1.54082262710661e-6, 1732.89925128741, 1769.59868329086,
          586.406885304906]
    d1 = [1.0, 38840.0743174942, 99.9009680656931, 171.874844843596,
          75.7101579899442, 83.0160130941424]
    d2 = [1.8888797407042, 5.35488690539183, 1.94673781342617,
          0.483128792035557, 0.0113386564109086, 2.64160073447322e-5]
    d3 = [255.936658313629, 9.42360945627147e-5, 81.2467063138386,
          180.100197053091, 89.0343496217014, 143.849128123195]
    with np.errstate(all='ignore'):
        # y < y1 : log-rational
        eta_a = np.log(_horner([1, n1[0], n1[1], n1[2], n1[3], n1[4], n1[5]], y) /
                       _horner([1, d1[0], d1[1], -d1[2], -d1[3], d1[4], d1[5]], y))
        # y1 <= y <= y2 : rational - 30
        eta_b = (_horner([n2[0], n2[1], -n2[2], n2[3], n2[4], n2[5], n2[6]], y) /
                 _horner([1, d2[0], -d2[1], d2[2], d2[3], d2[4], d2[5]], y)) - 30.0
        # y2 < y < y3 : exp-rational
        eta_c = np.exp(_horner([n3[0], -n3[1], n3[2], n3[3], n3[4], -n3[5], n3[6]], y) /
                       _horner([1, d3[0], d3[1], -d3[2], -d3[3], -d3[4], d3[5]], y))
    if apply_cap:
        eta = np.select([y < y1, (y >= y1) & (y <= y2), (y > y2) & (y < y3)],
                        [eta_a, eta_b, eta_c], default=_ETA_CAP)
        return np.minimum(eta, _ETA_CAP)
    # uncapped: keep the real exp-rational value for y2 < y < y3 (can exceed 20);
    # y >= y3 is past the FD ceiling -> diverges -> large finite sentinel.
    eta = np.select([y < y1, (y >= y1) & (y <= y2), (y > y2) & (y < y3)],
                    [eta_a, eta_b, eta_c], default=_ETA_UNCAPPED)
    return np.clip(np.nan_to_num(eta, nan=_ETA_UNCAPPED,
                                 posinf=_ETA_UNCAPPED, neginf=-_ETA_UNCAPPED),
                   -_ETA_UNCAPPED, _ETA_UNCAPPED)


# ---------------------------------------------------------------------------
# Derived-variable callables: signature fn(J_code, n_code) -> quantity in ngs
# units.  BNS_all.py registers these with (J:s, n:s) FLUID-frame variables in
# that order (energy density first, number density second).
# ---------------------------------------------------------------------------

def _to_ngs(J_code, n_code):
    """(J, n) in bns_nurates units [MeV nm^-3], [nm^-3], floored like the code."""
    J = np.maximum(np.asarray(J_code, float) * S_J, _FLOOR)
    n = np.maximum(np.asarray(n_code, float) * S_N, _FLOOR)
    return J, n


def calc_y(J_code, n_code):
    """Reconstructed degeneracy invariant y = n^4/J^3/(4pi/(hc)^3) [dimensionless]."""
    J, n = _to_ngs(J_code, n_code)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return n * (n / J) ** 3 / kBS_FourPi_hc3

def calc_eta_t(J_code, n_code):
    """Reconstructed degeneracy parameter eta_t = mu/T, UNCAPPED (diagnostic):
    shows the actual value the grey moments imply, not the sim's clamp at 20."""
    return eta_t_from_y(calc_y(J_code, n_code), apply_cap=False)

def calc_temp_t(J_code, n_code):
    """Reconstructed spectral temperature temp_t = F2/F3 * J/n [MeV].
    Uses the CAPPED eta_t (apply_cap=True) to match what the sim feeds the
    opacities -- this quantity is meant to reproduce the sim's temp_t."""
    J, n = _to_ngs(J_code, n_code)
    eta = eta_t_from_y(calc_y(J_code, n_code), apply_cap=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return fdi_p2(eta) / fdi_p3(eta) * (J / n)


def calc_avge(J_code, n_code):
    """Fluid-frame mean neutrino energy <e> = J/n [MeV].

    rad_J is MeV/fm^3 and rad_n is fm^-3, so J/n is already in MeV (the sim
    rescales J on output specifically so this ratio comes out in MeV)."""
    J = np.asarray(J_code, float)
    n = np.asarray(n_code, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(n > 0., J / n, 0.)


def calc_Ynu(n_nu_code, rho_code):
    """Neutrino number fraction per baryon Y_nu = n_nu / n_b [dimensionless].

    n_nu = rad_n [fm^-3]; n_b = rho * CODE2NB_FM3 (rest-mass density -> baryon
    fm^-3 with m_b = 939.56535 MeV, the DD2 table value).  Both comoving."""
    n_nu = np.asarray(n_nu_code, float)
    nb = np.asarray(rho_code, float) * au.CODE2NB_FM3
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(nb > 0., n_nu / nb, 0.)


# --- heavy-lepton (nux) variants ------------------------------------------
# In M1 "nux" means mu+tau, so the athenak wrapper feeds bns_nurates 0.5*(J,n)
# for species 2/3 (see radiation_m1_calc_opacities_nurates.cpp; n[id_nux] =
# 0.5*n_nux, J[id_nux] = 0.5*j_nux) BEFORE CalculateDistrParamsFromM1.  So the
# faithful nux y / eta_t / temp_t use half the rad_J:2 / rad_n:2 fields.
#   y_nux  = 0.5 * y(J,n)          (n^4/J^3 with both halved -> factor 0.5)
#   eta_t  differs (function of the halved y)
#   temp_t: J/n ratio is unchanged by the 0.5, but eta_t (hence F2/F3) changes
#   <e>=J/n is unchanged by the 0.5 -> nux_avge just uses species 2 directly.

def _nux_half(J_code, n_code):
    return 0.5 * np.asarray(J_code, float), 0.5 * np.asarray(n_code, float)


def calc_y_nux(J_code, n_code):
    """nux degeneracy invariant y, with the M1 mu+tau 0.5 weighting."""
    return calc_y(*_nux_half(J_code, n_code))


def calc_eta_t_nux(J_code, n_code):
    """nux degeneracy eta_t = mu/T, UNCAPPED (diagnostic), 0.5-weighted."""
    Jh, nh = _nux_half(J_code, n_code)
    return eta_t_from_y(calc_y(Jh, nh), apply_cap=False)


def calc_temp_t_nux(J_code, n_code):
    """nux reconstructed spectral temperature [MeV], capped eta (sim-faithful)."""
    Jh, nh = _nux_half(J_code, n_code)
    J, n = _to_ngs(Jh, nh)
    eta = eta_t_from_y(calc_y(Jh, nh), apply_cap=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return fdi_p2(eta) / fdi_p3(eta) * (J / n)


# var_key (as used by athenak_units.plot_params / BNS_all.py) -> callable.
# fn(field_e, field_n): for the recon/avge kinds the pair is (J, n); for Y_nu
# it is (n_nu, rho).  BNS_all.py wires the correct file pair (and species) per
# kind.  'nu_*' = electron species (0/1, no weighting); 'nux_*' = heavy species
# (2/3) with the 0.5 mu+tau weighting.  nux_avge reuses calc_avge (J/n is
# invariant under the 0.5).
DERIVED = {
    'nu_y':       calc_y,
    'nu_eta_t':   calc_eta_t,
    'nu_temp_t':  calc_temp_t,
    'nu_avge':    calc_avge,
    'nu_Ynu':     calc_Ynu,
    'nux_y':      calc_y_nux,
    'nux_eta_t':  calc_eta_t_nux,
    'nux_temp_t': calc_temp_t_nux,
    'nux_avge':   calc_avge,
}
