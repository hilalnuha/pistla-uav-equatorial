"""Propagation models used by PI-STLA (Section 3 of the paper).

Implements
  * Free-space path loss, Eq. (3)
  * Sigmoid LoS probability and mean A2G path loss (Al-Hourani et al.), Eqs. (4)-(5)
  * ITU-R P.838-3 rain specific attenuation, Eqs. (7)-(8)
  * ITU-R P.840 cloud/fog specific attenuation (double-Debye model), Eqs. (9)-(10)
  * Simplified ITU-R P.676 gaseous attenuation below 57 GHz, Eq. (11)
  * ITU-R P.833 vegetation saturation loss, Eq. (13)
  * Radio refractivity, Eq. (12)
"""
import numpy as np

C_LIGHT = 299_792_458.0

# ---------------------------------------------------------------- A2G path loss
# (a, b, eta_LoS [dB], eta_NLoS [dB]) -- Al-Hourani, Kandeepan & Lardner (2014)
ENVIRONMENTS = {
    "suburban":    (4.88, 0.43, 0.1, 21.0),
    "urban":       (9.61, 0.16, 1.0, 20.0),
    "dense urban": (12.08, 0.11, 1.6, 23.0),
    "high-rise":   (27.23, 0.08, 2.3, 34.0),
}


def fspl_db(d_m, f_ghz):
    """Free-space path loss in dB, Eq. (3). d in metres, f in GHz."""
    d_m = np.maximum(np.asarray(d_m, float), 1.0)
    return 20 * np.log10(d_m) + 20 * np.log10(np.asarray(f_ghz) * 1e9) - 147.55


def p_los(theta_deg, env="urban"):
    """LoS probability as a sigmoid of the elevation angle (deg), Eq. (4)."""
    a, b, _, _ = ENVIRONMENTS[env]
    return 1.0 / (1.0 + a * np.exp(-b * (np.asarray(theta_deg, float) - a)))


def geometry(r_m, h_m, h_g=0.0):
    """Slant range d and elevation angle theta (deg) from ground range, Eq. (1)."""
    r_m = np.asarray(r_m, float)
    dh = h_m - h_g
    d = np.sqrt(r_m ** 2 + dh ** 2)
    theta = np.degrees(np.arctan2(dh, np.maximum(r_m, 1e-9)))
    return d, theta


def mean_a2g_pl_db(d_m, theta_deg, f_ghz, env="urban"):
    """Mean A2G path loss, Eq. (5)."""
    _, _, eta_los, eta_nlos = ENVIRONMENTS[env]
    pl = p_los(theta_deg, env)
    return fspl_db(d_m, f_ghz) + pl * eta_los + (1 - pl) * eta_nlos


# ---------------------------------------------------------------- ITU-R P.838-3
_KH = dict(a=[-5.33980, -0.35351, -0.23789, -0.94158],
           b=[-0.10008, 1.26970, 0.86036, 0.64552],
           c=[1.13098, 0.45400, 0.15354, 0.16817], m=-0.18961, cc=0.71147)
_KV = dict(a=[-3.80595, -3.44965, -0.39902, 0.50167],
           b=[0.56934, -0.22911, 0.73042, 1.07319],
           c=[0.81061, 0.51059, 0.11899, 0.27195], m=-0.16398, cc=0.63297)
_AH = dict(a=[-0.14318, 0.29591, 0.32177, -5.37610, 16.1721],
           b=[1.82442, 0.77564, 0.63773, -0.96230, -3.29980],
           c=[-0.55187, 0.19822, 0.13164, 1.47828, 3.43990], m=0.67849, cc=-1.95537)
_AV = dict(a=[-0.07771, 0.56727, -0.20238, -48.2991, 48.5833],
           b=[2.33840, 0.95545, 1.14520, 0.791669, 0.791459],
           c=[-0.76284, 0.54039, 0.26809, 0.116226, 0.116479], m=-0.053739, cc=0.83433)


def _p838_sum(t, lf):
    s = sum(a * np.exp(-((lf - b) / c) ** 2) for a, b, c in zip(t["a"], t["b"], t["c"]))
    return s + t["m"] * lf + t["cc"]


def p838_coeffs(f_ghz, pol="H"):
    """Return (k, alpha) of ITU-R P.838-3 for horizontal ('H') or vertical ('V') polarisation, Eq. (8)."""
    lf = np.log10(np.asarray(f_ghz, float))
    tk, ta = (_KH, _AH) if pol == "H" else (_KV, _AV)
    k = 10 ** _p838_sum(tk, lf)
    alpha = _p838_sum(ta, lf)
    return k, alpha


def gamma_rain(f_ghz, rain_mmh, pol="H"):
    """Rain specific attenuation in dB/km, Eq. (7)."""
    k, alpha = p838_coeffs(f_ghz, pol)
    return k * np.asarray(rain_mmh, float) ** alpha


# ---------------------------------------------------------------- ITU-R P.840
def k_cloud(f_ghz, T_kelvin=293.15):
    """Cloud liquid-water specific attenuation coefficient (dB/km)/(g/m^3), Eqs. (9)-(10)."""
    f = np.asarray(f_ghz, float)
    th = 300.0 / T_kelvin
    e0 = 77.66 + 103.3 * (th - 1)
    e1 = 0.0671 * e0
    e2 = 3.52
    fp = 20.20 - 146 * (th - 1) + 316 * (th - 1) ** 2
    fs = 39.8 * fp
    e_pp = f * (e0 - e1) / (fp * (1 + (f / fp) ** 2)) + f * (e1 - e2) / (fs * (1 + (f / fs) ** 2))
    e_p = (e0 - e1) / (1 + (f / fp) ** 2) + (e1 - e2) / (1 + (f / fs) ** 2) + e2
    eta = (2 + e_p) / e_pp
    return 0.819 * f / (e_pp * (1 + eta ** 2))


def gamma_cloud(f_ghz, M_gm3=1.0, T_kelvin=293.15):
    return k_cloud(f_ghz, T_kelvin) * M_gm3


# ---------------------------------------------------------------- simplified P.676
def gamma_gas(f_ghz, rho_w=20.0):
    """Gaseous specific attenuation (dB/km), sea level, f < 57 GHz, Eq. (11).

    Includes the far-wing 183 and 325 GHz water-vapour lines that the printed
    equation omits for brevity.
    """
    f = np.asarray(f_ghz, float)
    g_o = (7.19e-3 + 6.09 / (f ** 2 + 0.227) + 4.81 / ((f - 57) ** 2 + 1.50)) * f ** 2 * 1e-3
    g_w = (0.050 + 0.0021 * rho_w + 3.6 / ((f - 22.2) ** 2 + 8.5)
           + 10.6 / ((f - 183.3) ** 2 + 9.0) + 8.9 / ((f - 325.4) ** 2 + 26.3)) * f ** 2 * rho_w * 1e-4
    return g_o + g_w


def atm_attenuation_db(f_ghz, d_km, rain_mmh=0.0, M_gm3=1.0, rho_w=20.0, r_p=1.0):
    """Total tropospheric attenuation over a homogeneous slant path, Eq. (6) with n = 1."""
    return (gamma_gas(f_ghz, rho_w) + gamma_cloud(f_ghz, M_gm3) + r_p * gamma_rain(f_ghz, rain_mmh)) * d_km


# ---------------------------------------------------------------- refractivity & vegetation
def refractivity(P_hpa, T_k, e_hpa):
    """Radio refractivity N (N-units), Eq. (12)."""
    return 77.6 * P_hpa / T_k + 3.732e5 * e_hpa / T_k ** 2


def modified_refractivity(N, h_km):
    return N + 157.0 * h_km


def vegetation_loss_db(d_v_m, gamma_v_db_per_m, A_m_db):
    """ITU-R P.833 saturating vegetation loss, Eq. (13)."""
    return A_m_db * (1 - np.exp(-d_v_m * gamma_v_db_per_m / A_m_db))
