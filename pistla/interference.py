"""Poisson-field aggregate interference and SINR (Section 3.7), Eqs. (19)-(22)."""
import numpy as np
from scipy import special

from .propagation import geometry, mean_a2g_pl_db, atm_attenuation_db, fspl_db, ENVIRONMENTS

N0_DBM_HZ = -174.0


def noise_dbm(B_hz=20e6, F_db=7.0):
    return N0_DBM_HZ + 10 * np.log10(B_hz) + F_db


def db2lin(x):
    return 10 ** (np.asarray(x, float) / 10)


def lin2db(x):
    return 10 * np.log10(np.asarray(x, float))


def simulate_sinr(n_real, f_ghz, r0_m, h_m, lam_I, env="urban", rain_mmh=0.0,
                  Pt_dbm=20.0, Gt_dbi=2.0, Gr_dbi=5.0, B_hz=20e6, F_db=7.0,
                  M_gm3=1.0, rho_w=20.0, R_max_m=3000.0, Pj_dbm=(0.0, 10.0),
                  p_a=(0.1, 0.5), Gj_dbi=0.0, los_desired=False, rng=None):
    """Monte Carlo SINR (dB) of the UAV link, Eqs. (19)-(20).

    Interferers form a homogeneous PPP of density lam_I (m^-2) on the ground
    within a disk of radius R_max_m centred below the UAV. Each has a uniform
    power in Pj_dbm, a Bernoulli(p_a) activity with p_a ~ U(p_a), Rayleigh
    fading and the mean A2G path loss of Eq. (5). The desired link also has
    Rayleigh fading and suffers the tropospheric attenuation of Eq. (6).
    """
    rng = np.random.default_rng(rng)
    d0, th0 = geometry(r0_m, h_m)
    if los_desired:  # GCS sited with a clear LoS to the UAV (excess loss eta_LoS only)
        pl0 = fspl_db(d0, f_ghz) + ENVIRONMENTS[env][2]
    else:
        pl0 = mean_a2g_pl_db(d0, th0, f_ghz, env)
    pl0 = pl0 + atm_attenuation_db(f_ghz, d0 / 1e3, rain_mmh, M_gm3, rho_w)
    S_bar = db2lin(Pt_dbm + Gt_dbi + Gr_dbi - pl0)  # mW
    N = db2lin(noise_dbm(B_hz, F_db))
    area = np.pi * R_max_m ** 2
    out = np.empty(n_real)
    for k in range(n_real):
        S = S_bar * rng.exponential()
        n = rng.poisson(lam_I * area)
        I = 0.0
        if n:
            rr = R_max_m * np.sqrt(rng.random(n))
            dj, thj = geometry(rr, h_m)
            plj = mean_a2g_pl_db(dj, thj, f_ghz, env)
            pa = rng.uniform(*p_a)
            act = rng.random(n) < pa
            Pj = rng.uniform(*Pj_dbm, size=n)
            I = np.sum(act * db2lin(Pj + Gj_dbi + Gr_dbi - plj) * rng.exponential(size=n))
        out[k] = lin2db(S / (I + N))
    return out


def success_prob_closed_form(gamma_th_db, lam_I, p_a, d0_m, beta, S_bar_mw, N_mw):
    """Interference-limited success probability, Eq. (21) (Rayleigh, l(d)=d^-beta).

    Note: Eq. (21) assumes the interferers have the same mean received power law
    and transmit power as the desired link (normalised form of Haenggi 2012).
    """
    g = db2lin(gamma_th_db)
    k = 2 * np.pi / beta
    return np.exp(-lam_I * p_a * np.pi * d0_m ** 2 * g ** (2 / beta) * k / np.sin(k)) * np.exp(-g * N_mw / S_bar_mw)


def qfunc(x):
    return 0.5 * special.erfc(np.asarray(x, float) / np.sqrt(2))


def capacity_bps(sinr_lin, B_hz=20e6):
    """Shannon capacity, Eq. (22)."""
    return B_hz * np.log2(1 + np.asarray(sinr_lin, float))


def ber_mqam(sinr_lin, M=4):
    """Approximate M-QAM BER, Eq. (22)."""
    return 4 / np.log2(M) * (1 - 1 / np.sqrt(M)) * qfunc(np.sqrt(3 * np.asarray(sinr_lin, float) / (M - 1)))


def interference_margin_db(I_mw, N_mw):
    """Interference term of the margin decomposition, Eq. (29)."""
    return 10 * np.log10(1 + np.asarray(I_mw, float) / N_mw)
