"""Equatorial ionospheric scintillation models (Section 3.6), Eqs. (14)-(18)."""
import numpy as np
from scipy import integrate, special


def s4_index(intensity):
    """Amplitude scintillation index S4 of a detrended intensity series, Eq. (14)."""
    I = np.asarray(intensity, float)
    return np.sqrt((np.mean(I ** 2) - np.mean(I) ** 2) / np.mean(I) ** 2)


def sigma_phi(phase):
    """Phase scintillation index (rad), Eq. (14)."""
    p = np.asarray(phase, float)
    return np.sqrt(np.mean(p ** 2) - np.mean(p) ** 2)


def rot_roti(tec, t_s, window=None):
    """Rate of TEC and ROTI, Eq. (15). Returns (ROT, ROTI)."""
    rot = np.diff(np.asarray(tec, float)) / np.diff(np.asarray(t_s, float))
    roti = np.sqrt(np.mean(rot ** 2) - np.mean(rot) ** 2)
    return rot, roti


def nakagami_m(s4):
    """Nakagami shape parameter tied to S4, m = 1/S4^2 (S4 <= 1), Eq. (16)."""
    s4 = np.clip(np.asarray(s4, float), 1e-6, 1.0)
    return 1.0 / s4 ** 2


def ber_bpsk_awgn(snr_lin):
    return 0.5 * special.erfc(np.sqrt(snr_lin))


def ber_bpsk_nakagami(snr_db, s4):
    """Average BPSK BER under Nakagami-m fading, Eq. (17) (numerical integration)."""
    m = float(nakagami_m(s4))
    gbar = 10 ** (np.asarray(snr_db, float) / 10)

    def one(g):
        f = lambda x: 0.5 * special.erfc(np.sqrt(x)) * np.exp(
            m * np.log(m) + (m - 1) * np.log(max(x, 1e-300)) - special.gammaln(m) - m * np.log(g) - m * x / g)
        val, _ = integrate.quad(f, 0, np.inf, limit=200)
        return val

    return np.vectorize(one)(gbar)


def ber_bpsk_nakagami_mc(snr_db, s4, n=200_000, rng=None):
    """Monte Carlo counterpart of Eq. (17)."""
    rng = np.random.default_rng(rng)
    m = float(nakagami_m(s4))
    xi = rng.gamma(m, 1.0 / m, size=n)
    gbar = 10 ** (np.atleast_1d(snr_db) / 10)
    return np.array([np.mean(ber_bpsk_awgn(g * xi)) for g in gbar])


def p_loss_of_lock(cn0_mean_dbhz, s4, cn0_th_dbhz=26.0):
    """GNSS loss-of-lock probability, Eq. (18)."""
    m = nakagami_m(s4)
    ratio = 10 ** ((cn0_th_dbhz - np.asarray(cn0_mean_dbhz, float)) / 10)
    return special.gammainc(m, m * ratio)  # regularised lower incomplete gamma


def scint_fade_margin_db(s4, outage=0.01):
    """Fade margin Delta_sc(S4) that keeps the Nakagami-m outage below `outage` (used in Eq. (29))."""
    m = nakagami_m(s4)
    x = special.gammaincinv(m, outage) / m
    return -10 * np.log10(x)
