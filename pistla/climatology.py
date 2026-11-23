"""Synthetic, physics-motivated climatologies and fields for western Java.

These generators are NOT measurements. They encode the known physics used in
Section 5 of the paper (afternoon convective rain in the wet season,
post-sunset equinoctial scintillation, human-driven ISM occupancy) so that the
illustrative figures can be reproduced exactly from a fixed random seed.
"""
import numpy as np

HOURS = np.arange(24)
MONTHS = np.arange(1, 13)

JAKARTA = (106.83, -6.21)
BANDUNG = (107.61, -6.91)
DOMAIN = dict(lon=(106.0, 108.6), lat=(-7.8, -5.9))


def _circ_gauss(x, mu, sd, period):
    d = (x - mu + period / 2) % period - period / 2
    return np.exp(-0.5 * (d / sd) ** 2)


def convective_cycle(hour, peak=16.0, sd_rise=1.5, sd_decay=2.3):
    """Skewed diurnal cycle of convection: fast build-up to the 15-17 LT peak, slower evening decay."""
    d = (np.asarray(hour, float) - peak + 12) % 24 - 12
    return np.exp(-0.5 * (d / np.where(d < 0, sd_rise, sd_decay)) ** 2)


def rain_climatology():
    """Normalised rain index R_hm (24 x 12): 15-17 LT peak, Nov-Mar wet season."""
    diurnal = 0.12 + convective_cycle(HOURS)
    seasonal = 0.25 + 0.75 * (0.5 + 0.5 * np.cos(2 * np.pi * (MONTHS - 1.0) / 12))
    R = np.outer(diurnal, seasonal)
    return R / R.max()


def scint_climatology():
    """Normalised occurrence of S4 > 0.3 (24 x 12): 20-23 LT peak around the equinoxes."""
    diurnal = 0.02 + _circ_gauss(HOURS, 21.5, 1.6, 24)
    seasonal = 0.15 + _circ_gauss(MONTHS, 3.5, 1.0, 12) + _circ_gauss(MONTHS, 9.5, 1.0, 12)
    S = np.outer(diurnal, seasonal)
    return S / S.max()


def ism_diurnal(hour):
    """Human-driven ISM occupancy: daytime (13 LT) and evening (21 LT) peaks."""
    return 0.15 + 0.6 * _circ_gauss(hour, 13.0, 2.5, 24) + 0.85 * _circ_gauss(hour, 21.0, 2.0, 24)


def ism_climatology():
    """Normalised ISM-band occupancy (24 x 12): daytime and evening peaks, weak seasonality."""
    diurnal = ism_diurnal(HOURS)
    seasonal = 1.0 - 0.08 * np.cos(2 * np.pi * (MONTHS - 7) / 12)
    O = np.outer(diurnal, seasonal)
    return O / O.max()


# ------------------------------------------------------------------ spatial fields
def grid(n_lon=131, n_lat=96):
    lon = np.linspace(*DOMAIN["lon"], n_lon)
    lat = np.linspace(*DOMAIN["lat"], n_lat)
    return np.meshgrid(lon, lat)


def _blob(LON, LAT, lon0, lat0, slon, slat, rot=0.0):
    x, y = LON - lon0, LAT - lat0
    c, s = np.cos(rot), np.sin(rot)
    xr, yr = c * x + s * y, -s * x + c * y
    return np.exp(-0.5 * ((xr / slon) ** 2 + (yr / slat) ** 2))


def terrain(LON, LAT):
    """Stylised elevation (km): Salak-Gede-Pangrango massif and the ridges around Bandung."""
    return (2.2 * _blob(LON, LAT, 106.75, -6.75, 0.17, 0.22)
            + 2.0 * _blob(LON, LAT, 107.75, -7.12, 0.32, 0.13, rot=-0.6))


def rain_field(LON, LAT, hour, seed=7):
    """Rain rate (mm/h) for an illustrative wet-season day."""
    rng = np.random.default_rng(seed + hour)
    hgt = terrain(LON, LAT)
    conv = convective_cycle(hour)
    R = 150 * conv * (hgt / hgt.max()) ** 1.5
    # scattered lowland cells
    for _ in range(5):
        lo, la = rng.uniform(106.2, 108.4), rng.uniform(-7.7, -6.1)
        R += 130 * conv * rng.uniform(0.4, 1.0) * _blob(LON, LAT, lo, la, 0.07, 0.10)
    R += 25 * conv * _blob(LON, LAT, 106.9, -7.65, 0.06, 0.09)
    return R


def scint_field(LON, LAT, hour):
    """S4 field: post-sunset EPB band, uniform over this small domain with weak latitude trend."""
    act = _circ_gauss(hour, 21.5, 1.6, 24)
    return 0.15 + 0.6 * act * (1 + 0.1 * (LAT + 7.0))


def interferer_density(LON, LAT, hour):
    """ISM interferer density (m^-2): urban hot-spots modulated by the diurnal occupancy."""
    occ = ism_diurnal(hour)
    urban = _blob(LON, LAT, *JAKARTA, 0.12, 0.10) + 1.2 * _blob(LON, LAT, *BANDUNG, 0.14, 0.11)
    return 1e-8 + 1.2e-4 * occ * urban


def series_72h(seed=3):
    """Illustrative 72-h March sequence of rain rate, S4 and interferer density at one site."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 72, 0.25)
    hod = t % 24
    rain = np.zeros_like(t)
    for day, (amp, peak) in enumerate([(95, 15.5), (70, 16.5), (110, 16.0)]):
        rain += amp * convective_cycle(t - 24 * day, peak=peak, sd_rise=1.0, sd_decay=1.8) * (np.abs(t - 24 * day - peak) < 12)
    rain = np.maximum(rain + rng.normal(0, 2, t.size), 0)
    s4 = 0.12 + 0.75 * np.exp(-0.5 * (((hod - 21.5 + 12) % 24 - 12) / 1.3) ** 2) * np.repeat([0.9, 0.7, 1.0], t.size // 3)
    occ = ism_diurnal(hod)
    lam = 2.4e-5 * occ * (1 + 0.15 * rng.standard_normal(t.size)).clip(0.5)
    return t, rain, np.clip(s4, 0.05, 1.0), lam
