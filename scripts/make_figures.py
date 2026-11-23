"""Reproduce the illustrative figures (Figs. 4-10) and their data tables.

Usage:  python scripts/make_figures.py [--fast]
Outputs: figures/fig04.png ... figures/fig10.png and data/*.csv
All random draws use fixed seeds, so the outputs are bit-for-bit reproducible.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pistla import propagation as pr, scintillation as sc, interference as itf, stgp, climatology as cl  # noqa: E402

FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
COL = ["#1b9e77", "#1f4e9c", "#7557a6", "#b2182b"]

# Default link parameters (Table 3 of the paper)
P = dict(Pt=20.0, Gt=2.0, Gr=5.0, B=20e6, F=7.0, h=120.0, rho_w=20.0, M=1.0, gamma_th=5.0)
NOISE = itf.noise_dbm(P["B"], P["F"])  # -94.0 dBm


def save_csv(name, header, *cols):
    np.savetxt(os.path.join(DATA, name), np.column_stack(cols), delimiter=",", header=",".join(header),
               comments="", fmt="%.6g")


# ---------------------------------------------------------------- Fig. 4
def fig04():
    f = np.logspace(0, np.log10(40), 300)
    gr = pr.gamma_rain(f, 145)
    gg = pr.gamma_gas(f, P["rho_w"])
    gc = pr.gamma_cloud(f, P["M"])
    R = np.linspace(0, 150, 151)
    A24, A58 = pr.gamma_rain(2.4, R) * 5, pr.gamma_rain(5.8, R) * 5
    save_csv("fig04a_specific_attenuation.csv", ["f_GHz", "rain145_dB_km", "gas_dB_km", "cloud_dB_km"], f, gr, gg, gc)
    save_csv("fig04b_rain_attenuation_5km.csv", ["R_mm_h", "A_2p4GHz_dB", "A_5p8GHz_dB"], R, A24, A58)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].loglog(f, gr, color=COL[3], lw=2, label="rain, R = 145 mm/h")
    ax[0].loglog(f, gg, color=COL[1], lw=2, label="gases, ρ = 20 g/m³")
    ax[0].loglog(f, gc, color=COL[0], lw=2, label="cloud, M = 1 g/m³")
    for fx in (2.4, 5.8):
        ax[0].axvline(fx, ls=":", color="0.4")
    ax[0].set(xlabel="Frequency (GHz)", ylabel="Specific attenuation (dB/km)", title="(a) Tropical specific attenuation")
    ax[0].legend(fontsize=9)
    ax[1].plot(R, A24, color=COL[0], lw=2, label="2.4 GHz")
    ax[1].plot(R, A58, color=COL[3], lw=2, label="5.8 GHz")
    ax[1].axvspan(100, 150, color="0.85", alpha=0.6, label="typical $R_{0.01}$, W. Indonesia")
    ax[1].set(xlabel="Rain rate R (mm/h)", ylabel="Rain attenuation over 5 km (dB)", title="(b) Path attenuation, d = 5 km")
    ax[1].legend(fontsize=9)
    fig.savefig(os.path.join(FIG, "fig04.png"))
    plt.close(fig)
    return {"gamma_r_2.4@145": pr.gamma_rain(2.4, 145), "gamma_r_5.8@145": pr.gamma_rain(5.8, 145),
            "gamma_r_10@145": pr.gamma_rain(10, 145), "gamma_r_28@145": pr.gamma_rain(28, 145),
            "gamma_g_2.4": pr.gamma_gas(2.4), "gamma_c_2.4": pr.gamma_cloud(2.4),
            "A5km_5.8@145": pr.gamma_rain(5.8, 145) * 5, "A5km_2.4@145": pr.gamma_rain(2.4, 145) * 5}


# ---------------------------------------------------------------- Fig. 5
def snr_vs_range(r_m, f, env, R):
    d, th = pr.geometry(r_m, P["h"])
    pl = pr.mean_a2g_pl_db(d, th, f, env) + pr.atm_attenuation_db(f, d / 1e3, R, P["M"], P["rho_w"])
    return P["Pt"] + P["Gt"] + P["Gr"] - pl - NOISE


def usable_range_km(f, env, R=100.0):
    r = np.linspace(10, 8000, 8000)
    s = snr_vs_range(r, f, env, R)
    ok = np.where(s >= P["gamma_th"])[0]
    return r[ok.max()] / 1e3 if ok.size else 0.0


def fig05():
    th = np.linspace(0, 90, 361)
    r = np.linspace(50, 8000, 400)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    cols = {}
    for c, env in zip(COL, pr.ENVIRONMENTS):
        y = pr.p_los(th, env)
        cols[env] = y
        ax[0].plot(th, y, color=c, lw=2, label=env)
    save_csv("fig05a_plos.csv", ["theta_deg"] + list(cols), th, *cols.values())
    ax[0].set(xlabel="Elevation angle θ (deg)", ylabel="$P_{LoS}(θ)$", title="(a) LoS probability (sigmoid model)")
    ax[0].legend(fontsize=9)
    out = {}
    for f, ls in ((2.4, "-"), (5.8, "--")):
        for env, c in (("suburban", COL[0]), ("dense urban", COL[3])):
            s = snr_vs_range(r, f, env, 100)
            out[f"{f}GHz_{env}"] = s
            ax[1].plot(r / 1e3, s, ls, color=c, lw=2, label=f"{f} GHz, {env}")
    save_csv("fig05b_snr_vs_range_R100.csv", ["r_m"] + list(out), r, *out.values())
    ax[1].axhline(P["gamma_th"], ls=":", color="0.4")
    ax[1].text(8, P["gamma_th"] + 1, "QPSK threshold ≈ 5 dB", ha="right", fontsize=9)
    ax[1].set(xlabel="Ground distance r (km), h = 120 m", ylabel="SNR (dB), R = 100 mm/h",
              title="(b) Link budget with tropical rain")
    ax[1].legend(fontsize=9)
    fig.savefig(os.path.join(FIG, "fig05.png"))
    plt.close(fig)
    res = {f"PLoS30_{e}": float(pr.p_los(30, e)) for e in pr.ENVIRONMENTS}
    res.update({f"PLoS60_{e}": float(pr.p_los(60, e)) for e in pr.ENVIRONMENTS})
    for f in (2.4, 5.8):
        for env in ("suburban", "dense urban"):
            res[f"range_km_{f}_{env}"] = usable_range_km(f, env)
    return res


# ---------------------------------------------------------------- Fig. 6
def fig06():
    R, S, O = cl.rain_climatology(), cl.scint_climatology(), cl.ism_climatology()
    rho = stgp.climatology_correlation(R, S)
    mon = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for a, Z, t in zip(ax, (R, S, O), ("(a) Convective rain", "(b) Scintillation (S4 > 0.3)", "(c) ISM occupancy")):
        im = a.imshow(Z, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=(0.5, 12.5, -0.5, 23.5))
        a.set_xticks(range(1, 13), mon)
        a.set(title=t, xlabel="Month")
        a.grid(False)
    ax[0].set_ylabel("Local time (h)")
    fig.colorbar(im, ax=ax, fraction=0.02, label="Normalised index")
    fig.savefig(os.path.join(FIG, "fig06.png"))
    plt.close(fig)
    hh, mm = np.meshgrid(cl.HOURS, cl.MONTHS, indexing="ij")
    save_csv("fig06_climatology.csv", ["hour_LT", "month", "rain", "scint", "ism"],
             hh.ravel(), mm.ravel(), R.ravel(), S.ravel(), O.ravel())
    return {"rho_rs": rho, "rain_peak_LT": int(np.argmax(R.mean(1))), "scint_peak_LT": int(np.argmax(S.mean(1)))}


# ---------------------------------------------------------------- Fig. 7
F_MAP, R_LINK = 5.8, 1000.0  # reference link of the decision layer
SIGMA_PRED = 2.0  # dB, predictive std of Eq. (26) used for the maps
PATH_KM = 5.0     # effective path length through the rain field (km)
CN0_MAP = 33.0    # mean GNSS C/N0 (dB-Hz) of a low-elevation satellite


def mean_interference_mw(lam, f, h=120.0, Pj_dbm=5.0, p_a=0.3, Gr=5.0 - 10.0, R_max=3000.0, env="urban"):
    rr = np.linspace(1, R_max, 3000)
    d, th = pr.geometry(rr, h)
    l = 10 ** ((Pj_dbm + Gr - pr.mean_a2g_pl_db(d, th, f, env)) / 10)
    return lam * p_a * np.trapezoid(l * 2 * np.pi * rr, rr)


def margin_terms(f, d_km, R, S4, lam, env="urban"):
    """Components of the margin deficit, Eq. (29): troposphere, interference, scintillation."""
    A = pr.gamma_rain(f, R) * d_km
    N = 10 ** (NOISE / 10)
    I = mean_interference_mw(1.0, f, env=env) * np.asarray(lam)
    Im = itf.interference_margin_db(I, N)
    Sc = sc.scint_fade_margin_db(S4, outage=0.1) - sc.scint_fade_margin_db(0.05, outage=0.1)
    return A, Im, Sc


def fig07():
    LON, LAT = cl.grid()
    hgt = cl.terrain(LON, LAT)
    d, th = pr.geometry(R_LINK, P["h"])
    G_clear = P["Pt"] + P["Gt"] + P["Gr"] - (pr.fspl_db(d, F_MAP) + pr.ENVIRONMENTS["urban"][2]) - NOISE  # LoS-sited GCS
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    titles = {9: "09 LT (dry morning)", 16: "16 LT (convective peak)", 22: "22 LT (post-sunset EPB)"}
    rows = []
    for a, hr in zip(ax, (9, 16, 22)):
        R = cl.rain_field(LON, LAT, hr)
        S4 = cl.scint_field(LON, LAT, hr)
        lam = cl.interferer_density(LON, LAT, hr)
        Aatm, Im, Sc = margin_terms(F_MAP, PATH_KM, R, S4, lam)
        Yhat = G_clear - pr.gamma_gas(F_MAP) * PATH_KM - pr.gamma_cloud(F_MAP) * PATH_KM - Aatm - Im
        # mission availability: C2 availability (Eq. 28) x (1 - P_LoL) (Eq. 18)
        A = stgp.availability(Yhat, SIGMA_PRED, P["gamma_th"]) * (1 - sc.p_loss_of_lock(CN0_MAP, S4))
        im = a.pcolormesh(LON, LAT, A, cmap="RdYlGn", vmin=0, vmax=1, shading="auto")
        a.contour(LON, LAT, hgt, levels=[0.6, 1.4], colors="k", linewidths=0.6, alpha=0.6)
        a.plot(*cl.JAKARTA, "k^", ms=8)
        a.plot(*cl.BANDUNG, "ks", ms=8)
        if hr == 9:
            a.text(cl.JAKARTA[0] + 0.05, cl.JAKARTA[1] + 0.06, "Jakarta", fontsize=9)
            a.text(cl.BANDUNG[0] + 0.05, cl.BANDUNG[1] + 0.06, "Bandung", fontsize=9)
        a.set(title=titles[hr], xlabel="Longitude (°E)")
        a.grid(False)
        rows.append(np.column_stack([np.full(A.size, hr), LON.ravel(), LAT.ravel(), R.ravel(), S4.ravel(),
                                     lam.ravel(), A.ravel()]))
    ax[0].set_ylabel("Latitude (°)")
    fig.colorbar(im, ax=ax, fraction=0.02, label="Predicted link availability Â(s,t)")
    fig.savefig(os.path.join(FIG, "fig07.png"))
    plt.close(fig)
    M = np.vstack(rows)
    save_csv("fig07_availability_maps.csv", ["hour_LT", "lon", "lat", "rain_mm_h", "S4", "lambda_I_m2", "A_hat"], *M.T)
    return {f"Amin_{int(h)}LT": float(M[M[:, 0] == h, 6].min()) for h in (9, 16, 22)} | \
           {f"Amedian_{int(h)}LT": float(np.median(M[M[:, 0] == h, 6])) for h in (9, 16, 22)}


# ---------------------------------------------------------------- Fig. 8
L_BLDG = 10.0  # building-entry loss of indoor ISM emitters (dB), applied as Gj = -L_BLDG


def fig08(n_real=6000):
    common = dict(env="urban", R_max_m=3000.0, Pj_dbm=(0.0, 10.0), p_a=(0.1, 0.5), Gj_dbi=-L_BLDG)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    lams_a = [1e-7, 5e-7, 2e-6, 5e-6]
    med = {}
    cdf_cols = {}
    for c, lam in zip(COL, lams_a):
        s = np.sort(itf.simulate_sinr(n_real, 2.4, 1000.0, P["h"], lam, los_desired=True, rng=11, **common))
        y = np.arange(1, s.size + 1) / s.size
        ax[0].plot(s, y, color=c, lw=2, label=f"λ_I = {lam:.0e} m⁻²")
        med[lam] = float(np.median(s))
        cdf_cols[f"sinr_dB_lam{lam:.0e}"] = s
    save_csv("fig08a_sinr_samples_sorted.csv", ["cdf"] + list(cdf_cols), y, *cdf_cols.values())
    ax[0].axvline(P["gamma_th"], ls=":", color="0.4")
    ax[0].set(xlabel="SINR (dB)", ylabel="CDF", xlim=(-25, 40), title="(a) SINR CDF, 2.4 GHz, 1 km")
    ax[0].legend(fontsize=8)
    lams = np.logspace(-8, -5, 16)
    out = {}
    for c, R in zip((COL[0], COL[1], COL[3]), (0, 50, 100)):
        po = [np.mean(itf.simulate_sinr(n_real // 2, 5.8, 2500.0, P["h"], lam, rain_mmh=R, los_desired=True,
                                        Pt_dbm=28.0, Gr_dbi=7.0, rng=21 + i, **common) < P["gamma_th"])
              for i, lam in enumerate(lams)]
        out[R] = np.array(po)
        ax[1].semilogx(lams, po, "o-", ms=3, color=c, lw=2, label=f"R = {R} mm/h")
    save_csv("fig08b_outage_5p8GHz_2p5km.csv", ["lambda_I_m2", "Pout_R0", "Pout_R50", "Pout_R100"], lams, *out.values())
    ax[1].set(xlabel="Interferer density λ_I (m⁻²)", ylabel="Outage P(SINR < 5 dB)",
              title="(b) Outage, 5.8 GHz, 2.5 km")
    ax[1].legend(fontsize=9)
    fig.savefig(os.path.join(FIG, "fig08.png"))
    plt.close(fig)
    res = {f"median_shift_dB": med[1e-7] - med[5e-6]}
    for R in (0, 50, 100):
        res[f"Pout_low_R{R}"] = float(out[R][:6].mean())
        res[f"Pout_1e-5_R{R}"] = float(out[R][-1])
    return res


# ---------------------------------------------------------------- Fig. 9
def fig09():
    snr = np.linspace(0, 40, 81)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ber_cols = {"awgn": sc.ber_bpsk_awgn(10 ** (snr / 10))}
    ax[0].semilogy(snr, ber_cols["awgn"], "k:", lw=2, label="AWGN (no scintillation)")
    for c, s4 in zip([COL[0], COL[1], COL[2], COL[3], "#e08214"], (0.2, 0.4, 0.6, 0.8, 1.0)):
        b = sc.ber_bpsk_nakagami(snr, s4)
        ber_cols[f"S4_{s4}"] = b
        ax[0].semilogy(snr, b, color=c, lw=2, label=f"S4 = {s4} (m = {1 / s4 ** 2:.2f})")
    save_csv("fig09a_ber_bpsk.csv", ["snr_dB"] + list(ber_cols), snr, *ber_cols.values())
    ax[0].axhline(1e-5, ls="--", color="0.5", lw=1)
    ax[0].set(xlabel="Mean SNR (dB)", ylabel="Average BER (BPSK)", ylim=(1e-7, 0.5), title="(a) BER under Nakagami-m scintillation")
    ax[0].legend(fontsize=8)
    s4 = np.linspace(0.1, 1.0, 91)
    lol = {}
    for c, cn0 in zip((COL[3], COL[1], COL[0]), (35, 40, 45)):
        p = sc.p_loss_of_lock(cn0, s4)
        lol[f"CN0_{cn0}"] = p
        ax[1].semilogy(s4, p, color=c, lw=2, label=f"mean C/N0 = {cn0} dB-Hz")
    save_csv("fig09b_gnss_loss_of_lock.csv", ["S4"] + list(lol), s4, *lol.values())
    ax[1].set(xlabel="S4", ylabel="P(loss of lock)", ylim=(1e-6, 1), title="(b) GNSS loss of lock, threshold 26 dB-Hz")
    ax[1].legend(fontsize=9)
    fig.savefig(os.path.join(FIG, "fig09.png"))
    plt.close(fig)

    def req(s4v):
        g = np.linspace(0, 60, 601)
        b = sc.ber_bpsk_nakagami(g, s4v) if s4v else sc.ber_bpsk_awgn(10 ** (g / 10))
        return float(g[np.argmax(b <= 1e-5)])

    return {"req_snr_awgn": req(0), "req_snr_S4_0.4": req(0.4), "req_snr_S4_0.6": req(0.6), "req_snr_S4_0.8": req(0.8),
            "PLoL_35_S4_0.7": float(sc.p_loss_of_lock(35, 0.7)), "PLoL_35_S4_1.0": float(sc.p_loss_of_lock(35, 1.0))}


# ---------------------------------------------------------------- Fig. 10
def fig10():
    t, R, S4, lam = cl.series_72h()
    A, Im, Sc = margin_terms(5.8, PATH_KM, R, S4, lam)
    total = A + Im + Sc
    margin0 = 12.0  # clear-sky margin above the 5 dB threshold
    Ahat = stgp.availability(P["gamma_th"] + margin0 - total, 6.0, P["gamma_th"])
    save_csv("fig10_margin_decomposition_72h.csv", ["t_h", "rain_mm_h", "S4", "lambda_I_m2", "dM_rain_dB",
                                                    "dM_interf_dB", "dM_scint_dB", "dM_total_dB", "A_hat"],
             t, R, S4, lam, A, Im, Sc, total, Ahat)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    ax[0].stackplot(t, A, Im, Sc, colors=[COL[1], "#e08214", COL[2]], alpha=0.85,
                    labels=["rain (troposphere)", "interference", "scintillation"])
    ax[0].set(ylabel="Margin loss ΔM (dB)", title="Stressor decomposition, Eq. (29) — illustrative March sequence")
    ax[0].legend(fontsize=9, loc="upper left")
    ax[1].plot(t, Ahat, color=COL[0], lw=2)
    ax[1].set(xlabel="Time (h from 00 LT, day 1)", ylabel="Â(t)", ylim=(0, 1.05))
    ax[1].set_xticks(range(0, 73, 6))
    fig.savefig(os.path.join(FIG, "fig10.png"))
    plt.close(fig)
    return {"peak_total_dB": float(total.max()), "min_A": float(Ahat.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fewer Monte Carlo realisations for Fig. 8")
    a = ap.parse_args()
    results = {}
    for fn in (fig04, fig05, fig06, fig07, fig09, fig10):
        results.update(fn())
        print("done", fn.__name__)
    results.update(fig08(1000 if a.fast else 6000))
    print("done fig08")
    with open(os.path.join(DATA, "key_numbers.csv"), "w") as fh:
        fh.write("quantity,value\n")
        for k, v in results.items():
            fh.write(f"{k},{float(v):.6g}\n")
            print(f"{k:28s} {float(v):.4g}")


if __name__ == "__main__":
    main()
