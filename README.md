# PI-STLA — Physics-Informed Spatio-Temporal Link Availability for UAV links in equatorial regions

Code and data accompanying the MethodsX article

> **A physics-informed spatio-temporal method for mapping UAV link availability under tropical rain, equatorial ionospheric scintillation and radio interference**
> Muhammad Reza Firmansyah Tanjung, Niken Dwi Wahyu Cahyani, Sidik Prabowo — Telkom University, Bandung, Indonesia.

PI-STLA combines ITU-R gaseous, cloud and rain attenuation, a probabilistic air-to-ground path-loss model, an S4-driven Nakagami-*m* scintillation model and a Poisson-field aggregate-interference model into one SINR expression, and uses that expression as the physics-informed mean of a non-separable spatio-temporal Gaussian process. The output is a probabilistic link-availability map Â(**s**, *t*).

## Repository layout

| Path | Content |
|---|---|
| `pistla/propagation.py` | FSPL, sigmoid LoS probability, mean A2G path loss, ITU-R P.838-3 rain, P.840 cloud, simplified P.676 gases, P.833 vegetation, refractivity — Eqs. (1)–(13) |
| `pistla/scintillation.py` | S4, σφ, ROT/ROTI, Nakagami-*m* BER, GNSS loss-of-lock, scintillation fade margin — Eqs. (14)–(18) |
| `pistla/interference.py` | PPP aggregate interference Monte Carlo, SINR, closed-form success probability, capacity, M-QAM BER — Eqs. (19)–(22) |
| `pistla/stgp.py` | Gneiting non-separable covariance, universal kriging / GP prediction, log-marginal likelihood, availability, ρ_rs, delta method, Sobol indices — Eqs. (23)–(32) |
| `pistla/climatology.py` | Synthetic, physics-motivated climatologies and fields for western Java (seeded) |
| `scripts/make_figures.py` | Regenerates Figs. 4–10 and every number quoted in Section "Method validation" |
| `data/*.csv` | Numerical data behind each figure; `data/key_numbers.csv` lists the quoted values |
| `figures/` | Figures as used in the article (Figs. 1–3 are conceptual diagrams) |
| `tests/` | Regression tests against values quoted in the article |

## Quick start

```bash
pip install -r requirements.txt
python scripts/make_figures.py        # ~1 min; add --fast for fewer Monte Carlo runs
python -m pytest -q tests
```

All random draws use fixed seeds, so the CSV files and figures are reproducible.

## Important note on the data

The environmental fields (rain, S4, interferer density) in `pistla/climatology.py` are **synthetic** and physically motivated. They are **not field measurements**. They illustrate the magnitudes and structures that the proposed measurement campaign in western Java is designed to capture. The measurement protocol and the open, georeferenced dataset it will produce are described in the article.

## Default parameters (Table 3 of the article)

P_t = 20 dBm, G_t = 2 dBi, G_r = 5 dBi, B = 20 MHz, F = 7 dB (noise floor −94.0 dBm), h = 120 m, ρ_w = 20 g/m³, M = 1 g/m³, QPSK threshold 5 dB, GNSS tracking threshold 26 dB-Hz. Fig. 8 assumes a GCS sited with line of sight to the UAV; Fig. 8(b) uses a 28 dBm video transmitter and a 7 dBi receive antenna; indoor ISM emitters have 10 dB building-entry loss.

## Licence

Code: MIT (see `LICENSE`). Data and figures: CC BY 4.0.

## Citation

If you use this code, please cite the MethodsX article (details will be added on publication) and this repository (`CITATION.cff`).
