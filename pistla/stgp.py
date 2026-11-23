"""Spatio-temporal Gaussian-process layer of PI-STLA (Section 3.8), Eqs. (23)-(28), (30)-(32)."""
import numpy as np
from scipy import linalg, stats


def gneiting_cov(h_km, u_h, sigma2=4.0, a=0.5, c=0.8, alpha=0.5, gamma=0.5, beta_s=0.6, delta=0.0, d=2):
    """Non-separable Gneiting space-time covariance, Eq. (25).

    h_km : spatial lag (km); u_h : temporal lag (h). beta_s = 0 is separable.
    """
    psi = a * np.abs(u_h) ** (2 * alpha) + 1.0
    return sigma2 / psi ** (delta + beta_s * d / 2) * np.exp(-c * np.abs(h_km) ** (2 * gamma) / psi ** (beta_s * gamma))


def _pairwise(X1, X2):
    """X = columns (x_km, y_km, t_h). Returns spatial and temporal lag matrices."""
    hs = np.sqrt((X1[:, None, 0] - X2[None, :, 0]) ** 2 + (X1[:, None, 1] - X2[None, :, 1]) ** 2)
    ut = X1[:, None, 2] - X2[None, :, 2]
    return hs, ut


class PISTLA:
    """Universal kriging / GP regression with a physics-informed mean, Eqs. (23)-(28)."""

    def __init__(self, tau2=0.5, **cov_params):
        self.tau2 = tau2
        self.cov = dict(cov_params)

    def _K(self, X1, X2):
        hs, ut = _pairwise(X1, X2)
        return gneiting_cov(hs, ut, **self.cov)

    def fit(self, X, y, gamma_phys_db, covariates=None):
        """Estimate beta of Eq. (24) by GLS and store the kriging system."""
        F = np.column_stack([np.ones(len(y)), gamma_phys_db] + ([covariates] if covariates is not None else []))
        K = self._K(X, X) + self.tau2 * np.eye(len(y))
        self.L = linalg.cho_factor(K, lower=True)
        Ki_F = linalg.cho_solve(self.L, F)
        self.beta = np.linalg.solve(F.T @ Ki_F, Ki_F.T @ y)
        self.X, self.resid = X, y - F @ self.beta
        self.alpha_ = linalg.cho_solve(self.L, self.resid)
        return self

    def predict(self, Xs, gamma_phys_db, covariates=None):
        """Kriging mean and variance, Eq. (26)."""
        Fs = np.column_stack([np.ones(len(Xs)), gamma_phys_db] + ([covariates] if covariates is not None else []))
        ks = self._K(Xs, self.X)
        mean = Fs @ self.beta + ks @ self.alpha_
        v = linalg.cho_solve(self.L, ks.T)
        var = self.cov.get("sigma2", 4.0) + self.tau2 - np.sum(ks * v.T, axis=1)
        return mean, np.maximum(var, 1e-9)

    def log_marginal_likelihood(self, y, mu):
        """Eq. (27)."""
        r = y - mu
        n = len(y)
        logdet = 2 * np.sum(np.log(np.diag(self.L[0])))
        return -0.5 * r @ linalg.cho_solve(self.L, r) - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)


def availability(y_hat_db, sigma_db, gamma_th_db=5.0):
    """Predicted link availability A_hat(s,t), Eq. (28)."""
    return stats.norm.cdf((np.asarray(y_hat_db) - gamma_th_db) / np.asarray(sigma_db))


def climatology_correlation(R_hm, S_hm):
    """Rain-scintillation climatology cross-correlation rho_rs, Eq. (30)."""
    r, s = np.ravel(R_hm) - np.mean(R_hm), np.ravel(S_hm) - np.mean(S_hm)
    return float(np.sum(r * s) / np.sqrt(np.sum(r ** 2) * np.sum(s ** 2)))


def delta_method_var(grad, Sigma):
    """First-order SINR variance from input uncertainty, Eq. (31)."""
    g = np.asarray(grad, float)
    return float(g @ np.asarray(Sigma, float) @ g)


def sobol_first_order(model, samplers, n=4000, n_bins=20, rng=None):
    """First-order Sobol indices of Eq. (32) by binning (given-data estimator).

    model    : callable taking dict of arrays -> output array
    samplers : dict name -> callable(rng, n) returning samples
    """
    rng = np.random.default_rng(rng)
    X = {k: f(rng, n) for k, f in samplers.items()}
    Y = model(X)
    VY = np.var(Y)
    S = {}
    for k, x in X.items():
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
        means = np.array([Y[idx == b].mean() for b in range(n_bins)])
        S[k] = float(np.var(means) / VY)
    return S
