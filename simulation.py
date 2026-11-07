"""Simulate correlated GBM price paths for a fixed-share portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd


def parameters(returns: pd.DataFrame, zero_mean: bool = True) -> tuple[pd.Series, pd.DataFrame]:
    # Daily mean *log* return already contains the -sigma^2/2 drag; do not subtract it again.
    # Zero mean is the default: historical means are noisy forecasts and small next to volatility.
    mean = pd.Series(0.0, index=returns.columns) if zero_mean else returns.mean()
    return mean, returns.cov()


def correlated_shocks(cov: pd.DataFrame, n_paths: int, n_days: int,
                      rng: np.random.Generator) -> np.ndarray:
    # Cholesky: cov = L @ L.T. If z ~ N(0, I) then L @ z ~ N(0, cov).
    # Volatility lives inside L, so shocks are not scaled by sigma again.
    lower = np.linalg.cholesky(cov.to_numpy())
    z = rng.standard_normal((n_paths, n_days, len(lower)))
    return z @ lower.T  # shape: (paths, days, assets)
