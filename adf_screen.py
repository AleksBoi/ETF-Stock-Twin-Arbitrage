import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from main import load_data

# ~21 liquid large-caps spanning several sectors, to run a broad screen of
# ~200 pairs rather than a hand-picked "obvious twins" shortlist.
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",       # tech
    "JPM", "BAC", "WFC", "GS",                     # financials
    "JNJ", "PFE", "UNH",                           # healthcare
    "KO", "PEP", "PG", "CL", "WMT", "HD", "MCD",   # consumer
    "XOM", "CVX",                                  # energy
]


def fit_beta(log_y, log_x):
    """Fixed-beta OLS fit of log(y) on log(x) (Engle-Granger step 1)."""
    beta, alpha, r_value, _, _ = stats.linregress(log_x, log_y)
    return alpha, beta, r_value ** 2


def spread_from_beta(log_y, log_x, alpha, beta):
    return (log_y - (alpha + beta * log_x)).dropna()


def run_adf(spread):
    """Augmented Dickey-Fuller test. H0 = unit root (not mean-reverting);
    reject H0 (small p-value) => evidence the spread is stationary."""
    stat, pvalue, _, nobs, _, _ = adfuller(spread.dropna(), autolag="AIC")
    return stat, pvalue, nobs


def cointegration_stats(prices, y_col, x_col):
    """Fixed-beta OLS + ADF test for one pair, fit and tested on the same
    (full) sample. Returns the numbers needed to rank pairs."""
    log_y = np.log(prices[y_col])
    log_x = np.log(prices[x_col])

    alpha, beta, r2 = fit_beta(log_y, log_x)
    spread = spread_from_beta(log_y, log_x, alpha, beta)
    stat, pvalue, _ = run_adf(spread)
    return_corr = log_y.diff().corr(log_x.diff())

    return {
        "y": y_col,
        "x": x_col,
        "beta": beta,
        "r2": r2,
        "return_corr": return_corr,
        "adf_stat": stat,
        "adf_pvalue": pvalue,
    }


def screen_adf(tickers=UNIVERSE, prices=None):
    """Fixed-beta OLS + ADF cointegration test (cointegration_stats) over
    every unordered pair in `tickers`, sorted by ADF p-value ascending.

    Testing this many pairs at once means multiple-comparisons noise: at
    p < 0.05, expect ~5% of all pairs to look "significant" by chance alone.
    Treat the p-value ranking as a shortlist to validate out-of-sample, not
    a final verdict."""
    if prices is None:
        prices = load_data(tickers)
    rows = [
        cointegration_stats(prices, y_col, x_col)
        for y_col, x_col in itertools.combinations(tickers, 2)
    ]
    return pd.DataFrame(rows).sort_values("adf_pvalue").reset_index(drop=True)


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    pd.set_option("display.max_rows", None)

    results = screen_adf()
    print(f"{len(results)} pairs tested\n")
    print(results.to_string(index=False))
