import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from main import load_data, TICKERS

WINDOW = 60  # trailing days for the rolling regression


def rolling_ols(prices, window=WINDOW, y_col=TICKERS[0], x_col=TICKERS[1]):
    """Rolling regression of log(y) on log(x): y_t = alpha + beta * x_t.

    beta/alpha are estimated on the trailing `window` ending at t-1, then
    shifted forward one day before being applied to day t's actual price.
    This keeps spread_t out-of-sample: day t's price has zero influence on
    the line used to score day t, avoiding the look-ahead bias of judging
    a point against a line that was fit including that same point (which
    mechanically shrinks the residual toward 0 by construction of OLS).
    """
    log_y = np.log(prices[y_col])
    log_x = np.log(prices[x_col])

    cov = log_y.rolling(window).cov(log_x)
    var = log_x.rolling(window).var()
    beta = cov / var
    alpha = log_y.rolling(window).mean() - beta * log_x.rolling(window).mean()

    beta_lagged = beta.shift(1)
    alpha_lagged = alpha.shift(1)
    spread = log_y - (alpha_lagged + beta_lagged * log_x)

    return pd.DataFrame({"beta": beta_lagged, "alpha": alpha_lagged, "spread": spread})


def static_ols(prices, y_col=TICKERS[0], x_col=TICKERS[1]):
    """Single fixed-beta regression of log(y) on log(x) over the whole
    sample (Engle-Granger step 1). Unlike rolling_ols, alpha/beta don't
    change over time, so the resulting spread is a valid series to test
    for stationarity: it reflects one relationship, not an estimator that
    re-anchors to the recent ratio every day."""
    log_y = np.log(prices[y_col])
    log_x = np.log(prices[x_col])

    beta, alpha, r_value, _, _ = stats.linregress(log_x, log_y)
    spread = log_y - (alpha + beta * log_x)

    return spread, alpha, beta


def adf_test(spread, label=""):
    """Augmented Dickey-Fuller test for stationarity of the spread.
    H0 = spread has a unit root (not mean-reverting). Reject H0 (small
    p-value) => evidence the spread is stationary / mean-reverting."""
    spread = spread.dropna()
    stat, pvalue, _, nobs, crit, _ = adfuller(spread, autolag="AIC")

    tag = f" ({label})" if label else ""
    print(f"\nADF test{tag}: statistic={stat:.3f}, p-value={pvalue:.4f}, n={nobs}")
    for level, cv in crit.items():
        print(f"  {level} critical value: {cv:.3f}")
    if pvalue < 0.05:
        print("  => reject unit root: spread looks mean-reverting")
    else:
        print("  => cannot reject unit root: no significant mean reversion")

    return stat, pvalue


def half_life(spread):
    """Mean-reversion half-life from an AR(1) fit: spread_t = a + b*spread_{t-1}.
    half_life = -ln(2)/ln(b), in units of the spread's own sampling frequency
    (trading days here). Only meaningful when 0 < b < 1 (i.e. actually
    mean-reverting); a slow/negative or explosive b makes the number
    meaningless, so we flag that instead of printing a bogus half-life."""
    spread = spread.dropna()
    lagged = spread.shift(1).dropna()
    current = spread.loc[lagged.index]

    b, a, _, _, _ = stats.linregress(lagged, current)
    if not (0 < b < 1):
        print(f"\nhalf-life: undefined (AR(1) coefficient b={b:.3f} outside (0,1))")
        return None

    hl = -np.log(2) / np.log(b)
    print(f"\nhalf-life: {hl:.1f} trading days (AR(1) b={b:.3f})")
    return hl


def analyze_spread(spread):
    """Fit a Gaussian to the (out-of-sample) spread and report what fraction
    of the time it sits within various sigma bands, vs. what a normal
    distribution would predict."""
    spread = spread.dropna()
    mu, sigma = stats.norm.fit(spread)
    return mu, sigma


def plot_spread_distribution(spread, mu, sigma):
    spread = spread.dropna()
    fig, ax = plt.subplots()
    ax.hist(spread, bins=50, density=True, alpha=0.6, label="empirical")
    x = np.linspace(spread.min(), spread.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), label="fitted normal")
    ax.axvline(0, linestyle="--", color="black")
    ax.set_xlabel("Spread (log-price residual)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of out-of-sample spread")
    ax.legend()
    plt.show()
    return fig


def plot_rolling_ols(ols, y_col=TICKERS[0], x_col=TICKERS[1]):
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(ols.index, ols["beta"])
    axes[0].set_ylabel(f"Rolling beta ({y_col} on {x_col})")
    axes[0].set_title(f"Rolling OLS ({WINDOW}-day window)")

    axes[1].plot(ols.index, ols["spread"])
    axes[1].axhline(0, linestyle="--")
    axes[1].set_ylabel("Spread (residual)")
    axes[1].set_xlabel("Date")

    plt.show()
    return fig


if __name__ == "__main__":
    df = load_data()

    static_spread, alpha, beta = static_ols(df)
    adf_test(static_spread, label="fixed-beta")
    half_life(static_spread)

    ols = rolling_ols(df)
    plot_rolling_ols(ols)

    mu, sigma = analyze_spread(ols["spread"])
    plot_spread_distribution(ols["spread"], mu, sigma)
