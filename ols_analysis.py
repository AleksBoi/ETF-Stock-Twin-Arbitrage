import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from main import load_data

WINDOW = 60  # trailing days for the rolling regression


def rolling_ols(prices, window=WINDOW, y_col="AAPL", x_col="SPY"):
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


def analyze_spread(spread):
    """Fit a Gaussian to the (out-of-sample) spread and report what fraction
    of the time it sits within various sigma bands, vs. what a normal
    distribution would predict."""
    spread = spread.dropna()
    mu, sigma = stats.norm.fit(spread)

    print(f"spread: mean={mu:.4f}, std={sigma:.4f}, n={len(spread)}")
    print(f"{'|z| >':>8} {'empirical %':>14} {'gaussian %':>14}")
    for z in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        empirical_pct = (spread.abs() > z * sigma).mean() * 100
        gaussian_pct = 2 * (1 - stats.norm.cdf(z)) * 100
        print(f"{z:>8.1f} {empirical_pct:>13.1f}% {gaussian_pct:>13.1f}%")

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


def plot_rolling_ols(ols):
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(ols.index, ols["beta"])
    axes[0].set_ylabel("Rolling beta (AAPL on SPY)")
    axes[0].set_title(f"Rolling OLS ({WINDOW}-day window)")

    axes[1].plot(ols.index, ols["spread"])
    axes[1].axhline(0, linestyle="--")
    axes[1].set_ylabel("Spread (residual)")
    axes[1].set_xlabel("Date")

    plt.show()
    return fig


if __name__ == "__main__":
    df = load_data()

    ols = rolling_ols(df)
    print(ols.dropna().head())
    print(ols.dropna().tail())
    plot_rolling_ols(ols)

    mu, sigma = analyze_spread(ols["spread"])
    plot_spread_distribution(ols["spread"], mu, sigma)
