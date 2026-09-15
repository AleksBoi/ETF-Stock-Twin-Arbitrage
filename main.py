import yfinance as yf
import matplotlib.pyplot as plt

START_DATE = "2020-01-01"
TICKERS = ["AAPL", "SPY"]


def load_data(tickers=TICKERS, start=START_DATE):
    data = yf.download(tickers, start=start, auto_adjust=True)
    return data["Close"]


def rebase(prices, base=100):
    """Index each column to `base` at its first observation, so price
    levels become comparable regardless of each ticker's actual price."""
    return prices / prices.iloc[0] * base


def plot_rebased(rebased):
    fig, ax = plt.subplots()
    for ticker in rebased.columns:
        ax.plot(rebased.index, rebased[ticker], label=ticker)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Rebased price (start = {rebased.iloc[0, 0]:.0f})")
    ax.set_title("Rebased price comparison")
    ax.legend()
    plt.show()
    return fig


if __name__ == "__main__":
    df = load_data()
    rebased = rebase(df)
    plot_rebased(rebased)
