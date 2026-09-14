import yfinance as yf

START_DATE = "2010-01-01"
TICKERS = ["AAPL", "SPY"]


def load_data(tickers=TICKERS, start=START_DATE):
    data = yf.download(tickers, start=start, auto_adjust=True)
    return data


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.tail())
    print(df.shape)
