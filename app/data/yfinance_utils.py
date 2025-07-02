import yfinance as yf
import pandas as pd
from typing import Optional

def get_spot_price(ticker: str) -> Optional[float]:
    """
    Fetch the latest spot price for a given ticker using yfinance.
    Returns None if ticker is invalid or data unavailable.
    """
    try:
        data = yf.Ticker(ticker)
        price = data.history(period="1d").iloc[-1]["Close"]
        return float(price)
    except Exception as e:
        print(f"Error fetching spot price for {ticker}: {e}")
        return None

def get_historical_prices(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch historical price data for a given ticker.
    period: e.g. '1y', '6mo', '1mo', '5d'
    interval: e.g. '1d', '1h', '5m'
    Returns DataFrame with Date index and OHLCV columns, or None on error.
    """
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period=period, interval=interval)
        if hist.empty:
            return None
        return hist
    except Exception as e:
        print(f"Error fetching historical prices for {ticker}: {e}")
        return None

def get_option_chain(ticker: str):
    """
    Fetch the full option chain for a given ticker.
    Returns a dict with 'calls' and 'puts' DataFrames, or None on error.
    """
    try:
        data = yf.Ticker(ticker)
        expiries = data.options
        chains = {}
        for expiry in expiries:
            opt = data.option_chain(expiry)
            chains[expiry] = {'calls': opt.calls, 'puts': opt.puts}
        return chains
    except Exception as e:
        print(f"Error fetching option chain for {ticker}: {e}")
        return None 