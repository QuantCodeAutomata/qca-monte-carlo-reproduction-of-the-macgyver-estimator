"""
Data Loader Module.

Fetches daily adjusted close prices for equity tickers.
Primary source: Massive REST API (requires MASSIVE_TOKEN env var).
Fallback source: Yahoo Finance via yfinance.
Used for Experiment 2 (empirical DCC on 18 U.S. large-cap stocks, 1994-2004).
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _load_via_massive(
    tickers: List[str],
    start_date: str,
    end_date: str,
    api_key: str,
) -> pd.DataFrame:
    """Fetch adjusted daily close prices using the Massive REST API."""
    from massive import RESTClient  # noqa: PLC0415

    client = RESTClient(api_key=api_key)
    price_dict: dict = {}

    for ticker in tickers:
        aggs = []
        for a in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=50000,
            adjusted=True,
        ):
            aggs.append(a)

        if not aggs:
            print(f"WARNING: No data returned for {ticker}")
            continue

        records = []
        for a in aggs:
            ts = pd.Timestamp(a.timestamp, unit="ms")
            records.append({"date": ts.normalize(), "close": float(a.close)})

        df_ticker = (
            pd.DataFrame(records)
            .drop_duplicates("date")
            .set_index("date")
        )
        df_ticker.index = pd.DatetimeIndex(df_ticker.index)
        df_ticker.sort_index(inplace=True)
        price_dict[ticker] = df_ticker["close"]

    if not price_dict:
        raise RuntimeError("Massive API: no data retrieved for any ticker.")

    prices = pd.DataFrame(price_dict)
    prices.index = pd.DatetimeIndex(prices.index)
    prices.sort_index(inplace=True)
    return prices


def _load_via_yfinance(
    tickers: List[str],
    start_date: str,
    end_date: str,
    min_coverage: float = 0.80,
) -> pd.DataFrame:
    """Fetch adjusted daily close prices via Yahoo Finance (yfinance fallback).

    Downloads each ticker individually so that one failed ticker does not
    prevent the rest from loading.  Tickers whose non-NaN coverage falls below
    *min_coverage* of the maximum observed row count are silently dropped and
    a warning is printed.
    """
    import yfinance as yf  # noqa: PLC0415
    import warnings

    # yfinance end date is exclusive; add one day for inclusive download
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    end_str = end_dt.strftime("%Y-%m-%d")

    # Some Engle (2007) tickers have changed; map to current yfinance symbols.
    # Old GM (pre-2009 bankruptcy) is unavailable in yfinance; substitute the
    # re-listed GM that debuted 2010-11-18 — its history covers the target
    # 1994-2004 window only as reconstructed data.  We drop it when unavailable
    # and document the omission.
    TICKER_MAP: Dict[str, str] = {
        "AA": "AA",   "AXP": "AXP", "BA": "BA",   "CAT": "CAT",
        "DD": "DD",   "DIS": "DIS", "GE": "GE",   "GM": "GM",
        "IBM": "IBM", "IP": "IP",   "JNJ": "JNJ", "JPM": "JPM",
        "KO": "KO",   "MCD": "MCD", "MMM": "MMM", "MO": "MO",
        "MRK": "MRK", "MSFT": "MSFT",
        "SPY": "SPY",
    }

    frames: Dict[str, pd.Series] = {}
    for orig in tickers:
        yf_sym = TICKER_MAP.get(orig, orig)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    yf_sym,
                    start=start_date,
                    end=end_str,
                    auto_adjust=True,
                    progress=False,
                )
            if raw.empty or len(raw) == 0:
                print(f"  [yfinance] {orig} ({yf_sym}): no data returned — skipping.", flush=True)
                continue
            # yfinance ≥1.0 may return MultiIndex columns even for single ticker
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"].iloc[:, 0]
            else:
                close = raw["Close"]
            close.name = orig
            frames[orig] = close
        except Exception as exc:
            print(f"  [yfinance] {orig} ({yf_sym}): error ({exc}) — skipping.", flush=True)

    if not frames:
        raise RuntimeError("yfinance returned no data for any ticker.")

    prices = pd.DataFrame(frames)
    prices.index = pd.DatetimeIndex(prices.index).normalize()
    prices.sort_index(inplace=True)

    # Drop tickers whose non-NaN row count is below min_coverage threshold
    max_rows = prices.notna().sum().max()
    coverage = prices.notna().sum() / max_rows
    sparse = coverage[coverage < min_coverage].index.tolist()
    if sparse:
        print(
            f"  [yfinance] dropping sparse tickers (coverage < {min_coverage:.0%}): {sparse}",
            flush=True,
        )
        prices.drop(columns=sparse, inplace=True)

    return prices


def load_stock_data(
    tickers: List[str],
    start_date: str = "1994-01-01",
    end_date: str = "2004-12-31",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Load daily adjusted close prices for given tickers.

    Tries Massive REST API first (requires MASSIVE_TOKEN env var or api_key),
    then falls back to Yahoo Finance.  Aligns all tickers on the common
    intersection of trading dates; no imputation of missing values.

    Args:
        tickers:    List of ticker symbols (uppercase, e.g. ['AAPL', 'SPY']).
        start_date: Start date string 'YYYY-MM-DD'.
        end_date:   End date string 'YYYY-MM-DD'.
        api_key:    Massive API key; defaults to MASSIVE_TOKEN env var.

    Returns:
        prices: DataFrame of adjusted close prices, indexed by date (datetime),
                columns = tickers.  Shape (T_common, len(tickers)).
    """
    key = api_key or os.getenv("MASSIVE_API_KEY") or os.getenv("MASSIVE_TOKEN")

    # Minimum acceptable trading days — anything shorter triggers yfinance fallback
    MIN_DAYS = 1000

    if key:
        print("Loading data via Massive REST API ...", flush=True)
        try:
            prices_raw = _load_via_massive(tickers, start_date, end_date, key)
            prices_raw = prices_raw.dropna(axis=0, how="any")
            if len(prices_raw) >= MIN_DAYS:
                prices = prices_raw
            else:
                print(
                    f"  Massive returned only {len(prices_raw)} common days "
                    f"(need >= {MIN_DAYS}) — falling back to Yahoo Finance ...",
                    flush=True,
                )
                prices = _load_via_yfinance(tickers, start_date, end_date)
        except Exception as exc:
            print(f"  Massive API error ({exc}) — falling back to Yahoo Finance ...", flush=True)
            prices = _load_via_yfinance(tickers, start_date, end_date)
    else:
        print("MASSIVE_API_KEY/MASSIVE_TOKEN not set — falling back to Yahoo Finance ...", flush=True)
        prices = _load_via_yfinance(tickers, start_date, end_date)

    # Retain only dates where ALL tickers have data (common intersection)
    prices = prices.dropna(axis=0, how="any")

    if prices.empty:
        raise RuntimeError(
            f"No common trading dates found for tickers={tickers} "
            f"over {start_date}–{end_date}."
        )

    print(
        f"Loaded {prices.shape[1]} tickers, {prices.shape[0]} common trading days "
        f"({prices.index[0].date()} to {prices.index[-1].date()})",
        flush=True,
    )
    return prices


def compute_returns(
    prices: pd.DataFrame,
    log_returns: bool = False,
) -> pd.DataFrame:
    """Compute simple or log returns from an adjusted price DataFrame.

    Args:
        prices:      DataFrame of adjusted close prices.
        log_returns: If True, compute log returns; otherwise simple returns.

    Returns:
        returns: DataFrame of returns, same columns, T-1 rows (first row dropped).
    """
    if log_returns:
        rets = np.log(prices / prices.shift(1)).dropna()
    else:
        rets = prices.pct_change().dropna()

    return rets.astype(np.float64)
