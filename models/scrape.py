# ============================================
# scrape_build_stage2.py
# ============================================
import os
import math
import time
import json
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pandas_datareader import data as pdr

# -------------------- Config --------------------
FRED_SERIES = {
    "CPI": "CPIAUCSL",  # CPI (Index, monthly)
    "GDP": "GDP",  # Nominal GDP (Billions $, quarterly)
    "RATE": "FEDFUNDS",  # Effective Fed funds rate (%, monthly)
}
GICS_SECTORS = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
]


# ----------------- Tech Indicators --------------
def ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder's smoothing
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def macd_signal(
    close: pd.Series, fast=12, slow=26, signal=9
) -> Tuple[pd.Series, pd.Series]:
    macd = ema(close, fast) - ema(close, slow)
    sig = ema(macd, signal)
    return macd, sig


def add_tech(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    macd, sig = macd_signal(df["Close"])
    df["MACD"] = macd - sig
    df["RSI"] = rsi_wilder(df["Close"], 14)
    return df


# ----------------- Macro from FRED --------------
def fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    try:
        s = pdr.DataReader(series_id, "fred", start=start, end=end).iloc[:, 0]
        s.name = series_id
        return s
    except Exception as e:
        print(f"[FRED] {series_id} fetch failed: {e}")
        return pd.Series(dtype=float)


def align_macro_to_calendar(
    dates: pd.DatetimeIndex, start: str, end: str
) -> pd.DataFrame:
    out = pd.DataFrame(index=dates)
    for key, sid in FRED_SERIES.items():
        s = fetch_fred(sid, start, end)
        if s.empty:
            out[key] = np.nan
            out[f"days_since_{key.lower()}"] = np.nan
            continue
        s = s.sort_index()
        # Forward-fill to daily calendar
        daily = s.reindex(dates.union(s.index)).ffill().reindex(dates)
        # staleness in days = current_date - last_print_date
        last_date = (
            s.reindex(dates.union(s.index)).ffill().index.to_series().reindex(dates)
        )
        staleness = (
            dates.to_series().dt.normalize() - last_date.dt.normalize()
        ).dt.days.values
        out[key] = daily.values
        out[f"days_since_{key.lower()}"] = staleness
    return out


# -------------- Fundamentals / Ratios ----------
def try_fmp_ratios(ticker: str, api_key: Optional[str]) -> Optional[pd.DataFrame]:
    if not api_key:
        return None
    url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}"
    params = {"period": "quarter", "apikey": api_key}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        df = pd.DataFrame(data)
        # Keep only columns we need, with consistent names
        keep = {
            "grossProfitMargin": "GPM",
            "netProfitMargin": "NPM",
            "operatingProfitMargin": "OPM",
            "priceEarningsRatio": "PE",
            "debtEquityRatio": "DE",
            "freeCashFlowOperatingCashFlowRatio": "FCF_ratio",
            "date": "date",
        }
        cols = [c for c in keep.keys() if c in df.columns]
        df = df[cols + ["date"]].rename(columns=keep)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        return df
    except Exception as e:
        print(f"[FMP] ratios fetch failed for {ticker}: {e}")
        return None


def yf_financial_ratios(ticker: str) -> Optional[pd.DataFrame]:
    """
    Best-effort quarterly ratios from yfinance statements.
    """
    try:
        t = yf.Ticker(ticker)
        inc = t.quarterly_income_stmt  # rows as items, cols as periods
        bal = t.quarterly_balance_sheet
        cfs = t.quarterly_cashflow

        if (
            inc is None
            or bal is None
            or cfs is None
            or inc.empty
            or bal.empty
            or cfs.empty
        ):
            return None

        inc = inc.T  # index: period end
        bal = bal.T
        cfs = cfs.T

        # Required fields
        rev = inc.get("Total Revenue")
        gp = inc.get("Gross Profit")
        opi = inc.get("Operating Income")
        ni = inc.get("Net Income")
        debt = (
            bal.get("Total Debt")
            if "Total Debt" in bal.columns
            else (bal.get("Short Long Term Debt") + bal.get("Long Term Debt"))
        )
        equity = bal.get("Total Stockholder Equity")
        ocf = cfs.get("Operating Cash Flow")
        fcf = (
            cfs.get("Free Cash Flow")
            if "Free Cash Flow" in cfs.columns
            else (ocf - cfs.get("Capital Expenditure"))
        )

        df = pd.DataFrame(index=inc.index)
        df["GPM"] = (gp / rev).astype(float)
        df["NPM"] = (ni / rev).astype(float)
        df["OPM"] = (opi / rev).astype(float)
        df["DE"] = (debt / equity).astype(float)
        # P/E (trailing): price / trailing EPS; we approximate using yfinance trailingPE
        info = t.fast_info if hasattr(t, "fast_info") else {}
        pe = (
            getattr(info, "trailing_pe", None)
            if hasattr(info, "trailing_pe")
            else info.get("trailingPe", None) if isinstance(info, dict) else None
        )
        if pe is None:
            pe = t.info.get("trailingPE", np.nan) if hasattr(t, "info") else np.nan
        df["PE"] = float(pe) if pe is not None else np.nan
        if fcf is not None and ocf is not None:
            df["FCF_ratio"] = (fcf / ocf).astype(float)

        df = df.sort_index()
        return df
    except Exception as e:
        print(f"[YF] fundamentals failed for {ticker}: {e}")
        return None


def fundamentals_timeseries(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Returns a daily-forward-filled DataFrame with columns:
    [GPM, NPM, OPM, PE, DE, FCF_ratio, days_since_fund]
    """
    api_key = os.getenv("FMP_API_KEY", None)
    df_q = try_fmp_ratios(ticker, api_key) or yf_financial_ratios(ticker)
    if df_q is None or df_q.empty:
        # Fallback: NaNs
        rng = pd.date_range(start, end, freq="B")
        out = pd.DataFrame(
            index=rng,
            columns=["GPM", "NPM", "OPM", "PE", "DE", "FCF_ratio"],
            dtype=float,
        )
        out["days_since_fund"] = np.nan
        return out

    # Approximate availability lag: 45 days for 10-Q, 90 days for 10-K (we use 60 days generic)
    df_q = df_q.copy()
    df_q["available_date"] = df_q.index + pd.to_timedelta(60, unit="D")
    df_q = df_q.sort_values("available_date")

    # Build a step-function time series switching at available_date
    rng = pd.date_range(start, end, freq="B")
    out = pd.DataFrame(
        index=rng, columns=["GPM", "NPM", "OPM", "PE", "DE", "FCF_ratio"], dtype=float
    )
    last_avail = None
    cur_vals = None
    j = 0
    rows = df_q[
        ["available_date", "GPM", "NPM", "OPM", "PE", "DE", "FCF_ratio"]
    ].to_records(index=False)
    for d in rng:
        while j < len(rows) and rows[j].available_date <= d:
            last_avail = rows[j].available_date
            cur_vals = [
                rows[j].GPM,
                rows[j].NPM,
                rows[j].OPM,
                rows[j].PE,
                rows[j].DE,
                rows[j].FCF_ratio,
            ]
            j += 1
        if cur_vals is not None:
            out.loc[d, ["GPM", "NPM", "OPM", "PE", "DE", "FCF_ratio"]] = cur_vals
        else:
            out.loc[d] = np.nan
    # Staleness vs last available
    last_dates = pd.Series(index=rng, dtype="datetime64[ns]")
    cur = pd.NaT
    j = 0
    avails = df_q["available_date"].to_list()
    for d in rng:
        while j < len(avails) and avails[j] <= d:
            cur = avails[j]
            j += 1
        last_dates.loc[d] = cur
    out["days_since_fund"] = (
        rng.to_series().dt.normalize() - last_dates.dt.normalize()
    ).dt.days.values
    return out


# --------------- Prices / VIX / Sector --------------
def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return df
    df = df.rename(columns=str.title)  # Open, High, Low, Close, Volume, Adj Close
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df


def fetch_vix(start: str, end: str) -> pd.Series:
    vix = yf.download("^VIX", start=start, end=end, progress=False)["Close"]
    vix.name = "VIX"
    return vix


def get_sector_onehot(ticker: str) -> np.ndarray:
    try:
        info = yf.Ticker(ticker).info or {}
        sector = info.get("sector", None)
    except Exception:
        sector = None
    vec = np.zeros(len(GICS_SECTORS), dtype=np.float32)
    if sector in GICS_SECTORS:
        vec[GICS_SECTORS.index(sector)] = 1.0
    return vec


# --------------- Window stacking / labels --------------
def stack_windows(mat: np.ndarray, lookback: int) -> np.ndarray:
    """
    mat: [N_days, F]; returns [N_eff, lookback, F] using windows ending at each t0-1
    t0 indices are aligned to prediction anchor day; you’ll drop last H for label.
    """
    N, F = mat.shape
    if N < lookback:
        return np.zeros((0, lookback, F), dtype=mat.dtype)
    strides = (mat.strides[0], mat.strides[0], mat.strides[1])
    shape = (N - lookback + 1, lookback, F)
    return np.lib.stride_tricks.as_strided(mat, shape=shape, strides=strides).copy()


def fwd_return(close: pd.Series, horizon_days: int) -> pd.Series:
    return close.shift(-horizon_days) / close - 1.0


# --------------- Master builder ----------------
def build_stage2_for_ticker(
    ticker: str,
    start: str,
    end: str,
    lookback: int = 64,
    horizon_days: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns:
      X_seq : [B, T, 10]  with [O,H,L,C,V,SMA50,SMA200,MACD,RSI,VIX]
      X_tab : [B, F_tab]  with [GPM,NPM,OPM,PE,DE,FCF_ratio, GDP,CPI,RATE, GICS_onehot] + staleness [days_since_*]
      x_aux : [B, 1]      with [horizon_days]
      y     : [B]         forward return over horizon
      panel : pd.DataFrame merged panel for inspection (dates index)
    """
    # --- Prices & tech ---
    px = fetch_ohlcv(ticker, start, end)
    if px.empty:
        raise ValueError(f"No price data for {ticker}")
    # Flatten columns if MultiIndex (fix for yfinance >=0.2.36)
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = ["_".join(col).strip() for col in px.columns.values]
    # Robustly rename columns to standard names
    col_map = {}
    for col in px.columns:
        col_lower = col.lower()
        if "close" in col_lower:
            col_map[col] = "Close"
        elif "open" in col_lower:
            col_map[col] = "Open"
        elif "high" in col_lower:
            col_map[col] = "High"
        elif "low" in col_lower:
            col_map[col] = "Low"
        elif "volume" in col_lower:
            col_map[col] = "Volume"
    px = px.rename(columns=col_map)
    print("Columns after renaming:", px.columns)
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in px.columns]
    if missing:
        raise ValueError(
            f"Missing columns in px: {missing} (columns: {list(px.columns)})"
        )
    px = px[required_cols]
    px = add_tech(px)

    # --- VIX ---
    vix = fetch_vix(start, end)
    px = px.join(vix, how="left").ffill()
    # Ensure VIX column exists
    if "VIX" not in px.columns:
        px["VIX"] = np.nan

    # --- Macro (daily aligned + staleness) ---
    macro = align_macro_to_calendar(px.index, start, end)
    # --- Fundamentals time series (quarterly → step → daily) ---
    fund = fundamentals_timeseries(ticker, start, end)

    # Merge everything to price calendar
    panel = px.join(macro, how="left").join(fund, how="left")
    # Build staleness flags for macro missing
    for key in ["CPI", "GDP", "RATE"]:
        panel[f"is_missing_{key.lower()}"] = panel[key].isna().astype(np.int8)
    panel = panel.ffill()

    # Fill any remaining NaNs in technicals and macro/fundamentals
    panel = panel.bfill().ffill().fillna(0)

    # Sequence features
    seq_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA50",
        "SMA200",
        "MACD",
        "RSI",
        "VIX",
    ]
    seq_mat = panel[seq_cols].astype(np.float32).values
    X_seq_full = stack_windows(seq_mat, lookback)  # aligns to t from lookback-1..end

    # Tabular (as-of t0): take values at anchor day
    gics_oh = get_sector_onehot(ticker)
    gics_tile = np.repeat(gics_oh[None, :], X_seq_full.shape[0], axis=0)

    # Macro (last prints) and staleness aligned to t0 index (the last row of each window)
    anchor_idx = panel.index[lookback - 1 :]
    tab_df = pd.DataFrame(index=anchor_idx)
    tab_df[["GPM", "NPM", "OPM", "PE", "DE", "FCF_ratio"]] = panel.loc[
        anchor_idx, ["GPM", "NPM", "OPM", "PE", "DE", "FCF_ratio"]
    ].values
    tab_df[["GDP", "CPI", "RATE"]] = panel.loc[
        anchor_idx, ["GDP", "CPI", "RATE"]
    ].values
    tab_df[
        ["days_since_gdp", "days_since_cpi", "days_since_rate", "days_since_fund"]
    ] = panel.loc[
        anchor_idx,
        ["days_since_gdp", "days_since_cpi", "days_since_rate", "days_since_fund"],
    ].values
    X_tab_core = (
        tab_df[
            [
                "GPM",
                "NPM",
                "OPM",
                "PE",
                "DE",
                "FCF_ratio",
                "GDP",
                "CPI",
                "RATE",
                "days_since_gdp",
                "days_since_cpi",
                "days_since_rate",
                "days_since_fund",
            ]
        ]
        .astype(np.float32)
        .values
    )
    X_tab = np.concatenate([X_tab_core, gics_tile.astype(np.float32)], axis=1)

    # Aux & labels
    x_aux = np.full((X_seq_full.shape[0], 1), float(horizon_days), dtype=np.float32)
    y_series = fwd_return(panel["Close"], horizon_days)
    y = y_series.iloc[lookback - 1 :].values.astype(np.float32)

    # Drop last horizon where we don't have label
    cut = X_seq_full.shape[0] - horizon_days
    X_seq = X_seq_full[:cut]
    X_tab = X_tab[:cut]
    x_aux = x_aux[:cut]
    y = y[:cut]
    panel_out = panel.iloc[: lookback - 1 + cut]  # for inspection

    return X_seq, X_tab, x_aux, y, panel_out


# --------------- Multi-ticker wrapper ------------
def build_stage2_multi(
    tickers: List[str],
    start: str,
    end: str,
    lookback: int = 64,
    horizon_days: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    Xs, Tabs, Auxs, Ys, metas = [], [], [], [], {}
    for tk in tickers:
        print(f"[Build] {tk}")
        Xi, Ti, Ai, Yi, panel = build_stage2_for_ticker(
            tk, start, end, lookback, horizon_days
        )
        Xs.append(Xi)
        Tabs.append(Ti)
        Auxs.append(Ai)
        Ys.append(Yi)
        # Save panel for debugging
        metas[tk] = {
            "panel_head": panel.head(3),
            "panel_tail": panel.tail(3),
            "rows": len(panel),
        }
        # Optional: dump parquet
        out_path = f"panel_{tk.replace('^','_')}.parquet"
        panel.to_parquet(out_path)
        print(f"  ↳ panel → {out_path} ({len(panel)} rows)")
    X_seq = np.concatenate(Xs, axis=0)
    X_tab = np.concatenate(Tabs, axis=0)
    x_aux = np.concatenate(Auxs, axis=0)
    y = np.concatenate(Ys, axis=0)
    return X_seq, X_tab, x_aux, y, metas


# ------------------------ CLI demo ------------------------
if __name__ == "__main__":
    # Example: 5 tickers, last 5 years
    tickers = ["AAPL", "MSFT", "XOM", "JPM", "WMT"]
    start = "2018-01-01"
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    X_seq, X_tab, x_aux, y, meta = build_stage2_multi(
        tickers=tickers, start=start, end=end, lookback=64, horizon_days=10
    )
    print(
        "[Stage2] X_seq",
        X_seq.shape,
        "X_tab",
        X_tab.shape,
        "x_aux",
        x_aux.shape,
        "y",
        y.shape,
    )
    # Save arrays for your training script
    np.savez_compressed(
        "stage2_dataset.npz",
        X_seq=X_seq,
        X_tab=X_tab,
        x_aux=x_aux,
        y=y,
        note="Seq:[O,H,L,C,V,SMA50,SMA200,MACD,RSI,VIX]; Tab:[GPM,NPM,OPM,PE,DE,FCF_ratio,GDP,CPI,RATE,days_since_*,GICS(11)]",
    )
    print("→ wrote stage2_dataset.npz")
