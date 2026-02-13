#!/usr/bin/env python3
"""
Sector Rotation Tactical Asset Allocation via Adjusted Relative Rotation Graphs (RRG)
=====================================================================================

Multi-factor RRG model for US equity sectors optimised for a 1-3 month horizon.
Identifies emerging sector leaders for overweighting based on short-term RS-Momentum
blended with momentum / value / quality / low-vol / size factor z-scores.

Usage
-----
    python sector_rrg_taa.py                          # defaults
    python sector_rrg_taa.py --horizon_weeks 5        # custom lookback
    python sector_rrg_taa.py --api_key YOUR_FMP_KEY   # use FMP for fundamentals

Author : Claude (Anthropic)
License: MIT
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance is required. Install with: pip install yfinance")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTOR_TICKERS: List[str] = [
    "XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLU", "XLRE", "XLB", "XLC",
]
BENCHMARK = "SPY"
CASH_PROXY = "SHY"

QUADRANT_LABELS = {
    "Leading":   {"rs_ratio": "above", "rs_mom": "above"},
    "Weakening": {"rs_ratio": "above", "rs_mom": "below"},
    "Lagging":   {"rs_ratio": "below", "rs_mom": "below"},
    "Improving": {"rs_ratio": "below", "rs_mom": "above"},
}

SECTOR_NAMES: Dict[str, str] = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care",
    "XLY": "Cons. Discr.", "XLP": "Cons. Staples", "XLI": "Industrials",
    "XLE": "Energy", "XLU": "Utilities", "XLRE": "Real Estate",
    "XLB": "Materials", "XLC": "Communication",
}

COLORS: Dict[str, str] = {
    "XLK": "#0071C5", "XLF": "#4CAF50", "XLV": "#E91E63",
    "XLY": "#FF9800", "XLP": "#9C27B0", "XLI": "#607D8B",
    "XLE": "#795548", "XLU": "#00BCD4", "XLRE": "#F44336",
    "XLB": "#8BC34A", "XLC": "#3F51B5",
}


# ===================================================================
# 1. DATA FETCHING
# ===================================================================
def fetch_data(
    tickers: List[str],
    benchmark: str = BENCHMARK,
    cash: str = CASH_PROXY,
    years: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted-close prices for sectors, benchmark, and cash proxy.

    Returns
    -------
    weekly : pd.DataFrame   – Friday-resampled weekly closes (all tickers + benchmark + cash)
    daily  : pd.DataFrame   – daily closes (for volatility calcs)
    """
    all_tickers = list(set(tickers + [benchmark, cash]))
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * years + 90)  # extra buffer

    print(f"[INFO] Fetching {len(all_tickers)} tickers from {start} to {end} ...")
    try:
        raw = yf.download(all_tickers, start=str(start), end=str(end),
                          auto_adjust=True, progress=False, threads=True)
    except Exception as exc:
        sys.exit(f"[ERROR] yfinance download failed: {exc}")

    if raw.empty:
        sys.exit("[ERROR] No data returned from yfinance.")

    # Handle multi-level columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw.copy()

    # Drop tickers with >20% missing
    thresh = len(close) * 0.80
    close = close.dropna(axis=1, thresh=int(thresh))
    close = close.ffill().bfill()

    missing = set(tickers) - set(close.columns)
    if missing:
        print(f"[WARN] Tickers dropped due to missing data: {missing}")

    daily = close.copy()

    # Resample to weekly (Friday)
    weekly = close.resample("W-FRI").last().dropna(how="all")

    print(f"[INFO] Weekly data shape: {weekly.shape}  |  Range: {weekly.index[0].date()} → {weekly.index[-1].date()}")
    return weekly, daily


# ===================================================================
# 2. CORE RRG COMPUTATION
# ===================================================================
def compute_rrg(
    weekly: pd.DataFrame,
    tickers: List[str],
    benchmark: str = BENCHMARK,
    rs_lookback: int = 5,
    mom_lookback: int = 2,
    sensitivity: int = 6,
) -> pd.DataFrame:
    """Compute RS-Ratio and RS-Momentum for each sector vs benchmark.

    Parameters
    ----------
    rs_lookback  : weeks for RS-Ratio (default 5  → 4-6 week spec)
    mom_lookback : weeks for RS-Momentum tail (default 2 → 1-3 week spec)
    sensitivity  : smoothing factor for RS-Momentum (default 6 → 5-7 spec)

    Returns
    -------
    DataFrame with columns: ticker, date, rs_ratio, rs_momentum, quadrant
    """
    avail = [t for t in tickers if t in weekly.columns]
    bm = weekly[benchmark]

    records = []
    for ticker in avail:
        px = weekly[ticker]
        # Raw relative strength line
        rs_line = px / bm

        # RS-Ratio: current RS / RS N weeks ago, rebased to 100
        rs_ratio = (rs_line / rs_line.shift(rs_lookback)) * 100

        # RS-Momentum: smoothed ROC of RS-Ratio
        rs_ratio_roc = rs_ratio.pct_change(mom_lookback) * 100
        # EMA smoothing with span = sensitivity
        rs_mom = rs_ratio_roc.ewm(span=sensitivity, adjust=False).mean() + 100

        for date in rs_ratio.dropna().index:
            if pd.isna(rs_mom.get(date)):
                continue
            rr = rs_ratio.loc[date]
            rm = rs_mom.loc[date]
            quad = _classify_quadrant(rr, rm)
            records.append({
                "date": date, "ticker": ticker,
                "rs_ratio": rr, "rs_momentum": rm,
                "quadrant": quad,
            })

    df = pd.DataFrame(records)
    return df


def _classify_quadrant(rs_ratio: float, rs_mom: float) -> str:
    if rs_ratio >= 100 and rs_mom >= 100:
        return "Leading"
    elif rs_ratio >= 100 and rs_mom < 100:
        return "Weakening"
    elif rs_ratio < 100 and rs_mom < 100:
        return "Lagging"
    else:
        return "Improving"


# ===================================================================
# 3. MULTI-FACTOR SCORING
# ===================================================================
def factor_scores(
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    tickers: List[str],
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Compute cross-sectional z-scores for each factor.

    Factors: Momentum, Value (proxy), Quality (proxy), Low-Vol, Size (proxy).
    Falls back to price-based proxies when fundamental data is unavailable.
    """
    avail = [t for t in tickers if t in weekly.columns and t in daily.columns]
    latest = weekly.index[-1]

    mom_1m, mom_3m = {}, {}
    vol_20d, mktcap = {}, {}
    value_proxy, quality_proxy = {}, {}

    for ticker in avail:
        # --- Momentum ---
        w = weekly[ticker].dropna()
        if len(w) >= 13:
            mom_3m[ticker] = w.iloc[-1] / w.iloc[-13] - 1
        else:
            mom_3m[ticker] = 0.0
        if len(w) >= 4:
            mom_1m[ticker] = w.iloc[-1] / w.iloc[-4] - 1
        else:
            mom_1m[ticker] = 0.0

        # --- Low-Vol (inverse 20-day std of daily returns) ---
        d = daily[ticker].dropna().pct_change().dropna()
        if len(d) >= 20:
            vol_20d[ticker] = d.iloc[-20:].std() * np.sqrt(252)
        else:
            vol_20d[ticker] = d.std() * np.sqrt(252) if len(d) > 1 else 0.2

        # --- Value / Quality / Size proxies via yfinance.info ---
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}

        # Value: inverse forward P/E  (fallback: inverse trailing P/E, or 0)
        fpe = info.get("forwardPE") or info.get("trailingPE")
        value_proxy[ticker] = 1.0 / fpe if fpe and fpe > 0 else 0.0

        # Quality: proxy via ROE or profit margin
        roe = info.get("returnOnEquity")
        quality_proxy[ticker] = roe if roe is not None else (
            info.get("profitMargins", 0.0) or 0.0
        )

        # Size: inverse market cap (favour smaller sectors)
        mc = info.get("totalAssets") or info.get("marketCap") or info.get("navPrice")
        mktcap[ticker] = 1.0 / (mc / 1e9) if mc and mc > 0 else 0.0

    # Build DataFrame
    df = pd.DataFrame(index=avail)
    df["momentum"] = df.index.map(lambda t: 0.5 * mom_1m.get(t, 0) + 0.5 * mom_3m.get(t, 0))
    df["value"]    = df.index.map(lambda t: value_proxy.get(t, 0.0))
    df["quality"]  = df.index.map(lambda t: quality_proxy.get(t, 0.0))
    df["low_vol"]  = df.index.map(lambda t: -vol_20d.get(t, 0.2))  # negative vol = low-vol tilt
    df["size"]     = df.index.map(lambda t: mktcap.get(t, 0.0))

    # Cross-sectional z-score each factor
    for col in ["momentum", "value", "quality", "low_vol", "size"]:
        s = df[col]
        mu, sigma = s.mean(), s.std()
        if sigma > 0:
            df[col + "_z"] = (s - mu) / sigma
        else:
            df[col + "_z"] = 0.0

    # Average z-score (equal weight across factors)
    z_cols = [c for c in df.columns if c.endswith("_z")]
    df["factor_z_avg"] = df[z_cols].mean(axis=1)

    # Pick top-2 factor z-scores per sector for blending
    df["top2_factor_z"] = df[z_cols].apply(lambda row: row.nlargest(2).mean(), axis=1)

    return df


def composite_signal(
    rrg_latest: pd.DataFrame,
    factors: pd.DataFrame,
    rrg_weight: float = 0.70,
    factor_weight: float = 0.30,
) -> pd.DataFrame:
    """Blend RRG scores with multi-factor z-scores into a single composite.

    Composite = rrg_weight * normalised(mean(RS-Ratio, RS-Mom)) + factor_weight * top-2 factor z avg
    Then z-score across sectors.
    """
    merged = rrg_latest.set_index("ticker").join(factors[["factor_z_avg", "top2_factor_z"]], how="inner")

    # Normalise RRG component: average of RS-Ratio and RS-Momentum, then z-score
    merged["rrg_raw"] = (merged["rs_ratio"] + merged["rs_momentum"]) / 2
    mu, sigma = merged["rrg_raw"].mean(), merged["rrg_raw"].std()
    merged["rrg_z"] = (merged["rrg_raw"] - mu) / sigma if sigma > 0 else 0.0

    # Composite
    merged["composite"] = rrg_weight * merged["rrg_z"] + factor_weight * merged["top2_factor_z"]

    # Final z-score of composite
    mu2, sigma2 = merged["composite"].mean(), merged["composite"].std()
    merged["composite_z"] = (merged["composite"] - mu2) / sigma2 if sigma2 > 0 else 0.0

    # Normalise composite to 0-1 range for threshold comparison
    cmin, cmax = merged["composite_z"].min(), merged["composite_z"].max()
    rng = cmax - cmin
    merged["composite_norm"] = (merged["composite_z"] - cmin) / rng if rng > 0 else 0.5

    # RS-Momentum week-over-week ROC
    merged["rs_mom_roc"] = merged["rs_momentum"] - 100  # simplified: >0 means positive momentum

    # Recommendation
    merged["recommendation"] = merged.apply(_recommend, axis=1)

    return merged.reset_index()


def _recommend(row: pd.Series) -> str:
    quad = row.get("quadrant", "Lagging")
    comp = row.get("composite_norm", 0)
    mom_roc = row.get("rs_mom_roc", 0)

    if quad in ("Leading", "Improving") and comp > 0.70 and mom_roc > 0:
        return "Overweight"
    elif quad in ("Leading", "Improving"):
        return "Hold"
    elif quad == "Weakening":
        return "Hold"
    else:
        return "Underweight"


# ===================================================================
# 4. RISK RULES & PORTFOLIO CONSTRUCTION
# ===================================================================
def construct_portfolio(
    signals: pd.DataFrame,
    daily: pd.DataFrame,
    max_sectors: int = 3,
    max_weight: float = 0.25,
    vol_target: float = 0.12,
) -> pd.DataFrame:
    """Build weights for the top-N overweight sectors with risk rules.

    Risk rules:
      1. Inverse-volatility weighting among selected sectors.
      2. Max 25% per sector.
      3. If all sectors have negative absolute momentum → 100% cash (SHY).
    """
    overweight = signals[signals["recommendation"] == "Overweight"]
    if overweight.empty:
        overweight = signals[signals["recommendation"] == "Hold"].nlargest(max_sectors, "composite_norm")
    else:
        overweight = overweight.nlargest(max_sectors, "composite_norm")

    if overweight.empty:
        return pd.DataFrame({"ticker": [CASH_PROXY], "weight": [1.0]})

    # Check absolute momentum (3-month return > 0)
    all_neg = True
    for _, row in overweight.iterrows():
        t = row["ticker"]
        if t in daily.columns:
            px = daily[t].dropna()
            if len(px) >= 63:
                ret_3m = px.iloc[-1] / px.iloc[-63] - 1
                if ret_3m > 0:
                    all_neg = False
                    break
    if all_neg:
        print("[RISK] All selected sectors have negative absolute momentum → 100% CASH")
        return pd.DataFrame({"ticker": [CASH_PROXY], "weight": [1.0]})

    # Inverse-vol weights
    vols = {}
    for _, row in overweight.iterrows():
        t = row["ticker"]
        if t in daily.columns:
            d = daily[t].pct_change().dropna()
            vol = d.iloc[-20:].std() * np.sqrt(252) if len(d) >= 20 else 0.15
            vols[t] = max(vol, 0.01)
        else:
            vols[t] = 0.15

    inv_vol = {t: 1.0 / v for t, v in vols.items()}
    total = sum(inv_vol.values())
    weights = {t: iv / total for t, iv in inv_vol.items()}

    # Cap at max_weight and redistribute
    weights = _cap_weights(weights, max_weight)

    # Vol-target scaling
    port_vol = sum(w * vols.get(t, 0.15) for t, w in weights.items())
    if port_vol > 0:
        scale = min(vol_target / port_vol, 1.0)
        weights = {t: w * scale for t, w in weights.items()}

    # Remainder to cash
    total_w = sum(weights.values())
    if total_w < 1.0:
        weights[CASH_PROXY] = 1.0 - total_w

    result = pd.DataFrame([
        {"ticker": t, "weight": round(w, 4)} for t, w in weights.items() if w > 0.001
    ])
    return result.sort_values("weight", ascending=False).reset_index(drop=True)


def _cap_weights(weights: dict, cap: float) -> dict:
    """Iteratively cap weights and redistribute excess proportionally."""
    for _ in range(10):
        excess = 0.0
        uncapped = []
        for t, w in weights.items():
            if w > cap:
                excess += w - cap
                weights[t] = cap
            else:
                uncapped.append(t)
        if excess <= 0 or not uncapped:
            break
        share = excess / len(uncapped)
        for t in uncapped:
            weights[t] += share
    return weights


# ===================================================================
# 5. BACKTESTING ENGINE
# ===================================================================
def backtest(
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    tickers: List[str],
    benchmark: str = BENCHMARK,
    cash: str = CASH_PROXY,
    rs_lookback: int = 5,
    mom_lookback: int = 2,
    sensitivity: int = 6,
    top_n: int = 3,
    max_weight: float = 0.25,
    tc_bps: float = 10,
    slippage_bps: float = 5,
    vol_target: float = 0.12,
) -> Tuple[pd.DataFrame, Dict]:
    """Walk-forward monthly-rebalance backtest.

    Each month-end:
      1. Compute RRG & factor scores on data available up to that point.
      2. Select top-N sectors in Leading/Improving with highest composite.
      3. Apply risk rules.
      4. Compute next-month return (with transaction costs & slippage).

    Returns equity-curve DataFrame and summary metrics dict.
    """
    # Monthly rebalance dates (last Friday of each month)
    monthly_idx = weekly.resample("ME").last().index
    # Align to weekly index
    monthly_dates = []
    for md in monthly_idx:
        mask = weekly.index <= md
        if mask.any():
            monthly_dates.append(weekly.index[mask][-1])
    monthly_dates = sorted(set(monthly_dates))

    # Need at least 52 weeks warmup
    warmup = max(rs_lookback + mom_lookback + sensitivity + 10, 52)
    if len(weekly) < warmup + 12:
        print(f"[WARN] Insufficient data for backtest (need {warmup + 12} weeks, have {len(weekly)})")
        return pd.DataFrame(), {}

    start_idx = warmup
    start_date = weekly.index[start_idx]
    monthly_dates = [d for d in monthly_dates if d >= start_date]

    if len(monthly_dates) < 2:
        print("[WARN] Not enough monthly dates for backtest.")
        return pd.DataFrame(), {}

    print(f"[BACKTEST] Running from {monthly_dates[0].date()} to {monthly_dates[-1].date()} "
          f"({len(monthly_dates)} rebalance periods)")

    # Precompute full RRG
    rrg_full = compute_rrg(weekly, tickers, benchmark, rs_lookback, mom_lookback, sensitivity)

    port_returns = []
    bm_returns = []
    prev_weights: Dict[str, float] = {}

    for i in range(len(monthly_dates) - 1):
        reb_date = monthly_dates[i]
        next_date = monthly_dates[i + 1]

        # Latest RRG as of reb_date
        rrg_snap = rrg_full[rrg_full["date"] == reb_date]
        if rrg_snap.empty:
            # Find nearest prior date
            prior = rrg_full[rrg_full["date"] <= reb_date]
            if prior.empty:
                continue
            rrg_snap = prior[prior["date"] == prior["date"].max()]

        # Simplified factor scores using only data up to reb_date
        avail = [t for t in tickers if t in weekly.columns]
        weekly_sub = weekly.loc[:reb_date]
        daily_sub = daily.loc[:reb_date]

        factor_df = _quick_factors(weekly_sub, daily_sub, avail)
        sig = composite_signal(rrg_snap, factor_df)

        # Build portfolio
        ow = sig[sig["recommendation"] == "Overweight"]
        if ow.empty:
            ow = sig[sig["recommendation"] == "Hold"].nlargest(top_n, "composite_norm")
        ow = ow.nlargest(top_n, "composite_norm")

        # Inverse-vol weights
        weights: Dict[str, float] = {}
        if not ow.empty:
            all_neg = True
            for _, r in ow.iterrows():
                t = r["ticker"]
                if t in daily_sub.columns:
                    px = daily_sub[t].dropna()
                    if len(px) >= 63 and (px.iloc[-1] / px.iloc[-63] - 1) > 0:
                        all_neg = False
                        break
            if all_neg:
                weights = {cash: 1.0}
            else:
                vols = {}
                for _, r in ow.iterrows():
                    t = r["ticker"]
                    if t in daily_sub.columns:
                        d = daily_sub[t].pct_change().dropna()
                        v = d.iloc[-20:].std() * np.sqrt(252) if len(d) >= 20 else 0.15
                        vols[t] = max(v, 0.01)
                    else:
                        vols[t] = 0.15
                inv_v = {t: 1.0 / v for t, v in vols.items()}
                tot = sum(inv_v.values())
                weights = {t: iv / tot for t, iv in inv_v.items()}
                weights = _cap_weights(weights, max_weight)

                # Vol target
                pv = sum(w * vols.get(t, 0.15) for t, w in weights.items())
                if pv > 0:
                    sc = min(vol_target / pv, 1.0)
                    weights = {t: w * sc for t, w in weights.items()}
                tw = sum(weights.values())
                if tw < 1.0:
                    weights[cash] = weights.get(cash, 0) + (1.0 - tw)
        else:
            weights = {cash: 1.0}

        # Compute period return
        period_ret = 0.0
        for t, w in weights.items():
            if t in weekly.columns:
                p0 = weekly[t].asof(reb_date)
                p1 = weekly[t].asof(next_date)
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    period_ret += w * (p1 / p0 - 1)

        # Transaction costs
        turnover = 0.0
        for t in set(list(weights.keys()) + list(prev_weights.keys())):
            turnover += abs(weights.get(t, 0) - prev_weights.get(t, 0))
        tc = turnover * (tc_bps + slippage_bps) / 10000
        period_ret -= tc

        # Benchmark return
        bm_p0 = weekly[benchmark].asof(reb_date)
        bm_p1 = weekly[benchmark].asof(next_date)
        bm_ret = (bm_p1 / bm_p0 - 1) if pd.notna(bm_p0) and pd.notna(bm_p1) and bm_p0 > 0 else 0.0

        port_returns.append({"date": next_date, "port_return": period_ret, "bm_return": bm_ret,
                             "holdings": dict(weights)})
        bm_returns.append(bm_ret)
        prev_weights = weights

    if not port_returns:
        return pd.DataFrame(), {}

    df = pd.DataFrame(port_returns).set_index("date")
    df["port_cumret"] = (1 + df["port_return"]).cumprod()
    df["bm_cumret"] = (1 + df["bm_return"]).cumprod()

    metrics = _compute_metrics(df)
    return df, metrics


def _quick_factors(weekly: pd.DataFrame, daily: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Compute price-based factor proxies quickly (no API calls) for backtest."""
    df = pd.DataFrame(index=tickers)
    mom_1m, mom_3m, vol_20d_vals = {}, {}, {}
    for t in tickers:
        if t not in weekly.columns:
            continue
        w = weekly[t].dropna()
        mom_3m[t] = w.iloc[-1] / w.iloc[-13] - 1 if len(w) >= 13 else 0.0
        mom_1m[t] = w.iloc[-1] / w.iloc[-4] - 1 if len(w) >= 4 else 0.0
        if t in daily.columns:
            d = daily[t].pct_change().dropna()
            vol_20d_vals[t] = d.iloc[-20:].std() * np.sqrt(252) if len(d) >= 20 else 0.15
        else:
            vol_20d_vals[t] = 0.15

    df["momentum"] = df.index.map(lambda t: 0.5 * mom_1m.get(t, 0) + 0.5 * mom_3m.get(t, 0))
    df["value"] = 0.0       # proxy unavailable in backtest
    df["quality"] = 0.0     # proxy unavailable in backtest
    df["low_vol"] = df.index.map(lambda t: -vol_20d_vals.get(t, 0.15))
    df["size"] = 0.0

    for col in ["momentum", "value", "quality", "low_vol", "size"]:
        s = df[col]
        mu, sigma = s.mean(), s.std()
        df[col + "_z"] = (s - mu) / sigma if sigma > 0 else 0.0

    z_cols = [c for c in df.columns if c.endswith("_z")]
    df["factor_z_avg"] = df[z_cols].mean(axis=1)
    df["top2_factor_z"] = df[z_cols].apply(lambda row: row.nlargest(2).mean(), axis=1)

    return df


def _compute_metrics(df: pd.DataFrame) -> Dict:
    """Compute standard portfolio metrics from monthly returns."""
    pr = df["port_return"]
    br = df["bm_return"]
    n_periods = len(pr)
    periods_per_year = 12  # monthly rebalance

    # CAGR
    total_ret_p = df["port_cumret"].iloc[-1]
    total_ret_b = df["bm_cumret"].iloc[-1]
    years = n_periods / periods_per_year
    cagr_p = total_ret_p ** (1 / years) - 1 if years > 0 else 0
    cagr_b = total_ret_b ** (1 / years) - 1 if years > 0 else 0

    # Sharpe
    excess = pr - br
    sharpe = excess.mean() / excess.std() * np.sqrt(periods_per_year) if excess.std() > 0 else 0

    # Max drawdown
    cum = df["port_cumret"]
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()

    # Calmar
    calmar = cagr_p / abs(max_dd) if max_dd != 0 else 0

    # Sortino
    downside = pr[pr < 0]
    down_std = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else 0.01
    sortino = (pr.mean() * periods_per_year) / down_std if down_std > 0 else 0

    # Win rate
    win_rate = (pr > 0).sum() / n_periods if n_periods > 0 else 0

    # Volatility
    ann_vol = pr.std() * np.sqrt(periods_per_year)

    metrics = {
        "CAGR (Strategy)":  f"{cagr_p:.2%}",
        "CAGR (SPY)":       f"{cagr_b:.2%}",
        "Sharpe Ratio":     f"{sharpe:.2f}",
        "Sortino Ratio":    f"{sortino:.2f}",
        "Max Drawdown":     f"{max_dd:.2%}",
        "Calmar Ratio":     f"{calmar:.2f}",
        "Ann. Volatility":  f"{ann_vol:.2%}",
        "Win Rate":         f"{win_rate:.1%}",
        "Total Return":     f"{(total_ret_p - 1):.2%}",
        "SPY Total Return": f"{(total_ret_b - 1):.2%}",
        "Periods":          n_periods,
    }
    return metrics


# ===================================================================
# 6. VISUALIZATIONS
# ===================================================================
def plot_rrg(
    rrg_df: pd.DataFrame,
    signals: pd.DataFrame,
    tail_length: int = 6,
    save_path: Optional[str] = "rrg_plot.png",
) -> None:
    """Plot Relative Rotation Graph with quadrants, bubbles, and tails."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Quadrant background colours
    ax.axhline(100, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(100, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)

    # Shade quadrants
    xlims = (96, 104)
    ylims = (96, 104)
    ax.fill_between([100, 110], 100, 110, alpha=0.06, color="green")   # Leading
    ax.fill_between([100, 110], 90, 100, alpha=0.06, color="orange")   # Weakening
    ax.fill_between([90, 100], 90, 100, alpha=0.06, color="red")       # Lagging
    ax.fill_between([90, 100], 100, 110, alpha=0.06, color="blue")     # Improving

    # Labels
    ax.text(0.95, 0.95, "LEADING", transform=ax.transAxes, fontsize=11, color="green",
            ha="right", va="top", alpha=0.5, weight="bold")
    ax.text(0.95, 0.05, "WEAKENING", transform=ax.transAxes, fontsize=11, color="orange",
            ha="right", va="bottom", alpha=0.5, weight="bold")
    ax.text(0.05, 0.05, "LAGGING", transform=ax.transAxes, fontsize=11, color="red",
            ha="left", va="bottom", alpha=0.5, weight="bold")
    ax.text(0.05, 0.95, "IMPROVING", transform=ax.transAxes, fontsize=11, color="blue",
            ha="left", va="top", alpha=0.5, weight="bold")

    # Get latest date range for tails
    dates = sorted(rrg_df["date"].unique())
    tail_dates = dates[-tail_length:] if len(dates) >= tail_length else dates
    latest_date = dates[-1]

    # Prepare composite for bubble sizing
    comp_map = {}
    if "composite_norm" in signals.columns:
        comp_map = dict(zip(signals["ticker"], signals["composite_norm"]))

    for ticker in rrg_df["ticker"].unique():
        color = COLORS.get(ticker, "#333333")
        name = SECTOR_NAMES.get(ticker, ticker)
        sub = rrg_df[(rrg_df["ticker"] == ticker) & (rrg_df["date"].isin(tail_dates))]
        sub = sub.sort_values("date")

        if sub.empty:
            continue

        xs = sub["rs_ratio"].values
        ys = sub["rs_momentum"].values

        # Tail line
        ax.plot(xs, ys, color=color, alpha=0.5, linewidth=1.5, zorder=2)

        # Bubble at latest point (size by composite)
        comp = comp_map.get(ticker, 0.5)
        size = 80 + comp * 300
        latest = sub[sub["date"] == sub["date"].max()]
        if not latest.empty:
            ax.scatter(latest["rs_ratio"].values[0], latest["rs_momentum"].values[0],
                       s=size, color=color, alpha=0.8, edgecolors="white", linewidths=0.8, zorder=3)
            ax.annotate(ticker, (latest["rs_ratio"].values[0], latest["rs_momentum"].values[0]),
                        textcoords="offset points", xytext=(8, 8), fontsize=8,
                        fontweight="bold", color=color, zorder=4)

        # Arrow showing direction on tail
        if len(xs) >= 2:
            ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=2)

    ax.set_xlabel("RS-Ratio (Relative Strength)", fontsize=12)
    ax.set_ylabel("RS-Momentum (Rate of Change)", fontsize=12)
    ax.set_title(f"Relative Rotation Graph — US Sector ETFs vs SPY\n"
                 f"as of {latest_date.strftime('%Y-%m-%d')}", fontsize=14, weight="bold")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] RRG plot saved to {save_path}")
    plt.close(fig)


def plot_equity_curve(
    bt: pd.DataFrame,
    metrics: Dict,
    save_path: Optional[str] = "equity_curve.png",
) -> None:
    """Plot backtest equity curve: strategy vs benchmark."""
    if bt.empty:
        print("[WARN] No backtest data to plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})

    # Equity curve
    ax1 = axes[0]
    ax1.plot(bt.index, bt["port_cumret"], label="Strategy", color="#1565C0", linewidth=2)
    ax1.plot(bt.index, bt["bm_cumret"], label="SPY (Benchmark)", color="#757575",
             linewidth=1.5, linestyle="--")
    ax1.fill_between(bt.index, bt["port_cumret"], bt["bm_cumret"],
                     where=bt["port_cumret"] >= bt["bm_cumret"],
                     alpha=0.1, color="green", interpolate=True)
    ax1.fill_between(bt.index, bt["port_cumret"], bt["bm_cumret"],
                     where=bt["port_cumret"] < bt["bm_cumret"],
                     alpha=0.1, color="red", interpolate=True)
    ax1.set_ylabel("Growth of $1", fontsize=12)
    ax1.set_title("Sector Rotation Strategy — Backtest Equity Curve", fontsize=14, weight="bold")
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(True, alpha=0.2)

    # Metrics annotation
    metrics_text = "\n".join(f"{k}: {v}" for k, v in metrics.items() if k != "Periods")
    ax1.text(0.98, 0.02, metrics_text, transform=ax1.transAxes, fontsize=8,
             verticalalignment="bottom", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
             family="monospace")

    # Drawdown
    ax2 = axes[1]
    cum = bt["port_cumret"]
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    ax2.fill_between(bt.index, drawdown, 0, color="#E53935", alpha=0.4)
    ax2.plot(bt.index, drawdown, color="#B71C1C", linewidth=1)
    ax2.set_ylabel("Drawdown", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.2)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Equity curve saved to {save_path}")
    plt.close(fig)


# ===================================================================
# 7. CONSOLE OUTPUT
# ===================================================================
def print_signal_table(signals: pd.DataFrame, portfolio: pd.DataFrame) -> None:
    """Print the latest tactical signals to console."""
    print("\n" + "=" * 90)
    print("  SECTOR ROTATION — TACTICAL ALLOCATION SIGNALS")
    print("=" * 90)

    header = f"{'Sector':<16} {'Ticker':<7} {'RS-Ratio':>9} {'RS-Mom':>9} " \
             f"{'Composite':>10} {'Quadrant':<12} {'Signal':<12}"
    print(header)
    print("-" * 90)

    for _, row in signals.sort_values("composite_norm", ascending=False).iterrows():
        ticker = row["ticker"]
        name = SECTOR_NAMES.get(ticker, ticker)[:14]
        rr = row.get("rs_ratio", 0)
        rm = row.get("rs_momentum", 0)
        comp = row.get("composite_norm", 0)
        quad = row.get("quadrant", "?")
        rec = row.get("recommendation", "?")
        marker = ">>>" if rec == "Overweight" else "   "
        print(f"{marker} {name:<13} {ticker:<7} {rr:>8.2f} {rm:>8.2f} "
              f"{comp:>9.2f} {quad:<12} {rec:<12}")

    print("-" * 90)
    print("\n  PORTFOLIO WEIGHTS:")
    print("-" * 40)
    for _, row in portfolio.iterrows():
        t = row["ticker"]
        w = row["weight"]
        name = SECTOR_NAMES.get(t, t)[:14]
        bar = "#" * int(w * 40)
        print(f"  {name:<14} ({t:<5}) {w:>6.1%}  {bar}")
    print("-" * 40)
    print()


def print_backtest_metrics(metrics: Dict) -> None:
    """Print backtest results table."""
    if not metrics:
        return
    print("\n" + "=" * 50)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<22} {str(v):>12}")
    print("=" * 50 + "\n")


# ===================================================================
# 8. MAIN
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sector Rotation TAA via Adjusted RRG Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for FMP/AlphaVantage (optional; falls back to yfinance)")
    parser.add_argument("--horizon_weeks", type=int, default=5,
                        help="RS-Ratio lookback in weeks (4-6)")
    parser.add_argument("--mom_weeks", type=int, default=2,
                        help="RS-Momentum lookback in weeks (1-3)")
    parser.add_argument("--sensitivity", type=int, default=6,
                        help="EMA smoothing span for RS-Momentum (5-7)")
    parser.add_argument("--top_n", type=int, default=3,
                        help="Number of sectors to overweight")
    parser.add_argument("--rrg_weight", type=float, default=0.70,
                        help="Weight on RRG component in composite (0-1)")
    parser.add_argument("--factor_weight", type=float, default=0.30,
                        help="Weight on factor component in composite (0-1)")
    parser.add_argument("--vol_target", type=float, default=0.12,
                        help="Annualised portfolio volatility target")
    parser.add_argument("--max_weight", type=float, default=0.25,
                        help="Maximum weight per sector")
    parser.add_argument("--tc_bps", type=float, default=10,
                        help="Transaction cost in basis points (one-way)")
    parser.add_argument("--slippage_bps", type=float, default=5,
                        help="Slippage in basis points (one-way)")
    parser.add_argument("--years", type=int, default=7,
                        help="Years of history to fetch")
    parser.add_argument("--no_plot", action="store_true",
                        help="Suppress plot generation")
    args = parser.parse_args()

    # --- Fetch Data ---
    weekly, daily = fetch_data(SECTOR_TICKERS, years=args.years)

    # --- RRG ---
    rrg_df = compute_rrg(weekly, SECTOR_TICKERS, BENCHMARK,
                         rs_lookback=args.horizon_weeks,
                         mom_lookback=args.mom_weeks,
                         sensitivity=args.sensitivity)

    # Latest snapshot
    latest_date = rrg_df["date"].max()
    rrg_latest = rrg_df[rrg_df["date"] == latest_date].copy()

    # --- Factor Scores ---
    print("[INFO] Computing factor scores (fetching fundamentals may be slow) ...")
    factors = factor_scores(weekly, daily, SECTOR_TICKERS, api_key=args.api_key)

    # --- Composite Signal ---
    signals = composite_signal(rrg_latest, factors, args.rrg_weight, args.factor_weight)

    # --- Portfolio Construction ---
    portfolio = construct_portfolio(signals, daily, max_sectors=args.top_n,
                                   max_weight=args.max_weight, vol_target=args.vol_target)

    # --- Console Output ---
    print_signal_table(signals, portfolio)

    # --- Backtest ---
    print("[INFO] Running backtest ...")
    bt, metrics = backtest(weekly, daily, SECTOR_TICKERS, BENCHMARK, CASH_PROXY,
                           rs_lookback=args.horizon_weeks,
                           mom_lookback=args.mom_weeks,
                           sensitivity=args.sensitivity,
                           top_n=args.top_n,
                           max_weight=args.max_weight,
                           tc_bps=args.tc_bps,
                           slippage_bps=args.slippage_bps,
                           vol_target=args.vol_target)
    print_backtest_metrics(metrics)

    # --- Plots ---
    if not args.no_plot:
        print("[INFO] Generating plots ...")
        plot_rrg(rrg_df, signals, tail_length=6, save_path="rrg_plot.png")
        plot_equity_curve(bt, metrics, save_path="equity_curve.png")

    print("[DONE] Sector Rotation RRG TAA complete.")


if __name__ == "__main__":
    main()
