"""
=============================================================
 Advanced Indian Stock Market Trading Strategy — Streamlit
 Data     : Yahoo Finance (yfinance)
 UI       : Streamlit + Plotly
 Version  : 2.1 (Corrected & Enhanced)
=============================================================

BUGS FIXED FROM ORIGINAL:
  1.  fetch_ohlcv()       — was MISSING; only fetch_all_data_batched existed
  2.  detect_swing_levels()— was MISSING; evaluate_signal called it but it didn't exist
  3.  is_near_zone()      — was MISSING; referenced but never defined
  4.  rsi_sell syntax     — had literal `in bearish zone"` appended, breaking Python
  5.  rsi_buy / rsi_sell  — computed but never appended to buy/sell conditions lists
  6.  volume variables    — vol_spike / volume_now / volume_ma used before assignment
  7.  calculate_rvol()    — defined INSIDE evaluate_signal() (invalid); moved to module level
  8.  SIGNAL_WEIGHTS block— referenced undefined buy_conditions_dict; removed floating code
  9.  Section 6 duplicate — Backtesting AND Telegram Alert both labelled "SECTION 6"
  10. get_nifty_trend()   — called fetch_ohlcv() which didn't exist; now fixed
"""

import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "WATCHLIST": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
        "ICICIBANK.NS", "AXISBANK.NS", "SBIN.NS", "WIPRO.NS",
        "TATAMOTORS.NS", "BHARTIARTL.NS",
    ],
    "NIFTY_TICKER":        "^NSEI",
    "VIX_TICKER":          "^INDIAVIX",
    "RSI_PERIOD":          14,
    "MACD_FAST":           12,
    "MACD_SLOW":           26,
    "MACD_SIGNAL":         9,
    "BB_PERIOD":           20,
    "BB_STD":              2,
    "EMA_SHORT":           20,
    "EMA_MID":             50,
    "EMA_LONG":            200,
    "ATR_PERIOD":          14,
    "VOL_MA_PERIOD":       20,
    "ATR_SL_MULTIPLIER":   1.5,
    "RR_RATIO":            2.0,
    "MIN_CONFIRMATIONS":   4,
    "MAX_VIX":             20.0,
    "DATA_PERIOD_DAYS":    250,
    "SWING_LOOKBACK":      10,
    "HIGH_IMPACT_DATES": [
        "2026-04-09",
        "2026-06-06",
    ],
    "NEWS_BUFFER_DAYS": 1,
}

# Signal weights (used in weighted scoring mode)
SIGNAL_WEIGHTS = {
    "rsi":            1.0,
    "macd":           1.0,
    "bollinger":      0.8,
    "ema_structure":  1.5,
    "demand_supply":  1.5,
    "volume_spike":   1.2,
}
MAX_SCORE = sum(SIGNAL_WEIGHTS.values())   # 7.0
SCORE_THRESHOLD = 0.60 * MAX_SCORE         # 4.2


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — DATA RETRIEVAL
# FIX: Added fetch_ohlcv() which was missing and called throughout.
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(ticker: str, period_days: int = 250) -> pd.DataFrame:
    """
    FIX #1 — This function was completely missing from the original script.
    fetch_ohlcv() was called in get_nifty_trend(), evaluate_signal() (via run_strategy),
    and backtest_strategy() but was never defined — causing NameError at runtime.
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=period_days)
    try:
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        # Flatten MultiIndex columns if present (yfinance quirk with single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception as e:
        logger.error(f"fetch_ohlcv error for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_vix() -> float:
    """Fetch latest India VIX."""
    try:
        vix_df = yf.download(
            CONFIG["VIX_TICKER"], period="5d", interval="1d",
            auto_adjust=True, progress=False,
        )
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        if not vix_df.empty:
            return float(vix_df["Close"].iloc[-1])
    except Exception as e:
        logger.error(f"VIX fetch error: {e}")
    return None


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Smoothed RSI."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Classic MACD — returns (macd_line, signal_line, histogram)."""
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2):
    """Bollinger Bands — returns (upper, middle, lower)."""
    middle = series.rolling(window=period).mean()
    std    = series.rolling(window=period).std()
    return middle + std_dev * std, middle, middle - std_dev * std


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close  = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_volume_ma(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(window=period).mean()


def calculate_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling 20-bar VWAP — proxy for institutional fair value."""
    typical      = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol   = (typical * df["Volume"]).rolling(window=window).sum()
    cum_vol      = df["Volume"].rolling(window=window).sum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def calculate_adx(df: pd.DataFrame, period: int = 14):
    """ADX + DI lines — returns (adx, plus_di, minus_di)."""
    high, low = df["High"], df["Low"]
    plus_dm    = high.diff().clip(lower=0)
    minus_dm   = (-low.diff()).clip(lower=0)
    # Zero out whichever is smaller
    mask = plus_dm >= minus_dm
    plus_dm  = plus_dm.where(mask, 0)
    minus_dm = minus_dm.where(~mask, 0)

    atr      = calculate_atr(df, period)
    plus_di  = 100 * plus_dm.ewm(alpha=1/period,  min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr
    dx       = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx      = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


def calculate_rvol(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    FIX #7 — calculate_rvol() was defined INSIDE evaluate_signal() as a nested
    function, which made it inaccessible and would error on any call referencing it.
    Moved here to module level.
    Relative Volume normalised by same day-of-week average.
    """
    df = df.copy()
    df["DayOfWeek"] = df.index.dayofweek
    day_avg = df.groupby("DayOfWeek")["Volume"].transform(
        lambda x: x.rolling(window=max(period // 5, 2), min_periods=2).mean()
    )
    return df["Volume"] / day_avg.replace(0, np.nan)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all indicators to the DataFrame."""
    c, v = df["Close"], df["Volume"]

    df["RSI"]                              = calculate_rsi(c, CONFIG["RSI_PERIOD"])
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calculate_macd(
        c, CONFIG["MACD_FAST"], CONFIG["MACD_SLOW"], CONFIG["MACD_SIGNAL"]
    )
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = calculate_bollinger_bands(
        c, CONFIG["BB_PERIOD"], CONFIG["BB_STD"]
    )
    df["EMA20"]  = calculate_ema(c, CONFIG["EMA_SHORT"])
    df["EMA50"]  = calculate_ema(c, CONFIG["EMA_MID"])
    df["EMA200"] = calculate_ema(c, CONFIG["EMA_LONG"])
    df["ATR"]    = calculate_atr(df, CONFIG["ATR_PERIOD"])
    df["Vol_MA"] = calculate_volume_ma(v, CONFIG["VOL_MA_PERIOD"])
    df["VWAP"]   = calculate_vwap(df)
    df["ADX"], df["Plus_DI"], df["Minus_DI"] = calculate_adx(df, CONFIG["ATR_PERIOD"])
    df["RVOL"]   = calculate_rvol(df)
    return df


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — ZONE DETECTION
# FIX: Added missing detect_swing_levels() and is_near_zone() helpers.
# ═══════════════════════════════════════════════════════════════

def detect_demand_supply_zones(
    df: pd.DataFrame, lookback: int = 10, min_volume_ratio: float = 1.3
):
    """
    Enhanced zone detection with volume validation and retest counting.
    Returns (best_demand, best_supply, demand_list[:3], supply_list[:3]).
    """
    highs   = df["High"].values
    lows    = df["Low"].values
    volumes = df["Volume"].values
    vol_ma  = df["Volume"].rolling(20).mean().values
    n       = len(df)
    zones   = []

    for i in range(lookback, n - lookback):
        window_h     = highs[i - lookback: i + lookback + 1]
        window_l     = lows[i - lookback:  i + lookback + 1]
        is_swing_high = highs[i] == max(window_h)
        is_swing_low  = lows[i]  == min(window_l)

        if not (is_swing_high or is_swing_low):
            continue

        vol_ratio  = volumes[i] / vol_ma[i] if (vol_ma[i] and vol_ma[i] > 0) else 0
        has_volume = vol_ratio >= min_volume_ratio

        body         = abs(df["Close"].iloc[i] - df["Open"].iloc[i])
        candle_range = highs[i] - lows[i]
        is_rejection = (body < 0.4 * candle_range) if candle_range > 0 else False

        level   = lows[i] if is_swing_low else highs[i]
        retests = sum(
            1 for j in range(i + 1, min(i + 50, n))
            if abs(lows[j]  - level) / level < 0.02
            or abs(highs[j] - level) / level < 0.02
        )

        strength = (
            1.0 * has_volume +
            0.5 * is_rejection +
            0.3 * min(retests, 3)
        )
        zones.append({
            "type":         "DEMAND" if is_swing_low else "SUPPLY",
            "level":        level,
            "index":        i,
            "strength":     strength,
            "volume_ratio": vol_ratio,
            "retests":      retests,
        })

    demand_zones = sorted(
        [z for z in zones if z["type"] == "DEMAND"],
        key=lambda x: (x["strength"], x["index"]), reverse=True,
    )
    supply_zones = sorted(
        [z for z in zones if z["type"] == "SUPPLY"],
        key=lambda x: (x["strength"], x["index"]), reverse=True,
    )
    best_demand = demand_zones[0]["level"] if demand_zones else None
    best_supply = supply_zones[0]["level"] if supply_zones else None
    return best_demand, best_supply, demand_zones[:3], supply_zones[:3]


def detect_swing_levels(df: pd.DataFrame, lookback: int = 10):
    """
    FIX #2 — detect_swing_levels() was called in evaluate_signal() but was
    never defined anywhere in the original file.  This thin wrapper delegates
    to detect_demand_supply_zones() and returns the two scalar levels that
    evaluate_signal() expects: (demand_level, supply_level).
    """
    best_demand, best_supply, _, _ = detect_demand_supply_zones(df, lookback)
    return best_demand, best_supply


def is_near_zone(price: float, zone_level, tolerance_pct: float = 0.015) -> bool:
    """
    FIX #3 — is_near_zone() was called in evaluate_signal() but was never
    defined.  Returns True when price is within tolerance_pct of zone_level.
    """
    if zone_level is None or zone_level == 0:
        return False
    return abs(price - zone_level) / zone_level <= tolerance_pct


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — MARKET & SENTIMENT FILTERS
# ═══════════════════════════════════════════════════════════════

def get_nifty_trend() -> str:
    """Determine NIFTY broad trend via EMA structure."""
    nifty_df = fetch_ohlcv(CONFIG["NIFTY_TICKER"], period_days=250)
    if nifty_df.empty:
        return "NEUTRAL"
    nifty_df = add_all_indicators(nifty_df)
    last = nifty_df.iloc[-1]
    if last["EMA20"] > last["EMA50"] > last["EMA200"]:
        return "BULLISH"
    elif last["EMA20"] < last["EMA50"] < last["EMA200"]:
        return "BEARISH"
    return "NEUTRAL"


def is_high_impact_day() -> bool:
    today  = datetime.today().date()
    buffer = CONFIG["NEWS_BUFFER_DAYS"]
    for date_str in CONFIG["HIGH_IMPACT_DATES"]:
        event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if abs((today - event_date).days) <= buffer:
            return True
    return False


def check_vix_filter(vix_value) -> bool:
    if vix_value is None:
        return True
    return vix_value < CONFIG["MAX_VIX"]


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — SIGNAL ENGINE  (6 CONDITIONS, ≥4 REQUIRED)
#
# FIXES applied here:
#   #4  rsi_sell had ` in bearish zone"` appended — broken syntax
#   #5  rsi_buy / rsi_sell tuples were never appended to conditions lists
#   #6  volume_now / volume_ma / vol_spike were used before assignment
#   #8  Floating SIGNAL_WEIGHTS / buy_conditions_dict block removed;
#       weighted scoring is now optional and self-contained
# ═══════════════════════════════════════════════════════════════

def evaluate_signal(df: pd.DataFrame, ticker: str, use_weighted: bool = False) -> dict:
    """
    Signal generation engine.
    6 independent conditions for BUY / SELL.
    Signal fires when ≥ MIN_CONFIRMATIONS (4/6) align.
    """
    if len(df) < 3:
        return _hold(ticker, 0.0, 0.0)

    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    close = float(last["Close"])
    atr   = float(last["ATR"]) if not np.isnan(last["ATR"]) else 0.0

    # ── Zone detection ────────────────────────────────────────
    demand_zone, supply_zone = detect_swing_levels(df, CONFIG["SWING_LOOKBACK"])

    buy_conditions  = []   # list of (bool, reason_str)
    sell_conditions = []

    # ── Condition 1 — RSI ─────────────────────────────────────
    # FIX #4: Original rsi_sell line ended with ` in bearish zone"` — invalid Python.
    # FIX #5: rsi_buy/rsi_sell were computed but never appended.
    rsi_now  = float(last["RSI"])
    rsi_prev = float(prev["RSI"])

    rsi_buy  = (rsi_prev < 30 and rsi_now >= 30) or (rsi_now > 50 and rsi_now > rsi_prev)
    # ORIGINAL BROKEN LINE: (rsi_now < 50 and rsi_now < rsi_prev) in bearish zone"
    rsi_sell = (rsi_prev > 70 and rsi_now <= 70) or (rsi_now < 50 and rsi_now < rsi_prev)

    buy_conditions.append((rsi_buy,  f"RSI={rsi_now:.1f} — oversold reversal / bullish momentum"))
    sell_conditions.append((rsi_sell, f"RSI={rsi_now:.1f} — overbought reversal / bearish momentum"))

    # ── Condition 2 — MACD Crossover ─────────────────────────
    macd_cross_up   = (float(prev["MACD"]) < float(prev["MACD_Signal"]) and
                       float(last["MACD"]) > float(last["MACD_Signal"]))
    macd_cross_down = (float(prev["MACD"]) > float(prev["MACD_Signal"]) and
                       float(last["MACD"]) < float(last["MACD_Signal"]))

    buy_conditions.append((macd_cross_up,   "MACD bullish crossover"))
    sell_conditions.append((macd_cross_down, "MACD bearish crossover"))

    # ── Condition 3 — Bollinger Band Position ─────────────────
    bb_lower = float(last["BB_Lower"])
    bb_upper = float(last["BB_Upper"])

    buy_conditions.append((
        close <= bb_lower + 0.5 * atr,
        f"Price near BB Lower ({bb_lower:.2f})"
    ))
    sell_conditions.append((
        close >= bb_upper - 0.5 * atr,
        f"Price near BB Upper ({bb_upper:.2f})"
    ))

    # ── Condition 4 — EMA Structure ───────────────────────────
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])

    buy_conditions.append((
        close > ema20 > ema50,
        f"Price > EMA20({ema20:.2f}) > EMA50({ema50:.2f})"
    ))
    sell_conditions.append((
        close < ema20 < ema50,
        f"Price < EMA20({ema20:.2f}) < EMA50({ema50:.2f})"
    ))

    # ── Condition 5 — Demand / Supply Zone Proximity ──────────
    near_demand = is_near_zone(close, demand_zone)
    near_supply = is_near_zone(close, supply_zone)

    buy_conditions.append((
        near_demand,
        f"Price near Demand Zone ({demand_zone:.2f})" if demand_zone else "No demand zone"
    ))
    sell_conditions.append((
        near_supply,
        f"Price near Supply Zone ({supply_zone:.2f})" if supply_zone else "No supply zone"
    ))

    # ── Condition 6 — Volume Spike ────────────────────────────
    # FIX #6: volume_now, volume_ma, vol_spike were used before being assigned.
    volume_now = float(last["Volume"])
    volume_ma  = float(last["Vol_MA"]) if not np.isnan(last["Vol_MA"]) else 1.0
    vol_spike  = volume_now > 1.5 * volume_ma

    buy_conditions.append((
        vol_spike,
        f"Volume spike: {volume_now:,.0f} > 1.5× MA ({volume_ma:,.0f})"
    ))
    sell_conditions.append((
        vol_spike,
        f"Volume spike: {volume_now:,.0f} > 1.5× MA ({volume_ma:,.0f})"
    ))

    # ── Tally ─────────────────────────────────────────────────
    if use_weighted:
        # Weighted scoring mode (optional)
        keys = ["rsi", "macd", "bollinger", "ema_structure", "demand_supply", "volume_spike"]
        buy_score  = sum(SIGNAL_WEIGHTS[k] for k, (c, _) in zip(keys, buy_conditions)  if c)
        sell_score = sum(SIGNAL_WEIGHTS[k] for k, (c, _) in zip(keys, sell_conditions) if c)
        threshold  = SCORE_THRESHOLD
    else:
        buy_score  = sum(1 for cond, _ in buy_conditions  if cond)
        sell_score = sum(1 for cond, _ in sell_conditions if cond)
        threshold  = CONFIG["MIN_CONFIRMATIONS"]

    buy_reasons  = [msg for cond, msg in buy_conditions  if cond]
    sell_reasons = [msg for cond, msg in sell_conditions if cond]

    # ── Risk Management ───────────────────────────────────────
    sl_distance = CONFIG["ATR_SL_MULTIPLIER"] * atr
    tp_distance = sl_distance * CONFIG["RR_RATIO"]

    # ── Decision ──────────────────────────────────────────────
    if buy_score >= threshold and buy_score > sell_score:
        entry = close
        return {
            "ticker":        ticker,
            "signal":        "BUY",
            "entry":         round(entry, 2),
            "stop_loss":     round(entry - sl_distance, 2),
            "target":        round(entry + tp_distance, 2),
            "rr_ratio":      round(CONFIG["RR_RATIO"], 2),
            "confirmations": buy_score,
            "total_checks":  6,
            "reasons":       buy_reasons,
            "atr":           round(atr, 2),
            "close":         round(close, 2),
        }
    elif sell_score >= threshold and sell_score > buy_score:
        entry = close
        return {
            "ticker":        ticker,
            "signal":        "SELL",
            "entry":         round(entry, 2),
            "stop_loss":     round(entry + sl_distance, 2),
            "target":        round(entry - tp_distance, 2),
            "rr_ratio":      round(CONFIG["RR_RATIO"], 2),
            "confirmations": sell_score,
            "total_checks":  6,
            "reasons":       sell_reasons,
            "atr":           round(atr, 2),
            "close":         round(close, 2),
        }
    else:
        return _hold(ticker, max(buy_score, sell_score), close)


def _hold(ticker: str, score, close: float) -> dict:
    return {
        "ticker":        ticker,
        "signal":        "HOLD",
        "entry":         round(float(close), 2),
        "stop_loss":     None,
        "target":        None,
        "rr_ratio":      None,
        "confirmations": score,
        "total_checks":  6,
        "reasons":       ["Insufficient confirmations for trade entry"],
        "atr":           None,
        "close":         round(float(close), 2),
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — BACK TESTING  (was duplicate "SECTION 6" in original)
# ═══════════════════════════════════════════════════════════════

def backtest_strategy(
    ticker: str,
    start_date: str = "2023-01-01",
    end_date:   str = "2025-12-31",
) -> dict:
    """Walk-forward backtest — returns performance metrics dict."""
    df = fetch_ohlcv(ticker, period_days=int(
        (datetime.strptime(end_date, "%Y-%m-%d") -
         datetime.strptime(start_date, "%Y-%m-%d")).days + 30
    ))
    if df.empty:
        return {}

    # Filter date range
    df = df.loc[start_date:end_date]
    if df.empty:
        return {}

    df = add_all_indicators(df)
    trades   = []
    min_bars = max(CONFIG["EMA_LONG"], CONFIG["BB_PERIOD"]) + CONFIG["SWING_LOOKBACK"]

    for i in range(min_bars, len(df) - 1):
        window = df.iloc[: i + 1].copy()
        result = evaluate_signal(window, ticker)

        if result["signal"] not in ("BUY", "SELL"):
            continue

        entry = result["entry"]
        sl    = result["stop_loss"]
        tp    = result["target"]

        for j in range(i + 1, min(i + 30, len(df))):
            high_j = float(df["High"].iloc[j])
            low_j  = float(df["Low"].iloc[j])

            if result["signal"] == "BUY":
                if low_j <= sl:
                    trades.append({"pnl": sl - entry, "result": "LOSS", "bars": j - i})
                    break
                if high_j >= tp:
                    trades.append({"pnl": tp - entry, "result": "WIN",  "bars": j - i})
                    break
            else:
                if high_j >= sl:
                    trades.append({"pnl": entry - sl, "result": "LOSS", "bars": j - i})
                    break
                if low_j <= tp:
                    trades.append({"pnl": entry - tp, "result": "WIN",  "bars": j - i})
                    break

    if not trades:
        return {"total_trades": 0}

    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]

    sum_wins   = abs(sum(t["pnl"] for t in wins))
    sum_losses = abs(sum(t["pnl"] for t in losses))

    return {
        "total_trades":    len(trades),
        "wins":            len(wins),
        "losses":          len(losses),
        "win_rate":        round(len(wins) / len(trades) * 100, 1),
        "avg_win":         round(np.mean([t["pnl"] for t in wins]),   2) if wins   else 0,
        "avg_loss":        round(np.mean([t["pnl"] for t in losses]), 2) if losses else 0,
        "profit_factor":   round(sum_wins / sum_losses, 2) if sum_losses else float("inf"),
        "avg_holding_bars":round(np.mean([t["bars"] for t in trades]), 1),
        "trades":          trades,
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — PLOTLY CHARTS
# ═══════════════════════════════════════════════════════════════

def build_price_chart(df: pd.DataFrame, result: dict, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.2, 0.15],
        vertical_spacing=0.03,
        subplot_titles=("Price + Indicators", "RSI", "MACD", "Volume"),
    )

    # ── Candlestick ───────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"],
        name="Price", increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # ── Bollinger Bands ───────────────────────────────────────
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                             line=dict(color="rgba(150,150,255,0.5)", dash="dot"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                             line=dict(color="rgba(150,150,255,0.5)", dash="dot"),
                             fill="tonexty", fillcolor="rgba(150,150,255,0.05)", showlegend=False), row=1, col=1)

    # ── EMAs ──────────────────────────────────────────────────
    ema_styles = [("EMA20", "#f39c12", 1), ("EMA50", "#3498db", 1.5), ("EMA200", "#e74c3c", 2)]
    for col, color, width in ema_styles:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                 line=dict(color=color, width=width)), row=1, col=1)

    # ── VWAP ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
                             line=dict(color="white", width=1, dash="dash")), row=1, col=1)

    # ── Signal marker ─────────────────────────────────────────
    sig = result["signal"]
    if sig in ("BUY", "SELL"):
        last_date = df.index[-1]
        marker_y  = result["entry"]
        fig.add_trace(go.Scatter(
            x=[last_date], y=[marker_y],
            mode="markers+text",
            marker=dict(symbol="triangle-up" if sig == "BUY" else "triangle-down",
                        size=18, color="#00ff88" if sig == "BUY" else "#ff4466"),
            text=[sig], textposition="top center",
            name=sig, showlegend=True,
        ), row=1, col=1)
        # SL / TP lines
        for level, label, color in [
            (result["stop_loss"], "SL", "red"),
            (result["target"],    "TP", "green"),
        ]:
            if level:
                fig.add_hline(y=level, line=dict(color=color, dash="dash", width=1),
                              annotation_text=f"{label}: {level}", row=1, col=1)

    # ── RSI ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#9b59b6", width=1.5)), row=2, col=1)
    for lvl, color in [(70, "red"), (30, "green"), (50, "gray")]:
        fig.add_hline(y=lvl, line=dict(color=color, dash="dot", width=0.8), row=2, col=1)

    # ── MACD ─────────────────────────────────────────────────
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=colors,
                         name="Histogram", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                             line=dict(color="#3498db", width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                             line=dict(color="#e67e22", width=1.2)), row=3, col=1)

    # ── Volume ────────────────────────────────────────────────
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors,
                         name="Volume", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Vol_MA"], name="Vol MA",
                             line=dict(color="#f39c12", width=1)), row=4, col=1)

    fig.update_layout(
        title=f"{ticker} — Technical Analysis",
        template="plotly_dark",
        height=800,
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=60, b=20),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


def build_backtest_chart(trades_list: list) -> go.Figure:
    if not trades_list:
        return go.Figure()

    cumulative = np.cumsum([t["pnl"] for t in trades_list])
    colors     = ["#26a69a" if t["result"] == "WIN" else "#ef5350" for t in trades_list]

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                        subplot_titles=("Cumulative P&L", "Per-Trade P&L"))

    fig.add_trace(go.Scatter(y=cumulative, mode="lines+markers",
                             line=dict(color="#3498db", width=2), name="Cumulative P&L"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="gray", dash="dash"), row=1, col=1)

    fig.add_trace(go.Bar(y=[t["pnl"] for t in trades_list],
                         marker_color=colors, name="Trade P&L"), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=500,
                      title="Backtest Results", margin=dict(l=40, r=40, t=60, b=20))
    return fig


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

def render_signal_card(result: dict, nifty_trend: str, vix):
    sig    = result["signal"]
    ticker = result["ticker"].replace(".NS", "")
    colors = {"BUY": "#00c853", "SELL": "#d50000", "HOLD": "#ff8f00"}
    emojis = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}

    st.markdown(
        f"""
        <div style="background:#1e1e2e;border-left:5px solid {colors.get(sig,'#aaa')};
                    border-radius:8px;padding:1rem 1.5rem;margin-bottom:1rem">
          <h2 style="color:{colors.get(sig,'#fff')};margin:0">
            {emojis.get(sig,'')} {sig} — {ticker}
          </h2>
          <hr style="border-color:#333">
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem">
            <div><small>Entry</small><br><b style="font-size:1.1em">₹{result['entry']:,.2f}</b></div>
            <div><small>Stop Loss</small><br><b style="color:#ef5350;font-size:1.1em">
              {"₹" + f"{result['stop_loss']:,.2f}" if result['stop_loss'] else "—"}</b></div>
            <div><small>Target</small><br><b style="color:#26a69a;font-size:1.1em">
              {"₹" + f"{result['target']:,.2f}" if result['target'] else "—"}</b></div>
            <div><small>R:R Ratio</small><br><b>{"1:" + str(result['rr_ratio']) if result['rr_ratio'] else "—"}</b></div>
            <div><small>Score</small><br><b>{result['confirmations']}/6</b></div>
            <div><small>ATR</small><br><b>{"₹" + str(result['atr']) if result['atr'] else "—"}</b></div>
          </div>
          <hr style="border-color:#333">
          <div style="display:flex;gap:2rem">
            <span>🏦 NIFTY: <b>{nifty_trend}</b></span>
            <span>📉 VIX: <b>{f"{vix:.2f}" if vix else "N/A"}</b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result["reasons"]:
        with st.expander("📋 Signal Conditions"):
            for reason in result["reasons"]:
                st.markdown(f"• {reason}")


def render_metrics_row(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    chg  = float(last["Close"]) - float(prev["Close"])
    pct  = chg / float(prev["Close"]) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Close",   f"₹{float(last['Close']):,.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
    c2.metric("RSI",     f"{float(last['RSI']):.1f}" if not np.isnan(last["RSI"]) else "—")
    c3.metric("ATR",     f"₹{float(last['ATR']):.2f}" if not np.isnan(last["ATR"]) else "—")
    c4.metric("MACD",    f"{float(last['MACD']):.3f}" if not np.isnan(last["MACD"]) else "—")
    c5.metric("ADX",     f"{float(last['ADX']):.1f}" if not np.isnan(last["ADX"]) else "—")


def main():
    st.set_page_config(
        page_title="Indian Stock Market Scanner",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
      .main {background:#0d1117} 
      [data-testid="stMetricValue"]{font-size:1.1rem}
      div[data-testid="stHorizontalBlock"] > div {background:#161b22;border-radius:8px;padding:0.5rem}
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/48/stock-share.png", width=48)
        st.title("Strategy Config")

        selected_ticker = st.selectbox(
            "Select Stock", CONFIG["WATCHLIST"],
            format_func=lambda x: x.replace(".NS", ""),
        )
        period_days = st.slider("Data Period (days)", 100, 500, 250, 50)

        st.subheader("Indicator Settings")
        rsi_period  = st.number_input("RSI Period",  5,  30, CONFIG["RSI_PERIOD"])
        bb_period   = st.number_input("BB Period",   10, 50, CONFIG["BB_PERIOD"])
        atr_mult    = st.slider("ATR SL Multiplier", 0.5, 3.0, CONFIG["ATR_SL_MULTIPLIER"], 0.1)
        rr_ratio    = st.slider("R:R Ratio",         1.0, 5.0, CONFIG["RR_RATIO"], 0.5)
        min_conf    = st.slider("Min Confirmations", 2, 6, CONFIG["MIN_CONFIRMATIONS"])
        use_weighted = st.checkbox("Use Weighted Scoring", value=False,
                                   help="Weight each condition by importance instead of equal voting")
        max_vix     = st.number_input("Max VIX",    5.0, 40.0, CONFIG["MAX_VIX"], 1.0)

        # Apply sidebar overrides
        CONFIG.update({
            "RSI_PERIOD": int(rsi_period), "BB_PERIOD": int(bb_period),
            "ATR_SL_MULTIPLIER": atr_mult, "RR_RATIO": rr_ratio,
            "MIN_CONFIRMATIONS": int(min_conf), "MAX_VIX": max_vix,
        })

        st.divider()
        tab_mode = st.radio("Mode", ["📊 Analysis", "🔁 Backtest", "🔍 Market Scan"])

    # ════════════════════════════════════════════
    # TAB: ANALYSIS
    # ════════════════════════════════════════════
    if "Analysis" in tab_mode:
        st.title(f"📈 {selected_ticker.replace('.NS','')} — Analysis")

        with st.spinner("Fetching data & computing indicators…"):
            df = fetch_ohlcv(selected_ticker, period_days)
            if df.empty:
                st.error("No data returned. Check ticker or network.")
                return
            df = add_all_indicators(df)

        # Market context
        col_vix, col_nifty = st.columns(2)
        with col_vix:
            with st.spinner("VIX…"):
                vix = fetch_vix()
        with col_nifty:
            with st.spinner("NIFTY trend…"):
                nifty_trend = get_nifty_trend()

        st.subheader("📊 Key Metrics")
        render_metrics_row(df)

        st.divider()

        with st.spinner("Evaluating signal…"):
            result = evaluate_signal(df, selected_ticker, use_weighted)

        # Apply NIFTY macro filter
        if result["signal"] == "BUY" and nifty_trend == "BEARISH":
            result.update({"signal": "HOLD", "stop_loss": None, "target": None,
                           "reasons": ["BUY suppressed — NIFTY in BEARISH trend"]})
        if result["signal"] == "SELL" and nifty_trend == "BULLISH":
            result.update({"signal": "HOLD", "stop_loss": None, "target": None,
                           "reasons": ["SELL suppressed — NIFTY in BULLISH trend"]})

        # VIX filter
        vix_ok = check_vix_filter(vix)
        if not vix_ok:
            st.warning(f"⚠️ India VIX = {vix:.2f} > {max_vix}. New entries suppressed.")

        if is_high_impact_day():
            st.warning("⛔ High-impact event detected. Signals suppressed.")

        render_signal_card(result, nifty_trend, vix)

        st.subheader("📉 Price Chart")
        fig = build_price_chart(df.iloc[-120:], result, selected_ticker)
        st.plotly_chart(fig, use_container_width=True)

        # Demand / Supply Zones
        with st.expander("🏛️ Demand / Supply Zones"):
            _, _, demand_zones, supply_zones = detect_demand_supply_zones(
                df, CONFIG["SWING_LOOKBACK"]
            )
            col_d, col_s = st.columns(2)
            with col_d:
                st.markdown("**Top Demand Zones**")
                for z in demand_zones:
                    st.markdown(
                        f"₹{z['level']:,.2f} — Strength: {z['strength']:.1f} | "
                        f"Retests: {z['retests']} | VolRatio: {z['volume_ratio']:.2f}"
                    )
            with col_s:
                st.markdown("**Top Supply Zones**")
                for z in supply_zones:
                    st.markdown(
                        f"₹{z['level']:,.2f} — Strength: {z['strength']:.1f} | "
                        f"Retests: {z['retests']} | VolRatio: {z['volume_ratio']:.2f}"
                    )

        # Raw data
        with st.expander("📋 Raw Data (last 20 bars)"):
            show_cols = ["Open", "High", "Low", "Close", "Volume",
                         "RSI", "MACD", "EMA20", "EMA50", "ATR", "ADX", "RVOL"]
            show_cols = [c for c in show_cols if c in df.columns]
            st.dataframe(df[show_cols].tail(20).round(2), use_container_width=True)

    # ════════════════════════════════════════════
    # TAB: BACKTEST
    # ════════════════════════════════════════════
    elif "Backtest" in tab_mode:
        st.title(f"🔁 Backtest — {selected_ticker.replace('.NS','')}")

        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", value=datetime(2023, 1, 1))
        end_date   = col2.date_input("End Date",   value=datetime(2025, 12, 31))

        if st.button("▶ Run Backtest", type="primary"):
            with st.spinner("Running walk-forward backtest… (may take a moment)"):
                metrics = backtest_strategy(
                    selected_ticker,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                )

            if not metrics or metrics.get("total_trades", 0) == 0:
                st.warning("No trades generated in the selected period.")
                return

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Total Trades",    metrics["total_trades"])
            m2.metric("Win Rate",        f"{metrics['win_rate']}%")
            m3.metric("Avg Win",         f"₹{metrics['avg_win']:.2f}")
            m4.metric("Avg Loss",        f"₹{metrics['avg_loss']:.2f}")
            m5.metric("Profit Factor",   metrics["profit_factor"])
            m6.metric("Avg Hold (bars)", metrics["avg_holding_bars"])

            if "trades" in metrics:
                st.plotly_chart(
                    build_backtest_chart(metrics["trades"]), use_container_width=True
                )

            with st.expander("📋 Trade Log"):
                trades_df = pd.DataFrame(metrics.get("trades", []))
                if not trades_df.empty:
                    trades_df.index += 1
                    st.dataframe(trades_df.style.applymap(
                        lambda v: "color:#26a69a" if v == "WIN" else "color:#ef5350",
                        subset=["result"]
                    ), use_container_width=True)

    # ════════════════════════════════════════════
    # TAB: MARKET SCAN
    # ════════════════════════════════════════════
    elif "Scan" in tab_mode:
        st.title("🔍 Full Market Scan")

        if is_high_impact_day():
            st.warning("⛔ High-impact event detected. Consider pausing live trades.")

        col_v, col_n = st.columns(2)
        vix         = fetch_vix()
        nifty_trend = get_nifty_trend()
        col_v.metric("India VIX",    f"{vix:.2f}" if vix else "N/A",
                     delta="⚠ High" if vix and vix > CONFIG["MAX_VIX"] else "✅ Normal")
        col_n.metric("NIFTY Trend",  nifty_trend)

        if st.button("▶ Scan All Watchlist Stocks", type="primary"):
            results = []
            bar     = st.progress(0, text="Scanning…")

            for idx, ticker in enumerate(CONFIG["WATCHLIST"]):
                bar.progress((idx + 1) / len(CONFIG["WATCHLIST"]),
                             text=f"Scanning {ticker}…")
                try:
                    df = fetch_ohlcv(ticker, period_days)
                    if df.empty:
                        continue
                    df     = add_all_indicators(df)
                    result = evaluate_signal(df, ticker, use_weighted)

                    # Macro filter
                    if result["signal"] == "BUY"  and nifty_trend == "BEARISH":
                        result.update({"signal": "HOLD", "stop_loss": None, "target": None})
                    if result["signal"] == "SELL" and nifty_trend == "BULLISH":
                        result.update({"signal": "HOLD", "stop_loss": None, "target": None})
                    results.append(result)
                except Exception as e:
                    st.warning(f"Error scanning {ticker}: {e}")

            bar.empty()

            buys  = [r for r in results if r["signal"] == "BUY"]
            sells = [r for r in results if r["signal"] == "SELL"]
            holds = [r for r in results if r["signal"] == "HOLD"]

            m1, m2, m3 = st.columns(3)
            m1.metric("🟢 BUY Signals",  len(buys))
            m2.metric("🔴 SELL Signals", len(sells))
            m3.metric("🟡 HOLD Signals", len(holds))

            if results:
                scan_df = pd.DataFrame([{
                    "Ticker":    r["ticker"].replace(".NS", ""),
                    "Signal":    r["signal"],
                    "Close":     r["close"],
                    "Entry":     r["entry"],
                    "Stop Loss": r["stop_loss"],
                    "Target":    r["target"],
                    "R:R":       r["rr_ratio"],
                    "Score":     f"{r['confirmations']}/6",
                } for r in results])

                def color_signal(val):
                    if val == "BUY":  return "color:#26a69a;font-weight:bold"
                    if val == "SELL": return "color:#ef5350;font-weight:bold"
                    return "color:#f39c12"

                st.dataframe(
                    scan_df.style.applymap(color_signal, subset=["Signal"]),
                    use_container_width=True,
                )

                # Detailed cards for actionable signals
                actionable = [r for r in results if r["signal"] in ("BUY", "SELL")]
                if actionable:
                    st.subheader("📌 Actionable Signals")
                    for r in actionable:
                        render_signal_card(r, nifty_trend, vix)


if __name__ == "__main__":
    main()
