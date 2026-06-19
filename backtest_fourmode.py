"""
Vier-modus Gecombineerde Strategie Backtest
============================================
Voegt Buy & Hold toe als vierde modus aan het hybride V4+MR systeem.
Regime sterkte (vier niveaus via p25/p50/p75) bepaalt welke strategie actief is.

  sterke_bull  (>p75) : Buy & Hold — 80% van kapitaal, houd vast
  lichte_bull (p50-p75): V4 momentum — 25/40/60%, 3×ATR trailing stop
  neutraal    (p25-p50): V4 voorzichtig — 15/20/30%, 2×ATR trailing stop
  bear         (<p25)  : Mean reversion — 10/20%, 2×ATR stop

Portfolio: DOGE, SOL, ETH, AVAX, ADA — max 5 posities, één per crypto.
Exposure caps: 80% sterke_bull / 60% lichte_bull / 40% neutraal / 30% bear.

Stack: Python 3.11, yfinance, ta, pandas. Geen matplotlib, geen scikit-learn.
"""

import json, warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ta.momentum   import RSIIndicator
from ta.trend      import MACD as TaMACD
from ta.volatility import AverageTrueRange, BollingerBands

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS   = ["DOGE-USD", "SOL-USD", "ETH-USD", "AVAX-USD", "ADA-USD"]
START_CAP = 500.0
FEE_RT    = 0.004
MIN_POS   = 5.0

# Regimes
STERKE_BULL = "sterke_bull"
LICHTE_BULL = "lichte_bull"
NEUTRAAL    = "neutraal"
BEAR        = "bear"

REGIME_ORDER  = [BEAR, NEUTRAAL, LICHTE_BULL, STERKE_BULL]
REGIME_NUM    = {BEAR: 0, NEUTRAAL: 1, LICHTE_BULL: 2, STERKE_BULL: 3}

# Exposure caps per portfolio-regime
EXPO_CAP = {STERKE_BULL: 0.80, LICHTE_BULL: 0.60, NEUTRAAL: 0.40, BEAR: 0.30}

# B&H
BH_FRAC = 0.80   # 80% van beschikbaar kapitaal

# V4 lichte_bull
V4_SIZES      = {0: 0.25, 1: 0.25, 2: 0.40, 3: 0.60, 4: 0.60}
V4_ATR_ENTRY  = 4.0
V4_ATR_TRAIL  = 3.0
V4_RSI_LO     = 45; V4_RSI_HI = 75
V4_MA200_BELOW = 0.15

# V4 neutraal (voorzichtig)
V4C_SIZES     = {0: 0.15, 1: 0.15, 2: 0.20, 3: 0.30, 4: 0.30}
V4C_ATR_TRAIL = 2.0

# MR bear
MR_RSI_THR = 35
MR_BB_WIN  = 30; MR_BB_SIG = 2.5
MR_ATR     = 2.0
MR_FRAC    = {True: 0.20, False: 0.10}   # True = RSI<20 (diep oversold)

ATR_WIN = 14; MA50_WIN = 50; MA200_WIN = 200
MA200_LAG = 10; VOL_WIN = 20
MACD_F, MACD_S, MACD_SIG = 12, 26, 9

TRAIN_START = "2024-01-01"; TRAIN_END  = "2024-12-31"
VAL_START   = "2025-01-01"; VAL_END    = "2025-09-30"
FINAL_START = "2025-10-01"; FINAL_END  = "2026-06-17"
PERIODS = [
    ("2024",     TRAIN_START, TRAIN_END),
    ("Val 2025", VAL_START,   VAL_END),
    ("Finale",   FINAL_START, FINAL_END),
]

OUTPUT_FILE = Path(__file__).parent / "results_fourmode.json"


# ── Data ──────────────────────────────────────────────────────────────────────
def fetch_4h(ticker: str) -> pd.DataFrame:
    raw_1h = yf.download(ticker, start="2024-06-20", end=FINAL_END,
                         interval="1h", auto_adjust=True, progress=False)
    if isinstance(raw_1h.columns, pd.MultiIndex):
        raw_1h.columns = raw_1h.columns.get_level_values(0)
    raw_1h.index = pd.to_datetime(raw_1h.index).tz_localize(None)
    df_1h = raw_1h.resample("4h").agg(
        Open=("Open","first"), High=("High","max"),
        Low=("Low","min"),    Close=("Close","last"),
        Volume=("Volume","sum")
    ).dropna()

    raw_d = yf.download(ticker, start="2023-06-01", end="2024-06-25",
                        interval="1d", auto_adjust=True, progress=False)
    if isinstance(raw_d.columns, pd.MultiIndex):
        raw_d.columns = raw_d.columns.get_level_values(0)
    raw_d.index = pd.to_datetime(raw_d.index).tz_localize(None)
    raw_d = raw_d[["Open","High","Low","Close","Volume"]].dropna()

    rows = []
    for ts, row in raw_d.iterrows():
        for q in range(6):
            rows.append({"_ts": ts + pd.Timedelta(hours=q*4),
                         "Open":   float(row["Open"]),
                         "High":   float(row["High"]),
                         "Low":    float(row["Low"]),
                         "Close":  float(row["Close"]),
                         "Volume": float(row["Volume"]) / 6})
    warmup = pd.DataFrame(rows).set_index("_ts")
    warmup.index = pd.DatetimeIndex(warmup.index)
    warmup = warmup[warmup.index < df_1h.index.min()]
    df = pd.concat([warmup, df_1h]).sort_index()
    return df[~df.index.duplicated(keep="last")]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma50"]      = df["Close"].rolling(MA50_WIN).mean()
    df["ma200"]     = df["Close"].rolling(MA200_WIN).mean()
    df["ma200_lag"] = df["ma200"].shift(MA200_LAG)
    df["rsi"]       = RSIIndicator(df["Close"], window=14).rsi()
    _m = TaMACD(df["Close"], window_fast=MACD_F, window_slow=MACD_S, window_sign=MACD_SIG)
    df["macd"]      = _m.macd()
    df["macd_sig"]  = _m.macd_signal()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    df["prev_hist"] = df["macd_hist"].shift(1)
    df["vol_ma"]    = df["Volume"].rolling(VOL_WIN).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma"]
    df["atr"]       = AverageTrueRange(df["High"], df["Low"], df["Close"],
                                       window=ATR_WIN).average_true_range()
    bb = BollingerBands(df["Close"], window=MR_BB_WIN, window_dev=MR_BB_SIG)
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_mid"]    = bb.bollinger_mavg()
    df["bb_std"]    = df["Close"].rolling(MR_BB_WIN).std()
    return df.dropna()


def slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


# ── Regime (vier niveaus) ─────────────────────────────────────────────────────
def regime_score_row(row) -> float:
    ma200     = float(row["ma200"]); ma200_lag = float(row["ma200_lag"])
    close     = float(row["Close"]); rsi       = float(row["rsi"])
    ma200_s = (ma200 - ma200_lag) / ma200_lag if ma200_lag else 0.0
    prijs_s = (close - ma200)     / ma200     if ma200     else 0.0
    rsi_s   = (rsi - 50)          / 50
    return float(np.clip((ma200_s + prijs_s + rsi_s) / 3, -1.0, 1.0))


def calibrate_4level(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Bereken p25/p50/p75 van regime scores over 2024 trainingsdata."""
    sl = slice_window(df, TRAIN_START, TRAIN_END)
    scores = sl.apply(regime_score_row, axis=1).dropna().values
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    return (float(np.percentile(scores, 25)),
            float(np.percentile(scores, 50)),
            float(np.percentile(scores, 75)))


def get_regime_4(score: float, p25: float, p50: float, p75: float) -> str:
    if score > p75:  return STERKE_BULL
    if score > p50:  return LICHTE_BULL
    if score > p25:  return NEUTRAAL
    return BEAR


def portfolio_regime_4(regime_map: dict) -> str:
    """Gewogen meerderheidsstem; geeft meest voorkomend regime terug."""
    if not regime_map:
        return NEUTRAAL
    counts = {r: 0 for r in REGIME_ORDER}
    for r in regime_map.values():
        counts[r] += 1
    # Conservatief bij gelijkstand: kies lager regime
    best_n = max(counts.values())
    for r in REGIME_ORDER:   # laag → hoog, eerste met max count = conservatiefste
        if counts[r] == best_n:
            return r
    return NEUTRAAL


# ── Signalen ──────────────────────────────────────────────────────────────────
def v4_signal(row) -> bool:
    rsi   = float(row["rsi"]); close = float(row["Close"]); ma200 = float(row["ma200"])
    cross = float(row["macd"]) > float(row["macd_sig"]) and float(row["prev_hist"]) <= 0
    return (V4_RSI_LO <= rsi <= V4_RSI_HI and cross
            and float(row["Volume"]) > float(row["vol_ma"])
            and close >= ma200 * (1 - V4_MA200_BELOW))


def v4_strength(row) -> int:
    s = 0
    if 55 <= float(row["rsi"]) <= 70:                                           s += 1
    if float(row["macd_hist"]) > float(row["prev_hist"]) > 0:                   s += 1
    if float(row["vol_ratio"]) > 1.5:                                           s += 1
    if float(row["Close"]) > float(row["ma200"]) * 1.02:                        s += 1
    return s


def mr_signal(row) -> bool:
    close = float(row["Close"])
    return (float(row["rsi"]) < MR_RSI_THR
            and close < float(row["bb_lower"]) - float(row["bb_std"])
            and float(row["Volume"]) > float(row["vol_ma"]))


def mr_exit(row, stop_px: float) -> bool:
    return (float(row["rsi"]) > 50
            or float(row["Close"]) >= float(row["bb_mid"])
            or float(row["Close"]) <= stop_px)


# ── Positieklasse ─────────────────────────────────────────────────────────────
class Pos:
    __slots__ = ["ticker","mode","units","entry_px","stop_px",
                 "trail_high","invest","entry_ts","strength","trail_mult"]

    def __init__(self, ticker: str, mode: str, units: float, entry_px: float,
                 stop_px: float, invest: float, entry_ts, strength: int = 0,
                 trail_mult: float = 0.0):
        self.ticker     = ticker
        self.mode       = mode       # "bh", "v4", "v4c", "mr"
        self.units      = units
        self.entry_px   = entry_px
        self.stop_px    = stop_px    # 0 for B&H (no stop)
        self.trail_high = entry_px
        self.invest     = invest
        self.entry_ts   = entry_ts
        self.strength   = strength
        self.trail_mult = trail_mult  # ATR multiplier for trailing stop

    def update_trail(self, close: float, atr: float):
        if self.mode not in ("v4", "v4c"):
            return
        if close > self.trail_high:
            self.trail_high = close
            new_stop = close - self.trail_mult * atr
            if new_stop > self.stop_px:
                self.stop_px = new_stop

    def mark_value(self, close: float) -> float:
        return self.units * close


def _close_pos(p: Pos, close: float) -> Tuple[float, float]:
    proceeds  = p.units * close
    sell_fee  = proceeds * FEE_RT / 2
    buy_fee   = p.invest * FEE_RT / 2
    pnl       = proceeds - p.units * p.entry_px - sell_fee - buy_fee
    cap_delta = proceeds - sell_fee
    return pnl, cap_delta


def _trade_rec(p: Pos, exit_ts, exit_px: float, pnl: float, reason: str) -> dict:
    return {
        "ticker":      p.ticker,
        "mode":        p.mode,
        "entry_ts":    str(p.entry_ts),
        "exit_ts":     str(exit_ts),
        "entry_px":    p.entry_px,
        "exit_px":     exit_px,
        "pnl":         pnl,
        "invest":      p.invest,
        "win":         pnl > 0,
        "strength":    p.strength,
        "exit_reason": reason,
        "pnl_pct":     pnl / p.invest * 100 if p.invest > 0 else 0.0,
    }


# ── Hoofd backtest ─────────────────────────────────────────────────────────────
def run_fourmode(frames: dict, thresholds: dict,
                 start: str, end: str,
                 start_cap: float = START_CAP) -> dict:

    slices = {t: slice_window(frames[t], start, end) for t in TICKERS}

    idx = None
    for df in slices.values():
        idx = df.index if idx is None else idx.union(df.index)
    idx = idx.sort_values()

    capital   = start_cap
    positions: List[Pos] = []
    trades    = []
    equity    = []

    # Tijdsregistratie per modus per ticker (voor analyse)
    regime_log: Dict[str, List[str]] = {t: [] for t in TICKERS}
    mode_time:  Dict[str, int]       = {"bh": 0, "v4": 0, "v4c": 0, "mr": 0, "flat": 0}

    for ts in idx:

        # ── Regime per ticker ──────────────────────────────────────────────────
        ticker_regimes: Dict[str, str] = {}
        for t in TICKERS:
            if ts not in slices[t].index:
                ticker_regimes[t] = NEUTRAAL
                continue
            row   = slices[t].loc[ts]
            score = regime_score_row(row)
            thr   = thresholds[t]
            reg   = get_regime_4(score, thr["p25"], thr["p50"], thr["p75"])
            ticker_regimes[t] = reg
            regime_log[t].append(reg)

        port_reg = portfolio_regime_4(ticker_regimes)

        # Portfolio totaalwaarde voor exposure berekening
        total_val = capital
        for p in positions:
            c = (float(slices[p.ticker].loc[ts, "Close"])
                 if ts in slices[p.ticker].index else p.entry_px)
            total_val += p.mark_value(c)

        # ── Exits ─────────────────────────────────────────────────────────────
        to_close = []
        for p in positions:
            if ts not in slices[p.ticker].index:
                continue
            row   = slices[p.ticker].loc[ts]
            close = float(row["Close"])
            atr   = float(row["atr"])
            t_reg = ticker_regimes[p.ticker]

            p.update_trail(close, atr)

            should_exit = False; reason = ""

            if p.mode == "bh":
                # B&H blijft open zolang regime ≥ lichte_bull
                # Sluit als regime daalt naar neutraal of bear
                if t_reg in (NEUTRAAL, BEAR):
                    should_exit = True
                    reason = f"regime_exit:{t_reg}"

            elif p.mode in ("v4", "v4c"):
                if close <= p.stop_px:
                    should_exit = True; reason = "trail_stop"

            elif p.mode == "mr":
                if mr_exit(row, p.stop_px):
                    should_exit = True
                    reason = "atr_stop" if close <= p.stop_px else "mr_signal"

            if should_exit:
                pnl, cd = _close_pos(p, close)
                capital   += cd
                total_val  = total_val - p.mark_value(close) + cd
                trades.append(_trade_rec(p, ts, close, pnl, reason))
                to_close.append(p)
        positions = [p for p in positions if p not in to_close]

        # ── Entries ────────────────────────────────────────────────────────────
        max_expo  = EXPO_CAP[port_reg]
        invested  = sum(p.invest for p in positions)
        expo_frac = invested / total_val if total_val > 0 else 0.0

        for t in TICKERS:
            if ts not in slices[t].index:
                continue
            if any(p.ticker == t for p in positions):
                continue   # al open
            if len(positions) >= 5:
                break
            if expo_frac >= max_expo:
                break

            row   = slices[t].loc[ts]
            close = float(row["Close"])
            atr   = float(row["atr"])
            t_reg = ticker_regimes[t]

            if t_reg == STERKE_BULL:
                # B&H: koop 80% van beschikbaar kapitaal
                invest = capital * BH_FRAC
                if invest < MIN_POS:
                    continue
                fee   = invest * FEE_RT / 2
                units = (invest - fee) / close
                capital  -= invest
                invested += invest
                expo_frac = invested / total_val if total_val > 0 else 0.0
                positions.append(Pos(t, "bh", units, close, 0.0,
                                     invest, ts, 0, 0.0))

            elif t_reg == LICHTE_BULL:
                if not v4_signal(row):
                    continue
                s      = v4_strength(row)
                frac   = V4_SIZES[s]
                invest = capital * frac
                if invest < MIN_POS:
                    continue
                fee   = invest * FEE_RT / 2
                units = (invest - fee) / close
                stop  = close - V4_ATR_ENTRY * atr
                capital  -= invest
                invested += invest
                expo_frac = invested / total_val if total_val > 0 else 0.0
                positions.append(Pos(t, "v4", units, close, stop,
                                     invest, ts, s, V4_ATR_TRAIL))

            elif t_reg == NEUTRAAL:
                if not v4_signal(row):
                    continue
                s      = v4_strength(row)
                frac   = V4C_SIZES[s]
                invest = capital * frac
                if invest < MIN_POS:
                    continue
                fee   = invest * FEE_RT / 2
                units = (invest - fee) / close
                stop  = close - V4_ATR_ENTRY * atr
                capital  -= invest
                invested += invest
                expo_frac = invested / total_val if total_val > 0 else 0.0
                positions.append(Pos(t, "v4c", units, close, stop,
                                     invest, ts, s, V4C_ATR_TRAIL))

            else:  # BEAR
                if not mr_signal(row):
                    continue
                deep   = float(row["rsi"]) < 20
                frac   = MR_FRAC[deep]
                invest = capital * frac
                if invest < MIN_POS:
                    continue
                fee   = invest * FEE_RT / 2
                units = (invest - fee) / close
                stop  = close - MR_ATR * atr
                capital  -= invest
                invested += invest
                expo_frac = invested / total_val if total_val > 0 else 0.0
                positions.append(Pos(t, "mr", units, close, stop,
                                     invest, ts, 0, 0.0))

        # Mode-tijd bijhouden (per bar, welke modus zijn open posities)
        open_modes = {p.mode for p in positions}
        if not open_modes:
            mode_time["flat"] += 1
        else:
            for m in open_modes:
                mode_time[m] += 1

        # Equity snapshot
        port = capital
        for p in positions:
            c = (float(slices[p.ticker].loc[ts, "Close"])
                 if ts in slices[p.ticker].index else p.entry_px)
            port += p.mark_value(c)
        equity.append(float(port))

    # Sluit open posities
    for p in positions:
        df_t = slices[p.ticker]
        lc   = float(df_t["Close"].iloc[-1]); lts = df_t.index[-1]
        pnl, cd = _close_pos(p, lc)
        capital += cd
        trades.append(_trade_rec(p, lts, lc, pnl, "eod"))

    return _calc_stats(trades, equity, capital, start_cap,
                       regime_log, mode_time, start, end)


def _calc_stats(trades, equity, final_cap, start_cap,
                regime_log, mode_time, start, end) -> dict:
    total_ret = (final_cap - start_cap) / start_cap * 100
    n    = len(trades)
    wins = sum(1 for t in trades if t["win"])
    wr   = wins / n * 100 if n else 0.0
    gw   = sum(t["pnl"] for t in trades if t["win"])
    gl   = abs(sum(t["pnl"] for t in trades if not t["win"]))
    pf   = gw / gl if gl > 0 else float("inf")

    eq   = np.array(equity if equity else [start_cap])
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / peak * 100

    streak = 0; max_streak = 0
    for t in trades:
        if not t["win"]: streak += 1; max_streak = max(max_streak, streak)
        else:            streak = 0

    monthly = {}; monthly_n = {}
    for t in trades:
        m = str(t["exit_ts"])[:7]
        monthly[m]   = monthly.get(m, 0.0) + t["pnl"]
        monthly_n[m] = monthly_n.get(m, 0) + 1

    best_m  = max(monthly.items(), key=lambda x: x[1]) if monthly else ("—", 0.0)
    worst_m = min(monthly.items(), key=lambda x: x[1]) if monthly else ("—", 0.0)

    n_months = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 30.44, 1)

    # PnL per modus
    mode_pnl = {"bh": 0.0, "v4": 0.0, "v4c": 0.0, "mr": 0.0}
    mode_n   = {"bh": 0,   "v4": 0,   "v4c": 0,   "mr": 0}
    mode_wins= {"bh": 0,   "v4": 0,   "v4c": 0,   "mr": 0}
    for t in trades:
        m = t["mode"]
        mode_pnl[m]  = mode_pnl.get(m, 0.0) + t["pnl"]
        mode_n[m]    = mode_n.get(m, 0) + 1
        if t["win"]:
            mode_wins[m] = mode_wins.get(m, 0) + 1

    # Regime tijd per ticker
    regime_dist: Dict[str, Dict[str, float]] = {}
    for t, regimes in regime_log.items():
        total = len(regimes) or 1
        regime_dist[t] = {
            r: round(regimes.count(r) / total * 100, 1)
            for r in REGIME_ORDER
        }

    # Totale mode-tijd percentages
    total_bars = sum(mode_time.values()) or 1
    mode_pct = {m: round(v / total_bars * 100, 1) for m, v in mode_time.items()}

    return {
        "final_cap":        final_cap,
        "total_ret":        total_ret,
        "n_trades":         n,
        "trades_per_month": round(n / n_months, 1),
        "win_rate":         wr,
        "profit_factor":    pf,
        "max_dd":           float(dd.max()),
        "max_loss_streak":  max_streak,
        "best_month":       best_m,
        "worst_month":      worst_m,
        "mode_pnl":         {k: round(v, 2) for k, v in mode_pnl.items()},
        "mode_n":           mode_n,
        "mode_wins":        mode_wins,
        "mode_time_pct":    mode_pct,
        "regime_dist":      regime_dist,
        "monthly":          monthly,
        "monthly_n":        monthly_n,
        "trades":           trades,
        "equity":           equity,
    }


# ── B&H benchmarks ─────────────────────────────────────────────────────────────
def run_bh(frames: dict, tickers: list, start: str, end: str,
           start_cap: float = START_CAP) -> dict:
    per_ticker = start_cap / len(tickers)
    vals_list  = []
    for t in tickers:
        sl = slice_window(frames[t], start, end)
        if sl.empty: continue
        units = per_ticker / float(sl["Close"].iloc[0])
        vals_list.append(units * sl["Close"])
    if not vals_list:
        return {"total_ret": 0.0, "max_dd": 0.0, "final_cap": start_cap}
    combined  = sum(vals_list)
    final_cap = float(combined.iloc[-1])
    peak      = combined.cummax()
    dd        = (peak - combined) / peak * 100
    return {"total_ret": (final_cap - start_cap) / start_cap * 100,
            "max_dd":    float(dd.max()),
            "final_cap": final_cap}


# ── Output helpers ────────────────────────────────────────────────────────────
def _pfs(v):
    return "∞" if v == float("inf") else f"{v:.2f}"


def _compound(pres: dict, cap: float = START_CAP) -> float:
    return cap * (1 + pres["2024"]["total_ret"]/100) \
                * (1 + pres["Val 2025"]["total_ret"]/100) \
                * (1 + pres["Finale"]["total_ret"]/100)


def _maxdd(pres: dict) -> float:
    return max(pres["2024"]["max_dd"], pres["Val 2025"]["max_dd"],
               pres["Finale"]["max_dd"])


def print_hoofdvergelijking(fm_pres: dict, refs: dict):
    W = 115
    print(f"\n{'═'*W}")
    print("  HOOFDVERGELIJKING — €500 startkapitaal, compound drie periodes")
    print(f"{'═'*W}")
    print(f"  {'Strategie':<32} {'2024':>8} {'Val2025':>8} {'Finale':>8} {'Totaal':>8} "
          f"{'MaxDD':>7} {'Trades':>7} {'Win%':>7} {'PF':>7} {'Eind€':>7}")
    print(f"  {'─'*32} {'─'*8} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    def _row(label, pres, is_bh=False):
        r24 = pres["2024"]; rv = pres["Val 2025"]; rf = pres["Finale"]
        eind = _compound(pres)
        tot  = (eind - START_CAP) / START_CAP * 100
        mdd  = _maxdd(pres)
        if is_bh:
            print(f"  {label:<32} {r24['total_ret']:>+7.1f}% {rv['total_ret']:>+7.1f}% "
                  f"{rf['total_ret']:>+7.1f}% {tot:>+7.1f}% {mdd:>6.1f}% "
                  f"{'—':>7} {'—':>7} {'—':>7} {eind:>6.0f}€")
            return
        ntot = r24["n_trades"] + rv["n_trades"] + rf["n_trades"]
        wr   = (r24["win_rate"]*r24["n_trades"] + rv["win_rate"]*rv["n_trades"]
                + rf["win_rate"]*rf["n_trades"]) / (ntot or 1)
        pfl  = [pres[p]["profit_factor"] for p in ["2024","Val 2025","Finale"]
                if pres[p]["profit_factor"] not in (0, float("inf"))]
        avgpf = float(np.mean(pfl)) if pfl else float("inf")
        print(f"  {label:<32} {r24['total_ret']:>+7.1f}% {rv['total_ret']:>+7.1f}% "
              f"{rf['total_ret']:>+7.1f}% {tot:>+7.1f}% {mdd:>6.1f}% "
              f"{ntot:>7} {wr:>6.1f}% {_pfs(avgpf):>7} {eind:>6.0f}€")

    _row("Vier-modus gecombineerd", fm_pres)
    print(f"  {'─'*32} {'─'*8} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    for label, key in [
        ("Hybride V4+MR (baseline)",     "baseline"),
        ("All-in C (winner takes all)",  "allin_c"),
    ]:
        if key in refs:
            _row(label, refs[key])
    for label, key in [
        ("B&H DOGE+SOL gelijk gewogen", "bh_ds"),
        ("B&H BTC",                      "bh_btc"),
    ]:
        if key in refs:
            _row(label, refs[key], is_bh=True)
    print(f"{'═'*W}")


def print_modus_analyse(fm_pres: dict):
    W = 110
    print(f"\n{'═'*W}")
    print("  MODUS ANALYSE — hoeveel tijd en rendement per strategie")
    print(f"{'═'*W}")

    for pname in ["2024", "Val 2025", "Finale"]:
        r = fm_pres[pname]
        print(f"\n  ── {pname} ──────────────────────────────────────────────")
        print(f"  Rendement: {r['total_ret']:>+.1f}%  MaxDD: {r['max_dd']:.1f}%  "
              f"Trades: {r['n_trades']}  (€{r['final_cap']:.0f})")

        # Tijd in elke modus (gebaseerd op bars met open posities per mode)
        mt = r["mode_time_pct"]
        print(f"\n  Modus-tijd (% bars met open positie in die modus):")
        for mode, label in [("bh","B&H sterke_bull"), ("v4","V4 lichte_bull"),
                             ("v4c","V4c neutraal"), ("mr","MR bear"), ("flat","Flat (geen pos)")]:
            pct = mt.get(mode, 0.0)
            bar = "█" * int(pct / 2)
            print(f"    {label:<20} {pct:>5.1f}%  {bar}")

        # PnL per modus
        mp   = r["mode_pnl"]
        mn   = r["mode_n"]
        mw   = r["mode_wins"]
        print(f"\n  PnL per modus:")
        print(f"    {'Modus':<12} {'Trades':>7} {'Win%':>7} {'PnL €':>9} {'Bijdrage':>10}")
        print(f"    {'─'*12} {'─'*7} {'─'*7} {'─'*9} {'─'*10}")
        total_pnl = sum(mp.values()) or 1e-9
        for mode, label in [("bh","B&H"), ("v4","V4"), ("v4c","V4c"), ("mr","MR")]:
            n   = mn.get(mode, 0); pnl = mp.get(mode, 0.0)
            wins = mw.get(mode, 0)
            wr  = wins / n * 100 if n else 0.0
            bij = pnl / total_pnl * 100 if total_pnl != 0 else 0.0
            print(f"    {label:<12} {n:>7} {wr:>6.1f}% {pnl:>+8.1f}€ {bij:>9.1f}%")

    # Regime distributie per ticker (over volledige periode)
    print(f"\n  Regime distributie per ticker (eerste periode 2024):")
    rd = fm_pres["2024"]["regime_dist"]
    print(f"  {'Crypto':<8} {'sterke_bull':>12} {'lichte_bull':>12} "
          f"{'neutraal':>10} {'bear':>8}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*10} {'─'*8}")
    for t in TICKERS:
        name = t.replace("-USD","")
        d = rd.get(t, {})
        print(f"  {name:<8} {d.get(STERKE_BULL,0):>11.1f}% "
              f"{d.get(LICHTE_BULL,0):>11.1f}% "
              f"{d.get(NEUTRAAL,0):>9.1f}% "
              f"{d.get(BEAR,0):>7.1f}%")
    print(f"{'═'*W}")


def print_periode_detail(fm_pres: dict):
    W = 110
    print(f"\n{'═'*W}")
    print("  DETAIL PER PERIODE — Vier-modus gecombineerd")
    print(f"{'═'*W}")
    print(f"  {'Periode':<12} {'Ret':>8} {'MaxDD':>7} {'Trades':>8} "
          f"{'T/mnd':>6} {'Win%':>7} {'PF':>7} {'MaxVrl':>7} "
          f"{'BesteMaand':>14} {'SlechteMaand':>14}")
    print(f"  {'─'*12} {'─'*8} {'─'*7} {'─'*8} "
          f"{'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*14} {'─'*14}")
    for pname in ["2024","Val 2025","Finale"]:
        r  = fm_pres[pname]
        bm = r["best_month"]; wm = r["worst_month"]
        bm_str = f"{bm[0]}(+{bm[1]:.0f}€)" if bm[0] != "—" else "—"
        wm_str = f"{wm[0]}({wm[1]:.0f}€)"  if wm[0] != "—" else "—"
        print(f"  {pname:<12} {r['total_ret']:>+7.1f}% {r['max_dd']:>6.1f}% "
              f"{r['n_trades']:>8} {r['trades_per_month']:>6.1f} "
              f"{r['win_rate']:>6.1f}% {_pfs(r['profit_factor']):>7} "
              f"{r['max_loss_streak']:>7} {bm_str:>14} {wm_str:>14}")
    eind = _compound(fm_pres)
    tot  = (eind - START_CAP) / START_CAP * 100
    print(f"  {'─'*12}")
    print(f"  Compound totaal: {tot:>+.1f}%  →  €{eind:.0f}")
    print(f"{'═'*W}")


def print_bh_contributie(fm_pres: dict):
    W = 80
    print(f"\n{'═'*W}")
    print("  B&H MODUS BIJDRAGE AAN TOTAAL RENDEMENT")
    print(f"{'═'*W}")
    print(f"  {'Periode':<12} {'B&H PnL':>10} {'V4 PnL':>10} "
          f"{'V4c PnL':>10} {'MR PnL':>10} {'Totaal':>10} {'B&H %':>8}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    for pname in ["2024","Val 2025","Finale"]:
        r  = fm_pres[pname]
        mp = r["mode_pnl"]
        bh = mp.get("bh", 0.0); v4 = mp.get("v4", 0.0)
        vc = mp.get("v4c", 0.0); mr = mp.get("mr", 0.0)
        tot = bh + v4 + vc + mr
        bh_pct = bh / tot * 100 if tot != 0 else 0.0
        print(f"  {pname:<12} {bh:>+9.1f}€ {v4:>+9.1f}€ "
              f"{vc:>+9.1f}€ {mr:>+9.1f}€ {tot:>+9.1f}€ {bh_pct:>7.1f}%")
    print(f"{'═'*W}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("  Vier-modus Gecombineerde Backtest (B&H + V4 + V4c + MR)")
    print("=" * 80)

    print("\n  Data ophalen …")
    frames     = {}
    thresholds = {}
    for t in TICKERS:
        print(f"  {t:<12} …", end=" ", flush=True)
        df = add_indicators(fetch_4h(t))
        frames[t] = df
        p25, p50, p75 = calibrate_4level(df)
        thresholds[t] = {"p25": p25, "p50": p50, "p75": p75}
        print(f"{len(df):>6} bars  |  bear<{p25:+.4f}  neut<{p50:+.4f}  "
              f"licht<{p75:+.4f}  sterk>")

    print(f"  {'BTC-USD':<12} …", end=" ", flush=True)
    btc_df = add_indicators(fetch_4h("BTC-USD"))
    print(f"{len(btc_df):>6} bars")

    # ── Vier-modus backtest ────────────────────────────────────────────────────
    print("\n  Vier-modus backtest …")
    fm_pres = {}; cap = START_CAP
    for pname, pstart, pend in PERIODS:
        print(f"  {pname} …", end=" ", flush=True)
        r = run_fourmode(frames, thresholds, pstart, pend, start_cap=cap)
        fm_pres[pname] = r
        cap = r["final_cap"]
        print(f"{r['n_trades']:>4} trades  {r['total_ret']:>+.1f}%  "
              f"MaxDD {r['max_dd']:.1f}%  €{r['final_cap']:.0f}  "
              f"[BH:{r['mode_n']['bh']} V4:{r['mode_n']['v4']} "
              f"V4c:{r['mode_n']['v4c']} MR:{r['mode_n']['mr']}]")

    # ── Referenties ────────────────────────────────────────────────────────────
    refs = {}

    # Baseline hybride (DOGE/SOL/ETH/AVAX/ADA, p33/p67, 7/10/15%)
    print("\n  Baseline hybride V4+MR (referentie) …")
    from backtest_portfolio import (fetch_4h as bpf4h, add_indicators as bpai,
                                     calibrate_single, run_portfolio as bprp,
                                     TICKERS as BP_TICKERS)
    bp_frames = {}; bp_thr = {}
    for t in BP_TICKERS:
        df = bpai(bpf4h(t))
        bp_frames[t] = df
        p33, p67 = calibrate_single(df)
        bp_thr[t] = {"p33": p33, "p67": p67}

    # reuse existing portfolio runner with baseline config
    from backtest_aggressive_v2 import VariantCfg, run_portfolio as agv2rp
    bl_cfg = VariantCfg("baseline","Baseline",BP_TICKERS,
                         {0:0.07,1:0.07,2:0.10,3:0.15,4:0.15},45,75,True,5)
    bl_pres = {}; cap = START_CAP
    for pname, pstart, pend in PERIODS:
        r = agv2rp(bp_frames, bp_thr, pstart, pend, bl_cfg, start_cap=cap)
        bl_pres[pname] = r; cap = r["final_cap"]
    refs["baseline"] = bl_pres
    print(f"  Baseline: {(_compound(bl_pres)-START_CAP)/START_CAP*100:>+.1f}%  "
          f"€{_compound(bl_pres):.0f}")

    # All-in C (winner takes all op DOGE+SOL)
    print("  All-in C referentie …")
    from backtest_allin import run_allin, TICKERS as AI_TICKERS
    ai_frames = {}; ai_thr = {}
    for t in AI_TICKERS:
        df = add_indicators(fetch_4h(t))
        ai_frames[t] = df
        from backtest_allin import calibrate as ai_cal
        p33, p67_ai = ai_cal(df)
        ai_thr[t.replace("-USD","")] = {"p33": p33, "p67": p67_ai}
    aic_pres = {}; cap = START_CAP
    for pname, pstart, pend in PERIODS:
        r = run_allin(ai_frames, ai_thr, pstart, pend, "C", start_cap=cap)
        aic_pres[pname] = r; cap = r["final_cap"]
    refs["allin_c"] = aic_pres
    print(f"  All-in C: {(_compound(aic_pres)-START_CAP)/START_CAP*100:>+.1f}%  "
          f"€{_compound(aic_pres):.0f}")

    # B&H benchmarks
    print("  B&H benchmarks …")
    bh_all = {**frames, "BTC-USD": btc_df}
    for bh_key, tickers, label in [
        ("bh_ds",  TICKERS,       "DOGE/SOL/ETH/AVAX/ADA"),
        ("bh_btc", ["BTC-USD"],   "BTC"),
    ]:
        pres = {}; cap = START_CAP
        for pname, pstart, pend in PERIODS:
            r = run_bh(bh_all, tickers, pstart, pend, start_cap=cap)
            pres[pname] = r; cap = r["final_cap"]
        refs[bh_key] = pres
        eind = _compound(pres)
        print(f"  B&H {label:<22} →  {(eind-START_CAP)/START_CAP*100:>+.1f}%  €{eind:.0f}")

    # ── Tabellen ───────────────────────────────────────────────────────────────
    print_hoofdvergelijking(fm_pres, refs)
    print_periode_detail(fm_pres)
    print_modus_analyse(fm_pres)
    print_bh_contributie(fm_pres)

    # ── Opslaan ────────────────────────────────────────────────────────────────
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k != "equity"}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float):
            return 999.99 if obj == float("inf") else round(obj, 4)
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    save = {
        "fourmode":   _clean(fm_pres),
        "refs":       _clean(refs),
        "thresholds": {t: {k: round(v, 6) for k, v in thr.items()}
                       for t, thr in thresholds.items()},
    }
    OUTPUT_FILE.write_text(json.dumps(save, indent=2, ensure_ascii=False))
    print(f"\n  Resultaten opgeslagen: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
