"""
All-In Backtest — DOGE + SOL
=============================
Test drie all-in varianten van het hybride V4+MR systeem op DOGE en SOL.

All-in A: één positie tegelijk (95% van kapitaal, max 1 open)
All-in B: twee posities tegelijk (47.5% elk, max 2 open)
All-in C: winner takes all (sterkste signaal krijgt 95%, andere geblokkeerd)

Vergelijk met V3 (25/40/60%) en B&H BTC.

Stack: Python 3.11, yfinance, ta, pandas. Geen matplotlib, geen scikit-learn.
"""

import json, warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ta.momentum   import RSIIndicator
from ta.trend      import MACD as TaMACD
from ta.volatility import AverageTrueRange, BollingerBands

# ── Constanten ────────────────────────────────────────────────────────────────
TICKERS   = ["DOGE-USD", "SOL-USD"]
START_CAP = 500.0
FEE_RT    = 0.004
MIN_POS   = 3.0

ALLIN_FRAC    = 0.95   # 95% per trade (5% reserve voor kosten)
ALLIN_B_FRAC  = 0.475  # 47.5% per trade als beide open

V4_RSI_LO         = 45; V4_RSI_HI = 75
V4_ATR_ENTRY      = 4.0
V4_ATR_TRAIL      = 3.0
V4_MA200_MAX_BELOW = 0.15

MR_RSI_THR = 35
MR_BB_WIN  = 30; MR_BB_SIG = 2.5
MR_ATR     = 2.0
MR_FRAC    = 0.475   # MR ook half in all-in B, vol in A/C

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

OUTPUT_FILE = Path(__file__).parent / "results_allin.json"


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


# ── Regime ────────────────────────────────────────────────────────────────────
def regime_score_row(row) -> float:
    ma200 = float(row["ma200"]); ma200_lag = float(row["ma200_lag"])
    close = float(row["Close"]); rsi       = float(row["rsi"])
    ma200_s = (ma200 - ma200_lag) / ma200_lag if ma200_lag else 0.0
    prijs_s = (close - ma200)     / ma200     if ma200     else 0.0
    rsi_s   = (rsi - 50)          / 50
    return float(np.clip((ma200_s + prijs_s + rsi_s) / 3, -1.0, 1.0))


def calibrate(df: pd.DataFrame) -> Tuple[float, float]:
    sl = slice_window(df, TRAIN_START, TRAIN_END)
    scores = sl.apply(regime_score_row, axis=1).dropna().values
    if len(scores) == 0:
        return 0.0, 0.0
    return float(np.percentile(scores, 33)), float(np.percentile(scores, 67))


def get_regime(score: float, p33: float, p67: float) -> str:
    if score > p67: return "bull"
    if score < p33: return "bear"
    return "neutraal"


# ── Signalen ──────────────────────────────────────────────────────────────────
def v4_signal(row) -> bool:
    rsi   = float(row["rsi"]); close = float(row["Close"]); ma200 = float(row["ma200"])
    cross = float(row["macd"]) > float(row["macd_sig"]) and float(row["prev_hist"]) <= 0
    return (V4_RSI_LO <= rsi <= V4_RSI_HI and cross
            and float(row["Volume"]) > float(row["vol_ma"])
            and close >= ma200 * (1 - V4_MA200_MAX_BELOW))


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


def v4_confidence(row) -> float:
    """Combineer RSI en MACD histogram kracht tot één score voor winner-takes-all."""
    rsi_score  = (float(row["rsi"]) - V4_RSI_LO) / (V4_RSI_HI - V4_RSI_LO)
    hist_score = min(abs(float(row["macd_hist"])) / (abs(float(row["atr"])) + 1e-9), 1.0)
    vol_score  = min(float(row["vol_ratio"]) / 2.0, 1.0)
    return (rsi_score + hist_score + vol_score) / 3.0


# ── Positieklasse ─────────────────────────────────────────────────────────────
class Pos:
    __slots__ = ["ticker","strategy","units","entry_px","stop_px",
                 "trail_high","invest","entry_ts","strength"]

    def __init__(self, ticker, strategy, units, entry_px, stop_px,
                 invest, entry_ts, strength=0):
        self.ticker     = ticker
        self.strategy   = strategy
        self.units      = units
        self.entry_px   = entry_px
        self.stop_px    = stop_px
        self.trail_high = entry_px
        self.invest     = invest
        self.entry_ts   = entry_ts
        self.strength   = strength

    def update_trail(self, close: float, atr: float):
        if self.strategy == "v4" and close > self.trail_high:
            self.trail_high = close
            new_stop = close - V4_ATR_TRAIL * atr
            if new_stop > self.stop_px:
                self.stop_px = new_stop

    def mark_value(self, close: float) -> float:
        return self.units * close


def _close(p: Pos, close: float):
    proceeds  = p.units * close
    sell_fee  = proceeds * FEE_RT / 2
    buy_fee   = p.invest * FEE_RT / 2
    pnl       = proceeds - p.units * p.entry_px - sell_fee - buy_fee
    cap_delta = proceeds - sell_fee
    return pnl, cap_delta


def _trade_rec(p: Pos, exit_ts, exit_px: float, pnl: float, reason: str) -> dict:
    return {
        "ticker":      p.ticker,
        "strategy":    p.strategy,
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


# ── Kern backtest engine ──────────────────────────────────────────────────────
def run_allin(frames: dict, thresholds: dict, start: str, end: str,
              variant: str, start_cap: float = START_CAP) -> dict:
    """
    variant:
      'A' — max 1 open positie totaal, 95% van kapitaal
      'B' — max 2 open posities, elk 47.5% van kapitaal
      'C' — winner takes all: sterkste signaal 95%, andere geblokkeerd
    """
    slices = {t: slice_window(frames[t], start, end) for t in TICKERS}

    idx = None
    for df in slices.values():
        idx = df.index if idx is None else idx.union(df.index)
    idx = idx.sort_values()

    capital   = start_cap
    positions = []   # lijst van Pos
    trades    = []
    equity    = []
    last_regime = "neutraal"

    for ts in idx:
        # ── Exits ─────────────────────────────────────────────────────────────
        to_close = []
        for p in positions:
            if ts not in slices[p.ticker].index:
                continue
            row   = slices[p.ticker].loc[ts]
            close = float(row["Close"])
            atr   = float(row["atr"])
            p.update_trail(close, atr)

            if p.strategy == "v4":
                should_exit = close <= p.stop_px; reason = "trail"
            else:
                should_exit = mr_exit(row, p.stop_px)
                reason = "atr" if close <= p.stop_px else "signal"

            if should_exit:
                pnl, cd = _close(p, close)
                capital += cd
                trades.append(_trade_rec(p, ts, close, pnl, reason))
                to_close.append(p)
        positions = [p for p in positions if p not in to_close]

        # ── Verzamel entry-kandidaten deze bar ─────────────────────────────────
        candidates = []   # (ticker, strategy, strength, confidence, row)
        for t in TICKERS:
            if ts not in slices[t].index:
                continue
            if any(p.ticker == t for p in positions):
                continue   # al open in dit crypto

            row  = slices[t].loc[ts]
            name = t.replace("-USD","")
            thr  = thresholds[name]
            sc   = regime_score_row(row)
            reg  = get_regime(sc, thr["p33"], thr["p67"])
            last_regime = reg

            if reg in ("bull", "neutraal"):
                if v4_signal(row):
                    s    = v4_strength(row)
                    conf = v4_confidence(row)
                    candidates.append((t, "v4", s, conf, row))
            else:  # bear
                if mr_signal(row):
                    candidates.append((t, "mr", 0, 0.5, row))

        # ── Entries op basis van variant ──────────────────────────────────────
        if variant == "A":
            # Max 1 positie totaal; blokkeer als al iets open is
            if len(positions) == 0 and candidates:
                # Kies eerste kandidaat (DOGE voor SOL als beide)
                t, strategy, s, conf, row = candidates[0]
                close = float(row["Close"]); atr = float(row["atr"])
                frac  = ALLIN_FRAC if strategy == "v4" else ALLIN_FRAC
                invest = capital * frac
                if invest >= MIN_POS:
                    fee   = invest * FEE_RT / 2
                    units = (invest - fee) / close
                    stop  = close - V4_ATR_ENTRY * atr if strategy == "v4" else close - MR_ATR * atr
                    capital -= invest
                    positions.append(Pos(t, strategy, units, close, stop, invest, ts, s))

        elif variant == "B":
            # Max 2 posities; elk 47.5%
            for t, strategy, s, conf, row in candidates:
                if len(positions) >= 2:
                    break
                close  = float(row["Close"]); atr = float(row["atr"])
                frac   = ALLIN_B_FRAC
                invest = capital * frac
                if invest >= MIN_POS:
                    fee   = invest * FEE_RT / 2
                    units = (invest - fee) / close
                    stop  = close - V4_ATR_ENTRY * atr if strategy == "v4" else close - MR_ATR * atr
                    capital -= invest
                    positions.append(Pos(t, strategy, units, close, stop, invest, ts, s))

        elif variant == "C":
            # Winner takes all: hoogste confidence krijgt 95%
            if len(positions) == 0 and candidates:
                # Sorteer op confidence aflopend; V4 over MR bij gelijke conf
                best = sorted(candidates, key=lambda x: x[3], reverse=True)[0]
                t, strategy, s, conf, row = best
                close  = float(row["Close"]); atr = float(row["atr"])
                frac   = ALLIN_FRAC
                invest = capital * frac
                if invest >= MIN_POS:
                    fee   = invest * FEE_RT / 2
                    units = (invest - fee) / close
                    stop  = close - V4_ATR_ENTRY * atr if strategy == "v4" else close - MR_ATR * atr
                    capital -= invest
                    positions.append(Pos(t, strategy, units, close, stop, invest, ts, s))

        # ── Equity snapshot ────────────────────────────────────────────────────
        port = capital
        for p in positions:
            c = (float(slices[p.ticker].loc[ts, "Close"])
                 if ts in slices[p.ticker].index else p.entry_px)
            port += p.mark_value(c)
        equity.append(float(port))

    # Sluit open posities
    for p in positions:
        df_t = slices[p.ticker]
        lc   = float(df_t["Close"].iloc[-1])
        lts  = df_t.index[-1]
        pnl, cd = _close(p, lc)
        capital += cd
        trades.append(_trade_rec(p, lts, lc, pnl, "eod"))

    return _calc_stats(trades, equity, capital, start_cap, start, end)


def _calc_stats(trades, equity, final_cap, start_cap, start, end):
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

    # Grootste winst en verlies per individuele trade
    best_trade  = max(trades, key=lambda t: t["pnl"]) if trades else None
    worst_trade = min(trades, key=lambda t: t["pnl"]) if trades else None

    n_months = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 30.44, 1)

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
        "best_trade_pnl":   round(best_trade["pnl"],  2) if best_trade  else 0.0,
        "best_trade_pct":   round(best_trade["pnl_pct"],  1) if best_trade  else 0.0,
        "worst_trade_pnl":  round(worst_trade["pnl"], 2) if worst_trade else 0.0,
        "worst_trade_pct":  round(worst_trade["pnl_pct"], 1) if worst_trade else 0.0,
        "best_trade_ts":    str(best_trade["exit_ts"])[:10]  if best_trade  else "—",
        "worst_trade_ts":   str(worst_trade["exit_ts"])[:10] if worst_trade else "—",
        "monthly":          monthly,
        "monthly_n":        monthly_n,
        "trades":           trades,
        "equity":           equity,
    }


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
        return {"total_ret": 0.0, "max_dd": 0.0, "final_cap": start_cap,
                "best_trade_pnl": 0.0, "worst_trade_pnl": 0.0}
    combined  = sum(vals_list)
    final_cap = float(combined.iloc[-1])
    peak      = combined.cummax()
    dd        = (peak - combined) / peak * 100
    return {"total_ret": (final_cap - start_cap) / start_cap * 100,
            "max_dd":    float(dd.max()),
            "final_cap": final_cap}


# ── Slechts-moment analyse ────────────────────────────────────────────────────
def worst_entry_analysis(frames: dict, thresholds: dict) -> dict:
    """
    Simuleer All-in A maar start op het slechtste instapmoment:
    de dag dat de gecombineerde DOGE+SOL waarde op zijn hoogste punt stond
    vóór de grootste drawdown in 2024-2026.
    """
    # Vind piek in gecombineerde DOGE+SOL waarde over de volledige periode
    all_vals = []
    for t in TICKERS:
        df = slice_window(frames[t], TRAIN_START, FINAL_END)
        if df.empty: continue
        units = START_CAP / 2 / float(df["Close"].iloc[0])
        all_vals.append(units * df["Close"])

    if not all_vals:
        return {}

    combined = sum(all_vals)
    peak_idx = combined.idxmax()
    peak_str = str(peak_idx.date())

    # Vind de periode die de piek bevat
    if peak_idx <= pd.Timestamp(TRAIN_END):
        pstart, pend = TRAIN_START, TRAIN_END
    elif peak_idx <= pd.Timestamp(VAL_END):
        pstart, pend = VAL_START, VAL_END
    else:
        pstart, pend = FINAL_START, FINAL_END

    # Simuleer All-in A startend met €500 op piek_datum als start
    # (sla alle eerdere data over)
    start_after = peak_str
    r = run_allin(frames, thresholds, start_after, FINAL_END, "A", START_CAP)

    return {
        "peak_date":    peak_str,
        "peak_val_eur": round(float(combined.loc[peak_idx]), 2),
        "final_cap":    round(r["final_cap"], 2),
        "total_ret":    round(r["total_ret"], 2),
        "max_dd":       round(r["max_dd"], 2),
        "n_trades":     r["n_trades"],
    }


# ── Output helpers ────────────────────────────────────────────────────────────
def _pfs(v):
    return "∞" if v == float("inf") else f"{v:.2f}"


def _compound(pres: dict, cap: float = START_CAP) -> float:
    return cap * (1 + pres["2024"]["total_ret"]/100) \
                * (1 + pres["Val 2025"]["total_ret"]/100) \
                * (1 + pres["Finale"]["total_ret"]/100)


def _maxdd(pres: dict) -> float:
    return max(pres["2024"]["max_dd"], pres["Val 2025"]["max_dd"], pres["Finale"]["max_dd"])


def _streaks(pres: dict) -> int:
    return max(pres["2024"]["max_loss_streak"],
               pres["Val 2025"]["max_loss_streak"],
               pres["Finale"]["max_loss_streak"])


def print_vergelijking(all_results: dict):
    W = 150
    print(f"\n{'═'*W}")
    print("  VERGELIJKING — €500 startkapitaal, compound drie periodes")
    print(f"{'═'*W}")
    print(f"  {'Variant':<28} {'2024':>8} {'Val2025':>8} {'Finale':>8} {'Totaal':>8} "
          f"{'MaxDD':>7} {'Trades':>7} {'Win%':>7} {'PF':>7} {'Eind€':>7} "
          f"{'MaxVrl':>7} {'BesteTrade':>12} {'SlechtTrade':>12}")
    print(f"  {'─'*28} {'─'*8} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} "
          f"{'─'*7} {'─'*12} {'─'*12}")

    variants = [
        ("All-in A: 1 pos, 95%",   "allin_a"),
        ("All-in B: 2 pos, 47.5%", "allin_b"),
        ("All-in C: winner 95%",   "allin_c"),
    ]
    for label, key in variants:
        pres = all_results[key]
        r24  = pres["2024"]; rv = pres["Val 2025"]; rf = pres["Finale"]
        eind = _compound(pres)
        tot  = (eind - START_CAP) / START_CAP * 100
        mdd  = _maxdd(pres)
        ntot = r24["n_trades"] + rv["n_trades"] + rf["n_trades"]
        wr   = (r24["win_rate"]*r24["n_trades"] + rv["win_rate"]*rv["n_trades"]
                + rf["win_rate"]*rf["n_trades"]) / (ntot or 1)
        pfl  = [pres[p]["profit_factor"] for p in ["2024","Val 2025","Finale"]
                if pres[p]["profit_factor"] not in (0, float("inf"))]
        avgpf = float(np.mean(pfl)) if pfl else float("inf")
        mls  = _streaks(pres)

        # Beste en slechtste trade over alle periodes
        all_trades = []
        for p in ["2024","Val 2025","Finale"]:
            all_trades.extend(pres[p].get("trades", []))
        bt = max(all_trades, key=lambda t: t["pnl"]) if all_trades else None
        wt = min(all_trades, key=lambda t: t["pnl"]) if all_trades else None
        bt_str = f"+{bt['pnl']:.0f}€ ({bt['pnl_pct']:>+.0f}%)" if bt else "—"
        wt_str = f"{wt['pnl']:.0f}€ ({wt['pnl_pct']:>+.0f}%)"  if wt else "—"

        print(f"  {label:<28} {r24['total_ret']:>+7.1f}% {rv['total_ret']:>+7.1f}% "
              f"{rf['total_ret']:>+7.1f}% {tot:>+7.1f}% "
              f"{mdd:>6.1f}% {ntot:>7} {wr:>6.1f}% {_pfs(avgpf):>7} "
              f"{eind:>6.0f}€ {mls:>7} {bt_str:>12} {wt_str:>12}")

    print(f"  {'─'*28} {'─'*8} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} "
          f"{'─'*7} {'─'*12} {'─'*12}")

    # V3 referentie
    pres = all_results["v3_ref"]
    r24  = pres["2024"]; rv = pres["Val 2025"]; rf = pres["Finale"]
    eind = _compound(pres)
    tot  = (eind - START_CAP) / START_CAP * 100
    mdd  = _maxdd(pres)
    ntot = r24["n_trades"] + rv["n_trades"] + rf["n_trades"]
    wr   = (r24["win_rate"]*r24["n_trades"] + rv["win_rate"]*rv["n_trades"]
            + rf["win_rate"]*rf["n_trades"]) / (ntot or 1)
    mls  = _streaks(pres)
    all_trades = []
    for p in ["2024","Val 2025","Finale"]:
        all_trades.extend(pres[p].get("trades", []))
    bt = max(all_trades, key=lambda t: t["pnl"]) if all_trades else None
    wt = min(all_trades, key=lambda t: t["pnl"]) if all_trades else None
    bt_str = f"+{bt['pnl']:.0f}€ ({bt['pnl_pct']:>+.0f}%)" if bt else "—"
    wt_str = f"{wt['pnl']:.0f}€ ({wt['pnl_pct']:>+.0f}%)"  if wt else "—"
    print(f"  {'V3: 25/40/60% (ref)':<28} {r24['total_ret']:>+7.1f}% {rv['total_ret']:>+7.1f}% "
          f"{rf['total_ret']:>+7.1f}% {tot:>+7.1f}% "
          f"{mdd:>6.1f}% {ntot:>7} {wr:>6.1f}% {'—':>7} "
          f"{eind:>6.0f}€ {mls:>7} {bt_str:>12} {wt_str:>12}")

    # B&H benchmarks
    for bh_label, bh_key in [("B&H DOGE+SOL", "bh_2x"), ("B&H BTC", "bh_btc")]:
        pres = all_results[bh_key]
        r24  = pres["2024"]; rv = pres["Val 2025"]; rf = pres["Finale"]
        eind = _compound(pres)
        tot  = (eind - START_CAP) / START_CAP * 100
        mdd  = _maxdd(pres)
        print(f"  {bh_label:<28} {r24['total_ret']:>+7.1f}% {rv['total_ret']:>+7.1f}% "
              f"{rf['total_ret']:>+7.1f}% {tot:>+7.1f}% {mdd:>6.1f}% "
              f"{'—':>7} {'—':>7} {'—':>7} {eind:>6.0f}€")
    print(f"{'═'*W}")


def print_periode_detail(all_results: dict):
    W = 100
    labels = [
        ("All-in A: één positie (95%)",    "allin_a"),
        ("All-in B: twee posities (47.5%)", "allin_b"),
        ("All-in C: winner takes all",      "allin_c"),
    ]
    for label, key in labels:
        pres = all_results[key]
        print(f"\n{'═'*W}")
        print(f"  DETAIL: {label}")
        print(f"{'─'*W}")
        print(f"  {'Periode':<12} {'Ret':>8} {'MaxDD':>7} {'Trades':>8} "
              f"{'T/mnd':>6} {'Win%':>7} {'PF':>7} {'MaxVrl':>7} "
              f"{'BesteMaand':>14} {'SlechteMaand':>14}")
        print(f"  {'─'*12} {'─'*8} {'─'*7} {'─'*8} "
              f"{'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*14} {'─'*14}")
        for pname in ["2024","Val 2025","Finale"]:
            r  = pres[pname]
            bm = r["best_month"]; wm = r["worst_month"]
            bm_str = f"{bm[0]}(+{bm[1]:.0f}€)" if bm[0] != "—" else "—"
            wm_str = f"{wm[0]}({wm[1]:.0f}€)"  if wm[0] != "—" else "—"
            print(f"  {pname:<12} {r['total_ret']:>+7.1f}% {r['max_dd']:>6.1f}% "
                  f"{r['n_trades']:>8} {r['trades_per_month']:>6.1f} "
                  f"{r['win_rate']:>6.1f}% {_pfs(r['profit_factor']):>7} "
                  f"{r['max_loss_streak']:>7} {bm_str:>14} {wm_str:>14}")
        eind = _compound(pres)
        tot  = (eind - START_CAP) / START_CAP * 100
        print(f"  {'─'*12}")
        print(f"  Compound totaal: {tot:>+.1f}%  →  €{eind:.0f}")

        # Beste en slechtste trade
        all_tr = []
        for p in ["2024","Val 2025","Finale"]:
            all_tr.extend(pres[p].get("trades", []))
        if all_tr:
            bt = max(all_tr, key=lambda t: t["pnl"])
            wt = min(all_tr, key=lambda t: t["pnl"])
            print(f"  Beste trade  : {bt['ticker']}  {bt['exit_ts'][:10]}  "
                  f"+{bt['pnl']:.2f}€ ({bt['pnl_pct']:>+.1f}%)")
            print(f"  Slechtste    : {wt['ticker']}  {wt['exit_ts'][:10]}  "
                  f"{wt['pnl']:.2f}€ ({wt['pnl_pct']:>+.1f}%)")
    print(f"\n{'═'*W}")


def print_worst_entry(worst: dict):
    if not worst:
        return
    W = 80
    print(f"\n{'═'*W}")
    print("  SLECHTSTE INSTAPMOMENT ANALYSE (All-in A)")
    print(f"{'═'*W}")
    print(f"  Gecombineerde DOGE+SOL piek: {worst['peak_date']}")
    print(f"  Waarde op piek (gelijk gewogen €500): €{worst['peak_val_eur']:.0f}")
    print(f"")
    print(f"  Als je op {worst['peak_date']} was ingestapt met All-in A:")
    print(f"  Eindkapitaal (jun 2026)  : €{worst['final_cap']:.0f}")
    print(f"  Rendement                : {worst['total_ret']:>+.1f}%")
    print(f"  MaxDD                    : {worst['max_dd']:.1f}%")
    print(f"  Aantal trades            : {worst['n_trades']}")
    print(f"{'═'*W}")


# ── V3 referentie (inline, geen externe import) ───────────────────────────────
V3_SIZES = {0: 0.25, 1: 0.25, 2: 0.40, 3: 0.60, 4: 0.60}

def run_v3_ref(frames: dict, thresholds: dict, start: str, end: str,
               start_cap: float = START_CAP) -> dict:
    """V3 (25/40/60%, max 2 posities) — zelfstandige implementatie."""
    slices = {t: slice_window(frames[t], start, end) for t in TICKERS}

    idx = None
    for df in slices.values():
        idx = df.index if idx is None else idx.union(df.index)
    idx = idx.sort_values()

    capital = start_cap; positions = []; trades = []; equity = []

    for ts in idx:
        to_close = []
        for p in positions:
            if ts not in slices[p.ticker].index:
                continue
            row = slices[p.ticker].loc[ts]
            close = float(row["Close"]); atr = float(row["atr"])
            p.update_trail(close, atr)
            if p.strategy == "v4":
                should_exit = close <= p.stop_px; reason = "trail"
            else:
                should_exit = mr_exit(row, p.stop_px)
                reason = "atr" if close <= p.stop_px else "signal"
            if should_exit:
                pnl, cd = _close(p, close)
                capital += cd
                trades.append(_trade_rec(p, ts, close, pnl, reason))
                to_close.append(p)
        positions = [p for p in positions if p not in to_close]

        for t in TICKERS:
            if ts not in slices[t].index:
                continue
            if len(positions) >= 2 or any(p.ticker == t for p in positions):
                continue
            row = slices[t].loc[ts]
            name = t.replace("-USD", "")
            thr  = thresholds[name]
            sc   = regime_score_row(row)
            reg  = get_regime(sc, thr["p33"], thr["p67"])
            close = float(row["Close"]); atr = float(row["atr"])

            if reg in ("bull", "neutraal"):
                if not v4_signal(row):
                    continue
                s      = v4_strength(row)
                frac   = V3_SIZES[s]
                invest = capital * frac
                if invest < MIN_POS:
                    continue
                fee   = invest * FEE_RT / 2
                units = (invest - fee) / close
                stop  = close - V4_ATR_ENTRY * atr
                capital -= invest
                positions.append(Pos(t, "v4", units, close, stop, invest, ts, s))
            else:
                if not mr_signal(row):
                    continue
                frac   = 0.25
                invest = capital * frac
                if invest < MIN_POS:
                    continue
                fee   = invest * FEE_RT / 2
                units = (invest - fee) / close
                stop  = close - MR_ATR * atr
                capital -= invest
                positions.append(Pos(t, "mr", units, close, stop, invest, ts, 0))

        port = capital
        for p in positions:
            c = (float(slices[p.ticker].loc[ts, "Close"])
                 if ts in slices[p.ticker].index else p.entry_px)
            port += p.mark_value(c)
        equity.append(float(port))

    for p in positions:
        df_t = slices[p.ticker]
        lc   = float(df_t["Close"].iloc[-1]); lts = df_t.index[-1]
        pnl, cd = _close(p, lc)
        capital += cd
        trades.append(_trade_rec(p, lts, lc, pnl, "eod"))

    return _calc_stats(trades, equity, capital, start_cap, start, end)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("  All-In Backtest — DOGE + SOL (drie varianten)")
    print("=" * 80)

    print("\n  Data ophalen …")
    frames     = {}
    thresholds = {}
    for t in TICKERS:
        print(f"  {t:<12} …", end=" ", flush=True)
        df = add_indicators(fetch_4h(t))
        frames[t] = df
        p33, p67 = calibrate(df)
        thresholds[t.replace("-USD","")] = {"p33": p33, "p67": p67}
        print(f"{len(df):>6} bars  |  bear < {p33:+.4f}  bull > {p67:+.4f}")

    # BTC voor benchmark
    print(f"  {'BTC-USD':<12} …", end=" ", flush=True)
    btc_df = add_indicators(fetch_4h("BTC-USD"))
    btc_frames = {"BTC-USD": btc_df}
    print(f"{len(btc_df):>6} bars")

    all_results = {}
    total_runs  = 3 * 3 + 3   # 3 varianten × 3 periodes + 3 benchmarks
    run_nr      = 0

    # All-in varianten
    for var_key, var_label in [("A","All-in A (1 pos, 95%)"),
                                ("B","All-in B (2 pos, 47.5%)"),
                                ("C","All-in C (winner takes all)")]:
        print(f"\n  ── {var_label}")
        pres = {}; cap = START_CAP
        for pname, pstart, pend in PERIODS:
            run_nr += 1
            print(f"  [{run_nr}/{total_runs}] {pname} …", end=" ", flush=True)
            r = run_allin(frames, thresholds, pstart, pend, var_key, start_cap=cap)
            pres[pname] = r
            cap = r["final_cap"]
            print(f"{r['n_trades']:>4} trades  {r['total_ret']:>+.1f}%  "
                  f"MaxDD {r['max_dd']:.1f}%  €{r['final_cap']:.0f}")
        all_results[f"allin_{var_key.lower()}"] = pres

    # V3 referentie
    print("\n  ── V3 referentie (25/40/60%) …")
    pres = {}; cap = START_CAP
    for pname, pstart, pend in PERIODS:
        r = run_v3_ref(frames, thresholds, pstart, pend, start_cap=cap)
        pres[pname] = r
        cap = r["final_cap"]
    all_results["v3_ref"] = pres
    eind = _compound(pres)
    print(f"  V3 ref: {(eind-START_CAP)/START_CAP*100:>+.1f}%  €{eind:.0f}")

    # B&H benchmarks
    print("\n  B&H benchmarks …")
    for bh_key, tickers, label in [
        ("bh_2x",  TICKERS,       "DOGE+SOL"),
        ("bh_btc", ["BTC-USD"],   "BTC"),
    ]:
        bh_all = {**frames, "BTC-USD": btc_df}
        pres = {}; cap = START_CAP
        for pname, pstart, pend in PERIODS:
            r = run_bh(bh_all, tickers, pstart, pend, start_cap=cap)
            pres[pname] = r
            cap = r["final_cap"]
        all_results[bh_key] = pres
        eind = _compound(pres)
        print(f"  B&H {label:<16} →  {(eind-START_CAP)/START_CAP*100:>+.1f}%  €{eind:.0f}")

    # Slechts-moment analyse
    print("\n  Slechtste instapmoment analyse …")
    worst = worst_entry_analysis(frames, thresholds)
    all_results["worst_entry"] = worst

    # Tabellen
    print_vergelijking(all_results)
    print_periode_detail(all_results)
    print_worst_entry(worst)

    # Opslaan
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

    OUTPUT_FILE.write_text(json.dumps(_clean(all_results), indent=2, ensure_ascii=False))
    print(f"\n  Resultaten opgeslagen: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
