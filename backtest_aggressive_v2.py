"""
Agressieve Varianten Backtest — Hybride V4+MR
==============================================
Test 7 agressieve varianten van het baseline hybride V4+MR portfolio systeem.
Elke variant gebruikt andere positiegrootte, RSI-drempel, MA200-filter en/of
tickers-selectie. Baseline = identiek aan backtest_portfolio.py.

Stack: Python 3.11, yfinance, ta, pandas. Geen matplotlib, geen scikit-learn.
"""

import json, warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

from ta.momentum   import RSIIndicator
from ta.trend      import MACD as TaMACD
from ta.volatility import AverageTrueRange, BollingerBands

# ── Constanten ────────────────────────────────────────────────────────────────
ALL_TICKERS  = ["DOGE-USD", "SOL-USD", "ETH-USD", "AVAX-USD", "ADA-USD"]
CONC_TICKERS = ["DOGE-USD", "SOL-USD"]

START_CAP = 500.0
FEE_RT    = 0.004
MIN_POS   = 5.0

V4_ATR_ENTRY      = 4.0
V4_ATR_TRAIL      = 3.0
V4_MA200_MAX_BELOW = 0.15   # alleen actief als ma200_filter=True

SDBF_RSI_LO = 50; SDBF_RSI_HI = 70
SDBF_ATR    = 2.0
MR_RSI_THR  = 35
MR_BB_WIN   = 30; MR_BB_SIG = 2.5
MR_ATR      = 2.0

ATR_WIN   = 14
MA50_WIN  = 50
MA200_WIN = 200
MA200_LAG = 10
VOL_WIN   = 20
MACD_F, MACD_S, MACD_SIG = 12, 26, 9

EXPO_BULL_NEUT = 0.70
EXPO_BEAR      = 0.40

TRAIN_START = "2024-01-01"; TRAIN_END  = "2024-12-31"
VAL_START   = "2025-01-01"; VAL_END    = "2025-09-30"
FINAL_START = "2025-10-01"; FINAL_END  = "2026-06-17"
PERIODS = [
    ("2024",     TRAIN_START, TRAIN_END),
    ("Val 2025", VAL_START,   VAL_END),
    ("Finale",   FINAL_START, FINAL_END),
]

OUTPUT_FILE = Path(__file__).parent / "results_aggressive_v2.json"


# ── Variant configuratie ──────────────────────────────────────────────────────
@dataclass
class VariantCfg:
    name:         str
    label:        str
    tickers:      List[str]
    v4_sizes:     Dict[int, float]   # strength 0-4 → fractie van kapitaal
    rsi_lo:       int
    rsi_hi:       int
    ma200_filter: bool               # True = close >= ma200*0.85 vereist
    max_open:     int


BASELINE_SIZES = {0: 0.07, 1: 0.07, 2: 0.10, 3: 0.15, 4: 0.15}
V1_SIZES       = {0: 0.20, 1: 0.20, 2: 0.30, 3: 0.50, 4: 0.50}
V3_SIZES       = {0: 0.25, 1: 0.25, 2: 0.40, 3: 0.60, 4: 0.60}

VARIANTS: List[VariantCfg] = [
    VariantCfg("baseline", "Baseline",       ALL_TICKERS,  BASELINE_SIZES, 45, 75, True,  5),
    VariantCfg("v1",       "V1: Grote pos",  ALL_TICKERS,  V1_SIZES,       45, 75, True,  5),
    VariantCfg("v2",       "V2: Min filters",ALL_TICKERS,  BASELINE_SIZES, 40, 80, False, 5),
    VariantCfg("v3",       "V3: DOGE+SOL",   CONC_TICKERS, V3_SIZES,       45, 75, True,  2),
    VariantCfg("v4",       "V4: V1+V2",      ALL_TICKERS,  V1_SIZES,       40, 80, False, 5),
    VariantCfg("v5",       "V5: V1+V3",      CONC_TICKERS, V3_SIZES,       45, 75, True,  2),
    VariantCfg("v6",       "V6: V2+V3",      CONC_TICKERS, V3_SIZES,       40, 80, False, 2),
    VariantCfg("v7",       "V7: V1+V2+V3",   CONC_TICKERS, V3_SIZES,       40, 80, False, 2),
]


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
    df["bb_mr_lower"] = bb.bollinger_lband()
    df["bb_mr_mid"]   = bb.bollinger_mavg()
    df["bb_mr_std"]   = df["Close"].rolling(MR_BB_WIN).std()
    return df.dropna()


def slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


# ── Regime ────────────────────────────────────────────────────────────────────
def regime_score_row(row) -> float:
    ma200     = float(row["ma200"]); ma200_lag = float(row["ma200_lag"])
    close     = float(row["Close"]); rsi       = float(row["rsi"])
    ma200_s = (ma200 - ma200_lag) / ma200_lag if ma200_lag != 0 else 0.0
    prijs_s = (close - ma200)     / ma200     if ma200     != 0 else 0.0
    rsi_s   = (rsi   - 50)        / 50
    return float(np.clip((ma200_s + prijs_s + rsi_s) / 3, -1.0, 1.0))


def calibrate_single(df: pd.DataFrame) -> tuple:
    sl = slice_window(df, TRAIN_START, TRAIN_END)
    scores = sl.apply(regime_score_row, axis=1).dropna().values
    if len(scores) == 0:
        return 0.0, 0.0
    return float(np.percentile(scores, 33)), float(np.percentile(scores, 67))


def get_regime(score: float, p33: float, p67: float) -> str:
    if score > p67: return "bull"
    if score < p33: return "bear"
    return "neutraal"


def portfolio_regime(ticker_regimes: list) -> str:
    counts = {"bull": 0, "neutraal": 0, "bear": 0}
    for r in ticker_regimes:
        counts[r] = counts.get(r, 0) + 1
    n = len(ticker_regimes)
    if counts["bear"] > n / 2:  return "bear"
    if counts["bull"] > n / 2:  return "bull"
    return "neutraal"


# ── Signalen ──────────────────────────────────────────────────────────────────
def v4_entry(row, cfg: VariantCfg) -> bool:
    rsi   = float(row["rsi"])
    close = float(row["Close"])
    ma200 = float(row["ma200"])
    macd_cross = (float(row["macd"]) > float(row["macd_sig"])
                  and float(row["prev_hist"]) <= 0)
    vol_ok = float(row["Volume"]) > float(row["vol_ma"])
    rsi_ok = cfg.rsi_lo <= rsi <= cfg.rsi_hi
    ma200_ok = (close >= ma200 * (1 - V4_MA200_MAX_BELOW)) if cfg.ma200_filter else True
    return rsi_ok and macd_cross and vol_ok and ma200_ok


def v4_strength(row) -> int:
    s = 0
    rsi = float(row["rsi"])
    if 55 <= rsi <= 70:                                                           s += 1
    if float(row["macd_hist"]) > 0 and float(row["macd_hist"]) > float(row["prev_hist"]): s += 1
    if float(row["vol_ratio"]) > 1.5:                                             s += 1
    if float(row["Close"]) > float(row["ma200"]) * 1.02:                          s += 1
    return s


def sdbf_entry(row) -> bool:
    return (SDBF_RSI_LO <= float(row["rsi"]) <= SDBF_RSI_HI
            and float(row["macd"]) > float(row["macd_sig"])
            and float(row["prev_hist"]) <= 0
            and float(row["Volume"]) > float(row["vol_ma"])
            and float(row["Close"]) > float(row["ma50"])
            and float(row["ma200"]) > float(row["ma200_lag"]))


def mr_entry(row) -> bool:
    close = float(row["Close"])
    return (float(row["rsi"]) < MR_RSI_THR
            and close < (float(row["bb_mr_lower"]) - float(row["bb_mr_std"]))
            and float(row["Volume"]) > float(row["vol_ma"]))


def mr_size(row) -> float:
    return 0.10 if float(row["rsi"]) < 20 else 0.07


def sdbf_exit(row, stop_px: float) -> bool:
    return float(row["Close"]) <= stop_px or float(row["macd"]) < float(row["macd_sig"])


def mr_exit(row, stop_px: float) -> bool:
    return (float(row["rsi"]) > 50
            or float(row["Close"]) >= float(row["bb_mr_mid"])
            or float(row["Close"]) <= stop_px)


# ── Positieklasse ─────────────────────────────────────────────────────────────
class Pos:
    __slots__ = ["ticker","strategy","units","entry_px","stop_px",
                 "trail_high","invest","entry_ts","strength","regime_at_entry"]

    def __init__(self, ticker, strategy, units, entry_px, stop_px,
                 invest, entry_ts, strength, regime_at_entry):
        self.ticker          = ticker
        self.strategy        = strategy
        self.units           = units
        self.entry_px        = entry_px
        self.stop_px         = stop_px
        self.trail_high      = entry_px
        self.invest          = invest
        self.entry_ts        = entry_ts
        self.strength        = strength
        self.regime_at_entry = regime_at_entry

    def update_trail(self, close: float, atr: float):
        if self.strategy == "v4" and close > self.trail_high:
            self.trail_high = close
            new_stop = close - V4_ATR_TRAIL * atr
            if new_stop > self.stop_px:
                self.stop_px = new_stop

    def mark_value(self, close: float) -> float:
        return self.units * close


def _close_pos(p: Pos, close: float):
    proceeds  = p.units * close
    sell_fee  = proceeds * FEE_RT / 2
    buy_fee   = p.invest * FEE_RT / 2
    pnl       = proceeds - p.units * p.entry_px - sell_fee - buy_fee
    cap_delta = proceeds - sell_fee
    return pnl, cap_delta


def _trade_rec(p: Pos, exit_ts, exit_px: float, pnl: float,
               exit_reason: str, regime_now: str) -> dict:
    return {
        "ticker":          p.ticker,
        "strategy":        p.strategy,
        "entry_ts":        str(p.entry_ts),
        "exit_ts":         str(exit_ts),
        "entry_px":        p.entry_px,
        "exit_px":         exit_px,
        "pnl":             pnl,
        "invest":          p.invest,
        "win":             pnl > 0,
        "strength":        p.strength,
        "exit_reason":     exit_reason,
        "regime_at_entry": p.regime_at_entry,
        "regime_at_exit":  regime_now,
    }


# ── Portfolio backtest (per variant) ─────────────────────────────────────────
def run_portfolio(frames: dict, thresholds: dict,
                  start: str, end: str,
                  cfg: VariantCfg,
                  start_cap: float = START_CAP) -> dict:
    slices = {t: slice_window(frames[t], start, end) for t in cfg.tickers}

    idx = None
    for df in slices.values():
        idx = df.index if idx is None else idx.union(df.index)
    idx = idx.sort_values()

    capital      = start_cap
    positions    = []
    trades       = []
    equity       = []
    regime_log   = []
    pos_cnt_log  = []
    last_regime  = "neutraal"

    for ts in idx:
        ticker_regimes = {}
        for t in cfg.tickers:
            if ts in slices[t].index:
                score = regime_score_row(slices[t].loc[ts])
                p33   = thresholds[t]["p33"]
                p67   = thresholds[t]["p67"]
                ticker_regimes[t] = get_regime(score, p33, p67)

        port_regime = portfolio_regime(list(ticker_regimes.values())) \
                      if ticker_regimes else "neutraal"
        regime_log.append(port_regime)
        last_regime = port_regime

        total_val = capital
        for p in positions:
            c = (float(slices[p.ticker].loc[ts, "Close"])
                 if ts in slices[p.ticker].index else p.entry_px)
            total_val += p.mark_value(c)

        for t in cfg.tickers:
            if ts not in slices[t].index:
                continue
            row   = slices[t].loc[ts]
            close = float(row["Close"])
            atr   = float(row["atr"])
            t_regime = ticker_regimes.get(t, "neutraal")

            # Exits
            to_close = []
            for p in positions:
                if p.ticker != t:
                    continue
                p.update_trail(close, atr)
                if p.strategy == "v4":
                    should_exit = close <= p.stop_px; reason = "trail"
                elif p.strategy == "sdbf":
                    should_exit = sdbf_exit(row, p.stop_px)
                    reason = "atr" if close <= p.stop_px else "macd"
                else:
                    should_exit = mr_exit(row, p.stop_px)
                    reason = "atr" if close <= p.stop_px else "signal"
                if should_exit:
                    pnl, cd = _close_pos(p, close)
                    capital   += cd
                    total_val  = total_val - p.mark_value(close) + cd
                    trades.append(_trade_rec(p, ts, close, pnl, reason, t_regime))
                    to_close.append(p)
            positions = [p for p in positions if p not in to_close]

            # Entries
            if len(positions) >= cfg.max_open:
                continue
            if any(p.ticker == t for p in positions):
                continue

            max_expo  = EXPO_BULL_NEUT if port_regime in ("bull","neutraal") else EXPO_BEAR
            invested  = sum(p.invest for p in positions)
            expo_frac = invested / total_val if total_val > 0 else 0.0
            if expo_frac >= max_expo:
                continue

            if t_regime in ("bull", "neutraal"):
                if not v4_entry(row, cfg):
                    continue
                s      = v4_strength(row)
                frac   = cfg.v4_sizes[s]
                invest = capital * frac
                if invest < MIN_POS:
                    continue
                fee   = invest * FEE_RT / 2
                units = (invest - fee) / close
                stop  = close - V4_ATR_ENTRY * atr
                capital -= invest
                positions.append(Pos(t, "v4", units, close, stop,
                                     invest, ts, s, t_regime))
            else:  # bear
                if sdbf_entry(row):
                    invest = capital * 0.07
                    if invest >= MIN_POS:
                        fee   = invest * FEE_RT / 2
                        units = (invest - fee) / close
                        stop  = close - SDBF_ATR * atr
                        capital -= invest
                        positions.append(Pos(t, "sdbf", units, close, stop,
                                             invest, ts, 0, t_regime))
                elif mr_entry(row):
                    frac   = mr_size(row)
                    invest = capital * frac
                    if invest >= MIN_POS:
                        fee   = invest * FEE_RT / 2
                        units = (invest - fee) / close
                        stop  = close - MR_ATR * atr
                        capital -= invest
                        positions.append(Pos(t, "mr", units, close, stop,
                                             invest, ts, 0, t_regime))

        port = capital
        for p in positions:
            c = (float(slices[p.ticker].loc[ts, "Close"])
                 if ts in slices[p.ticker].index else p.entry_px)
            port += p.mark_value(c)
        equity.append(float(port))
        pos_cnt_log.append(len(positions))

    for p in positions:
        df_t = slices[p.ticker]
        lc   = float(df_t["Close"].iloc[-1])
        lts  = df_t.index[-1]
        pnl, cd = _close_pos(p, lc)
        capital += cd
        trades.append(_trade_rec(p, lts, lc, pnl, "eod", last_regime))

    return _calc_stats(trades, equity, capital, start_cap, regime_log, pos_cnt_log, start, end)


def _calc_stats(trades, equity, final_cap, start_cap,
                regime_log, pos_cnt_log, start, end):
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

    rl    = np.array(regime_log) if regime_log else np.array([])
    n_reg = len(rl) or 1
    pcl   = np.array(pos_cnt_log) if pos_cnt_log else np.array([0])

    monthly = {}; monthly_n = {}
    for t in trades:
        m = str(t["exit_ts"])[:7]
        monthly[m]   = monthly.get(m, 0.0) + t["pnl"]
        monthly_n[m] = monthly_n.get(m, 0) + 1

    best_m  = max(monthly.items(), key=lambda x: x[1]) if monthly else ("—", 0.0)
    worst_m = min(monthly.items(), key=lambda x: x[1]) if monthly else ("—", 0.0)

    # Trades per maand (gem over actieve maanden)
    n_months = max(
        (pd.Timestamp(end) - pd.Timestamp(start)).days / 30.44, 1
    )
    tpm = n / n_months

    return {
        "final_cap":       final_cap,
        "total_ret":       total_ret,
        "n_trades":        n,
        "trades_per_month": round(tpm, 1),
        "win_rate":        wr,
        "profit_factor":   pf,
        "max_dd":          float(dd.max()),
        "max_loss_streak": max_streak,
        "best_month":      best_m,
        "worst_month":     worst_m,
        "regime_bull_pct": float((rl == "bull").sum()     / n_reg * 100),
        "regime_neut_pct": float((rl == "neutraal").sum() / n_reg * 100),
        "regime_bear_pct": float((rl == "bear").sum()     / n_reg * 100),
        "avg_open_pos":    float(pcl.mean()),
        "max_open_obs":    int(pcl.max()),
        "monthly":         monthly,
        "monthly_n":       monthly_n,
        "trades":          trades,
        "equity":          equity,
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


def _compound_eind(pres: dict, cap: float = START_CAP) -> float:
    r24 = pres["2024"]["total_ret"]
    rv  = pres["Val 2025"]["total_ret"]
    rf  = pres["Finale"]["total_ret"]
    return cap * (1 + r24/100) * (1 + rv/100) * (1 + rf/100)


def _max_dd_all(pres: dict) -> float:
    return max(pres["2024"]["max_dd"], pres["Val 2025"]["max_dd"], pres["Finale"]["max_dd"])


def _totaal_trades(pres: dict) -> int:
    return sum(pres[p]["n_trades"] for p in ["2024","Val 2025","Finale"])


def _gewogen_winrate(pres: dict) -> float:
    n_tot = _totaal_trades(pres) or 1
    return sum(pres[p]["win_rate"] * pres[p]["n_trades"]
               for p in ["2024","Val 2025","Finale"]) / n_tot


def _gewogen_pf(pres: dict):
    vals = [pres[p]["profit_factor"] for p in ["2024","Val 2025","Finale"]
            if pres[p]["profit_factor"] not in (0, float("inf"))]
    return float(np.mean(vals)) if vals else float("inf")


def print_vergelijking(all_results: dict):
    W = 145
    print(f"\n{'═'*W}")
    print("  HOOFDVERGELIJKING — €500 startkapitaal, compound over drie periodes")
    print(f"{'═'*W}")
    hdr = (f"  {'Variant':<24} {'2024':>8} {'Val2025':>8} {'Finale':>8} {'Totaal':>8} "
           f"{'MaxDD':>7} {'Trades':>7} {'T/mnd':>6} {'Win%':>7} {'PF':>7} {'Eind€':>7} "
           f"{'MaxVerlies':>11}")
    print(hdr)
    print(f"  {'─'*24} {'─'*8} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*7} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*11}")

    for cfg in VARIANTS:
        pres  = all_results[cfg.name]
        r24   = pres["2024"]; rv = pres["Val 2025"]; rf = pres["Finale"]
        eind  = _compound_eind(pres)
        tot   = (eind - START_CAP) / START_CAP * 100
        mdd   = _max_dd_all(pres)
        ntot  = _totaal_trades(pres)
        tpm   = round(ntot / 30, 1)   # ruwe totaal / periodes in maanden (30)
        wr    = _gewogen_winrate(pres)
        pf    = _gewogen_pf(pres)
        mls   = max(pres[p]["max_loss_streak"] for p in ["2024","Val 2025","Finale"])
        print(f"  {cfg.label:<24} {r24['total_ret']:>+7.1f}% {rv['total_ret']:>+7.1f}% "
              f"{rf['total_ret']:>+7.1f}% {tot:>+7.1f}% "
              f"{mdd:>6.1f}% {ntot:>7} {tpm:>6.1f} {wr:>6.1f}% {_pfs(pf):>7} "
              f"{eind:>6.0f}€ {mls:>11}")

    print(f"  {'─'*24} {'─'*8} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*7} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*11}")

    for label, bh in [("B&H 5x gelijk gew", "bh_5x"),
                      ("B&H DOGE+SOL",       "bh_2x"),
                      ("B&H BTC",             "bh_btc")]:
        pres = all_results[bh]
        r24  = pres["2024"]; rv = pres["Val 2025"]; rf = pres["Finale"]
        eind = _compound_eind(pres)
        tot  = (eind - START_CAP) / START_CAP * 100
        mdd  = _max_dd_all(pres)
        print(f"  {label:<24} {r24['total_ret']:>+7.1f}% {rv['total_ret']:>+7.1f}% "
              f"{rf['total_ret']:>+7.1f}% {tot:>+7.1f}% {mdd:>6.1f}% "
              f"{'—':>7} {'—':>6} {'—':>7} {'—':>7} {eind:>6.0f}€ {'—':>11}")
    print(f"{'═'*W}")


def print_variant_detail(all_results: dict):
    W = 100
    for cfg in VARIANTS:
        pres = all_results[cfg.name]
        print(f"\n{'═'*W}")
        print(f"  DETAIL: {cfg.label}")
        print(f"  Tickers: {', '.join(t.replace('-USD','') for t in cfg.tickers)}  "
              f"| RSI {cfg.rsi_lo}-{cfg.rsi_hi}  "
              f"| MA200-filter: {'ja' if cfg.ma200_filter else 'nee'}  "
              f"| Max pos: {cfg.max_open}")
        sizes = f"{cfg.v4_sizes[0]*100:.0f}%/{cfg.v4_sizes[2]*100:.0f}%/{cfg.v4_sizes[3]*100:.0f}%"
        print(f"  V4 sizes: {sizes} (zwak/normaal/sterk)")
        print(f"{'─'*W}")
        print(f"  {'Periode':<12} {'Ret':>8} {'MaxDD':>7} {'Trades':>8} "
              f"{'T/mnd':>6} {'Win%':>7} {'PF':>7} {'MaxVerlies':>11} "
              f"{'BesteM':>15} {'SlechtsteM':>15}")
        print(f"  {'─'*12} {'─'*8} {'─'*7} {'─'*8} "
              f"{'─'*6} {'─'*7} {'─'*7} {'─'*11} {'─'*15} {'─'*15}")
        for pname in ["2024","Val 2025","Finale"]:
            r  = pres[pname]
            bm = r["best_month"]; wm = r["worst_month"]
            bm_str = f"{bm[0]} (+{bm[1]:.0f}€)" if bm[0] != "—" else "—"
            wm_str = f"{wm[0]} ({wm[1]:.0f}€)"  if wm[0] != "—" else "—"
            print(f"  {pname:<12} {r['total_ret']:>+7.1f}% {r['max_dd']:>6.1f}% "
                  f"{r['n_trades']:>8} {r['trades_per_month']:>6.1f} "
                  f"{r['win_rate']:>6.1f}% {_pfs(r['profit_factor']):>7} "
                  f"{r['max_loss_streak']:>11} {bm_str:>15} {wm_str:>15}")
        eind = _compound_eind(pres)
        tot  = (eind - START_CAP) / START_CAP * 100
        print(f"  {'─'*12}")
        print(f"  {'Compound totaal':<12} {tot:>+7.1f}%  →  {eind:.0f}€")
    print(f"\n{'═'*W}")


def print_risico_profiel(all_results: dict):
    W = 100
    print(f"\n{'═'*W}")
    print("  RISICO PROFIEL VERGELIJKING")
    print(f"{'═'*W}")
    print(f"  {'Variant':<24} {'Totaal ret':>11} {'MaxDD':>7} {'Win%':>7} {'PF':>7} "
          f"{'MaxVerlies':>11} {'Eind€':>7} {'Risico':>8}")
    print(f"  {'─'*24} {'─'*11} {'─'*7} {'─'*7} {'─'*7} {'─'*11} {'─'*7} {'─'*8}")

    rows = []
    for cfg in VARIANTS:
        pres = all_results[cfg.name]
        eind = _compound_eind(pres)
        tot  = (eind - START_CAP) / START_CAP * 100
        mdd  = _max_dd_all(pres)
        wr   = _gewogen_winrate(pres)
        pf   = _gewogen_pf(pres)
        mls  = max(pres[p]["max_loss_streak"] for p in ["2024","Val 2025","Finale"])
        rows.append((cfg.label, tot, mdd, wr, pf, mls, eind))

    for label, tot, mdd, wr, pf, mls, eind in rows:
        # Risicoscore: MaxDD > 40% = hoog, 25-40% = middel, <25% = laag
        risico = "HOOG" if mdd > 40 else ("MIDDEL" if mdd > 25 else "LAAG")
        print(f"  {label:<24} {tot:>+10.1f}% {mdd:>6.1f}% {wr:>6.1f}% {_pfs(pf):>7} "
              f"{mls:>11} {eind:>6.0f}€ {risico:>8}")
    print(f"{'═'*W}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("  Agressieve Varianten Backtest — Hybride V4+MR (7 varianten + baseline)")
    print("=" * 80)

    # Data ophalen (alle 5 tickers nodig, ook voor concentrated variants)
    print("\n  Data ophalen …")
    frames     = {}
    thresholds = {}
    for t in ALL_TICKERS:
        print(f"  {t:<12} …", end=" ", flush=True)
        df = add_indicators(fetch_4h(t))
        frames[t] = df
        p33, p67 = calibrate_single(df)
        thresholds[t] = {"p33": p33, "p67": p67}
        print(f"{len(df):>6} bars  |  bear < {p33:+.4f}  bull > {p67:+.4f}")

    # BTC data voor B&H vergelijking
    print(f"  {'BTC-USD':<12} …", end=" ", flush=True)
    btc_df = add_indicators(fetch_4h("BTC-USD"))
    print(f"{len(btc_df):>6} bars")

    # Backtest alle varianten × alle periodes
    all_results = {}
    total_runs  = len(VARIANTS) * len(PERIODS)
    run_nr      = 0

    for cfg in VARIANTS:
        print(f"\n  ── {cfg.label} ({'×'.join(t.replace('-USD','') for t in cfg.tickers)},"
              f" RSI {cfg.rsi_lo}-{cfg.rsi_hi},"
              f" MA200={'ja' if cfg.ma200_filter else 'nee'},"
              f" sizes={cfg.v4_sizes[0]*100:.0f}/{cfg.v4_sizes[2]*100:.0f}/{cfg.v4_sizes[3]*100:.0f}%,"
              f" max {cfg.max_open} pos)")
        pres = {}
        cap  = START_CAP
        for pname, pstart, pend in PERIODS:
            run_nr += 1
            print(f"  [{run_nr}/{total_runs}] {pname} …", end=" ", flush=True)
            r = run_portfolio(frames, thresholds, pstart, pend, cfg, start_cap=cap)
            pres[pname] = r
            cap = r["final_cap"]   # compound
            print(f"{r['n_trades']:>4} trades  {r['total_ret']:>+.1f}%  "
                  f"MaxDD {r['max_dd']:.1f}%  €{r['final_cap']:.0f}")
        all_results[cfg.name] = pres

    # B&H benchmarks
    print("\n  B&H benchmarks …")
    for bh_key, tickers, label in [
        ("bh_5x",  ALL_TICKERS,  "5x gelijk gewogen"),
        ("bh_2x",  CONC_TICKERS, "DOGE+SOL"),
        ("bh_btc", ["BTC-USD"],  "BTC"),
    ]:
        bh_frames = {**frames, "BTC-USD": btc_df}
        pres = {}
        cap  = START_CAP
        for pname, pstart, pend in PERIODS:
            r = run_bh(bh_frames, tickers, pstart, pend, start_cap=cap)
            pres[pname] = r
            cap = r["final_cap"]
        all_results[bh_key] = pres
        eind = _compound_eind(pres)
        print(f"  B&H {label:<20} →  {(eind-START_CAP)/START_CAP*100:>+.1f}%  €{eind:.0f}")

    # Tabellen
    print_vergelijking(all_results)
    print_variant_detail(all_results)
    print_risico_profiel(all_results)

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

    save = {
        "variants": [
            {
                "name":  cfg.name,
                "label": cfg.label,
                "tickers": cfg.tickers,
                "rsi_lo":  cfg.rsi_lo,
                "rsi_hi":  cfg.rsi_hi,
                "ma200_filter": cfg.ma200_filter,
                "max_open":     cfg.max_open,
                "v4_sizes":     {str(k): v for k, v in cfg.v4_sizes.items()},
            }
            for cfg in VARIANTS
        ],
        "results":    _clean(all_results),
        "thresholds": thresholds,
    }
    OUTPUT_FILE.write_text(json.dumps(save, indent=2, ensure_ascii=False))
    print(f"\n  Resultaten opgeslagen: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
