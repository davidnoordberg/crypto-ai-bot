"""
Vier-modus V2 — Snellere B&H Exit Triggers
===========================================
Verbetert de vier-modus strategie met drie vroege exit triggers voor B&H posities:

  Trigger 1 (MA50)     : prijs kruist onder MA50
  Trigger 2 (ATR)      : ATR > 2× 20-periode gemiddelde (extreme volatiliteit)
  Trigger 3 (VolDown)  : rode candle >2% EN volume > 2× gemiddelde

Drie varianten:
  A : alleen MA50
  B : MA50 + ATR spike
  C : MA50 + ATR spike + volume spike down

Na trigger-exit: herinstap in B&H pas na regime-herstel
(regime moet eerst NIET sterke_bull worden, daarna weer sterke_bull).

Stack: Python 3.11, yfinance, ta, pandas. Geen matplotlib, geen scikit-learn.
"""

import json, warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from ta.momentum   import RSIIndicator
from ta.trend      import MACD as TaMACD
from ta.volatility import AverageTrueRange, BollingerBands

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS   = ["DOGE-USD", "SOL-USD", "ETH-USD", "AVAX-USD", "ADA-USD"]
START_CAP = 500.0
FEE_RT    = 0.004
MIN_POS   = 5.0

STERKE_BULL = "sterke_bull"
LICHTE_BULL = "lichte_bull"
NEUTRAAL    = "neutraal"
BEAR        = "bear"
REGIME_ORDER = [BEAR, NEUTRAAL, LICHTE_BULL, STERKE_BULL]

EXPO_CAP = {STERKE_BULL: 0.80, LICHTE_BULL: 0.60, NEUTRAAL: 0.40, BEAR: 0.30}

BH_FRAC        = 0.80
V4_SIZES       = {0: 0.25, 1: 0.25, 2: 0.40, 3: 0.60, 4: 0.60}
V4_ATR_ENTRY   = 4.0
V4_ATR_TRAIL   = 3.0
V4_RSI_LO      = 45;  V4_RSI_HI = 75
V4_MA200_BELOW = 0.15
V4C_SIZES      = {0: 0.15, 1: 0.15, 2: 0.20, 3: 0.30, 4: 0.30}
V4C_ATR_TRAIL  = 2.0
MR_RSI_THR     = 35
MR_BB_WIN      = 30;  MR_BB_SIG = 2.5
MR_ATR         = 2.0
MR_FRAC        = {True: 0.20, False: 0.10}

# B&H trigger parameters
MA50_WIN       = 50
ATR_SPIKE_MULT = 2.0    # ATR > N × 20-bar ATR gemiddelde
ATR_AVG_WIN    = 20
VOL_SPIKE_MULT = 2.0    # volume > N × vol_ma
VOL_RED_THR    = 0.02   # candle body > 2% van open prijs

ATR_WIN = 14; MA200_WIN = 200; MA200_LAG = 10; VOL_WIN = 20
MACD_F, MACD_S, MACD_SIG = 12, 26, 9

TRAIN_START = "2024-01-01"; TRAIN_END  = "2024-12-31"
VAL_START   = "2025-01-01"; VAL_END    = "2025-09-30"
FINAL_START = "2025-10-01"; FINAL_END  = "2026-06-17"
PERIODS     = [("2024", TRAIN_START, TRAIN_END),
               ("Val 2025", VAL_START, VAL_END),
               ("Finale", FINAL_START, FINAL_END)]

OUTPUT_FILE = Path(__file__).parent / "results_fourmode_v2.json"

# Look-ahead venster voor trigger-effectiviteit meting (in bars)
LOOKAHEAD_BARS = 20


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
                         "Open": float(row["Open"]), "High": float(row["High"]),
                         "Low":  float(row["Low"]),  "Close": float(row["Close"]),
                         "Volume": float(row["Volume"]) / 6})
    warmup = pd.DataFrame(rows).set_index("_ts")
    warmup.index = pd.DatetimeIndex(warmup.index)
    warmup = warmup[warmup.index < df_1h.index.min()]
    df = pd.concat([warmup, df_1h]).sort_index()
    return df[~df.index.duplicated(keep="last")]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma50"]       = df["Close"].rolling(MA50_WIN).mean()
    df["prev_ma50"]  = df["ma50"].shift(1)
    df["prev_close"] = df["Close"].shift(1)
    df["ma200"]      = df["Close"].rolling(MA200_WIN).mean()
    df["ma200_lag"]  = df["ma200"].shift(MA200_LAG)
    df["rsi"]        = RSIIndicator(df["Close"], window=14).rsi()
    _m = TaMACD(df["Close"], window_fast=MACD_F, window_slow=MACD_S, window_sign=MACD_SIG)
    df["macd"]       = _m.macd()
    df["macd_sig"]   = _m.macd_signal()
    df["macd_hist"]  = df["macd"] - df["macd_sig"]
    df["prev_hist"]  = df["macd_hist"].shift(1)
    df["vol_ma"]     = df["Volume"].rolling(VOL_WIN).mean()
    df["vol_ratio"]  = df["Volume"] / df["vol_ma"]
    df["atr"]        = AverageTrueRange(df["High"], df["Low"], df["Close"],
                                        window=ATR_WIN).average_true_range()
    df["atr_avg"]    = df["atr"].rolling(ATR_AVG_WIN).mean()   # voor ATR spike
    bb = BollingerBands(df["Close"], window=MR_BB_WIN, window_dev=MR_BB_SIG)
    df["bb_lower"]   = bb.bollinger_lband()
    df["bb_mid"]     = bb.bollinger_mavg()
    df["bb_std"]     = df["Close"].rolling(MR_BB_WIN).std()
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


def calibrate_4level(df: pd.DataFrame) -> Tuple[float, float, float]:
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
    counts = {r: 0 for r in REGIME_ORDER}
    for r in regime_map.values():
        counts[r] += 1
    best_n = max(counts.values())
    for r in REGIME_ORDER:
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
    if 55 <= float(row["rsi"]) <= 70:                                          s += 1
    if float(row["macd_hist"]) > float(row["prev_hist"]) > 0:                  s += 1
    if float(row["vol_ratio"]) > 1.5:                                          s += 1
    if float(row["Close"]) > float(row["ma200"]) * 1.02:                       s += 1
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


# ── B&H vroege exit triggers ──────────────────────────────────────────────────
def bh_trigger_ma50(row) -> bool:
    """Prijs kruist onder MA50 (was erboven vorige bar)."""
    close      = float(row["Close"])
    ma50       = float(row["ma50"])
    prev_close = float(row["prev_close"])
    prev_ma50  = float(row["prev_ma50"])
    return close < ma50 and prev_close >= prev_ma50


def bh_trigger_atr(row) -> bool:
    """ATR > ATR_SPIKE_MULT × 20-periode ATR gemiddelde."""
    atr     = float(row["atr"])
    atr_avg = float(row["atr_avg"])
    return atr_avg > 0 and atr > ATR_SPIKE_MULT * atr_avg


def bh_trigger_vol_down(row) -> bool:
    """Grote rode candle (>2% daling) met volume > VOL_SPIKE_MULT × gemiddelde."""
    open_  = float(row["Open"]); close = float(row["Close"])
    vol    = float(row["Volume"]); vol_ma = float(row["vol_ma"])
    red_pct = (open_ - close) / open_ if open_ > 0 else 0.0
    return (close < open_
            and red_pct > VOL_RED_THR
            and vol > VOL_SPIKE_MULT * vol_ma)


def check_bh_triggers(row, variant: str) -> str:
    """
    Controleer triggers op basis van variant.
    Retourneert triggernaam als actief, anders "".
    """
    if bh_trigger_ma50(row):
        return "ma50"
    if variant in ("B", "C") and bh_trigger_atr(row):
        return "atr_spike"
    if variant == "C" and bh_trigger_vol_down(row):
        return "vol_down"
    return ""


# ── Positieklasse ─────────────────────────────────────────────────────────────
class Pos:
    __slots__ = ["ticker","mode","units","entry_px","stop_px",
                 "trail_high","invest","entry_ts","strength","trail_mult"]

    def __init__(self, ticker, mode, units, entry_px, stop_px,
                 invest, entry_ts, strength=0, trail_mult=0.0):
        self.ticker     = ticker
        self.mode       = mode
        self.units      = units
        self.entry_px   = entry_px
        self.stop_px    = stop_px
        self.trail_high = entry_px
        self.invest     = invest
        self.entry_ts   = entry_ts
        self.strength   = strength
        self.trail_mult = trail_mult

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
    return pnl, proceeds - sell_fee


def _trade_rec(p: Pos, exit_ts, exit_px: float, pnl: float,
               reason: str, trigger: str = "") -> dict:
    return {
        "ticker":        p.ticker,
        "mode":          p.mode,
        "entry_ts":      str(p.entry_ts),
        "exit_ts":       str(exit_ts),
        "entry_px":      p.entry_px,
        "exit_px":       exit_px,
        "pnl":           pnl,
        "invest":        p.invest,
        "win":           pnl > 0,
        "strength":      p.strength,
        "exit_reason":   reason,
        "trigger":       trigger,   # "" | "ma50" | "atr_spike" | "vol_down"
        "pnl_pct":       pnl / p.invest * 100 if p.invest > 0 else 0.0,
    }


# ── Look-ahead analyse (trigger effectiviteit) ─────────────────────────────────
def compute_trigger_stats(trades: list, frames: dict) -> dict:
    """
    Voor elke trigger-exit: bereken wat er daarna gebeurde in LOOKAHEAD_BARS.
      - verlies_voorkomen_pct : hoeveel % de prijs daalde na exit (pos = goed)
      - false_positive         : prijs steeg >2% na exit (onterecht gesloten)
    """
    stats: Dict[str, dict] = {
        "ma50":      {"fires": 0, "verlies_pct": [], "false_pos": 0, "pnl_at_exit": []},
        "atr_spike": {"fires": 0, "verlies_pct": [], "false_pos": 0, "pnl_at_exit": []},
        "vol_down":  {"fires": 0, "verlies_pct": [], "false_pos": 0, "pnl_at_exit": []},
    }
    for tr in trades:
        trig = tr.get("trigger", "")
        if not trig or trig not in stats:
            continue
        s = stats[trig]
        s["fires"] += 1
        s["pnl_at_exit"].append(tr["pnl"])

        # Look-ahead
        ticker   = tr["ticker"]
        exit_ts  = pd.Timestamp(tr["exit_ts"])
        exit_px  = tr["exit_px"]
        df_full  = frames[ticker]
        future   = df_full[df_full.index > exit_ts].head(LOOKAHEAD_BARS)

        if future.empty:
            continue

        future_min = float(future["Close"].min())
        future_max = float(future["Close"].max())

        # Verlies voorkomen: hoeveel % lag de minimumprijs onder exit_px
        drop_pct = (exit_px - future_min) / exit_px * 100
        s["verlies_pct"].append(drop_pct)

        # False positive: prijs steeg meer dan 2% boven exit_px (hadden we kunnen houden)
        if future_max > exit_px * 1.02:
            s["false_pos"] += 1

    # Aggregeer
    result = {}
    for trig, s in stats.items():
        fires  = s["fires"]
        vp     = s["verlies_pct"]
        pnl_ex = s["pnl_at_exit"]
        result[trig] = {
            "fires":                fires,
            "gem_verlies_voorkomen": round(float(np.mean(vp)),  2) if vp  else 0.0,
            "max_verlies_voorkomen": round(float(np.max(vp)),   2) if vp  else 0.0,
            "false_positives":      s["false_pos"],
            "false_pos_pct":        round(s["false_pos"] / fires * 100, 1) if fires else 0.0,
            "gem_pnl_at_exit":      round(float(np.mean(pnl_ex)), 2) if pnl_ex else 0.0,
        }
    return result


# ── Backtest engine ───────────────────────────────────────────────────────────
def run_fourmode_v2(frames: dict, thresholds: dict,
                    start: str, end: str,
                    variant: str,
                    start_cap: float = START_CAP) -> dict:
    """
    variant: "orig" | "A" | "B" | "C"
    "orig" = geen triggers (identiek aan backtest_fourmode.py)
    """
    slices = {t: slice_window(frames[t], start, end) for t in TICKERS}

    idx = None
    for df in slices.values():
        idx = df.index if idx is None else idx.union(df.index)
    idx = idx.sort_values()

    capital    = start_cap
    positions: List[Pos] = []
    trades     = []
    equity     = []
    mode_time  = {"bh": 0, "v4": 0, "v4c": 0, "mr": 0, "flat": 0}

    # Herinstap cooldown per ticker na trigger-exit
    # Twee fasen: False = klaar voor instap, True = wacht tot regime <> sterke_bull
    bh_cooldown: Dict[str, bool] = {t: False for t in TICKERS}

    for ts in idx:
        # ── Regime ────────────────────────────────────────────────────────────
        ticker_regimes: Dict[str, str] = {}
        for t in TICKERS:
            if ts not in slices[t].index:
                ticker_regimes[t] = NEUTRAAL
                continue
            row   = slices[t].loc[ts]
            score = regime_score_row(row)
            thr   = thresholds[t]
            ticker_regimes[t] = get_regime_4(score, thr["p25"], thr["p50"], thr["p75"])

            # Cooldown resetten zodra regime NIET meer sterke_bull is
            if bh_cooldown[t] and ticker_regimes[t] != STERKE_BULL:
                bh_cooldown[t] = False

        port_reg  = portfolio_regime_4(ticker_regimes)
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

            should_exit = False; reason = ""; trigger = ""

            if p.mode == "bh":
                # 1. Regime-gebaseerde exit (neutraal of bear)
                if t_reg in (NEUTRAAL, BEAR):
                    should_exit = True
                    reason      = f"regime_exit:{t_reg}"
                # 2. Vroege trigger exits (alleen als variant != "orig")
                elif variant != "orig":
                    trig = check_bh_triggers(row, variant)
                    if trig:
                        should_exit = True
                        reason      = f"trigger_{trig}"
                        trigger     = trig
                        bh_cooldown[p.ticker] = True   # wacht op regime-herstel

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
                trades.append(_trade_rec(p, ts, close, pnl, reason, trigger))
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
                continue
            if len(positions) >= 5:
                break
            if expo_frac >= max_expo:
                break

            row   = slices[t].loc[ts]
            close = float(row["Close"])
            atr   = float(row["atr"])
            t_reg = ticker_regimes[t]

            if t_reg == STERKE_BULL:
                # B&H: blokkeer als cooldown actief is
                if bh_cooldown[t]:
                    continue
                invest = capital * BH_FRAC
                if invest < MIN_POS:
                    continue
                fee   = invest * FEE_RT / 2
                units = (invest - fee) / close
                capital  -= invest; invested += invest
                expo_frac = invested / total_val if total_val > 0 else 0.0
                positions.append(Pos(t, "bh", units, close, 0.0, invest, ts))

            elif t_reg == LICHTE_BULL:
                if not v4_signal(row):
                    continue
                s      = v4_strength(row)
                invest = capital * V4_SIZES[s]
                if invest < MIN_POS:
                    continue
                fee   = invest * FEE_RT / 2
                units = (invest - fee) / close
                stop  = close - V4_ATR_ENTRY * atr
                capital  -= invest; invested += invest
                expo_frac = invested / total_val if total_val > 0 else 0.0
                positions.append(Pos(t, "v4", units, close, stop, invest, ts, s, V4_ATR_TRAIL))

            elif t_reg == NEUTRAAL:
                if not v4_signal(row):
                    continue
                s      = v4_strength(row)
                invest = capital * V4C_SIZES[s]
                if invest < MIN_POS:
                    continue
                fee   = invest * FEE_RT / 2
                units = (invest - fee) / close
                stop  = close - V4_ATR_ENTRY * atr
                capital  -= invest; invested += invest
                expo_frac = invested / total_val if total_val > 0 else 0.0
                positions.append(Pos(t, "v4c", units, close, stop, invest, ts, s, V4C_ATR_TRAIL))

            else:  # BEAR
                if not mr_signal(row):
                    continue
                deep   = float(row["rsi"]) < 20
                invest = capital * MR_FRAC[deep]
                if invest < MIN_POS:
                    continue
                fee   = invest * FEE_RT / 2
                units = (invest - fee) / close
                stop  = close - MR_ATR * atr
                capital  -= invest; invested += invest
                expo_frac = invested / total_val if total_val > 0 else 0.0
                positions.append(Pos(t, "mr", units, close, stop, invest, ts))

        # Mode-tijd logging
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

    return _calc_stats(trades, equity, capital, start_cap, mode_time, start, end)


def _calc_stats(trades, equity, final_cap, start_cap, mode_time, start, end) -> dict:
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

    mode_pnl  = {"bh": 0.0, "v4": 0.0, "v4c": 0.0, "mr": 0.0}
    mode_n    = {"bh": 0,   "v4": 0,   "v4c": 0,   "mr": 0}
    mode_wins = {"bh": 0,   "v4": 0,   "v4c": 0,   "mr": 0}
    for t in trades:
        m = t["mode"]
        mode_pnl[m]  = mode_pnl.get(m, 0.0) + t["pnl"]
        mode_n[m]    = mode_n.get(m, 0) + 1
        if t["win"]:
            mode_wins[m] = mode_wins.get(m, 0) + 1

    total_bars = sum(mode_time.values()) or 1
    mode_pct   = {m: round(v / total_bars * 100, 1) for m, v in mode_time.items()}

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


def _totaal_trigger_stats(all_trig: dict) -> dict:
    """Combineer trigger stats over alle periodes."""
    combined: Dict[str, dict] = {}
    for trig in ("ma50", "atr_spike", "vol_down"):
        fires = sum(all_trig[p].get(trig, {}).get("fires", 0) for p in all_trig)
        fp    = sum(all_trig[p].get(trig, {}).get("false_positives", 0) for p in all_trig)
        vp_vals = []
        for p in all_trig:
            v = all_trig[p].get(trig, {})
            if v.get("fires", 0) > 0:
                vp_vals.append(v.get("gem_verlies_voorkomen", 0.0))
        combined[trig] = {
            "fires":                 fires,
            "false_positives":       fp,
            "false_pos_pct":         round(fp / fires * 100, 1) if fires else 0.0,
            "gem_verlies_voorkomen": round(float(np.mean(vp_vals)), 2) if vp_vals else 0.0,
        }
    return combined


def print_hoofdvergelijking(all_results: dict):
    W = 120
    print(f"\n{'═'*W}")
    print("  HOOFDVERGELIJKING — €500 startkapitaal, compound drie periodes")
    print(f"{'═'*W}")
    print(f"  {'Strategie':<32} {'2024':>8} {'Val2025':>8} {'Finale':>8} {'Totaal':>8} "
          f"{'MaxDD':>7} {'Trades':>7} {'Win%':>7} {'PF':>7} {'Eind€':>7}")
    print(f"  {'─'*32} {'─'*8} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    VARIANTEN = [
        ("orig", "Vier-modus origineel"),
        ("A",    "Variant A: MA50"),
        ("B",    "Variant B: MA50+ATR"),
        ("C",    "Variant C: alle drie"),
    ]
    for key, label in VARIANTEN:
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
        print(f"  {label:<32} {r24['total_ret']:>+7.1f}% {rv['total_ret']:>+7.1f}% "
              f"{rf['total_ret']:>+7.1f}% {tot:>+7.1f}% "
              f"{mdd:>6.1f}% {ntot:>7} {wr:>6.1f}% {_pfs(avgpf):>7} {eind:>6.0f}€")

    print(f"  {'─'*32} {'─'*8} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    # Referenties (statische waarden uit vorige backtests)
    refs = [
        ("Hybride V4+MR baseline",    "+43.2%", "+20.1%",  "-9.8%",  "+55.2%", "16.2%", "266", "43.2%", "1.62", "776€"),
        ("All-in C (winner)",         "+86.3%", "+40.5%",  "+7.5%", "+181.4%", "47.4%",  "67", "50.7%", "1.39","1407€"),
        ("B&H BTC",                  "+109.1%", "+22.2%", "-42.5%",  "+46.8%", "52.0%",   "—",     "—",    "—", "734€"),
    ]
    for r in refs:
        print(f"  {r[0]:<32} {r[1]:>8} {r[2]:>8} {r[3]:>8} {r[4]:>8} "
              f"{r[5]:>7} {r[6]:>7} {r[7]:>7} {r[8]:>7} {r[9]:>7}")
    print(f"{'═'*W}")


def print_trigger_analyse(all_results: dict, frames: dict):
    W = 100
    print(f"\n{'═'*W}")
    print("  TRIGGER ANALYSE — effectiviteit per variant")
    print(f"{'═'*W}")

    TRIGGERS = [("ma50","MA50 breuk"), ("atr_spike","ATR spike"), ("vol_down","Vol spike down")]

    for var_key, var_label in [("A","Variant A"), ("B","Variant B"), ("C","Variant C")]:
        pres = all_results[var_key]
        print(f"\n  ── {var_label} ──────────────────────────────────────────────")

        # Bereken trigger stats per periode
        all_trig: Dict[str, dict] = {}
        for pname in ["2024","Val 2025","Finale"]:
            ts_obj = compute_trigger_stats(pres[pname].get("trades",[]), frames)
            all_trig[pname] = ts_obj

        combined = _totaal_trigger_stats(all_trig)

        print(f"  {'Trigger':<20} {'Fires':>7} {'Gem verlies vk%':>16} "
              f"{'False pos':>10} {'False pos%':>11}")
        print(f"  {'─'*20} {'─'*7} {'─'*16} {'─'*10} {'─'*11}")
        for trig_key, trig_label in TRIGGERS:
            c = combined.get(trig_key, {})
            fires = c.get("fires", 0)
            if fires == 0 and var_key == "A" and trig_key != "ma50":
                continue   # niet actief in variant A
            if fires == 0 and var_key == "B" and trig_key == "vol_down":
                continue
            vvp   = c.get("gem_verlies_voorkomen", 0.0)
            fp    = c.get("false_positives", 0)
            fpp   = c.get("false_pos_pct", 0.0)
            print(f"  {trig_label:<20} {fires:>7} {vvp:>15.1f}% {fp:>10} {fpp:>10.1f}%")

        # BH trades vergeleken met origineel
        orig_bh_n = sum(pres[p]["mode_n"].get("bh",0) for p in ["2024","Val 2025","Finale"])
        orig_n    = sum(all_results["orig"][p]["mode_n"].get("bh",0) for p in ["2024","Val 2025","Finale"])
        print(f"\n  B&H trades: {var_label} {orig_bh_n}  vs origineel {orig_n}  "
              f"(verschil {orig_bh_n - orig_n:+d})")

        # Totale trigger exits
        total_fires = sum(combined.get(t,{}).get("fires",0) for t,_ in TRIGGERS)
        print(f"  Totale trigger exits: {total_fires}")

    print(f"\n{'═'*W}")


def print_periode_detail(all_results: dict):
    W = 110
    for var_key, var_label in [("orig","Origineel (geen triggers)"),
                                ("A","Variant A: MA50"),
                                ("B","Variant B: MA50+ATR"),
                                ("C","Variant C: alle drie")]:
        pres = all_results[var_key]
        eind = _compound(pres)
        tot  = (eind - START_CAP) / START_CAP * 100
        mdd  = _maxdd(pres)
        print(f"\n  {var_label}  →  {tot:>+.1f}%  €{eind:.0f}  MaxDD {mdd:.1f}%")
        print(f"  {'Periode':<12} {'Ret':>8} {'MaxDD':>7} {'Trades':>8} "
              f"{'Win%':>7} {'PF':>7} {'B&H trades':>11} {'BH PnL €':>10} {'V4 PnL €':>10}")
        print(f"  {'─'*12} {'─'*8} {'─'*7} {'─'*8} "
              f"{'─'*7} {'─'*7} {'─'*11} {'─'*10} {'─'*10}")
        for pname in ["2024","Val 2025","Finale"]:
            r = pres[pname]
            print(f"  {pname:<12} {r['total_ret']:>+7.1f}% {r['max_dd']:>6.1f}% "
                  f"{r['n_trades']:>8} {r['win_rate']:>6.1f}% {_pfs(r['profit_factor']):>7} "
                  f"{r['mode_n'].get('bh',0):>11} "
                  f"{r['mode_pnl'].get('bh',0.0):>+9.1f}€ "
                  f"{r['mode_pnl'].get('v4',0.0):>+9.1f}€")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("  Vier-modus V2 — Snellere B&H Exit Triggers (Varianten A/B/C)")
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
        print(f"{len(df):>6} bars  |  bear<{p25:+.4f}  neut<{p50:+.4f}  licht<{p75:+.4f}")

    all_results = {}
    VARIANTEN = [("orig","Origineel"), ("A","Variant A"), ("B","Variant B"), ("C","Variant C")]

    for var_key, var_label in VARIANTEN:
        print(f"\n  ── {var_label}")
        pres = {}; cap = START_CAP
        for pname, pstart, pend in PERIODS:
            print(f"  {pname} …", end=" ", flush=True)
            r = run_fourmode_v2(frames, thresholds, pstart, pend, var_key, start_cap=cap)
            pres[pname] = r
            cap = r["final_cap"]
            bh_n = r["mode_n"].get("bh",0)
            trig_exits = sum(1 for t in r["trades"] if t.get("trigger"))
            print(f"{r['n_trades']:>4} trades  {r['total_ret']:>+.1f}%  "
                  f"MaxDD {r['max_dd']:.1f}%  €{r['final_cap']:.0f}  "
                  f"[BH:{bh_n} trig_exits:{trig_exits}]")
        all_results[var_key] = pres

    # Tabellen
    print_hoofdvergelijking(all_results)
    print_periode_detail(all_results)
    print_trigger_analyse(all_results, frames)

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
        "variants":   {k: _clean(v) for k, v in all_results.items()},
        "thresholds": {t: {k: round(v, 6) for k, v in thr.items()}
                       for t, thr in thresholds.items()},
    }
    OUTPUT_FILE.write_text(json.dumps(save, indent=2, ensure_ascii=False))
    print(f"\n  Resultaten opgeslagen: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
