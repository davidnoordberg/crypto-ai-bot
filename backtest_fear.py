"""
Fear Contrarian Backtest
=========================
Koopt tijdens extreme marktpaniek (Fear & Greed < 20, RSI < 45).
Verkoopt wanneer angst wegebt (FNG > 50), trailing stop (3×ATR) of RSI > 65.

Periode : jan 2024 – jun 2026
Data    : yfinance 4h (BTC-USD, ETH-USD, SOL-USD, DOGE-USD) + Alternative.me FNG daily
Kosten  : 0.4% roundtrip
"""

import json, warnings, requests
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ta.momentum   import RSIIndicator
from ta.volatility import AverageTrueRange

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS    = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"]
START_CAP  = 500.0
FEE_RT     = 0.004
MIN_POS    = 3.0
MAX_OPEN   = 3
ATR_WIN    = 14
ATR_TRAIL  = 3.0
RSI_WIN    = 14

FNG_ENTRY_MAX  = 20   # FNG onder dit niveau → entry mogelijk
FNG_EXIT_MIN   = 50   # FNG boven dit niveau → exit alle fear-posities
RSI_ENTRY_MAX  = 45   # RSI onder dit niveau bij entry
RSI_EXIT_MIN   = 65   # RSI boven dit niveau → exit

# Dynamic sizing op basis van FNG niveau
def fng_size(fng_score: int) -> float:
    if fng_score < 10:  return 0.40
    if fng_score < 15:  return 0.30
    return 0.20          # 15–20

TRAIN_START = "2024-01-01"; TRAIN_END  = "2024-12-31"
VAL_START   = "2025-01-01"; VAL_END    = "2025-09-30"
FINAL_START = "2025-10-01"; FINAL_END  = "2026-06-17"
PERIODS     = [("2024", TRAIN_START, TRAIN_END),
               ("Val 2025", VAL_START, VAL_END),
               ("Finale", FINAL_START, FINAL_END)]

OUTPUT_FILE = Path(__file__).parent / "results_fear.json"

# ── Fear & Greed historische data ─────────────────────────────────────────────
def fetch_fng_history() -> pd.Series:
    """
    Haalt tot 900 dagen Fear & Greed op via Alternative.me.
    Retourneert een dagelijkse Series geïndexeerd op datum (zonder tijd).
    """
    print("  Fear & Greed geschiedenis ophalen …", end=" ", flush=True)
    url = "https://api.alternative.me/fng/?limit=900&format=json"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()["data"]
    except Exception as e:
        print(f"FOUT: {e}")
        return pd.Series(dtype=int)

    records = []
    for d in data:
        ts    = pd.Timestamp(int(d["timestamp"]), unit="s").normalize()
        score = int(d["value"])
        records.append({"date": ts, "fng": score})

    s = pd.DataFrame(records).set_index("date")["fng"].sort_index()
    print(f"{len(s)} dagen  ({s.index[0].date()} … {s.index[-1].date()})")
    return s


def align_fng_to_4h(fng_daily: pd.Series, idx_4h: pd.DatetimeIndex) -> pd.Series:
    """
    Brengt dagelijkse FNG in lijn met 4-uurs index via forward-fill.
    Elke 4h bar krijgt de FNG van die dag (of de laatste bekende dag).
    """
    dates  = idx_4h.normalize()
    result = np.full(len(idx_4h), 50, dtype=int)   # fallback = neutraal

    for i, d in enumerate(dates):
        # Zoek de meest recente datum in fng_daily ≤ d
        avail = fng_daily[fng_daily.index <= d]
        if not avail.empty:
            result[i] = int(avail.iloc[-1])

    return pd.Series(result, index=idx_4h, name="fng")


# ── Price data ────────────────────────────────────────────────────────────────
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


def add_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < 50:
        return None
    df = df.copy()
    df["rsi"] = RSIIndicator(df["Close"], window=RSI_WIN).rsi()
    df["atr"] = AverageTrueRange(df["High"], df["Low"], df["Close"],
                                 window=ATR_WIN).average_true_range()
    return df.dropna()


def slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


# ── Positieklasse ─────────────────────────────────────────────────────────────
class Pos:
    __slots__ = ["ticker", "units", "entry_px", "stop_px", "trail_high",
                 "invest", "entry_ts", "entry_fng"]

    def __init__(self, ticker, units, entry_px, stop_px, invest, entry_ts, entry_fng):
        self.ticker     = ticker
        self.units      = units
        self.entry_px   = entry_px
        self.stop_px    = stop_px
        self.trail_high = entry_px
        self.invest     = invest
        self.entry_ts   = entry_ts
        self.entry_fng  = entry_fng

    def update_trail(self, close: float, atr: float):
        if close > self.trail_high:
            self.trail_high = close
            new_stop = close - ATR_TRAIL * atr
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


def _trade_rec(p: Pos, exit_ts, exit_px: float, pnl: float, reason: str,
               exit_fng: int, bars_held: int) -> dict:
    return {
        "ticker":     p.ticker,
        "entry_ts":   str(p.entry_ts),
        "exit_ts":    str(exit_ts),
        "entry_px":   p.entry_px,
        "exit_px":    exit_px,
        "pnl":        pnl,
        "invest":     p.invest,
        "win":        pnl > 0,
        "pnl_pct":    pnl / p.invest * 100 if p.invest > 0 else 0.0,
        "exit_reason": reason,
        "entry_fng":  p.entry_fng,
        "exit_fng":   exit_fng,
        "bars_held":  bars_held,
        "hours_held": bars_held * 4,
    }


# ── Backtest engine ───────────────────────────────────────────────────────────
def run_fear(slices: dict, fng_aligned: Dict[str, pd.Series],
             start: str, end: str, start_cap: float = START_CAP) -> dict:
    """
    Voert de Fear Contrarian strategie uit over één periode.

    fng_aligned: per ticker een Series met FNG waarden op 4h index.
    """
    # Bouw gemeenschappelijke tijdindex
    idx = None
    for t in TICKERS:
        sl = slices.get(t)
        if sl is not None and not sl.empty:
            idx = sl.index if idx is None else idx.union(sl.index)
    if idx is None or len(idx) == 0:
        return {}
    idx = idx.sort_values()

    capital    = start_cap
    positions: List[Pos] = []
    trades     = []
    equity     = []

    # Cooldown per ticker: True = wacht tot FNG < 20 opnieuw
    cooldown: Dict[str, bool] = {t: False for t in TICKERS}

    # Statistieken
    fng_below20_bars = 0

    for ts in idx:
        # FNG op dit moment (gebruik eerste ticker als bron — FNG is marktbreed)
        fng_now = 50
        for t in TICKERS:
            fs = fng_aligned.get(t)
            if fs is not None and ts in fs.index:
                fng_now = int(fs[ts])
                break

        if fng_now < FNG_ENTRY_MAX:
            fng_below20_bars += 1

        # ── Cooldown bijwerken ───────────────────────────────────────────────
        for t in TICKERS:
            if cooldown[t] and fng_now < FNG_ENTRY_MAX:
                cooldown[t] = False   # FNG terug onder 20 → cooldown voorbij

        # ── Exits ─────────────────────────────────────────────────────────────
        to_close = []
        for p in positions:
            sl = slices.get(p.ticker)
            if sl is None or ts not in sl.index:
                continue
            row   = sl.loc[ts]
            close = float(row["Close"])
            atr   = float(row["atr"])
            rsi   = float(row["rsi"])

            p.update_trail(close, atr)

            # Exit 1: FNG boven 50
            # Exit 2: trailing stop
            # Exit 3: RSI boven 65
            reason = ""
            if fng_now >= FNG_EXIT_MIN:
                reason = f"fng_recovery:{fng_now}"
            elif close <= p.stop_px:
                reason = "trail_stop"
            elif rsi >= RSI_EXIT_MIN:
                reason = f"rsi_exit:{rsi:.0f}"

            if reason:
                bars_held = len(idx[(idx >= p.entry_ts) & (idx <= ts)])
                pnl, cd   = _close_pos(p, close)
                capital  += cd
                trades.append(_trade_rec(p, ts, close, pnl, reason,
                                         fng_now, bars_held))
                to_close.append(p)
                cooldown[p.ticker] = True   # wacht op nieuwe FNG < 20
        positions = [p for p in positions if p not in to_close]

        # ── Entries ────────────────────────────────────────────────────────────
        if fng_now < FNG_ENTRY_MAX and len(positions) < MAX_OPEN:
            frac = fng_size(fng_now)

            for t in TICKERS:
                if len(positions) >= MAX_OPEN:
                    break
                if any(p.ticker == t for p in positions):
                    continue
                if cooldown[t]:
                    continue

                sl = slices.get(t)
                if sl is None or ts not in sl.index:
                    continue
                row   = sl.loc[ts]
                close = float(row["Close"])
                atr   = float(row["atr"])
                rsi   = float(row["rsi"])

                if rsi >= RSI_ENTRY_MAX:
                    continue

                invest = capital * frac
                if invest < MIN_POS:
                    continue

                fee   = invest * FEE_RT / 2
                units = (invest - fee) / close
                stop  = close - ATR_TRAIL * atr

                capital -= invest
                positions.append(Pos(t, units, close, stop, invest, ts, fng_now))

        # Equity snapshot
        port = capital
        for p in positions:
            sl = slices.get(p.ticker)
            c  = (float(sl.loc[ts, "Close"]) if sl is not None and ts in sl.index
                  else p.entry_px)
            port += p.mark_value(c)
        equity.append(float(port))

    # Sluit open posities aan einde
    for p in positions:
        sl  = slices.get(p.ticker)
        lc  = float(sl["Close"].iloc[-1]); lts = sl.index[-1]
        bars_held = len(idx[(idx >= p.entry_ts) & (idx <= lts)])
        pnl, cd   = _close_pos(p, lc)
        capital  += cd
        trades.append(_trade_rec(p, lts, lc, pnl, "eod",
                                 fng_now if 'fng_now' in dir() else 50, bars_held))

    return _calc_stats(trades, equity, capital, start_cap,
                       fng_below20_bars, start, end)


def _calc_stats(trades, equity, final_cap, start_cap,
                fng_below20_bars, start, end) -> dict:
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

    monthly = {}
    for t in trades:
        m = str(t["exit_ts"])[:7]
        monthly[m] = monthly.get(m, 0.0) + t["pnl"]

    best_m  = max(monthly.items(), key=lambda x: x[1]) if monthly else ("—", 0.0)
    worst_m = min(monthly.items(), key=lambda x: x[1]) if monthly else ("—", 0.0)

    n_months = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 30.44, 1)

    # Exit reden verdeling
    exit_dist: Dict[str, int] = {}
    for t in trades:
        r = t["exit_reason"].split(":")[0]
        exit_dist[r] = exit_dist.get(r, 0) + 1

    # Holding tijd
    avg_hours = (float(np.mean([t["hours_held"] for t in trades]))
                 if trades else 0.0)

    # FNG bij entry verdeling
    fng_buckets = {"<10": 0, "10-15": 0, "15-20": 0}
    for t in trades:
        fg = t["entry_fng"]
        if fg < 10:       fng_buckets["<10"] += 1
        elif fg < 15:     fng_buckets["10-15"] += 1
        else:             fng_buckets["15-20"] += 1

    return {
        "final_cap":          final_cap,
        "total_ret":          total_ret,
        "n_trades":           n,
        "trades_per_month":   round(n / n_months, 1),
        "win_rate":           wr,
        "profit_factor":      pf,
        "max_dd":             float(dd.max()),
        "max_loss_streak":    max_streak,
        "best_month":         best_m,
        "worst_month":        worst_m,
        "avg_holding_hours":  round(avg_hours, 1),
        "fng_below20_bars":   fng_below20_bars,
        "fng_below20_days":   round(fng_below20_bars / 6, 1),  # 6 bars = 1 dag
        "exit_distribution":  exit_dist,
        "fng_entry_dist":     fng_buckets,
        "monthly":            monthly,
        "trades":             trades,
        "equity":             equity,
    }


# ── Referentie backtests (B&H BTC) ───────────────────────────────────────────
def run_bh_btc(frames: dict, start: str, end: str,
               start_cap: float = START_CAP) -> dict:
    sl = frames["BTC-USD"]
    sl = sl[(sl.index >= pd.Timestamp(start)) & (sl.index <= pd.Timestamp(end))]
    if sl.empty:
        return {"total_ret": 0.0, "max_dd": 0.0, "final_cap": start_cap}
    units     = start_cap / float(sl["Close"].iloc[0])
    vals      = units * sl["Close"]
    final_cap = float(vals.iloc[-1])
    peak      = vals.cummax()
    dd        = (peak - vals) / peak * 100
    return {"total_ret": (final_cap - start_cap) / start_cap * 100,
            "max_dd":    float(dd.max()),
            "final_cap": final_cap}


# ── Output ────────────────────────────────────────────────────────────────────
def _pfs(v: float) -> str:
    return "∞" if v == float("inf") else f"{v:.2f}"


def _compound(pres: dict, cap: float = START_CAP) -> float:
    return cap * (1 + pres["2024"]["total_ret"]/100) \
                * (1 + pres["Val 2025"]["total_ret"]/100) \
                * (1 + pres["Finale"]["total_ret"]/100)


def print_vergelijking(fear_pres: dict, bh_pres: dict):
    W = 115
    print(f"\n{'═'*W}")
    print("  HOOFDVERGELIJKING — €500 startkapitaal, compound drie periodes")
    print(f"{'═'*W}")
    print(f"  {'Strategie':<30} {'2024':>8} {'Val2025':>8} {'Finale':>8} "
          f"{'Totaal':>8} {'MaxDD':>7} {'Trades':>7} {'Win%':>7} {'PF':>7} {'Eind€':>7}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    # Fear Contrarian
    r24  = fear_pres["2024"]; rv = fear_pres["Val 2025"]; rf = fear_pres["Finale"]
    eind = _compound(fear_pres)
    tot  = (eind - START_CAP) / START_CAP * 100
    mdd  = max(r24["max_dd"], rv["max_dd"], rf["max_dd"])
    ntot = r24["n_trades"] + rv["n_trades"] + rf["n_trades"]
    wr   = ((r24["win_rate"]*r24["n_trades"] + rv["win_rate"]*rv["n_trades"]
             + rf["win_rate"]*rf["n_trades"]) / (ntot or 1))
    pf_vals = [fear_pres[p]["profit_factor"] for p in ["2024","Val 2025","Finale"]
               if fear_pres[p]["profit_factor"] not in (0, float("inf"))]
    avgpf = float(np.mean(pf_vals)) if pf_vals else float("inf")
    print(f"  {'Fear Contrarian':<30} {r24['total_ret']:>+7.1f}% {rv['total_ret']:>+7.1f}% "
          f"{rf['total_ret']:>+7.1f}% {tot:>+7.1f}% "
          f"{mdd:>6.1f}% {ntot:>7} {wr:>6.1f}% {_pfs(avgpf):>7} {eind:>6.0f}€")

    # Referenties
    refs = [
        ("Hybride V4+MR baseline", "+43.2%", "+20.1%",  "-9.8%",  "+55.2%", "16.2%", "266", "43.2%", "1.62", "776€"),
        ("B&H BTC",               "+109.1%", "+22.2%", "-42.5%",  "+48.6%", "52.0%",   "—",     "—",    "—",  "734€"),
    ]
    for r in refs:
        print(f"  {r[0]:<30} {r[1]:>8} {r[2]:>8} {r[3]:>8} {r[4]:>8} "
              f"{r[5]:>7} {r[6]:>7} {r[7]:>7} {r[8]:>7} {r[9]:>7}")
    print(f"{'═'*W}")


def print_detail(fear_pres: dict, current_fng: int):
    print(f"\n  {'─'*90}")
    print("  DETAIL PER PERIODE")
    print(f"  {'─'*90}")
    print(f"  {'Periode':<12} {'Ret':>8} {'MaxDD':>7} {'Trades':>8} {'Win%':>7} "
          f"{'PF':>7} {'AvgHours':>10} {'FNG<20d':>8} {'FNG exit':>9} {'Trail':>7} {'RSI':>5}")
    print(f"  {'─'*12} {'─'*8} {'─'*7} {'─'*8} {'─'*7} "
          f"{'─'*7} {'─'*10} {'─'*8} {'─'*9} {'─'*7} {'─'*5}")

    for pname in ["2024","Val 2025","Finale"]:
        r   = fear_pres[pname]
        ed  = r["exit_distribution"]
        print(f"  {pname:<12} {r['total_ret']:>+7.1f}% {r['max_dd']:>6.1f}% "
              f"{r['n_trades']:>8} {r['win_rate']:>6.1f}% {_pfs(r['profit_factor']):>7} "
              f"{r['avg_holding_hours']:>9.0f}u "
              f"{r['fng_below20_days']:>7.0f}d "
              f"{ed.get('fng_recovery',0):>9} "
              f"{ed.get('trail_stop',0):>7} "
              f"{ed.get('rsi_exit',0):>5}")

    print(f"\n  FNG entry verdeling (totaal over alle periodes):")
    total_fng = {"<10": 0, "10-15": 0, "15-20": 0}
    for pname in ["2024","Val 2025","Finale"]:
        for k, v in fear_pres[pname]["fng_entry_dist"].items():
            total_fng[k] = total_fng.get(k, 0) + v
    for k, v in total_fng.items():
        print(f"    FNG {k}: {v} trades")

    print(f"\n  Huidige Fear & Greed: {current_fng}")
    if current_fng < FNG_ENTRY_MAX:
        print(f"  → Bot zou NU actief zijn (FNG {current_fng} < {FNG_ENTRY_MAX})")
        frac = fng_size(current_fng)
        print(f"  → Positiegrootte: {frac*100:.0f}% per trade (fng={current_fng})")
    else:
        print(f"  → Bot wacht (FNG {current_fng} >= {FNG_ENTRY_MAX})")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Fear Contrarian Backtest")
    print("=" * 70)

    # 1. FNG historische data
    fng_hist = fetch_fng_history()
    current_fng = int(fng_hist.iloc[-1]) if not fng_hist.empty else 50
    print(f"  Huidige FNG (meest recent): {current_fng}")

    # 2. Price data
    print("\n  Price data ophalen …")
    frames_raw = {}
    for t in TICKERS:
        print(f"  {t:<12} …", end=" ", flush=True)
        df = fetch_4h(t)
        df = add_indicators(df)
        frames_raw[t] = df
        print(f"{len(df):>6} bars" if df is not None else "MISLUKT")

    # 3. Backtest per periode (compound)
    print("\n  Backtest draaien …")
    fear_pres = {}
    cap = START_CAP

    for pname, pstart, pend in PERIODS:
        # Snijd slices
        slices = {t: slice_window(frames_raw[t], pstart, pend)
                  for t in TICKERS if frames_raw[t] is not None}

        # Align FNG aan 4h index per ticker
        fng_aligned = {}
        for t, sl in slices.items():
            fng_aligned[t] = align_fng_to_4h(fng_hist, sl.index)

        print(f"  {pname} (€{cap:.0f} start) …", end=" ", flush=True)
        r = run_fear(slices, fng_aligned, pstart, pend, start_cap=cap)
        fear_pres[pname] = r
        cap = r["final_cap"]

        n_fng = r["fng_below20_days"]
        print(f"{r['n_trades']:>4} trades  {r['total_ret']:>+.1f}%  "
              f"MaxDD {r['max_dd']:.1f}%  €{r['final_cap']:.0f}  "
              f"[FNG<20: {n_fng:.0f}d  WR:{r['win_rate']:.0f}%  "
              f"AvgHold:{r['avg_holding_hours']:.0f}u]")

    # 4. Output tabellen
    print_vergelijking(fear_pres, {})
    print_detail(fear_pres, current_fng)

    # 5. Opslaan
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
        "strategy": "fear_contrarian",
        "config": {
            "fng_entry_max": FNG_ENTRY_MAX,
            "fng_exit_min":  FNG_EXIT_MIN,
            "rsi_entry_max": RSI_ENTRY_MAX,
            "rsi_exit_min":  RSI_EXIT_MIN,
            "atr_trail":     ATR_TRAIL,
            "max_open":      MAX_OPEN,
            "fee_rt":        FEE_RT,
        },
        "current_fng":  current_fng,
        "results":      {k: _clean(v) for k, v in fear_pres.items()},
    }
    OUTPUT_FILE.write_text(json.dumps(save, indent=2, ensure_ascii=False))
    print(f"  Resultaten opgeslagen: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
