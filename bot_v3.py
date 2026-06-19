"""
Aggressive V3 Paper Trading Bot — DOGE + SOL
============================================
Portfolio: alleen DOGE-EUR en SOL-EUR via Bitvavo publieke API.
Strategie: hybride V4+MR identiek aan bot.py maar met grotere posities.

Positiegrootte V4: 25% zwak / 40% normaal / 60% sterk signaal.
Max twee open posities tegelijk (één per crypto).
State:     state_v3.json in GitHub repo — onafhankelijk van bot.py (state.json).
Schedule:  GitHub Actions cron '30 */4 * * *' (30 min na bot.py).
Notificaties: zelfde Slack webhook als bot.py, duidelijk gelabeld als V3.

Stack: Python 3.11, anthropic, ta, pandas, requests, feedparser.
"""

from __future__ import annotations
import os, json, re, sys, warnings, traceback, base64
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
import numpy as np
import feedparser

warnings.filterwarnings("ignore")

from ta.momentum   import RSIIndicator
from ta.trend      import MACD as TaMACD
from ta.volatility import AverageTrueRange, BollingerBands

try:
    import anthropic as _anthropic
    _ANTHR_CLIENT: Optional[_anthropic.Anthropic] = None
except ImportError:
    _anthropic = None
    _ANTHR_CLIENT = None

# ── Config ────────────────────────────────────────────────────────────────────
BOT_ID    = "v3"
TICKERS   = ["DOGE-EUR", "SOL-EUR"]
START_CAP = 500.0
FEE_RT    = 0.004
MIN_POS   = 3.0

GH_API        = "https://api.github.com"
GH_REPO       = os.environ.get("GITHUB_REPOSITORY", "davidnoordberg/crypto-ai-bot")
GH_STATE_FILE = "state_v3.json"   # eigen bestand — onafhankelijk van bot.py

# V3: grotere positiegrootte dan baseline
# strength 0-1 → zwak (25%), strength 2 → normaal (40%), strength 3-4 → sterk (60%)
V4_RSI_LO    = 45; V4_RSI_HI = 75
V4_ATR_ENTRY = 4.0
V4_ATR_TRAIL = 3.0
V4_MA200_MAX_BELOW = 0.15
V4_SIZES = {0: 0.25, 1: 0.25, 2: 0.40, 3: 0.60, 4: 0.60}

MR_RSI_THR = 35
MR_BB_WIN  = 30; MR_BB_SIG = 2.5
MR_ATR     = 2.0

# MR sizing schaalt mee: 7→25%, 10→40% (zelfde verhouding als baseline)
MR_SIZES = {"normal": 0.25, "deep": 0.40}   # normal = RSI≥20, deep = RSI<20

ATR_WIN = 14; MA50_WIN = 50; MA200_WIN = 200
MA200_LAG = 10; VOL_WIN = 20
MACD_F, MACD_S, MACD_SIG_W = 12, 26, 9

EXPO_BULL_NEUT = 0.70
EXPO_BEAR      = 0.40
MAX_OPEN       = 2

LLM_MODEL   = "claude-haiku-4-5-20251001"
PARAMS_FILE = Path(__file__).parent / "best_params_hybrid.json"

BITVAVO_BASE = "https://api.bitvavo.com/v2"
CANDLE_LIMIT = 250


# ── Client setup ──────────────────────────────────────────────────────────────
def _init_clients():
    global _ANTHR_CLIENT
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if _anthropic and api_key:
        _ANTHR_CLIENT = _anthropic.Anthropic(api_key=api_key)


def _name(ticker: str) -> str:
    return ticker.replace("-EUR", "")


# ── Bitvavo data ──────────────────────────────────────────────────────────────
def fetch_candles(market: str, interval: str = "4h",
                  limit: int = CANDLE_LIMIT) -> Optional[pd.DataFrame]:
    url = f"{BITVAVO_BASE}/{market}/candles"
    try:
        r = requests.get(url, params={"interval": interval, "limit": limit},
                         timeout=15)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        print(f"  [WARN] Bitvavo candles {market}: {e}")
        return None

    if not raw or isinstance(raw, dict):
        print(f"  [WARN] Bitvavo lege of fout response voor {market}: {raw}")
        return None

    df = pd.DataFrame(raw, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("ts").sort_index()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def fetch_all_candles() -> dict[str, Optional[pd.DataFrame]]:
    result = {}
    for t in TICKERS:
        print(f"  Bitvavo {t} …", end=" ", flush=True)
        df = fetch_candles(t)
        if df is not None:
            print(f"{len(df)} bars")
        else:
            print("MISLUKT")
        result[t] = df
    return result


def fetch_fng() -> dict:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        d = r.json()["data"][0]
        return {"score": int(d["value"]), "label": d["value_classification"]}
    except Exception:
        return {"score": 50, "label": "Neutraal"}


# ── Indicatoren ───────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < MA200_WIN + 10:
        return None
    df = df.copy()
    df["ma50"]      = df["Close"].rolling(MA50_WIN).mean()
    df["ma200"]     = df["Close"].rolling(MA200_WIN).mean()
    df["ma200_lag"] = df["ma200"].shift(MA200_LAG)
    df["rsi"]       = RSIIndicator(df["Close"], window=14).rsi()
    _m = TaMACD(df["Close"], window_fast=MACD_F, window_slow=MACD_S,
                window_sign=MACD_SIG_W)
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


def last_row(df: pd.DataFrame):
    return df.iloc[-1] if df is not None and not df.empty else None


# ── Regime grenzen ────────────────────────────────────────────────────────────
def load_thresholds() -> dict:
    if PARAMS_FILE.exists():
        raw = json.loads(PARAMS_FILE.read_text())
        return {k.replace("-EUR", "").replace("-USD", ""): v for k, v in raw.items()}
    print("  [WARN] best_params_hybrid.json niet gevonden — gebruik 0/0 fallback.")
    return {_name(t): {"p33": 0.0, "p67": 0.0} for t in TICKERS}


def regime_score(row) -> float:
    ma200     = float(row["ma200"]); ma200_lag = float(row["ma200_lag"])
    close     = float(row["Close"]); rsi       = float(row["rsi"])
    ma200_s = (ma200 - ma200_lag) / ma200_lag if ma200_lag else 0.0
    prijs_s = (close - ma200)     / ma200     if ma200     else 0.0
    rsi_s   = (rsi - 50)          / 50
    return float(np.clip((ma200_s + prijs_s + rsi_s) / 3, -1.0, 1.0))


def get_regime(score: float, p33: float, p67: float) -> str:
    if score > p67: return "bull"
    if score < p33: return "bear"
    return "neutraal"


def portfolio_regime(regime_map: dict) -> str:
    """Meerderheidsstem van DOGE+SOL regimes."""
    c = {"bull": 0, "neutraal": 0, "bear": 0}
    for r in regime_map.values():
        c[r] = c.get(r, 0) + 1
    n = len(regime_map) or 1
    if c["bear"] > n / 2:  return "bear"
    if c["bull"] > n / 2:  return "bull"
    return "neutraal"


# ── Signaallogica ─────────────────────────────────────────────────────────────
def v4_entry_check(row) -> bool:
    rsi   = float(row["rsi"]); close = float(row["Close"]); ma200 = float(row["ma200"])
    cross = float(row["macd"]) > float(row["macd_sig"]) and float(row["prev_hist"]) <= 0
    return (V4_RSI_LO <= rsi <= V4_RSI_HI and cross
            and float(row["Volume"]) > float(row["vol_ma"])
            and close >= ma200 * (1 - V4_MA200_MAX_BELOW))


def v4_strength_score(row) -> int:
    s = 0
    if 55 <= float(row["rsi"]) <= 70:                                          s += 1
    if float(row["macd_hist"]) > float(row["prev_hist"]) > 0:                  s += 1
    if float(row["vol_ratio"]) > 1.5:                                          s += 1
    if float(row["Close"]) > float(row["ma200"]) * 1.02:                       s += 1
    return s


def mr_entry_check(row) -> bool:
    close = float(row["Close"])
    return (float(row["rsi"]) < MR_RSI_THR
            and close < float(row["bb_lower"]) - float(row["bb_std"])
            and float(row["Volume"]) > float(row["vol_ma"]))


def mr_size_frac(row) -> float:
    return MR_SIZES["deep"] if float(row["rsi"]) < 20 else MR_SIZES["normal"]


def v4_exit_check(row, stop_px: float) -> bool:
    return float(row["Close"]) <= stop_px


def mr_exit_check(row, stop_px: float) -> bool:
    return (float(row["rsi"]) > 50
            or float(row["Close"]) >= float(row["bb_mid"])
            or float(row["Close"]) <= stop_px)


# ── Nieuws & LLM ──────────────────────────────────────────────────────────────
def fetch_news(ticker: str) -> list[str]:
    coin = _name(ticker).lower()
    coin_map = {"doge": "dogecoin", "sol": "solana"}
    coin_slug = coin_map.get(coin, coin)
    url = f"https://cryptopanic.com/news/{coin_slug}/rss/"
    try:
        feed = feedparser.parse(url)
        headlines = [e.title.strip() for e in feed.entries[:5] if e.get("title")]
        if headlines:
            return headlines
    except Exception:
        pass
    try:
        feed = feedparser.parse("https://www.coindesk.com/arc/outboundfeeds/rss/")
        return [e.title.strip() for e in feed.entries[:5] if e.get("title")]
    except Exception:
        return ["Geen nieuws beschikbaar"]


def _candle_label(row) -> str:
    o = float(row["Open"]); c = float(row["Close"])
    h = float(row["High"]); l = float(row["Low"])
    body  = abs(c - o); range_ = h - l or 1e-9
    dir_  = "bullish" if c > o else "bearish"
    size  = "groot" if body / range_ > 0.6 else ("klein" if body / range_ < 0.2 else "normaal")
    return f"{dir_} {size} lichaam"


def _ma200_label(row) -> str:
    ma200 = float(row["ma200"]); ma200_lag = float(row["ma200_lag"])
    slope_pct = (ma200 - ma200_lag) / ma200_lag * 100 if ma200_lag else 0
    if slope_pct > 0.1:   return f"stijgend (+{slope_pct:.2f}% / 10 bars)"
    if slope_pct < -0.1:  return f"dalend ({slope_pct:.2f}% / 10 bars)"
    return "zijwaarts"


def _prijs_label(row) -> str:
    close = float(row["Close"]); ma200 = float(row["ma200"])
    pct   = (close - ma200) / ma200 * 100
    if pct > 5:  return f"+{pct:.1f}% boven MA200"
    if pct > 0:  return f"+{pct:.1f}% net boven MA200"
    return f"{pct:.1f}% onder MA200"


def _cross_asset_label(regime_map: dict, skip_ticker: str) -> str:
    others = {t: r for t, r in regime_map.items() if t != skip_ticker}
    if not others:
        return "geen cross-asset data"
    bulls = sum(1 for r in others.values() if r == "bull")
    bears = sum(1 for r in others.values() if r == "bear")
    n = len(others)
    if bulls == n:  return f"andere asset ook bullish"
    if bears == n:  return f"andere asset ook bearish"
    return "andere asset neutraal"


def ask_llm(ticker: str, row, regime: str, strategy: str,
            regime_map: dict, fng: dict, headlines: list) -> dict:
    """Vraag Haiku om entry beslissing. V3: positiegrootte 25/40/60%."""
    fallback = {"beslissing": "KOOP", "confidence": 0.5,
                "positiegrootte": 40, "nieuws_sentiment": "neutraal",
                "reden": "LLM niet beschikbaar"}

    if _ANTHR_CLIENT is None:
        return fallback

    hl_txt = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines[:5]))
    prompt = f"""Crypto: {_name(ticker)}
Datum: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC
Entry prijs: {float(row['Close']):.6f} EUR
Regime: {regime}
Strategie: {strategy}
Bot: Agressief V3 (grotere posities, alleen DOGE+SOL)

Technische context:
RSI: {float(row['rsi']):.1f}
MACD histogram: {float(row['macd_hist']):.6f}
ATR ratio: {float(row['atr'])/float(row['Close'])*100:.2f}% van prijs
Volume ratio: {float(row['vol_ratio']):.2f}x gemiddelde
MA200 richting: {_ma200_label(row)}
Prijspositie: {_prijs_label(row)}
Cross-asset momentum: {_cross_asset_label(regime_map, ticker)}
Laatste candle: {_candle_label(row)}

Marktsentiment:
Fear & Greed Index: {fng['score']} ({fng['label']})

Recent nieuws (laatste 24u):
{hl_txt}

Het hybride V4+MR systeem heeft een entry signaal gegenereerd.
Dit is een agressieve bot met grote posities (25-60% van kapitaal).
Beoordeel of dit een goed instapmoment is gezien technische context EN nieuws.
Exit is een trailing stop — beoordeel alleen de entry.

Geef je antwoord UITSLUITEND als geldig JSON (geen tekst erbuiten):
{{
  "beslissing": "KOOP" of "NIETS",
  "confidence": 0.0 tot 1.0,
  "positiegrootte": 25, 40, of 60,
  "nieuws_sentiment": "bullish", "neutraal", of "bearish",
  "reden": "maximaal 2 zinnen"
}}"""

    try:
        resp = _ANTHR_CLIENT.messages.create(
            model=LLM_MODEL, max_tokens=200,
            messages=[{"role": "user", "content": prompt}])
        raw = resp.content[0].text.strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return fallback
        parsed = json.loads(m.group())
        bes = str(parsed.get("beslissing", "KOOP")).upper()
        if bes not in ("KOOP", "NIETS"): bes = "KOOP"
        ps = int(parsed.get("positiegrootte", 40))
        if ps not in (25, 40, 60): ps = 40
        ns = str(parsed.get("nieuws_sentiment", "neutraal")).lower()
        if ns not in ("bullish", "neutraal", "bearish"): ns = "neutraal"
        return {"beslissing": bes,
                "confidence":  float(parsed.get("confidence", 0.5)),
                "positiegrootte": ps,
                "nieuws_sentiment": ns,
                "reden": str(parsed.get("reden", ""))[:300]}
    except Exception as e:
        print(f"  [WARN] LLM fout voor {ticker}: {e}")
        return fallback


# ── GitHub state (Contents API) ────────────────────────────────────────────────
_INITIAL_STATE = {
    "bot_id":            BOT_ID,
    "capital_eur":       START_CAP,
    "open_positions":    {},
    "total_trades":      0,
    "wins":              0,
    "gross_win":         0.0,
    "gross_loss":        0.0,
    "win_rate":          0.0,
    "profit_factor":     0.0,
    "total_return_pct":  0.0,
    "trades":            [],
    "signals":           [],
}

_state_sha: Optional[str] = None


def _gh_headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN", "")
    h = {"Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _gh_read(path: str) -> tuple[Optional[dict], Optional[str]]:
    url = f"{GH_API}/repos/{GH_REPO}/contents/{path}"
    try:
        r = requests.get(url, headers=_gh_headers(), timeout=15)
        if r.status_code == 404:
            return None, None
        r.raise_for_status()
        data = r.json()
        content = json.loads(base64.b64decode(data["content"]).decode())
        return content, data["sha"]
    except Exception as e:
        print(f"  [WARN] GitHub read {path}: {e}")
        return None, None


def _gh_write(path: str, content: dict, sha: Optional[str], message: str):
    url = f"{GH_API}/repos/{GH_REPO}/contents/{path}"
    encoded = base64.b64encode(
        json.dumps(content, indent=2, ensure_ascii=False).encode()
    ).decode()
    body: dict = {"message": message, "content": encoded}
    if sha:
        body["sha"] = sha
    try:
        r = requests.put(url, headers=_gh_headers(), json=body, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"  [WARN] GitHub write {path}: {e}")


def load_state() -> dict:
    global _state_sha
    content, sha = _gh_read(GH_STATE_FILE)
    if content is None:
        print("  [INFO] state_v3.json bestaat nog niet — initialiseer met startkapitaal.")
        _state_sha = None
        return _INITIAL_STATE.copy()
    _state_sha = sha
    state = _INITIAL_STATE.copy()
    state.update(content)
    return state


def save_state(state: dict):
    global _state_sha
    total_val = state["capital_eur"] + sum(
        float(p.get("current_val", p.get("invest", 0)))
        for p in state["open_positions"].values())
    ret_pct = (total_val - START_CAP) / START_CAP * 100

    gw = state["gross_win"]; gl = state["gross_loss"]
    pf = gw / gl if gl > 0 else 0.0
    wr = state["wins"] / state["total_trades"] * 100 \
         if state["total_trades"] > 0 else 0.0

    state["bot_id"]            = BOT_ID
    state["win_rate"]          = round(wr, 4)
    state["profit_factor"]     = round(pf, 4)
    state["total_return_pct"]  = round(ret_pct, 4)

    clean_positions = {}
    for t, p in state["open_positions"].items():
        clean_positions[t] = {k: v for k, v in p.items() if k != "current_val"}
    state["open_positions"] = clean_positions

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_pos   = len(state["open_positions"])
    message = (f"bot-v3: {now_str} — "
               f"€{state['capital_eur']:.2f} vrij, "
               f"{n_pos} pos open, "
               f"{state['total_trades']} trades")

    _gh_write(GH_STATE_FILE, state, _state_sha, message)
    print(f"  [GitHub] state_v3.json opgeslagen  "
          f"(ret {ret_pct:+.1f}%  trades {state['total_trades']})")


def log_trade(trade: dict, state: dict):
    clean = {k: (round(v, 6) if isinstance(v, float) else v)
             for k, v in trade.items() if v is not None}
    state.setdefault("trades", []).append(clean)
    action = trade.get("action", "?")
    ticker = trade.get("ticker", "?")
    pnl    = trade.get("pnl_eur") or 0.0
    reason = trade.get("exit_reason", "")
    print(f"  [TRADE] {action} {ticker:<12}  pnl {pnl:>+.2f}€  {reason}")


def log_signal(sig: dict, state: dict):
    clean = {k: (round(v, 6) if isinstance(v, float) else v)
             for k, v in sig.items()}
    state.setdefault("signals", []).append(clean)
    blk = sig.get("entry_blocked", False)
    print(f"  [SIGNAL] {sig['ticker']:<12}  {sig['signal_type']}  "
          f"regime={sig['regime']}  "
          f"{'GEBLOKKEERD' if blk else 'KOOP'}")


# ── Positie helpers ────────────────────────────────────────────────────────────
def _close_position(pos: dict, close_px: float) -> tuple[float, float]:
    units    = float(pos["units"])
    entry_px = float(pos["entry_px"])
    invest   = float(pos["invest"])
    proceeds = units * close_px
    sell_fee = proceeds * FEE_RT / 2
    buy_fee  = invest   * FEE_RT / 2
    pnl      = proceeds - units * entry_px - sell_fee - buy_fee
    cap_back = proceeds - sell_fee
    return pnl, cap_back


def _update_trail(pos: dict, close: float, atr: float) -> dict:
    if pos.get("strategy") != "v4":
        return pos
    trail_high = float(pos.get("trail_high", pos["entry_px"]))
    if close > trail_high:
        pos["trail_high"] = close
        new_stop = close - V4_ATR_TRAIL * atr
        if new_stop > float(pos["stop_px"]):
            pos["stop_px"] = new_stop
    return pos


# ── Exit checks ────────────────────────────────────────────────────────────────
def check_exits(state: dict, indicators: dict) -> tuple[dict, list]:
    exits   = []
    to_del  = []
    now_str = datetime.now(timezone.utc).isoformat()

    for ticker, pos in list(state["open_positions"].items()):
        df = indicators.get(ticker)
        if df is None or df.empty:
            continue
        row   = last_row(df)
        close = float(row["Close"])
        atr   = float(row["atr"])

        _update_trail(pos, close, atr)

        strategy = pos.get("strategy", "v4")
        if strategy == "v4":
            should_exit = v4_exit_check(row, float(pos["stop_px"]))
            reason      = "trail_stop"
        else:
            should_exit = mr_exit_check(row, float(pos["stop_px"]))
            reason = "atr_stop" if close <= float(pos["stop_px"]) else "mr_signal"

        if should_exit:
            pnl, cap_back = _close_position(pos, close)
            state["capital_eur"] += cap_back
            state["total_trades"] += 1
            if pnl > 0:
                state["wins"]      += 1
                state["gross_win"] += pnl
            else:
                state["gross_loss"] += abs(pnl)

            pnl_pct = pnl / float(pos["invest"]) * 100
            trade = {
                "timestamp":            now_str,
                "ticker":               ticker,
                "action":               "SELL",
                "price":                round(close, 6),
                "position_size_pct":    round(float(pos["invest"]) / START_CAP * 100, 2),
                "position_size_eur":    round(float(pos["invest"]), 4),
                "entry_price":          round(float(pos["entry_px"]), 6),
                "exit_price":           round(close, 6),
                "pnl_pct":              round(pnl_pct, 4),
                "pnl_eur":              round(pnl, 4),
                "exit_reason":          reason,
                "regime":               pos.get("regime_at_entry", "?"),
                "strategy":             strategy,
                "llm_beslissing":       pos.get("llm_beslissing", ""),
                "llm_confidence":       pos.get("llm_confidence", 0.0),
                "llm_nieuws_sentiment": pos.get("llm_nieuws_sentiment", ""),
                "llm_reden":            pos.get("llm_reden", ""),
            }
            log_trade(trade, state)
            exits.append(trade)
            to_del.append(ticker)
            print(f"  EXIT  {ticker:<12}  {reason:<12}  "
                  f"€{close:.4f}  PnL {pnl_pct:>+.1f}% ({pnl:>+.2f}€)")
        else:
            pos["current_val"] = float(pos["units"]) * close
            state["open_positions"][ticker] = pos

    for t in to_del:
        del state["open_positions"][t]

    return state, exits


# ── Entry signalen ────────────────────────────────────────────────────────────
def generate_entries(state: dict, indicators: dict,
                     regime_map: dict, thresholds: dict,
                     fng: dict) -> tuple[dict, list]:
    signals   = []
    port_reg  = portfolio_regime(regime_map)
    max_expo  = EXPO_BULL_NEUT if port_reg in ("bull", "neutraal") else EXPO_BEAR
    now_str   = datetime.now(timezone.utc).isoformat()

    total_val = state["capital_eur"] + sum(
        float(p.get("current_val", p.get("invest", 0)))
        for p in state["open_positions"].values())
    invested  = sum(float(p["invest"]) for p in state["open_positions"].values())

    for ticker in TICKERS:
        if ticker in state["open_positions"]:
            continue
        if len(state["open_positions"]) >= MAX_OPEN:
            break

        df = indicators.get(ticker)
        if df is None or df.empty:
            continue
        row   = last_row(df)
        t_reg = regime_map.get(ticker, "neutraal")
        close = float(row["Close"])
        atr   = float(row["atr"])

        expo_frac = invested / total_val if total_val > 0 else 0.0
        if expo_frac >= max_expo:
            break

        strategy = None
        frac     = 0.0

        if t_reg in ("bull", "neutraal"):
            if v4_entry_check(row):
                strategy = "v4"
                s        = v4_strength_score(row)
                frac     = V4_SIZES[s]
        else:  # bear
            if mr_entry_check(row):
                strategy = "mr"
                frac     = mr_size_frac(row)

        if strategy is None:
            continue

        headlines = fetch_news(ticker)
        llm       = ask_llm(ticker, row, t_reg, strategy, regime_map, fng, headlines)

        name  = _name(ticker)
        thr   = thresholds.get(name, {"p33": 0.0, "p67": 0.0})
        score = regime_score(row)
        sig_row = {
            "timestamp":      now_str,
            "ticker":         ticker,
            "regime":         t_reg,
            "rsi":            round(float(row["rsi"]), 2),
            "macd_hist":      round(float(row["macd_hist"]), 6),
            "atr_ratio":      round(float(row["atr"]) / close * 100, 4),
            "volume_ratio":   round(float(row["vol_ratio"]), 4),
            "ma200_slope":    round((float(row["ma200"]) - float(row["ma200_lag"]))
                                    / float(row["ma200_lag"]) * 100, 4),
            "trailing_stop":  0.0,
            "signal_type":    strategy,
            "llm_beslissing": llm["beslissing"],
            "llm_confidence": llm["confidence"],
            "entry_blocked":  llm["beslissing"] == "NIETS",
            "block_reason":   llm["reden"] if llm["beslissing"] == "NIETS" else "",
        }
        log_signal(sig_row, state)
        signals.append({**sig_row, "llm": llm, "row": row,
                         "close": close, "atr": atr, "frac": frac})

        if llm["beslissing"] == "NIETS":
            print(f"  SKIP  {ticker:<12}  LLM NIETS  ({llm['reden'][:60]})")
            continue

        # V3: LLM kiest uit 25/40/60%, niet 7/10/15%
        llm_ps   = llm["positiegrootte"]
        eff_frac = llm_ps / 100.0
        invest   = state["capital_eur"] * eff_frac
        if invest < MIN_POS:
            print(f"  SKIP  {ticker:<12}  te klein (€{invest:.2f})")
            continue

        fee   = invest * FEE_RT / 2
        units = (invest - fee) / close
        stop  = close - V4_ATR_ENTRY * atr if strategy == "v4" else close - MR_ATR * atr

        pos = {
            "strategy":             strategy,
            "units":                round(units, 8),
            "entry_px":             round(close, 6),
            "stop_px":              round(stop, 6),
            "trail_high":           round(close, 6),
            "invest":               round(invest, 4),
            "entry_ts":             now_str,
            "strength":             v4_strength_score(row) if strategy == "v4" else 0,
            "regime_at_entry":      t_reg,
            "llm_beslissing":       llm["beslissing"],
            "llm_confidence":       llm["confidence"],
            "llm_nieuws_sentiment": llm["nieuws_sentiment"],
            "llm_reden":            llm["reden"],
        }
        state["capital_eur"] -= invest
        state["open_positions"][ticker] = pos
        invested += invest

        trade = {
            "timestamp":            now_str,
            "ticker":               ticker,
            "action":               "BUY",
            "price":                round(close, 6),
            "position_size_pct":    round(eff_frac * 100, 2),
            "position_size_eur":    round(invest, 4),
            "entry_price":          round(close, 6),
            "exit_price":           None,
            "pnl_pct":              None,
            "pnl_eur":              None,
            "exit_reason":          None,
            "regime":               t_reg,
            "strategy":             strategy,
            "llm_beslissing":       llm["beslissing"],
            "llm_confidence":       llm["confidence"],
            "llm_nieuws_sentiment": llm["nieuws_sentiment"],
            "llm_reden":            llm["reden"],
        }
        log_trade(trade, state)
        print(f"  BUY   {ticker:<12}  {strategy:<4}  {t_reg:<9}  "
              f"€{close:.4f}  {eff_frac*100:.0f}%  "
              f"(LLM conf {llm['confidence']:.2f})")

    return state, signals


# ── Slack notificatie (V3 label) ──────────────────────────────────────────────
def _emoji(pnl: float) -> str:
    return ":chart_with_upwards_trend:" if pnl >= 0 else ":chart_with_downwards_trend:"


def send_slack(state: dict, exits: list, signals: list, fng: dict):
    url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not url:
        print("  [INFO] Geen SLACK_WEBHOOK_URL — geen notificatie.")
        return

    now_cet   = datetime.now(timezone.utc) + timedelta(hours=2)
    now_str   = now_cet.strftime("%d-%m-%Y %H:%M")
    gw        = state["gross_win"]; gl = state["gross_loss"]
    pf        = gw / gl if gl > 0 else 0.0
    wr        = state["win_rate"]
    total_val = state["capital_eur"] + sum(
        float(p.get("current_val", p.get("invest", 0)))
        for p in state["open_positions"].values())
    ret_pct   = (total_val - START_CAP) / START_CAP * 100
    ret_icon  = ":arrow_up_small:" if ret_pct >= 0 else ":arrow_down_small:"

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text",
                     "text": f"Agressieve Bot V3 (DOGE+SOL) — {now_str} CET"}
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn",
                 "text": f"*Portfolio V3*\n€{total_val:.2f}  {ret_icon} {ret_pct:>+.1f}%"},
                {"type": "mrkdwn",
                 "text": f"*Vrij kapitaal*\n€{state['capital_eur']:.2f}"},
                {"type": "mrkdwn",
                 "text": f"*Trades*\n{state['total_trades']}  |  WR {wr:.1f}%  |  PF {pf:.2f}"},
                {"type": "mrkdwn",
                 "text": f"*Fear & Greed*\n{fng['score']} — {fng['label']}"},
            ]
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn",
                          "text": ":rocket: V3 sizing: 25% zwak | 40% normaal | 60% sterk"}]
        },
        {"type": "divider"},
    ]

    if state["open_positions"]:
        pos_text = ""
        for t, p in state["open_positions"].items():
            cv   = float(p.get("current_val", float(p["units"]) * float(p["entry_px"])))
            upnl = cv - float(p["invest"])
            sign = "+" if upnl >= 0 else ""
            size_pct = float(p["invest"]) / START_CAP * 100
            pos_text += (f"`{_name(t):<5}` {p['strategy'].upper()}  "
                         f"{size_pct:.0f}%  "
                         f"entry €{float(p['entry_px']):.4f}  "
                         f"stop €{float(p['stop_px']):.4f}  "
                         f"uPnL {sign}{upnl:.2f}€\n")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f":file_folder: *Open posities ({len(state['open_positions'])}/2)*\n{pos_text.strip()}"}
        })
    else:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": ":file_folder: Geen open posities"}
        })

    if exits:
        exit_text = ""
        for e in exits:
            icon = ":white_check_mark:" if e["pnl_eur"] >= 0 else ":x:"
            exit_text += (f"{icon} `{_name(e['ticker'])}` SELL "
                          f"€{e['exit_price']:.4f}  "
                          f"PnL {e['pnl_pct']:>+.1f}% ({e['pnl_eur']:>+.2f}€)  "
                          f"_{e['exit_reason']}_\n")
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f":closed_lock_with_key: *Exits deze candle*\n{exit_text.strip()}"}
        })

    if signals:
        sig_text = ""
        for s in signals:
            llm  = s.get("llm", {})
            blk  = s.get("entry_blocked", False)
            icon = ":no_entry_sign:" if blk else ":white_check_mark:"
            sig_text += (f"{icon} `{_name(s['ticker']):<5}` "
                         f"{s['signal_type'].upper()}  regime={s['regime']}  "
                         f"LLM *{llm.get('beslissing','?')}*  "
                         f"conf={llm.get('confidence',0):.2f}  "
                         f"{llm.get('nieuws_sentiment','?')}\n")
            if llm.get("reden"):
                sig_text += f"  _{llm['reden'][:100]}_\n"
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f":bell: *Signalen deze candle*\n{sig_text.strip()}"}
        })
    else:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": ":bell: Geen nieuwe signalen"}
        })

    try:
        r = requests.post(url, json={"blocks": blocks}, timeout=10)
        if r.status_code != 200:
            print(f"  [WARN] Slack HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"  [WARN] Slack fout: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print(f"  Agressieve Bot V3 (DOGE+SOL)  —  "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print("=" * 65)

    _init_clients()

    print("\n  1. Portfolio state laden (state_v3.json) …")
    state = load_state()
    print(f"     Kapitaal: €{state['capital_eur']:.2f}  |  "
          f"Posities: {len(state['open_positions'])}  |  "
          f"Trades: {state['total_trades']}")

    print("\n  2. Live data ophalen (Bitvavo) …")
    raw_frames = fetch_all_candles()

    failed = [t for t, df in raw_frames.items() if df is None]
    if len(failed) == len(TICKERS):
        print("  [FOUT] Alle Bitvavo requests mislukt — bot stopt.")
        sys.exit(1)
    if failed:
        print(f"  [WARN] Geen data voor: {', '.join(failed)}")

    print("\n  3. Indicatoren berekenen …")
    indicators = {}
    for t, df in raw_frames.items():
        ind = add_indicators(df) if df is not None else None
        indicators[t] = ind
        if ind is not None:
            print(f"     {t:<12}  {len(ind)} bruikbare bars")
        else:
            print(f"     {t:<12}  te weinig data (min {MA200_WIN+10} bars nodig)")

    print("\n  4. Regime detectie …")
    thresholds = load_thresholds()
    regime_map: dict[str, str] = {}
    for t in TICKERS:
        df = indicators.get(t)
        if df is None or df.empty:
            regime_map[t] = "neutraal"
            continue
        row  = last_row(df)
        name = _name(t)
        thr  = thresholds.get(name, {"p33": 0.0, "p67": 0.0})
        sc   = regime_score(row)
        reg  = get_regime(sc, thr["p33"], thr["p67"])
        regime_map[t] = reg
        print(f"     {t:<12}  score={sc:>+.4f}  →  {reg}")
    port_reg = portfolio_regime(regime_map)
    print(f"     Portfolio-regime: {port_reg}")

    print("\n  5. Exits controleren …")
    state, exits = check_exits(state, indicators)
    if not exits:
        print("     Geen exits.")

    fng = fetch_fng()
    print(f"\n  Fear & Greed: {fng['score']} ({fng['label']})")

    print("\n  6. Entry signalen genereren + LLM filter …")
    state, signals = generate_entries(state, indicators, regime_map,
                                      thresholds, fng)
    if not signals:
        print("     Geen entry signalen.")

    print("\n  7. State opslaan in GitHub (state_v3.json) …")
    save_state(state)

    print("\n  8. Slack notificatie …")
    send_slack(state, exits, signals, fng)

    total_val = state["capital_eur"] + sum(
        float(p.get("current_val", p.get("invest", 0)))
        for p in state["open_positions"].values())
    print(f"\n{'='*65}")
    print(f"  Portfolio V3: €{total_val:.2f}  ({(total_val-START_CAP)/START_CAP*100:>+.1f}%)")
    print(f"  Open posities: {len(state['open_positions'])}  |  "
          f"Exits: {len(exits)}  |  Signalen: {len(signals)}")
    print(f"{'='*65}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
