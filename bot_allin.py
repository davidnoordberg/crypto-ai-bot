"""
All-In C Paper Trading Bot
===========================
Assets  : DOGE-EUR en SOL-EUR via Bitvavo publieke API.
Logica  : Winner takes all. Als beide crypto's een entry signaal geven,
          kiest het systeem degene met de hoogste confidence score
          (RSI momentum + MACD histogram + volume). Die krijgt 95% van
          beschikbaar kapitaal. Als slechts één signaal actief is: die
          krijgt 95%. Max één open positie tegelijk.
Strategie: Hybride V4+MR. V4 momentum met 3× ATR trailing stop in
          bull/neutraal regime. Mean reversion in bear regime.
State   : state_allin.json in GitHub repo (GitHub Contents API).
Schedule: 30 */4 * * * (30 minuten na Bot 1).
Notif   : Slack Incoming Webhook.

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

from shared import check_meta_config
from ta.volatility import AverageTrueRange, BollingerBands

try:
    import anthropic as _anthropic
    _ANTHR_CLIENT: Optional[_anthropic.Anthropic] = None
except ImportError:
    _anthropic = None
    _ANTHR_CLIENT = None

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS   = ["DOGE-EUR", "SOL-EUR"]
START_CAP = 500.0
FEE_RT    = 0.004
MIN_POS   = 3.0
MAX_OPEN  = 1          # winner takes all → max 1 positie
ALLIN_FRAC = 0.95      # 95% van beschikbaar kapitaal per trade

GH_API        = "https://api.github.com"
GH_REPO       = os.environ.get("GITHUB_REPOSITORY", "davidnoordberg/crypto-ai-bot")
GH_STATE_FILE = "state_allin.json"

V4_RSI_LO    = 45; V4_RSI_HI = 75
V4_ATR_ENTRY = 4.0
V4_ATR_TRAIL = 3.0
V4_MA200_MAX_BELOW = 0.15

MR_RSI_THR = 35
MR_BB_WIN  = 30; MR_BB_SIG = 2.5
MR_ATR     = 2.0

ATR_WIN = 14; MA50_WIN = 50; MA200_WIN = 200
MA200_LAG = 10; VOL_WIN = 20
MACD_F, MACD_S, MACD_SIG_W = 12, 26, 9

EXPO_BULL_NEUT = 0.95   # Bij bull/neutraal: tot 95% geïnvesteerd
EXPO_BEAR      = 0.40   # Bij bear: max 40%

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
        r = requests.get(url, params={"interval": interval, "limit": limit}, timeout=15)
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
        print(f"{len(df)} bars" if df is not None else "MISLUKT")
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


# ── Regime ────────────────────────────────────────────────────────────────────
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


def mr_entry_check(row) -> bool:
    close = float(row["Close"])
    return (float(row["rsi"]) < MR_RSI_THR
            and close < float(row["bb_lower"]) - float(row["bb_std"])
            and float(row["Volume"]) > float(row["vol_ma"]))


def v4_exit_check(row, stop_px: float) -> bool:
    return float(row["Close"]) <= stop_px


def mr_exit_check(row, stop_px: float) -> bool:
    return (float(row["rsi"]) > 50
            or float(row["Close"]) >= float(row["bb_mid"])
            or float(row["Close"]) <= stop_px)


# ── Winner-takes-all: confidence score ────────────────────────────────────────
def v4_confidence(row) -> float:
    """
    Composite confidence voor winner selectie.
    Combinatie van RSI momentum, MACD histogram kracht en volume ratio.
    Waarde 0.0–1.0.
    """
    rsi        = float(row["rsi"])
    atr        = float(row["atr"])
    macd_hist  = float(row["macd_hist"])
    vol_ratio  = float(row["vol_ratio"])
    rsi_score  = np.clip((rsi - V4_RSI_LO) / (V4_RSI_HI - V4_RSI_LO), 0.0, 1.0)
    hist_score = min(abs(macd_hist) / (atr + 1e-9), 1.0)
    vol_score  = min(vol_ratio / 2.0, 1.0)
    return float((rsi_score + hist_score + vol_score) / 3.0)


def mr_confidence(row) -> float:
    """Confidence voor MR signalen: hoe dieper RSI/BB oversold, hoe hoger."""
    rsi       = float(row["rsi"])
    close     = float(row["Close"])
    bb_lower  = float(row["bb_lower"])
    bb_std    = float(row["bb_std"])
    rsi_score = np.clip((MR_RSI_THR - rsi) / MR_RSI_THR, 0.0, 1.0)
    bb_depth  = np.clip((bb_lower - close) / (bb_std + 1e-9), 0.0, 1.0)
    return float((rsi_score + bb_depth) / 2.0)


# ── Nieuws & LLM ──────────────────────────────────────────────────────────────
def fetch_news(ticker: str) -> list[str]:
    coin_map = {"doge": "dogecoin", "sol": "solana"}
    coin     = _name(ticker).lower()
    slug     = coin_map.get(coin, coin)
    url      = f"https://cryptopanic.com/news/{slug}/rss/"
    try:
        feed = feedparser.parse(url)
        heads = [e.title.strip() for e in feed.entries[:5] if e.get("title")]
        if heads:
            return heads
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
    body   = abs(c - o); range_ = h - l or 1e-9
    dir_   = "bullish" if c > o else "bearish"
    size   = "groot" if body / range_ > 0.6 else ("klein" if body / range_ < 0.2 else "normaal")
    return f"{dir_} {size} lichaam"


def _ma200_label(row) -> str:
    ma200     = float(row["ma200"]); ma200_lag = float(row["ma200_lag"])
    slope_pct = (ma200 - ma200_lag) / ma200_lag * 100 if ma200_lag else 0
    if slope_pct > 0.1:  return f"stijgend (+{slope_pct:.2f}% / 10 bars)"
    if slope_pct < -0.1: return f"dalend ({slope_pct:.2f}% / 10 bars)"
    return "zijwaarts"


def _prijs_label(row) -> str:
    close = float(row["Close"]); ma200 = float(row["ma200"])
    pct   = (close - ma200) / ma200 * 100
    if pct > 5:  return f"+{pct:.1f}% boven MA200"
    if pct > 0:  return f"+{pct:.1f}% net boven MA200"
    return f"{pct:.1f}% onder MA200"


def ask_llm(ticker: str, row, regime: str, strategy: str,
            competitor: str, fng: dict, headlines: list) -> dict:
    """
    Vraag Haiku: KOOP of NIETS voor dit all-in signaal.
    competitor: naam van de andere crypto die ook overwogen werd (of "").
    """
    fallback = {"beslissing": "KOOP", "confidence": 0.5,
                "nieuws_sentiment": "neutraal",
                "reden": "LLM niet beschikbaar"}

    if _ANTHR_CLIENT is None:
        return fallback

    hl_txt       = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines[:5]))
    competitor_txt = (f"Concurrent (niet gekozen): {competitor}"
                      if competitor else "Enig signaal op dit moment")

    prompt = f"""Crypto: {_name(ticker)}
Datum: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC
Entry prijs: {float(row['Close']):.6f} EUR
Regime: {regime}
Strategie: {strategy.upper()} (All-in C — winner takes all)
Positiegrootte: VAST 95% van beschikbaar kapitaal
{competitor_txt}

Technische context:
RSI: {float(row['rsi']):.1f}
MACD histogram: {float(row['macd_hist']):.6f}
ATR ratio: {float(row['atr'])/float(row['Close'])*100:.2f}% van prijs
Volume ratio: {float(row['vol_ratio']):.2f}x gemiddelde
MA200 richting: {_ma200_label(row)}
Prijspositie: {_prijs_label(row)}
Laatste candle: {_candle_label(row)}

Marktsentiment:
Fear & Greed Index: {fng['score']} ({fng['label']})

Recent nieuws (laatste 24u):
{hl_txt}

Dit is een ALL-IN signaal: 95% van het kapitaal gaat in deze ene trade.
Beoordeel extra kritisch: alleen instappen als de setup sterk is.
Geef je antwoord UITSLUITEND als geldig JSON (geen tekst erbuiten):
{{
  "beslissing": "KOOP" of "NIETS",
  "confidence": 0.0 tot 1.0,
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
        ns = str(parsed.get("nieuws_sentiment", "neutraal")).lower()
        if ns not in ("bullish", "neutraal", "bearish"): ns = "neutraal"
        return {"beslissing": bes,
                "confidence":  float(parsed.get("confidence", 0.5)),
                "nieuws_sentiment": ns,
                "reden": str(parsed.get("reden", ""))[:300]}
    except Exception as e:
        print(f"  [WARN] LLM fout voor {ticker}: {e}")
        return fallback


# ── GitHub state ───────────────────────────────────────────────────────────────
_INITIAL_STATE = {
    "bot_id":            "allin",
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
        data    = r.json()
        content = json.loads(base64.b64decode(data["content"]).decode())
        return content, data["sha"]
    except Exception as e:
        print(f"  [WARN] GitHub read {path}: {e}")
        return None, None


def _gh_write(path: str, content: dict, sha: Optional[str], message: str):
    url     = f"{GH_API}/repos/{GH_REPO}/contents/{path}"
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
        print("  [INFO] state_allin.json bestaat nog niet — initialiseer met startkapitaal.")
        _state_sha = None
        return _INITIAL_STATE.copy()
    _state_sha = sha
    state = _INITIAL_STATE.copy()
    state.update(content)
    return state


def save_state(state: dict):
    global _state_sha

    # Bereken portfolio waarde VOOR het strippen van current_val
    open_positions_value = sum(
        float(p.get("current_val", float(p["units"]) * float(p["entry_px"])))
        for p in state["open_positions"].values())
    portfolio_total = state["capital_eur"] + open_positions_value
    ret_pct = (portfolio_total - START_CAP) / START_CAP * 100

    gw = state["gross_win"]; gl = state["gross_loss"]
    pf = gw / gl if gl > 0 else 0.0
    wr = state["wins"] / state["total_trades"] * 100 \
         if state["total_trades"] > 0 else 0.0

    state["win_rate"]             = round(wr, 4)
    state["profit_factor"]        = round(pf, 4)
    state["total_return_pct"]     = round(ret_pct, 4)
    state["portfolio_total"]      = round(portfolio_total, 4)
    state["open_positions_value"] = round(open_positions_value, 4)

    clean_positions = {}
    for t, p in state["open_positions"].items():
        clean_positions[t] = {k: v for k, v in p.items() if k != "current_val"}
    state["open_positions"] = clean_positions

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    message = (f"allin: {now_str} — "
               f"€{state['capital_eur']:.2f} vrij, "
               f"{len(state['open_positions'])} pos open, "
               f"{state['total_trades']} trades")

    _gh_write(GH_STATE_FILE, state, _state_sha, message)
    print(f"  [GitHub] state_allin.json opgeslagen  "
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
    clean = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sig.items()}
    state.setdefault("signals", []).append(clean)
    blk = sig.get("entry_blocked", False)
    print(f"  [SIGNAL] {sig['ticker']:<12}  {sig['signal_type']}  "
          f"regime={sig['regime']}  "
          f"{'GEBLOKKEERD' if blk else 'WINNER' if sig.get('is_winner') else 'VERLIEZER'}")


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


# ── Exits ─────────────────────────────────────────────────────────────────────
def check_exits(state: dict, indicators: dict) -> tuple[dict, list]:
    exits  = []
    to_del = []
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
                "confidence_at_entry":  pos.get("confidence_at_entry", 0.0),
                "llm_beslissing":       pos.get("llm_beslissing", ""),
                "llm_confidence":       pos.get("llm_confidence", 0.0),
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


# ── Entries — winner takes all ─────────────────────────────────────────────────
def generate_entries(state: dict, indicators: dict,
                     regime_map: dict, thresholds: dict,
                     fng: dict) -> tuple[dict, list]:
    """
    Verzamel alle entry signalen voor DOGE en SOL.
    Als meerdere signalen: kies degene met hoogste confidence score.
    De winner krijgt 95% van beschikbaar kapitaal.
    """
    signals  = []
    now_str  = datetime.now(timezone.utc).isoformat()

    # Al een open positie? Niets doen.
    if len(state["open_positions"]) >= MAX_OPEN:
        print("  Positielimiet bereikt (max 1) — geen nieuwe entries.")
        return state, signals

    port_reg  = portfolio_regime(regime_map)
    max_expo  = EXPO_BULL_NEUT if port_reg in ("bull", "neutraal") else EXPO_BEAR
    total_val = state["capital_eur"]
    if total_val / total_val < max_expo:  # vrijwel altijd True als geen open positie
        pass

    # Verzamel kandidaten
    candidates = []
    for ticker in TICKERS:
        df = indicators.get(ticker)
        if df is None or df.empty:
            continue
        row    = last_row(df)
        t_reg  = regime_map.get(ticker, "neutraal")
        close  = float(row["Close"])
        atr    = float(row["atr"])
        strategy = None

        if t_reg in ("bull", "neutraal"):
            if v4_entry_check(row):
                strategy   = "v4"
                conf_score = v4_confidence(row)
        else:  # bear
            if mr_entry_check(row):
                strategy   = "mr"
                conf_score = mr_confidence(row)

        if strategy is None:
            continue

        candidates.append({
            "ticker":     ticker,
            "strategy":   strategy,
            "regime":     t_reg,
            "close":      close,
            "atr":        atr,
            "conf_score": conf_score,
            "row":        row,
        })

    if not candidates:
        print("  Geen entry signalen voor DOGE of SOL.")
        return state, signals

    # Sorteer op confidence — hoogste wint
    candidates.sort(key=lambda x: x["conf_score"], reverse=True)
    winner     = candidates[0]
    losers     = candidates[1:]  # normaal max 1 verliezer

    winner_name = _name(winner["ticker"])
    loser_names = [_name(c["ticker"]) for c in losers]
    competitor  = loser_names[0] if loser_names else ""

    print(f"  Kandidaten: {[_name(c['ticker']) for c in candidates]}")
    print(f"  Winner: {winner_name}  (conf {winner['conf_score']:.3f})"
          + (f"  vs {competitor} (conf {losers[0]['conf_score']:.3f})" if competitor else ""))

    # LLM filter op de winner
    headlines = fetch_news(winner["ticker"])
    llm = ask_llm(winner["ticker"], winner["row"], winner["regime"],
                  winner["strategy"], competitor, fng, headlines)

    is_blocked = llm["beslissing"] == "NIETS"

    # Log alle kandidaten als signaal
    for i, cand in enumerate(candidates):
        is_win = (i == 0)
        sc     = regime_score(cand["row"])
        thr    = thresholds.get(_name(cand["ticker"]), {"p33": 0.0, "p67": 0.0})
        sig_row = {
            "timestamp":       now_str,
            "ticker":          cand["ticker"],
            "regime":          cand["regime"],
            "rsi":             round(float(cand["row"]["rsi"]), 2),
            "macd_hist":       round(float(cand["row"]["macd_hist"]), 6),
            "atr_ratio":       round(float(cand["row"]["atr"]) / cand["close"] * 100, 4),
            "volume_ratio":    round(float(cand["row"]["vol_ratio"]), 4),
            "signal_type":     cand["strategy"],
            "conf_score":      round(cand["conf_score"], 4),
            "is_winner":       is_win,
            "entry_blocked":   is_blocked if is_win else True,
            "block_reason":    (llm["reden"] if is_blocked else "") if is_win else "niet_gekozen",
            "llm_beslissing":  llm["beslissing"] if is_win else "NIETS",
            "llm_confidence":  llm["confidence"] if is_win else 0.0,
        }
        log_signal(sig_row, state)
        signals.append(sig_row)

    if is_blocked:
        print(f"  SKIP  {winner['ticker']:<12}  LLM NIETS  ({llm['reden'][:60]})")
        return state, signals

    # Positie openen: 95% van beschikbaar kapitaal
    invest = state["capital_eur"] * ALLIN_FRAC
    if invest < MIN_POS:
        print(f"  SKIP  {winner['ticker']:<12}  te weinig kapitaal (€{invest:.2f})")
        return state, signals

    fee   = invest * FEE_RT / 2
    units = (invest - fee) / winner["close"]
    stop  = (winner["close"] - V4_ATR_ENTRY * winner["atr"]
             if winner["strategy"] == "v4"
             else winner["close"] - MR_ATR * winner["atr"])

    pos = {
        "strategy":             winner["strategy"],
        "units":                round(units, 8),
        "entry_px":             round(winner["close"], 6),
        "stop_px":              round(stop, 6),
        "trail_high":           round(winner["close"], 6),
        "invest":               round(invest, 4),
        "entry_ts":             now_str,
        "regime_at_entry":      winner["regime"],
        "confidence_at_entry":  round(winner["conf_score"], 4),
        "competitor":           competitor,
        "llm_beslissing":       llm["beslissing"],
        "llm_confidence":       llm["confidence"],
        "llm_nieuws_sentiment": llm["nieuws_sentiment"],
        "llm_reden":            llm["reden"],
    }
    state["capital_eur"] -= invest
    state["open_positions"][winner["ticker"]] = pos

    trade = {
        "timestamp":            now_str,
        "ticker":               winner["ticker"],
        "action":               "BUY",
        "price":                round(winner["close"], 6),
        "position_size_pct":    round(ALLIN_FRAC * 100, 2),
        "position_size_eur":    round(invest, 4),
        "entry_price":          round(winner["close"], 6),
        "exit_price":           None,
        "pnl_pct":              None,
        "pnl_eur":              None,
        "exit_reason":          None,
        "regime":               winner["regime"],
        "strategy":             winner["strategy"],
        "confidence_at_entry":  round(winner["conf_score"], 4),
        "competitor":           competitor,
        "llm_beslissing":       llm["beslissing"],
        "llm_confidence":       llm["confidence"],
        "llm_nieuws_sentiment": llm["nieuws_sentiment"],
        "llm_reden":            llm["reden"],
    }
    log_trade(trade, state)
    print(f"  BUY   {winner['ticker']:<12}  {winner['strategy']:<4}  {winner['regime']:<9}  "
          f"€{winner['close']:.4f}  {ALLIN_FRAC*100:.0f}%  "
          f"conf={winner['conf_score']:.3f}  LLM={llm['confidence']:.2f}")

    return state, signals


# ── Slack notificatie ──────────────────────────────────────────────────────────
def send_slack(state: dict, exits: list, signals: list, fng: dict):
    if not exits and not signals and not state.get("open_positions"):
        return
    url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not url:
        print("  [INFO] Geen SLACK_WEBHOOK_URL — geen notificatie.")
        return

    now_cet              = datetime.now(timezone.utc) + timedelta(hours=2)
    now_str              = now_cet.strftime("%d-%m-%Y %H:%M")
    gw                   = state["gross_win"]; gl = state["gross_loss"]
    pf                   = round(gw / gl, 2) if gl > 0 else 0.0
    wr                   = state["win_rate"]
    portfolio_total      = state.get("portfolio_total",
                               state["capital_eur"] + state.get("open_positions_value", 0.0))
    open_positions_value = state.get("open_positions_value", 0.0)
    ret_pct              = state.get("total_return_pct",
                               (portfolio_total - START_CAP) / START_CAP * 100)
    ret_icon             = ":arrow_up_small:" if ret_pct >= 0 else ":arrow_down_small:"

    # Open positie samenvatting
    if state["open_positions"]:
        ticker, pos = next(iter(state["open_positions"].items()))
        cv    = float(pos.get("current_val", float(pos["units"]) * float(pos["entry_px"])))
        upnl  = cv - float(pos["invest"])
        sign  = "+" if upnl >= 0 else ""
        pos_text = (f"`{_name(ticker)}` {pos['strategy'].upper()}  "
                    f"entry €{float(pos['entry_px']):.4f}  "
                    f"stop €{float(pos['stop_px']):.4f}  "
                    f"uPnL {sign}{upnl:.2f}€")
    else:
        pos_text = "Geen"

    # Signaal samenvatting
    winner_sigs = [s for s in signals if s.get("is_winner")]
    if winner_sigs:
        ws  = winner_sigs[0]
        llm_bes  = ws.get("llm_beslissing", "?")
        llm_conf = ws.get("llm_confidence", 0.0)
        llm_blk  = ws.get("entry_blocked", False)
        sig_text = (f"`{_name(ws['ticker'])}` {ws['signal_type'].upper()}  "
                    f"conf={ws['conf_score']:.3f}  LLM *{llm_bes}* ({llm_conf:.2f})")
    else:
        sig_text = "Geen nieuwe signalen"

    llm_reden = ""
    if winner_sigs:
        # Haal reden op uit trade log (meest recente BUY of geblokkeerd signaal)
        recent_trades = [t for t in state.get("trades", []) if t.get("action") == "BUY"]
        if recent_trades:
            llm_reden = recent_trades[-1].get("llm_reden", "")

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text",
                     "text": f":rocket: Bot ALL-IN C (DOGE+SOL) — {now_str} CET"}
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn",
                 "text": f":briefcase: *Portfolio*\n€{portfolio_total:.2f}  "
                         f"{ret_icon} {ret_pct:>+.1f}% sinds start"},
                {"type": "mrkdwn",
                 "text": f":chart_with_upwards_trend: *Belegd*\n€{open_positions_value:.2f}"},
                {"type": "mrkdwn",
                 "text": f":moneybag: *Vrij kapitaal*\n€{state['capital_eur']:.2f}"},
                {"type": "mrkdwn",
                 "text": f":bar_chart: *Trades*\n{state['total_trades']}  |  "
                         f"Win rate: {wr:.1f}%  |  PF: {pf:.2f}"},
            ]
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn",
                 "text": f":scream: *Fear & Greed*\n{fng['score']} — {fng['label']}"},
            ]
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f":chart_with_upwards_trend: *Open positie*\n{pos_text}"}
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f":bell: *Signaal*\n{sig_text}"}
        },
    ]

    if llm_reden:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f":robot_face: *LLM*\n_{llm_reden[:200]}_"}
        })

    # Exits
    if exits:
        exit_lines = []
        for e in exits:
            icon = ":white_check_mark:" if e["pnl_eur"] >= 0 else ":x:"
            exit_lines.append(
                f"{icon} `{_name(e['ticker'])}` SELL €{e['exit_price']:.4f}  "
                f"PnL {e['pnl_pct']:>+.1f}% ({e['pnl_eur']:>+.2f}€)  _{e['exit_reason']}_"
            )
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": ":closed_lock_with_key: *Exits deze candle*\n"
                             + "\n".join(exit_lines)}
        })

    try:
        r = requests.post(url, json={"blocks": blocks}, timeout=10)
        if r.status_code != 200:
            print(f"  [WARN] Slack HTTP {r.status_code}: {r.text[:200]}")
        else:
            print("  [Slack] Notificatie verzonden.")
    except Exception as e:
        print(f"  [WARN] Slack fout: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print(f"  All-In C Bot (DOGE+SOL)  —  "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print("=" * 65)

    _init_clients()

    if not check_meta_config("allin"):
        return

    print("\n  1. Portfolio state laden …")
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
            print(f"     {t:<12}  te weinig data")

    print("\n  4. Regime detectie …")
    thresholds = load_thresholds()
    regime_map: dict[str, str] = {}
    for t in TICKERS:
        df = indicators.get(t)
        if df is None or df.empty:
            regime_map[t] = "neutraal"
            continue
        row  = last_row(df)
        thr  = thresholds.get(_name(t), {"p33": 0.0, "p67": 0.0})
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

    print("\n  6. Winner-takes-all entry logica …")
    state, signals = generate_entries(state, indicators, regime_map, thresholds, fng)
    if not signals:
        print("     Geen signalen.")

    print("\n  7. State opslaan (state_allin.json) …")
    save_state(state)

    print("\n  8. Slack notificatie …")
    send_slack(state, exits, signals, fng)

    print("\n  Klaar.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
