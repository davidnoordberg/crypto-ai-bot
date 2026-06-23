"""
Fear Contrarian Paper Trading Bot
===================================
Koopt tijdens extreme marktpaniek (Fear & Greed < 20, RSI < 45).
Verkoopt wanneer angst wegebt (FNG > 50), trailing stop (3×ATR) of RSI > 65.

Assets  : BTC-EUR, ETH-EUR, SOL-EUR, DOGE-EUR via Bitvavo publieke API.
Bot ID  : 'fear'
State   : state_fear.json in GitHub repo.
Schedule: 45 */4 * * * (45 minuten na Bot 1).

Entry:
  - Fear & Greed < 20 (Extreme Fear)
  - RSI(14) < 45
  - Dynamic sizing: FNG 15-20 → 20%, FNG 10-15 → 30%, FNG < 10 → 40%
  - Max 3 open posities tegelijk

Exit (eerste die triggert):
  - FNG stijgt boven 50 → verkoop alle fear-posities
  - Trailing stop 3×ATR14
  - RSI stijgt boven 65

Herinstap cooldown: na exit wacht per ticker tot FNG opnieuw < 20.

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

warnings.filterwarnings("ignore")

from ta.momentum   import RSIIndicator
from ta.volatility import AverageTrueRange

from shared import check_meta_config

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS   = ["BTC-EUR", "ETH-EUR", "SOL-EUR", "DOGE-EUR"]
START_CAP = 500.0
FEE_RT    = 0.004
MIN_POS   = 3.0
MAX_OPEN  = 3

ATR_WIN    = 14
ATR_TRAIL  = 3.0
RSI_WIN    = 14

FNG_ENTRY_MAX = 20    # entry alleen als FNG onder dit niveau
FNG_EXIT_MIN  = 50    # exit alle posities als FNG boven dit niveau
RSI_ENTRY_MAX = 45    # entry alleen als RSI onder dit niveau
RSI_EXIT_MIN  = 65    # exit als RSI boven dit niveau

GH_API        = "https://api.github.com"
GH_REPO       = os.environ.get("GITHUB_REPOSITORY", "davidnoordberg/crypto-ai-bot")
GH_STATE_FILE = "state_fear.json"

BITVAVO_BASE = "https://api.bitvavo.com/v2"
CANDLE_LIMIT = 250
MA200_WIN    = 200    # voor indicator warmup check


def fng_size(fng_score: int) -> float:
    """Dynamic positiegrootte op basis van FNG niveau."""
    if fng_score < 10:  return 0.40
    if fng_score < 15:  return 0.30
    return 0.20


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
    """Haal huidige Fear & Greed Index op."""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        d = r.json()["data"][0]
        return {"score": int(d["value"]), "label": d["value_classification"]}
    except Exception:
        return {"score": 50, "label": "Neutraal"}


# ── Indicatoren ───────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < ATR_WIN + 5:
        return None
    df = df.copy()
    df["rsi"] = RSIIndicator(df["Close"], window=RSI_WIN).rsi()
    df["atr"] = AverageTrueRange(df["High"], df["Low"], df["Close"],
                                 window=ATR_WIN).average_true_range()
    return df.dropna()


def last_row(df: pd.DataFrame):
    return df.iloc[-1] if df is not None and not df.empty else None


# ── GitHub state ───────────────────────────────────────────────────────────────
_INITIAL_STATE = {
    "bot_id":            "fear",
    "capital_eur":       START_CAP,
    "open_positions":    {},
    "cooldown_tickers":  [],   # tickers in cooldown (wachten op FNG < 20 opnieuw)
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
        print("  [INFO] state_fear.json bestaat nog niet — initialiseer met startkapitaal.")
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
    state["win_rate"]             = round(state["wins"] / state["total_trades"] * 100, 4) \
                                    if state["total_trades"] > 0 else 0.0
    state["profit_factor"]        = round(gw / gl, 4) if gl > 0 else 0.0
    state["total_return_pct"]     = round(ret_pct, 4)
    state["portfolio_total"]      = round(portfolio_total, 4)
    state["open_positions_value"] = round(open_positions_value, 4)

    clean_pos = {}
    for t, p in state["open_positions"].items():
        clean_pos[t] = {k: v for k, v in p.items() if k != "current_val"}
    state["open_positions"] = clean_pos

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    message = (f"fear: {now_str} — "
               f"€{state['capital_eur']:.2f} vrij, "
               f"{len(state['open_positions'])} pos open, "
               f"{state['total_trades']} trades")
    _gh_write(GH_STATE_FILE, state, _state_sha, message)
    print(f"  [GitHub] state_fear.json opgeslagen  "
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
    trail_high = float(pos.get("trail_high", pos["entry_px"]))
    if close > trail_high:
        pos["trail_high"] = close
        new_stop = close - ATR_TRAIL * atr
        if new_stop > float(pos["stop_px"]):
            pos["stop_px"] = new_stop
    return pos


# ── Exits ─────────────────────────────────────────────────────────────────────
def check_exits(state: dict, indicators: dict, fng: dict) -> tuple[dict, list]:
    """
    Controleert drie exit condities:
    1. FNG > 50  → verkoop ALLE fear-posities (angst wegebt)
    2. Trailing stop 3×ATR
    3. RSI > 65
    """
    exits   = []
    to_del  = []
    now_str = datetime.now(timezone.utc).isoformat()
    fng_score = fng["score"]

    for ticker, pos in list(state["open_positions"].items()):
        df = indicators.get(ticker)
        if df is None or df.empty:
            continue
        row   = last_row(df)
        close = float(row["Close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        _update_trail(pos, close, atr)

        reason = ""
        if fng_score >= FNG_EXIT_MIN:
            reason = f"fng_recovery:{fng_score}"
        elif close <= float(pos["stop_px"]):
            reason = "trail_stop"
        elif rsi >= RSI_EXIT_MIN:
            reason = f"rsi_exit:{rsi:.0f}"

        if reason:
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
                "timestamp":          now_str,
                "ticker":             ticker,
                "action":             "SELL",
                "price":              round(close, 6),
                "position_size_pct":  round(float(pos["invest"]) / START_CAP * 100, 2),
                "position_size_eur":  round(float(pos["invest"]), 4),
                "entry_price":        round(float(pos["entry_px"]), 6),
                "exit_price":         round(close, 6),
                "pnl_pct":            round(pnl_pct, 4),
                "pnl_eur":            round(pnl, 4),
                "exit_reason":        reason,
                "entry_fng":          pos.get("entry_fng", 0),
                "exit_fng":           fng_score,
            }
            log_trade(trade, state)
            exits.append(trade)
            to_del.append(ticker)

            # Cooldown activeren: wacht op FNG < 20 opnieuw
            cooldown = state.get("cooldown_tickers", [])
            if ticker not in cooldown:
                cooldown.append(ticker)
            state["cooldown_tickers"] = cooldown

            print(f"  EXIT  {ticker:<12}  {reason:<22}  "
                  f"€{close:.4f}  PnL {pnl_pct:>+.1f}% ({pnl:>+.2f}€)")
        else:
            pos["current_val"] = float(pos["units"]) * close
            state["open_positions"][ticker] = pos

    for t in to_del:
        del state["open_positions"][t]

    return state, exits


# ── Cooldown beheer ───────────────────────────────────────────────────────────
def update_cooldowns(state: dict, fng: dict):
    """
    Reset cooldown voor tickers waarbij FNG nu weer onder 20 is.
    De cooldown was actief na een exit en wacht op nieuw extreem fear signaal.
    """
    if fng["score"] < FNG_ENTRY_MAX:
        state["cooldown_tickers"] = []
        print(f"  FNG={fng['score']} < {FNG_ENTRY_MAX}: alle cooldowns gereset.")
    else:
        n = len(state.get("cooldown_tickers", []))
        if n:
            print(f"  FNG={fng['score']} >= {FNG_ENTRY_MAX}: {n} ticker(s) in cooldown.")


# ── Entries ────────────────────────────────────────────────────────────────────
CORR_THRESHOLD  = 0.85
CORR_CANDLES    = 48


def correlation_block(ticker: str, indicators: dict, open_positions: dict) -> Optional[str]:
    """Geeft reden terug als ticker gecorreleerd is met een open positie, anders None."""
    if not open_positions:
        return None
    df_new = indicators.get(ticker)
    if df_new is None or len(df_new) < CORR_CANDLES:
        return None
    ret_new = df_new["Close"].pct_change().dropna().iloc[-CORR_CANDLES:]
    for open_ticker in open_positions:
        if open_ticker == ticker:
            continue
        df_open = indicators.get(open_ticker)
        if df_open is None or len(df_open) < CORR_CANDLES:
            continue
        ret_open = df_open["Close"].pct_change().dropna().iloc[-CORR_CANDLES:]
        aligned = ret_new.align(ret_open, join="inner")[0]
        aligned_open = ret_new.align(ret_open, join="inner")[1]
        if len(aligned) < 10:
            continue
        corr = float(aligned.corr(aligned_open))
        if corr >= CORR_THRESHOLD:
            return (f"{_name(open_ticker)} en {_name(ticker)} correlatie {corr:.2f}")
    return None


def generate_entries(state: dict, indicators: dict, fng: dict) -> tuple[dict, list, list]:
    """
    Entry condities:
    - FNG < 20 (Extreme Fear)
    - RSI < 45
    - Ticker niet in cooldown
    - Max 3 open posities
    - Correlation agent: blokkeer als correlatie >= 0.85 met open positie
    """
    signals        = []
    corr_blocks    = []
    now_str        = datetime.now(timezone.utc).isoformat()
    fng_score      = fng["score"]

    if fng_score >= FNG_ENTRY_MAX:
        print(f"  FNG={fng_score} >= {FNG_ENTRY_MAX}: geen entry condities.")
        return state, signals, corr_blocks

    frac    = fng_size(fng_score)
    cooldown = state.get("cooldown_tickers", [])

    for ticker in TICKERS:
        if len(state["open_positions"]) >= MAX_OPEN:
            print(f"  Max posities bereikt ({MAX_OPEN}) — stop entries.")
            break

        if ticker in state["open_positions"]:
            continue

        if ticker in cooldown:
            print(f"  {ticker:<12}  in cooldown — skip")
            continue

        df = indicators.get(ticker)
        if df is None or df.empty:
            continue

        # Correlation check
        corr_reason = correlation_block(ticker, indicators, state["open_positions"])
        if corr_reason:
            print(f"  {ticker:<12}  geblokkeerd door correlation agent: {corr_reason}")
            corr_blocks.append({"ticker": _name(ticker), "reden": corr_reason})
            continue

        row   = last_row(df)
        close = float(row["Close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if rsi >= RSI_ENTRY_MAX:
            print(f"  {ticker:<12}  RSI {rsi:.1f} >= {RSI_ENTRY_MAX} — skip")
            continue

        # Entry check geslaagd
        invest = state["capital_eur"] * frac
        if invest < MIN_POS:
            print(f"  {ticker:<12}  te weinig kapitaal (€{invest:.2f})")
            continue

        fee   = invest * FEE_RT / 2
        units = (invest - fee) / close
        stop  = close - ATR_TRAIL * atr

        pos = {
            "units":       round(units, 8),
            "entry_px":    round(close, 6),
            "stop_px":     round(stop, 6),
            "trail_high":  round(close, 6),
            "invest":      round(invest, 4),
            "entry_ts":    now_str,
            "entry_fng":   fng_score,
            "entry_rsi":   round(rsi, 2),
            "size_pct":    round(frac * 100, 1),
        }
        state["capital_eur"] -= invest
        state["open_positions"][ticker] = pos

        trade = {
            "timestamp":         now_str,
            "ticker":            ticker,
            "action":            "BUY",
            "price":             round(close, 6),
            "position_size_pct": round(frac * 100, 2),
            "position_size_eur": round(invest, 4),
            "entry_price":       round(close, 6),
            "exit_price":        None,
            "pnl_pct":           None,
            "pnl_eur":           None,
            "exit_reason":       None,
            "entry_fng":         fng_score,
            "entry_rsi":         round(rsi, 2),
        }
        log_trade(trade, state)

        sig = {
            "timestamp": now_str,
            "ticker":    ticker,
            "fng":       fng_score,
            "rsi":       round(rsi, 2),
            "atr":       round(atr, 6),
            "invest":    round(invest, 4),
            "size_pct":  round(frac * 100, 1),
            "stop_px":   round(stop, 6),
        }
        signals.append(sig)
        state.setdefault("signals", []).append(
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sig.items()}
        )

        print(f"  BUY   {ticker:<12}  FNG={fng_score}  RSI={rsi:.1f}  "
              f"€{close:.4f}  {frac*100:.0f}%  stop=€{stop:.4f}")

    return state, signals, corr_blocks


# ── Slack notificatie ──────────────────────────────────────────────────────────
def send_slack(state: dict, exits: list, signals: list, fng: dict, corr_blocks: list = []):
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
    n_pos                = len(state["open_positions"])

    # Open posities tekst
    if state["open_positions"]:
        pos_lines = []
        for t, p in state["open_positions"].items():
            cv   = float(p.get("current_val", float(p["units"]) * float(p["entry_px"])))
            upnl = cv - float(p["invest"])
            sign = "+" if upnl >= 0 else ""
            pos_lines.append(
                f"`{_name(t):<5}` entry €{float(p['entry_px']):.4f}  "
                f"stop €{float(p['stop_px']):.4f}  "
                f"uPnL {sign}{upnl:.2f}€  "
                f"FNG_in={p.get('entry_fng','?')}"
            )
        pos_text = "\n".join(pos_lines)
    else:
        pos_text = "Geen"

    # Signalen tekst
    if signals:
        sig_lines = [
            f"`{_name(s['ticker'])}` FNG={s['fng']}  RSI={s['rsi']:.1f}  "
            f"{s['size_pct']:.0f}%  stop=€{s['stop_px']:.4f}"
            for s in signals
        ]
        sig_text = "\n".join(sig_lines)
    elif fng["score"] < FNG_ENTRY_MAX:
        sig_text = f"FNG={fng['score']} — extreme fear actief maar geen RSI signalen"
    else:
        sig_text = f"Wacht op extreme fear (huidig FNG={fng['score']})"

    fng_icon = ":scream:" if fng["score"] < FNG_ENTRY_MAX else ":relieved:"

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text",
                     "text": f":skull_and_crossbones: Bot FEAR CONTRARIAN — {now_str} CET"}
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn",
                 "text": f":briefcase: *Portfolio*\n€{portfolio_total:.2f}  "
                         f"({'↑' if ret_pct >= 0 else '↓'} {ret_pct:>+.1f}% sinds start)"},
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
                 "text": f"{fng_icon} *Fear & Greed*\n{fng['score']} — {fng['label']}"},
                {"type": "mrkdwn",
                 "text": f":free: *Vrij kapitaal*\n€{state['capital_eur']:.2f}"},
            ]
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f":chart_with_upwards_trend: *Open posities ({n_pos}/{MAX_OPEN})*\n{pos_text}"}
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f":bell: *Signalen*\n{sig_text}"}
        },
    ]

    # Correlation blocks
    if corr_blocks:
        corr_lines = [
            f":link: Trade geblokkeerd door Correlation Agent\nTicker: {b['ticker']}\nReden: {b['reden']}"
            for b in corr_blocks
        ]
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n\n".join(corr_lines)}
        })

    # Exits
    if exits:
        exit_lines = []
        for e in exits:
            icon  = ":white_check_mark:" if (e.get("pnl_eur") or 0) >= 0 else ":x:"
            reason = e.get("exit_reason", "?")
            pnl_e  = e.get("pnl_eur", 0) or 0
            pnl_p  = e.get("pnl_pct", 0) or 0
            exit_lines.append(
                f"{icon} `{_name(e['ticker'])}` SELL €{e['exit_price']:.4f}  "
                f"PnL {pnl_p:>+.1f}% ({pnl_e:>+.2f}€)  _{reason}_  "
                f"FNG_in={e.get('entry_fng','?')}→{e.get('exit_fng','?')}"
            )
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": ":closed_lock_with_key: *Exits deze candle*\n"
                             + "\n".join(exit_lines)}
        })

    # Cooldown status
    cooldown = state.get("cooldown_tickers", [])
    if cooldown:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn",
                          "text": f":hourglass: Cooldown actief: "
                                  f"{', '.join(_name(t) for t in cooldown)} "
                                  f"(wacht op FNG < {FNG_ENTRY_MAX})"}]
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
    print(f"  Fear Contrarian Bot  —  "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print("=" * 65)

    if not check_meta_config("fear"):
        return

    print("\n  1. Portfolio state laden …")
    state = load_state()
    print(f"     Kapitaal: €{state['capital_eur']:.2f}  |  "
          f"Posities: {len(state['open_positions'])}  |  "
          f"Trades: {state['total_trades']}  |  "
          f"Cooldown: {state.get('cooldown_tickers', [])}")

    print("\n  2. Fear & Greed Index ophalen …")
    fng = fetch_fng()
    print(f"     FNG: {fng['score']} — {fng['label']}")

    print("\n  3. Live data ophalen (Bitvavo) …")
    raw_frames = fetch_all_candles()
    failed = [t for t, df in raw_frames.items() if df is None]
    if len(failed) == len(TICKERS):
        print("  [FOUT] Alle Bitvavo requests mislukt — bot stopt.")
        sys.exit(1)
    if failed:
        print(f"  [WARN] Geen data voor: {', '.join(failed)}")

    print("\n  4. Indicatoren berekenen …")
    indicators = {}
    for t, df in raw_frames.items():
        ind = add_indicators(df) if df is not None else None
        indicators[t] = ind
        if ind is not None:
            row = last_row(ind)
            print(f"     {t:<12}  {len(ind)} bars  "
                  f"RSI={float(row['rsi']):.1f}  "
                  f"ATR={float(row['atr']):.4f}")
        else:
            print(f"     {t:<12}  te weinig data")

    print("\n  5. Cooldown bijwerken …")
    update_cooldowns(state, fng)

    print("\n  6. Exits controleren …")
    state, exits = check_exits(state, indicators, fng)
    if not exits:
        print("     Geen exits.")

    print("\n  7. Entry signalen genereren …")
    state, signals, corr_blocks = generate_entries(state, indicators, fng)
    if not signals:
        print("     Geen entries.")

    print("\n  8. State opslaan (state_fear.json) …")
    save_state(state)

    print("\n  9. Slack notificatie …")
    send_slack(state, exits, signals, fng, corr_blocks)

    print("\n  Klaar.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
