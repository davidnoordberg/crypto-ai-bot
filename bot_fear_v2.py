"""
Fear Contrarian V2 Paper Trading Bot
======================================
Koopt tijdens extreme marktpaniek (Fear & Greed < 20, RSI < 45).
Verkoopt wanneer angst wegebt (FNG > 50), trailing stop (3×ATR) of RSI > 65.

Assets  : BTC-EUR, ETH-EUR, SOL-EUR, DOGE-EUR via Bitvavo publieke API.
Bot ID  : fear_v2
State   : state_fear_v2.json in GitHub repo.

Entry:
  - Fear & Greed < 20 (Extreme Fear)
  - RSI(14) < 45
  - Dynamic sizing: FNG 15-20 → 20%, FNG 10-15 → 30%, FNG < 10 → 40%
  - Multi-agent beslissing per entry. valid_sizes = [20, 30, 40]
  - Max 3 open posities tegelijk

Exit (eerste die triggert):
  - FNG stijgt boven 50 → verkoop alle fear-posities
  - Trailing stop 3×ATR14
  - RSI stijgt boven 65

Herinstap cooldown: na exit wacht per ticker tot FNG opnieuw < 20.
"""

from __future__ import annotations
import os, json, sys, warnings, traceback
from datetime import datetime, timezone
from typing import Optional

import warnings
warnings.filterwarnings("ignore")

from shared import (
    init_clients, fetch_all_candles, fetch_fng, fetch_news,
    add_indicators, last_row,
    gh_read, gh_write,
    log_agent_decision, send_slack_blocks, portfolio_blocks, agent_decision_block,
    multi_agent_decision,
    check_meta_config,
)

# ── Config ────────────────────────────────────────────────────────────────────
BOT_ID        = "fear_v2"
TICKERS       = ["BTC-EUR", "ETH-EUR", "SOL-EUR", "DOGE-EUR"]
START_CAP     = 500.0
FEE_RT        = 0.004
MIN_POS       = 3.0
MAX_OPEN      = 3

ATR_WIN   = 14
ATR_TRAIL = 3.0
RSI_WIN   = 14

FNG_ENTRY_MAX = 20
FNG_EXIT_MIN  = 50
RSI_ENTRY_MAX = 45
RSI_EXIT_MIN  = 65

GH_STATE_FILE = "state_fear_v2.json"


def fng_size(fng_score: int) -> float:
    """Dynamic positiegrootte op basis van FNG niveau."""
    if fng_score < 10: return 0.40
    if fng_score < 15: return 0.30
    return 0.20


def _name(ticker: str) -> str:
    return ticker.replace("-EUR", "")


# ── State ─────────────────────────────────────────────────────────────────────
_INITIAL_STATE = {
    "bot_id":            BOT_ID,
    "capital_eur":       START_CAP,
    "open_positions":    {},
    "cooldown_tickers":  [],
    "total_trades":      0,
    "wins":              0,
    "gross_win":         0.0,
    "gross_loss":        0.0,
    "win_rate":          0.0,
    "profit_factor":     0.0,
    "total_return_pct":  0.0,
    "portfolio_total":   START_CAP,
    "open_positions_value": 0.0,
    "trades":            [],
    "signals":           [],
}

_state_sha: Optional[str] = None


def load_state() -> dict:
    global _state_sha
    content, sha = gh_read(GH_STATE_FILE)
    if content is None:
        print(f"  [INFO] {GH_STATE_FILE} bestaat nog niet — initialiseer met startkapitaal.")
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
    message = (f"fear_v2: {now_str} — "
               f"€{state['capital_eur']:.2f} vrij, "
               f"{len(state['open_positions'])} pos open, "
               f"{state['total_trades']} trades")
    gh_write(GH_STATE_FILE, state, _state_sha, message)
    print(f"  [GitHub] {GH_STATE_FILE} opgeslagen  "
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
    exits     = []
    to_del    = []
    now_str   = datetime.now(timezone.utc).isoformat()
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
                "timestamp":         now_str,
                "ticker":            ticker,
                "action":            "SELL",
                "price":             round(close, 6),
                "position_size_pct": round(float(pos["invest"]) / START_CAP * 100, 2),
                "position_size_eur": round(float(pos["invest"]), 4),
                "entry_price":       round(float(pos["entry_px"]), 6),
                "exit_price":        round(close, 6),
                "pnl_pct":           round(pnl_pct, 4),
                "pnl_eur":           round(pnl, 4),
                "exit_reason":       reason,
                "entry_fng":         pos.get("entry_fng", 0),
                "exit_fng":          fng_score,
            }
            log_trade(trade, state)
            exits.append(trade)
            to_del.append(ticker)

            # Cooldown activeren
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
    if fng["score"] < FNG_ENTRY_MAX:
        state["cooldown_tickers"] = []
        print(f"  FNG={fng['score']} < {FNG_ENTRY_MAX}: alle cooldowns gereset.")
    else:
        n = len(state.get("cooldown_tickers", []))
        if n:
            print(f"  FNG={fng['score']} >= {FNG_ENTRY_MAX}: {n} ticker(s) in cooldown.")


# ── Entries ────────────────────────────────────────────────────────────────────
CORR_THRESHOLD = 0.85
CORR_CANDLES   = 48


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
        aligned, aligned_open = ret_new.align(ret_open, join="inner")
        if len(aligned) < 10:
            continue
        corr = float(aligned.corr(aligned_open))
        if corr >= CORR_THRESHOLD:
            return f"{_name(open_ticker)} en {_name(ticker)} correlatie {corr:.2f}"
    return None


def generate_entries(state: dict, indicators: dict,
                     fng: dict) -> tuple[dict, list, list]:
    """
    Entry condities:
    - FNG < 20, RSI < 45, niet in cooldown, max 3 open posities
    - Multi-agent beslissing per entry. decision["positiegrootte_pct"]/100 overschrijft fng_size().
    """
    signals        = []
    all_decisions  = []
    now_str        = datetime.now(timezone.utc).isoformat()
    fng_score      = fng["score"]

    if fng_score >= FNG_ENTRY_MAX:
        print(f"  FNG={fng_score} >= {FNG_ENTRY_MAX}: geen entry condities.")
        return state, signals, all_decisions

    base_frac = fng_size(fng_score)
    valid_sizes = [20, 30, 40]
    cooldown  = state.get("cooldown_tickers", [])

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

        row   = last_row(df)
        close = float(row["Close"])
        atr   = float(row["atr"])
        rsi   = float(row["rsi"])

        if rsi >= RSI_ENTRY_MAX:
            print(f"  {ticker:<12}  RSI {rsi:.1f} >= {RSI_ENTRY_MAX} — skip")
            continue

        # Correlation agent check
        corr_reason = correlation_block(ticker, indicators, state["open_positions"])
        if corr_reason:
            print(f"  {ticker:<12}  geblokkeerd door correlation agent: {corr_reason}")
            all_decisions.append({
                "beslissing": "NIETS", "reden": f"Correlation agent: {corr_reason}",
                "confidence": 0.0, "consensus": "geblokkeerd",
                "doorslaggevende_factor": "correlatie",
                "positiegrootte_pct": 0, "sentiment_score": 0, "sentiment_label": "",
                "technische_score": 0, "setup_kwaliteit": "", "risico_score": 0,
                "risico_label": "", "agents_failed": [], "llm_used": False,
                "corr_block": True, "corr_reden": corr_reason, "ticker": _name(ticker),
            })
            continue

        # Multi-agent beslissing
        headlines = fetch_news(ticker)
        decision  = multi_agent_decision(
            ticker       = ticker,
            row          = row,
            df           = df,
            regime       = "neutraal",   # fear bot werkt niet met regime
            strategy     = "fear",
            regime_map   = {},
            fng          = fng,
            headlines    = headlines,
            state        = state,
            proposed_frac= base_frac,
            valid_sizes  = valid_sizes,
            bot_id       = BOT_ID,
            max_open     = MAX_OPEN,
        )
        all_decisions.append(decision)

        # Supabase logging
        log_agent_decision({
            "bot_id":                BOT_ID,
            "ticker":                ticker,
            "timestamp":             now_str,
            "regime":                "fear",
            "strategy":              "fear",
            "entry_fng":             fng_score,
            "entry_rsi":             round(rsi, 2),
            "finale_beslissing":     decision["beslissing"],
            "positiegrootte_pct":    decision["positiegrootte_pct"],
            "confidence":            decision["confidence"],
            "consensus":             decision["consensus"],
            "doorslaggevende_factor":decision["doorslaggevende_factor"],
            "reden":                 decision["reden"],
            "sentiment_score":       decision["sentiment_score"],
            "sentiment_label":       decision["sentiment_label"],
            "technische_score":      decision["technische_score"],
            "setup_kwaliteit":       decision["setup_kwaliteit"],
            "risico_score":          decision["risico_score"],
            "risico_label":          decision["risico_label"],
            "agents_failed":         json.dumps(decision["agents_failed"]),
            "llm_used":              decision["llm_used"],
        })

        if decision["beslissing"] == "NIETS":
            print(f"  SKIP  {ticker:<12}  multi-agent NIETS  ({decision['reden'][:60]})")
            sig = {
                "timestamp":     now_str,
                "ticker":        ticker,
                "fng":           fng_score,
                "rsi":           round(rsi, 2),
                "atr":           round(atr, 6),
                "size_pct":      round(base_frac * 100, 1),
                "entry_blocked": True,
                "llm_beslissing":decision["beslissing"],
                "llm_confidence":decision["confidence"],
            }
            state.setdefault("signals", []).append(
                {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sig.items()}
            )
            signals.append(sig)
            continue

        # Positiegrootte: LLM override, maar minimaal base_frac
        frac   = decision["positiegrootte_pct"] / 100.0
        invest = state["capital_eur"] * frac
        if invest < MIN_POS:
            print(f"  {ticker:<12}  te weinig kapitaal (€{invest:.2f})")
            continue

        fee   = invest * FEE_RT / 2
        units = (invest - fee) / close
        stop  = close - ATR_TRAIL * atr

        pos = {
            "units":         round(units, 8),
            "entry_px":      round(close, 6),
            "stop_px":       round(stop, 6),
            "trail_high":    round(close, 6),
            "invest":        round(invest, 4),
            "entry_ts":      now_str,
            "entry_fng":     fng_score,
            "entry_rsi":     round(rsi, 2),
            "size_pct":      round(frac * 100, 1),
            "llm_beslissing":decision["beslissing"],
            "llm_confidence":decision["confidence"],
            "llm_reden":     decision["reden"],
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
            "llm_beslissing":    decision["beslissing"],
            "llm_confidence":    decision["confidence"],
            "llm_reden":         decision["reden"],
        }
        log_trade(trade, state)

        sig = {
            "timestamp":     now_str,
            "ticker":        ticker,
            "fng":           fng_score,
            "rsi":           round(rsi, 2),
            "atr":           round(atr, 6),
            "invest":        round(invest, 4),
            "size_pct":      round(frac * 100, 1),
            "stop_px":       round(stop, 6),
            "entry_blocked": False,
            "llm_beslissing":decision["beslissing"],
            "llm_confidence":decision["confidence"],
        }
        state.setdefault("signals", []).append(
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sig.items()}
        )
        signals.append(sig)

        print(f"  BUY   {ticker:<12}  FNG={fng_score}  RSI={rsi:.1f}  "
              f"€{close:.4f}  {frac*100:.0f}%  stop=€{stop:.4f}  "
              f"LLM={decision['confidence']:.2f}")

    return state, signals, all_decisions


# ── Slack ──────────────────────────────────────────────────────────────────────
def send_slack(state: dict, exits: list, signals: list,
               fng: dict, all_decisions: list):
    blocks = portfolio_blocks("Bot FEAR V2 (multi-agent)", state, fng, START_CAP)

    # Multi-agent beslissingen (eerste entry)
    for decision in all_decisions[:1]:
        blocks.append(agent_decision_block(decision))

    # Open posities
    n_pos = len(state["open_positions"])
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

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn",
                 "text": f":chart_with_upwards_trend: *Open posities ({n_pos}/{MAX_OPEN})*\n{pos_text}"}
    })

    # Signalen
    new_entries = [s for s in signals if not s.get("entry_blocked")]
    if new_entries:
        sig_lines = [
            f"`{_name(s['ticker'])}` FNG={s['fng']}  RSI={s['rsi']:.1f}  "
            f"{s['size_pct']:.0f}%  stop=€{s['stop_px']:.4f}"
            for s in new_entries
        ]
        sig_text = "\n".join(sig_lines)
    elif fng["score"] < FNG_ENTRY_MAX:
        sig_text = f"FNG={fng['score']} — extreme fear actief maar geen entries"
    else:
        sig_text = f"Wacht op extreme fear (huidig FNG={fng['score']})"

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": f":bell: *Signalen*\n{sig_text}"}
    })

    # Correlation blocks
    corr_blocked = [d for d in all_decisions if d.get("corr_block")]
    if corr_blocked:
        corr_lines = [
            f":link: Trade geblokkeerd door Correlation Agent\nTicker: {d['ticker']}\nReden: {d['corr_reden']}"
            for d in corr_blocked
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
            icon   = ":white_check_mark:" if (e.get("pnl_eur") or 0) >= 0 else ":x:"
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

    send_slack_blocks(blocks)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print(f"  Fear Contrarian V2 Bot  —  "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print("=" * 65)

    init_clients()

    if not check_meta_config("fear_v2"):
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
    raw_frames = fetch_all_candles(TICKERS)
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
    state, signals, all_decisions = generate_entries(state, indicators, fng)
    if not signals:
        print("     Geen entries.")

    print(f"\n  8. State opslaan ({GH_STATE_FILE}) …")
    save_state(state)

    print("\n  9. Slack notificatie …")
    send_slack(state, exits, signals, fng, all_decisions)

    print("\n  Klaar.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
