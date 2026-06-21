"""
Bot BASELINE V2 — Hybrid V4+MR Paper Trading Bot (multi-agent)
===============================================================
Portfolio: DOGE, SOL, ETH, AVAX, ADA via Bitvavo publieke API (EUR pairs).
Strategie: V4 momentum (bull/neutraal) + MR mean reversion (bear).
Beslissing: multi_agent_decision() uit shared.py (vier-agent pipeline).
State:     state_baseline_v2.json in GitHub repo.
"""

from __future__ import annotations
import os, sys, warnings, traceback
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from shared import (
    init_clients,
    fetch_all_candles,
    fetch_fng,
    fetch_news,
    add_indicators,
    last_row,
    regime_score,
    get_regime,
    portfolio_regime,
    load_thresholds,
    gh_read,
    gh_write,
    log_agent_decision,
    send_slack_blocks,
    portfolio_blocks,
    agent_decision_block,
    multi_agent_decision,
    check_meta_config,
)

# ── Config ────────────────────────────────────────────────────────────────────
BOT_ID        = "baseline_v2"
GH_STATE_FILE = "state_baseline_v2.json"

TICKERS   = ["DOGE-EUR", "SOL-EUR", "ETH-EUR", "AVAX-EUR", "ADA-EUR"]
START_CAP = 500.0
FEE_RT    = 0.004
MIN_POS   = 3.0

V4_RSI_LO    = 45; V4_RSI_HI = 75
V4_ATR_ENTRY = 4.0
V4_ATR_TRAIL = 3.0
V4_MA200_MAX_BELOW = 0.15
V4_SIZES     = {0: 0.07, 1: 0.07, 2: 0.10, 3: 0.15, 4: 0.15}
LLM_VALID_SIZES = [7, 10, 15]

MR_RSI_THR = 35
MR_BB_WIN  = 30; MR_BB_SIG = 2.5
MR_ATR     = 2.0

EXPO_BULL_NEUT = 0.70
EXPO_BEAR      = 0.40
MAX_OPEN       = 5

# SHA van het huidige state bestand
_state_sha: Optional[str] = None

_INITIAL_STATE = {
    "capital_eur":        START_CAP,
    "open_positions":     {},
    "total_trades":       0,
    "wins":               0,
    "gross_win":          0.0,
    "gross_loss":         0.0,
    "win_rate":           0.0,
    "profit_factor":      0.0,
    "total_return_pct":   0.0,
    "trades":             [],
    "signals":            [],
}


def _name(ticker: str) -> str:
    return ticker.replace("-EUR", "")


# ── State helpers ──────────────────────────────────────────────────────────────
def load_state() -> dict:
    global _state_sha
    content, sha = gh_read(GH_STATE_FILE)
    if content is None:
        print(f"  [INFO] {GH_STATE_FILE} bestaat nog niet — initialiseer.")
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
        for p in state["open_positions"].values()
    )
    portfolio_total = state["capital_eur"] + open_positions_value
    ret_pct = (portfolio_total - START_CAP) / START_CAP * 100

    gw = state["gross_win"]; gl = state["gross_loss"]
    pf = gw / gl if gl > 0 else 0.0
    wr = state["wins"] / state["total_trades"] * 100 if state["total_trades"] > 0 else 0.0

    state["win_rate"]             = round(wr, 4)
    state["profit_factor"]        = round(pf, 4)
    state["total_return_pct"]     = round(ret_pct, 4)
    state["portfolio_total"]      = round(portfolio_total, 4)
    state["open_positions_value"] = round(open_positions_value, 4)

    # Verwijder runtime-only velden
    clean_positions = {}
    for t, p in state["open_positions"].items():
        clean_positions[t] = {k: v for k, v in p.items() if k != "current_val"}
    state["open_positions"] = clean_positions

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_pos   = len(state["open_positions"])
    message = (f"bot: {now_str} — "
               f"€{state['capital_eur']:.2f} vrij, "
               f"{n_pos} pos open, "
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


def log_signal(sig: dict, state: dict):
    clean = {k: (round(v, 6) if isinstance(v, float) else v)
             for k, v in sig.items()}
    state.setdefault("signals", []).append(clean)
    blk = sig.get("entry_blocked", False)
    print(f"  [SIGNAL] {sig['ticker']:<12}  {sig['signal_type']}  "
          f"regime={sig['regime']}  "
          f"{'GEBLOKKEERD' if blk else 'KOOP'}")


# ── Signaallogica ─────────────────────────────────────────────────────────────
def v4_entry_check(row) -> bool:
    rsi   = float(row["rsi"]); close = float(row["Close"]); ma200 = float(row["ma200"])
    cross = float(row["macd_hist"]) > 0 and float(row["prev_hist"]) <= 0
    return (V4_RSI_LO <= rsi <= V4_RSI_HI and cross
            and float(row["Volume"]) > float(row["vol_ma"])
            and close >= ma200 * (1 - V4_MA200_MAX_BELOW))


def v4_strength_score(row) -> int:
    s = 0
    if 55 <= float(row["rsi"]) <= 70:                                    s += 1
    if float(row["macd_hist"]) > float(row["prev_hist"]) > 0:            s += 1
    if float(row["vol_ratio"]) > 1.5:                                    s += 1
    if float(row["Close"]) > float(row["ma200"]) * 1.02:                 s += 1
    return s


def mr_entry_check(row) -> bool:
    close = float(row["Close"])
    return (float(row["rsi"]) < MR_RSI_THR
            and close < float(row["bb_lower"]) - float(row["bb_std"])
            and float(row["Volume"]) > float(row["vol_ma"]))


def mr_size_frac(row) -> float:
    return 0.10 if float(row["rsi"]) < 20 else 0.07


def v4_exit_check(row, stop_px: float) -> bool:
    return float(row["Close"]) <= stop_px


def mr_exit_check(row, stop_px: float) -> bool:
    return (float(row["rsi"]) > 50
            or float(row["Close"]) >= float(row["bb_mid"])
            or float(row["Close"]) <= stop_px)


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
                "regime":             pos.get("regime_at_entry", "?"),
                "strategy":           strategy,
                "llm_beslissing":     pos.get("llm_beslissing", ""),
                "llm_confidence":     pos.get("llm_confidence", 0.0),
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
    signals    = []
    port_reg   = portfolio_regime(regime_map)
    max_expo   = EXPO_BULL_NEUT if port_reg in ("bull", "neutraal") else EXPO_BEAR
    now_str    = datetime.now(timezone.utc).isoformat()

    total_val = state["capital_eur"] + sum(
        float(p.get("current_val", p.get("invest", 0)))
        for p in state["open_positions"].values()
    )
    invested = sum(float(p["invest"]) for p in state["open_positions"].values())

    for ticker in TICKERS:
        if ticker in state["open_positions"]:
            continue
        if len(state["open_positions"]) >= MAX_OPEN:
            break

        df = indicators.get(ticker)
        if df is None or df.empty:
            continue
        row    = last_row(df)
        t_reg  = regime_map.get(ticker, "neutraal")
        close  = float(row["Close"])
        atr    = float(row["atr"])

        expo_frac = invested / total_val if total_val > 0 else 0.0
        if expo_frac >= max_expo:
            break

        strategy = None
        frac     = 0.0

        if t_reg in ("bull", "neutraal"):
            if v4_entry_check(row):
                strategy = "v4"
                s    = v4_strength_score(row)
                frac = V4_SIZES[s]
        else:
            if mr_entry_check(row):
                strategy = "mr"
                frac     = mr_size_frac(row)

        if strategy is None:
            continue

        headlines = fetch_news(ticker)

        # Multi-agent beslissing
        decision = multi_agent_decision(
            ticker=ticker,
            row=row,
            df=df,
            regime=t_reg,
            strategy=strategy,
            regime_map=regime_map,
            fng=fng,
            headlines=headlines,
            state=state,
            proposed_frac=frac,
            valid_sizes=LLM_VALID_SIZES,
            bot_id=BOT_ID,
            max_open=MAX_OPEN,
        )

        # Log naar Supabase
        log_agent_decision({
            "bot_id":              BOT_ID,
            "ticker":              ticker,
            "timestamp":           now_str,
            "beslissing":          decision["beslissing"],
            "positiegrootte_pct":  decision["positiegrootte_pct"],
            "confidence":          decision["confidence"],
            "consensus":           decision["consensus"],
            "doorslaggevende_factor": decision["doorslaggevende_factor"],
            "reden":               decision["reden"],
            "sentiment_score":     decision["sentiment_score"],
            "sentiment_label":     decision["sentiment_label"],
            "technische_score":    decision["technische_score"],
            "setup_kwaliteit":     decision["setup_kwaliteit"],
            "risico_score":        decision["risico_score"],
            "risico_label":        decision["risico_label"],
            "agents_failed":       decision["agents_failed"],
            "llm_used":            decision["llm_used"],
            "regime":              t_reg,
            "strategy":            strategy,
        })

        name  = _name(ticker)
        score = regime_score(row)
        sig_row = {
            "timestamp":      now_str,
            "ticker":         ticker,
            "regime":         t_reg,
            "rsi":            round(float(row["rsi"]), 2),
            "macd_hist":      round(float(row["macd_hist"]), 6),
            "atr_ratio":      round(float(row["atr"]) / close * 100, 4),
            "volume_ratio":   round(float(row["vol_ratio"]), 4),
            "signal_type":    strategy,
            "llm_beslissing": decision["beslissing"],
            "llm_confidence": decision["confidence"],
            "entry_blocked":  decision["beslissing"] == "NIETS",
            "block_reason":   decision["reden"] if decision["beslissing"] == "NIETS" else "",
        }
        log_signal(sig_row, state)
        signals.append({**sig_row, "decision": decision, "row": row,
                        "close": close, "atr": atr, "frac": frac})

        if decision["beslissing"] == "NIETS":
            print(f"  SKIP  {ticker:<12}  NIETS  ({decision['reden'][:60]})")
            continue

        eff_frac = decision["positiegrootte_pct"] / 100.0
        invest   = state["capital_eur"] * eff_frac
        if invest < MIN_POS:
            print(f"  SKIP  {ticker:<12}  te klein (€{invest:.2f})")
            continue

        fee   = invest * FEE_RT / 2
        units = (invest - fee) / close
        stop  = close - V4_ATR_ENTRY * atr if strategy == "v4" else close - MR_ATR * atr

        pos = {
            "strategy":        strategy,
            "units":           round(units, 8),
            "entry_px":        round(close, 6),
            "stop_px":         round(stop, 6),
            "trail_high":      round(close, 6),
            "invest":          round(invest, 4),
            "entry_ts":        now_str,
            "strength":        v4_strength_score(row) if strategy == "v4" else 0,
            "regime_at_entry": t_reg,
            "llm_beslissing":  decision["beslissing"],
            "llm_confidence":  decision["confidence"],
        }
        state["capital_eur"] -= invest
        state["open_positions"][ticker] = pos
        invested += invest

        trade = {
            "timestamp":         now_str,
            "ticker":            ticker,
            "action":            "BUY",
            "price":             round(close, 6),
            "position_size_pct": round(eff_frac * 100, 2),
            "position_size_eur": round(invest, 4),
            "entry_price":       round(close, 6),
            "exit_price":        None,
            "pnl_pct":           None,
            "pnl_eur":           None,
            "exit_reason":       None,
            "regime":            t_reg,
            "strategy":          strategy,
            "llm_beslissing":    decision["beslissing"],
            "llm_confidence":    decision["confidence"],
        }
        log_trade(trade, state)
        print(f"  BUY   {ticker:<12}  {strategy:<4}  {t_reg:<9}  "
              f"€{close:.4f}  {eff_frac*100:.0f}%  "
              f"(conf {decision['confidence']:.2f}  consensus={decision['consensus']})")

    return state, signals


# ── Slack notificatie ─────────────────────────────────────────────────────────
def send_slack(state: dict, exits: list, signals: list, fng: dict):
    blocks = portfolio_blocks("Bot BASELINE V2 (multi-agent)", state, fng, START_CAP)

    # Open posities
    if state["open_positions"]:
        pos_text = ""
        for t, p in state["open_positions"].items():
            cv   = float(p.get("current_val", float(p["units"]) * float(p["entry_px"])))
            upnl = cv - float(p["invest"])
            sign = "+" if upnl >= 0 else ""
            pos_text += (f"`{_name(t):<5}` {p['strategy'].upper()}  "
                         f"entry €{float(p['entry_px']):.4f}  "
                         f"stop €{float(p['stop_px']):.4f}  "
                         f"uPnL {sign}{upnl:.2f}€\n")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f":file_folder: *Open posities ({len(state['open_positions'])})*\n{pos_text.strip()}"}
        })
    else:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": ":file_folder: Geen open posities"}
        })

    # Exits
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

    # Agent beslissingen per signaal
    if signals:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": f":bell: *Signalen deze candle ({len(signals)})*"}
        })
        for s in signals:
            decision = s.get("decision", {})
            if decision:
                blocks.append(agent_decision_block(decision))
    else:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": ":bell: Geen nieuwe signalen"}
        })

    send_slack_blocks(blocks)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print(f"  Bot BASELINE V2 (multi-agent)  —  "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print("=" * 65)

    init_clients()

    if not check_meta_config("baseline_v2"):
        return

    print("\n  1. Portfolio state laden …")
    state = load_state()
    print(f"     Kapitaal: €{state['capital_eur']:.2f}  |  "
          f"Posities: {len(state['open_positions'])}  |  "
          f"Trades: {state['total_trades']}")

    print("\n  2. Live data ophalen (Bitvavo) …")
    raw_frames = fetch_all_candles(TICKERS)

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

    print("\n  6. Entry signalen genereren + multi-agent filter …")
    state, signals = generate_entries(state, indicators, regime_map, thresholds, fng)
    if not signals:
        print("     Geen entry signalen.")

    print("\n  7. State opslaan in GitHub …")
    save_state(state)

    print("\n  8. Slack notificatie …")
    send_slack(state, exits, signals, fng)

    total_val = state["capital_eur"] + sum(
        float(p.get("current_val", p.get("invest", 0)))
        for p in state["open_positions"].values()
    )
    print(f"\n{'='*65}")
    print(f"  Portfolio: €{total_val:.2f}  ({(total_val-START_CAP)/START_CAP*100:>+.1f}%)")
    print(f"  Open posities: {len(state['open_positions'])}  |  "
          f"Exits: {len(exits)}  |  Signalen: {len(signals)}")
    print(f"{'='*65}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
