"""
All-In V2 Paper Trading Bot
============================
Assets  : DOGE-EUR en SOL-EUR via Bitvavo publieke API.
Logica  : Winner takes all. Beide tickers worden beoordeeld op v4_confidence.
          De winner wordt door de multi-agent pipeline gehaald.
          Max één open positie tegelijk.
Strategie: Hybride V4+MR. V4 momentum met 3× ATR trailing stop in
          bull/neutraal regime. Mean reversion in bear regime.
State   : state_allin_v2.json in GitHub repo.
Bot ID  : allin_v2
"""

from __future__ import annotations
import os, json, sys, warnings, traceback
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from shared import (
    init_clients, fetch_all_candles, fetch_fng, fetch_news,
    add_indicators, last_row, price_change_pct,
    regime_score, get_regime, portfolio_regime, load_thresholds,
    gh_read, gh_write,
    log_agent_decision, send_slack_blocks, portfolio_blocks, agent_decision_block,
    multi_agent_decision,
)

# ── Config ────────────────────────────────────────────────────────────────────
BOT_ID        = "allin_v2"
TICKERS       = ["DOGE-EUR", "SOL-EUR"]
START_CAP     = 500.0
FEE_RT        = 0.004
MIN_POS       = 3.0
MAX_OPEN      = 1
ALLIN_FRAC    = 0.95

GH_STATE_FILE = "state_allin_v2.json"

V4_RSI_LO    = 45; V4_RSI_HI = 75
V4_ATR_ENTRY = 4.0
V4_ATR_TRAIL = 3.0
V4_MA200_MAX_BELOW = 0.15

MR_RSI_THR = 35
MR_BB_WIN  = 30; MR_BB_SIG = 2.5
MR_ATR     = 2.0


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
    "portfolio_total":   START_CAP,
    "open_positions_value": 0.0,
    "trades":            [],
    "signals":           [],
}

_state_sha: Optional[str] = None


def _name(ticker: str) -> str:
    return ticker.replace("-EUR", "")


# ── v4_confidence ─────────────────────────────────────────────────────────────
def v4_confidence(row) -> float:
    rsi_score  = np.clip((float(row["rsi"]) - 45) / (75 - 45), 0.0, 1.0)
    hist_score = min(abs(float(row["macd_hist"])) / (abs(float(row["atr"])) + 1e-9), 1.0)
    vol_score  = min(float(row["vol_ratio"]) / 2.0, 1.0)
    return float((rsi_score + hist_score + vol_score) / 3.0)


def mr_confidence(row) -> float:
    rsi      = float(row["rsi"])
    close    = float(row["Close"])
    bb_lower = float(row["bb_lower"])
    bb_std   = float(row["bb_std"])
    rsi_sc   = np.clip((MR_RSI_THR - rsi) / MR_RSI_THR, 0.0, 1.0)
    bb_depth = np.clip((bb_lower - close) / (bb_std + 1e-9), 0.0, 1.0)
    return float((rsi_sc + bb_depth) / 2.0)


# ── Signaallogica ─────────────────────────────────────────────────────────────
def v4_entry_check(row) -> bool:
    rsi   = float(row["rsi"])
    close = float(row["Close"])
    ma200 = float(row["ma200"])
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


# ── State ─────────────────────────────────────────────────────────────────────
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

    clean_positions = {}
    for t, p in state["open_positions"].items():
        clean_positions[t] = {k: v for k, v in p.items() if k != "current_val"}
    state["open_positions"] = clean_positions

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    message = (f"allin_v2: {now_str} — "
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
                "timestamp":           now_str,
                "ticker":              ticker,
                "action":              "SELL",
                "price":               round(close, 6),
                "position_size_pct":   round(float(pos["invest"]) / START_CAP * 100, 2),
                "position_size_eur":   round(float(pos["invest"]), 4),
                "entry_price":         round(float(pos["entry_px"]), 6),
                "exit_price":          round(close, 6),
                "pnl_pct":             round(pnl_pct, 4),
                "pnl_eur":             round(pnl, 4),
                "exit_reason":         reason,
                "regime":              pos.get("regime_at_entry", "?"),
                "strategy":            strategy,
                "confidence_at_entry": pos.get("confidence_at_entry", 0.0),
                "llm_beslissing":      pos.get("llm_beslissing", ""),
                "llm_confidence":      pos.get("llm_confidence", 0.0),
                "llm_reden":           pos.get("llm_reden", ""),
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
                     fng: dict) -> tuple[dict, list, Optional[dict]]:
    """
    Verzamel entry signalen voor DOGE en SOL.
    Winner = hoogste v4_confidence. Roep multi_agent_decision aan voor de winner.
    """
    now_str  = datetime.now(timezone.utc).isoformat()
    last_decision: Optional[dict] = None

    if len(state["open_positions"]) >= MAX_OPEN:
        print("  Positielimiet bereikt (max 1) — geen nieuwe entries.")
        return state, [], None

    # Verzamel kandidaten
    candidates = []
    for ticker in TICKERS:
        df = indicators.get(ticker)
        if df is None or df.empty:
            continue
        row      = last_row(df)
        t_reg    = regime_map.get(ticker, "neutraal")
        close    = float(row["Close"])
        atr      = float(row["atr"])
        strategy = None
        conf_score = 0.0

        if t_reg in ("bull", "neutraal"):
            if v4_entry_check(row):
                strategy   = "v4"
                conf_score = v4_confidence(row)
        else:
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
            "df":         df,
        })

    if not candidates:
        print("  Geen entry signalen voor DOGE of SOL.")
        return state, [], None

    # Sorteer op confidence — hoogste wint
    candidates.sort(key=lambda x: x["conf_score"], reverse=True)
    winner  = candidates[0]
    losers  = candidates[1:]
    competitor = _name(losers[0]["ticker"]) if losers else ""

    print(f"  Kandidaten: {[_name(c['ticker']) for c in candidates]}")
    print(f"  Winner: {_name(winner['ticker'])}  (conf {winner['conf_score']:.3f})"
          + (f"  vs {competitor} (conf {losers[0]['conf_score']:.3f})" if competitor else ""))

    # Multi-agent beslissing voor de winner
    headlines  = fetch_news(winner["ticker"])
    valid_sizes = [25, 50, 75, 95]

    decision = multi_agent_decision(
        ticker       = winner["ticker"],
        row          = winner["row"],
        df           = winner["df"],
        regime       = winner["regime"],
        strategy     = winner["strategy"],
        regime_map   = regime_map,
        fng          = fng,
        headlines    = headlines,
        state        = state,
        proposed_frac= ALLIN_FRAC,
        valid_sizes  = valid_sizes,
        bot_id       = BOT_ID,
        max_open     = MAX_OPEN,
    )
    last_decision = decision

    # Supabase logging
    log_agent_decision({
        "bot_id":                BOT_ID,
        "ticker":                winner["ticker"],
        "timestamp":             now_str,
        "regime":                winner["regime"],
        "strategy":              winner["strategy"],
        "conf_score":            round(winner["conf_score"], 4),
        "competitor":            competitor,
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

    # Log loser signalen
    signals = []
    for i, cand in enumerate(candidates):
        is_win = (i == 0)
        sig = {
            "timestamp":      now_str,
            "ticker":         cand["ticker"],
            "regime":         cand["regime"],
            "rsi":            round(float(cand["row"]["rsi"]), 2),
            "macd_hist":      round(float(cand["row"]["macd_hist"]), 6),
            "atr_ratio":      round(float(cand["row"]["atr"]) / cand["close"] * 100, 4),
            "volume_ratio":   round(float(cand["row"]["vol_ratio"]), 4),
            "signal_type":    cand["strategy"],
            "conf_score":     round(cand["conf_score"], 4),
            "is_winner":      is_win,
            "entry_blocked":  (decision["beslissing"] == "NIETS") if is_win else True,
            "llm_beslissing": decision["beslissing"] if is_win else "NIETS",
            "llm_confidence": decision["confidence"] if is_win else 0.0,
        }
        state.setdefault("signals", []).append(
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sig.items()}
        )
        signals.append(sig)

    if decision["beslissing"] == "NIETS":
        print(f"  SKIP  {winner['ticker']:<12}  multi-agent NIETS  "
              f"({decision['reden'][:60]})")
        return state, signals, last_decision

    # Positiegrootte bepalen: LLM override, cap op 0.95
    frac = min(decision["positiegrootte_pct"] / 100.0, 0.95)
    invest = state["capital_eur"] * frac
    if invest < MIN_POS:
        print(f"  SKIP  {winner['ticker']:<12}  te weinig kapitaal (€{invest:.2f})")
        return state, signals, last_decision

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
        "llm_beslissing":       decision["beslissing"],
        "llm_confidence":       decision["confidence"],
        "llm_reden":            decision["reden"],
    }
    state["capital_eur"] -= invest
    state["open_positions"][winner["ticker"]] = pos

    trade = {
        "timestamp":           now_str,
        "ticker":              winner["ticker"],
        "action":              "BUY",
        "price":               round(winner["close"], 6),
        "position_size_pct":   round(frac * 100, 2),
        "position_size_eur":   round(invest, 4),
        "entry_price":         round(winner["close"], 6),
        "exit_price":          None,
        "pnl_pct":             None,
        "pnl_eur":             None,
        "exit_reason":         None,
        "regime":              winner["regime"],
        "strategy":            winner["strategy"],
        "confidence_at_entry": round(winner["conf_score"], 4),
        "competitor":          competitor,
        "llm_beslissing":      decision["beslissing"],
        "llm_confidence":      decision["confidence"],
        "llm_reden":           decision["reden"],
    }
    log_trade(trade, state)
    print(f"  BUY   {winner['ticker']:<12}  {winner['strategy']:<4}  {winner['regime']:<9}  "
          f"€{winner['close']:.4f}  {frac*100:.0f}%  "
          f"conf={winner['conf_score']:.3f}  LLM={decision['confidence']:.2f}")

    return state, signals, last_decision


# ── Slack ──────────────────────────────────────────────────────────────────────
def send_slack(state: dict, exits: list, signals: list,
               fng: dict, last_decision: Optional[dict]):
    blocks = portfolio_blocks("Bot ALL-IN V2 (multi-agent, winner-takes-all)",
                              state, fng, START_CAP)

    # Multi-agent beslissing block
    if last_decision:
        blocks.append(agent_decision_block(last_decision))

    # Open positie
    if state["open_positions"]:
        ticker, pos = next(iter(state["open_positions"].items()))
        cv   = float(pos.get("current_val", float(pos["units"]) * float(pos["entry_px"])))
        upnl = cv - float(pos["invest"])
        sign = "+" if upnl >= 0 else ""
        pos_text = (f"`{_name(ticker)}` {pos['strategy'].upper()}  "
                    f"entry €{float(pos['entry_px']):.4f}  "
                    f"stop €{float(pos['stop_px']):.4f}  "
                    f"uPnL {sign}{upnl:.2f}€")
    else:
        pos_text = "Geen"

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn",
                 "text": f":chart_with_upwards_trend: *Open positie*\n{pos_text}"}
    })

    # Signalen
    winner_sigs = [s for s in signals if s.get("is_winner")]
    if winner_sigs:
        ws = winner_sigs[0]
        sig_text = (f"`{_name(ws['ticker'])}` {ws['signal_type'].upper()}  "
                    f"conf={ws['conf_score']:.3f}  LLM *{ws['llm_beslissing']}* "
                    f"({ws['llm_confidence']:.2f})")
    else:
        sig_text = "Geen nieuwe signalen"

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": f":bell: *Signaal*\n{sig_text}"}
    })

    # Exits
    if exits:
        exit_lines = []
        for e in exits:
            icon = ":white_check_mark:" if (e.get("pnl_eur") or 0) >= 0 else ":x:"
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

    send_slack_blocks(blocks)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print(f"  All-In V2 Bot (DOGE+SOL)  —  "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print("=" * 65)

    init_clients()

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
        row = last_row(df)
        thr = thresholds.get(_name(t), {"p33": 0.0, "p67": 0.0})
        sc  = regime_score(row)
        reg = get_regime(sc, thr["p33"], thr["p67"])
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
    state, signals, last_decision = generate_entries(
        state, indicators, regime_map, thresholds, fng)
    if not signals:
        print("     Geen signalen.")

    print(f"\n  7. State opslaan ({GH_STATE_FILE}) …")
    save_state(state)

    print("\n  8. Slack notificatie …")
    send_slack(state, exits, signals, fng, last_decision)

    print("\n  Klaar.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
