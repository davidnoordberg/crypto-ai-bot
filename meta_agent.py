"""
meta_agent.py — Live Meta Agent (Fix1+Fix2)
==========================================
Draait elke 4 uur. Beslist welke bots actief zijn op basis van:
  - Fear & Greed Index (Alternative.me)
  - Crypto regime scores (BTC, ETH, SOL, DOGE, AVAX, ADA via Bitvavo)
  - ATR ratio
  Fix1: 3-daagse (18 candle) bevestiging voor configuratiewissel
  Fix2: Bear bescherming — fear bot altijd actief als >50% bear afgelopen 7d
"""

from __future__ import annotations
import os, json, base64, warnings
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

from ta.momentum   import RSIIndicator
from ta.trend      import MACD as TaMACD
from ta.volatility import AverageTrueRange

# ── Constanten ─────────────────────────────────────────────────────────────────
BITVAVO_BASE    = "https://api.bitvavo.com/v2"
GH_API          = "https://api.github.com"
CANDLE_LIMIT    = 250
ATR_WIN         = 14
MA200_WIN       = 200
MA200_LAG       = 10
VOL_WIN         = 20

REGIME_TICKERS  = ["BTC-EUR", "ETH-EUR", "SOL-EUR", "DOGE-EUR", "AVAX-EUR", "ADA-EUR"]
BASE_BOTS       = ["baseline", "v3", "allin", "fear"]
ALL_BOTS        = ["baseline", "baseline_v2", "v3", "v3_v2", "allin", "allin_v2", "fear", "fear_v2"]
BOT_LABELS      = {
    "baseline": "🛡️ Baseline", "baseline_v2": "🛡️ Baseline V2",
    "v3": "🎯 V3", "v3_v2": "🎯 V3 V2",
    "allin": "🚀 All-in", "allin_v2": "🚀 All-in V2",
    "fear": "😱 Fear", "fear_v2": "💀 Fear V2",
}

CONFIRM_CANDLES = 18   # Fix1: 3 dagen × 6 candles per dag
BEAR_LOOKBACK   = 42   # Fix2: 7 dagen × 6 candles per dag
BEAR_THRESHOLD  = 0.50


# ── Bitvavo ────────────────────────────────────────────────────────────────────
def fetch_candles(market: str, limit: int = CANDLE_LIMIT) -> Optional[pd.DataFrame]:
    url = f"{BITVAVO_BASE}/{market}/candles"
    try:
        r = requests.get(url, params={"interval": "4h", "limit": limit}, timeout=15)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        print(f"  [WARN] Bitvavo {market}: {e}")
        return None
    if not raw or isinstance(raw, dict):
        return None
    df = pd.DataFrame(raw, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("ts").sort_index()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


# ── Fear & Greed ───────────────────────────────────────────────────────────────
def fetch_fng_history(limit: int = 42) -> list:
    try:
        r = requests.get(
            f"https://api.alternative.me/fng/?limit={limit}&format=json", timeout=10)
        return r.json()["data"]
    except Exception as e:
        print(f"  [WARN] FNG ophalen mislukt: {e}")
        return []


def current_fng(fng_data: list) -> dict:
    if not fng_data:
        return {"score": 50, "label": "Neutraal", "trend": "stable"}
    latest = fng_data[0]
    score  = int(latest["value"])
    label  = latest["value_classification"]
    trend  = "stable"
    if len(fng_data) >= 7:
        week_avg = sum(int(d["value"]) for d in fng_data[1:8]) / 7
        if score < week_avg - 3:
            trend = "dalend"
        elif score > week_avg + 3:
            trend = "stijgend"
    return {"score": score, "label": label, "trend": trend}


# ── Indicatoren ────────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or len(df) < MA200_WIN + 10:
        return None
    df = df.copy()
    close = df["Close"]
    df["ma200"]     = close.rolling(MA200_WIN).mean()
    df["ma200_lag"] = df["ma200"].shift(MA200_LAG)
    df["rsi"]       = RSIIndicator(close, window=14).rsi()
    _m = TaMACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd"]      = _m.macd()
    df["macd_sig"]  = _m.macd_signal()
    df["vol_ma"]    = df["Volume"].rolling(VOL_WIN).mean()
    atr_obj         = AverageTrueRange(df["High"], df["Low"], close, window=ATR_WIN)
    df["atr"]       = atr_obj.average_true_range()
    df["atr_avg20"] = df["atr"].rolling(20).mean()
    df["atr_ratio"] = df["atr"] / df["atr_avg20"].replace(0, np.nan)
    return df.dropna()


# ── Regime ─────────────────────────────────────────────────────────────────────
def regime_score(row) -> float:
    ma200     = float(row["ma200"])
    ma200_lag = float(row["ma200_lag"])
    close     = float(row["Close"])
    rsi       = float(row["rsi"])
    ma200_s = (ma200 - ma200_lag) / ma200_lag if ma200_lag else 0.0
    prijs_s = (close - ma200)     / ma200     if ma200     else 0.0
    rsi_s   = (rsi - 50) / 50.0
    return float(np.clip((ma200_s + prijs_s + rsi_s) / 3, -1.0, 1.0))


def compute_thresholds(data_ind: dict) -> tuple:
    scores = []
    for df in data_ind.values():
        if df is None or df.empty:
            continue
        # Gebruik eerste helft van data voor percentiel berekening (train split)
        for _, row in df.iloc[:len(df) // 2].iterrows():
            try:
                scores.append(regime_score(row))
            except Exception:
                pass
    if not scores:
        return -0.1, 0.1
    return float(np.percentile(scores, 33)), float(np.percentile(scores, 67))


def get_regime(score: float, p33: float, p67: float) -> str:
    if score > p67:  return "bull"
    if score < p33:  return "bear"
    return "neutraal"


# ── Meta agent beslislogica ────────────────────────────────────────────────────
def meta_agent_beslissing(fng: int, bull_regimes: int, bear_regimes: int,
                           atr_ratio: float) -> tuple:
    """Ruwe aanbeveling (voor Fix1/Fix2 filter). Zelfde logica als backtest_meta_v2.py."""
    if fng < 20:
        return ["fear"], ["baseline", "v3", "allin"]
    elif fng < 35 and bear_regimes >= 3:
        return ["fear", "baseline"], ["v3", "allin"]
    elif atr_ratio > 2.0:
        return ["baseline", "v3"], ["fear", "allin"]
    elif 35 <= fng < 55:
        return ["baseline", "v3"], ["fear", "allin"]
    elif fng >= 55 and bull_regimes >= 3:
        return ["baseline", "v3", "allin"], ["fear"]
    elif fng >= 75:
        return ["baseline", "v3", "allin", "fear"], []
    else:
        return ["baseline"], ["v3", "allin", "fear"]


def should_switch_config(current_active: list, recommended: list,
                          regime_history: list) -> bool:
    """Fix1: wissel alleen als aanbeveling consistent is voor CONFIRM_CANDLES candles."""
    if set(current_active) == set(recommended):
        return False
    if len(regime_history) < CONFIRM_CANDLES:
        return True
    recent     = regime_history[-CONFIRM_CANDLES:]
    consistent = all(set(r["recommended_active"]) == set(recommended) for r in recent)
    return consistent


def apply_bear_protection(active: list, paused: list,
                           regime_history: list) -> tuple:
    """Fix2: als >50% van de laatste BEAR_LOOKBACK candles >=3 bear regimes had, fear altijd aan."""
    if len(regime_history) < BEAR_LOOKBACK:
        return active, paused
    bear_pct = sum(
        1 for r in regime_history[-BEAR_LOOKBACK:]
        if r.get("bear_regimes", 0) >= 3
    ) / BEAR_LOOKBACK
    if bear_pct > BEAR_THRESHOLD and "fear" not in active:
        return active + ["fear"], [b for b in paused if b != "fear"]
    return active, paused


# ── GitHub ─────────────────────────────────────────────────────────────────────
def _gh_headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN", "")
    h = {"Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def gh_read_config() -> tuple:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    url  = f"{GH_API}/repos/{repo}/contents/bot_config.json"
    try:
        r = requests.get(url, headers=_gh_headers(), timeout=15)
        if r.status_code == 404:
            return None, None
        r.raise_for_status()
        data    = r.json()
        content = json.loads(base64.b64decode(data["content"]).decode())
        return content, data["sha"]
    except Exception as e:
        print(f"  [WARN] GitHub lees bot_config.json: {e}")
        return None, None


def gh_write_config(content: dict, sha: Optional[str], message: str):
    repo    = os.environ.get("GITHUB_REPOSITORY", "")
    url     = f"{GH_API}/repos/{repo}/contents/bot_config.json"
    encoded = base64.b64encode(
        json.dumps(content, indent=2, ensure_ascii=False).encode()
    ).decode()
    body: dict = {"message": message, "content": encoded}
    if sha:
        body["sha"] = sha
    try:
        r = requests.put(url, headers=_gh_headers(), json=body, timeout=15)
        r.raise_for_status()
        print("  [GitHub] bot_config.json bijgewerkt.")
    except Exception as e:
        print(f"  [WARN] GitHub schrijf bot_config.json: {e}")


# ── Slack ──────────────────────────────────────────────────────────────────────
def send_slack(blocks: list):
    url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not url:
        print("  [INFO] Geen SLACK_WEBHOOK_URL — geen notificatie.")
        return
    try:
        r = requests.post(url, json={"blocks": blocks}, timeout=10)
        if r.status_code != 200:
            print(f"  [WARN] Slack HTTP {r.status_code}: {r.text[:200]}")
        else:
            print("  [Slack] Notificatie verzonden.")
    except Exception as e:
        print(f"  [WARN] Slack fout: {e}")


def build_slack_blocks(config: dict, config_switched: bool,
                        old_active_base: list) -> list:
    now_cet    = datetime.now(timezone.utc) + timedelta(hours=2)
    now_str    = now_cet.strftime("%d-%m-%Y %H:%M")
    fng        = config["fng"]
    actief     = config["actieve_bots"]
    gepauzeerd = config["gepauzeerde_bots"]
    bear_act   = config["bear_bescherming_actief"]
    redenering = config["redenering"]
    wisselingen = config["wisselingen_deze_maand"]

    actief_str     = "  |  ".join(BOT_LABELS.get(b, b) for b in actief)
    gepauzeerd_str = "  |  ".join(BOT_LABELS.get(b, b) for b in gepauzeerd)
    bear_icon      = ":white_check_mark:" if bear_act else ":x:"

    header_text = (
        f":arrows_counterclockwise: META AGENT CONFIGURATIEWISSEL — {now_str} CET"
        if config_switched
        else f":brain: Meta Agent — {now_str} CET"
    )

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": header_text}},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn",
                 "text": f":bar_chart: *Markt*\n{config['markt_label'].capitalize()}"},
                {"type": "mrkdwn",
                 "text": f":scream: *Fear & Greed*\n{fng['score']} — {fng['label']} ({fng['trend']})"},
                {"type": "mrkdwn",
                 "text": f":bear: *Bear bescherming*\n{bear_icon} {'actief' if bear_act else 'inactief'}"},
                {"type": "mrkdwn",
                 "text": f":repeat: *Wisselingen deze maand*\n{wisselingen}"},
            ]
        },
        {"type": "divider"},
    ]

    if config_switched:
        old_full = [b for b in ALL_BOTS if b.replace("_v2", "") in old_active_base]
        old_str  = "  |  ".join(BOT_LABELS.get(b, b) for b in old_full) or "(geen)"
        blocks.append({
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f":arrow_right: *Oud*\n{old_str}"},
                {"type": "mrkdwn", "text": f":arrow_right: *Nieuw*\n{actief_str or '(geen)'}"},
            ]
        })
        blocks.append({"type": "divider"})

    blocks += [
        {
            "type": "section",
            "text": {"type": "mrkdwn",
                     "text": (f":white_check_mark: *Actief ({len(actief)}):*  {actief_str or '—'}\n"
                              f":double_vertical_bar: *Gepauzeerd ({len(gepauzeerd)}):*  "
                              f"{gepauzeerd_str or '—'}")}
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f":speech_balloon: _{redenering}_"}
        },
    ]
    return blocks


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Meta Agent — Fix1+Fix2")
    print("=" * 60)

    # 1. Data ophalen
    print("\nBitvavo candles ophalen ...")
    data_raw = {}
    for market in REGIME_TICKERS:
        print(f"  {market} ...", end=" ", flush=True)
        df = fetch_candles(market)
        print(f"{len(df)} bars" if df is not None else "MISLUKT")
        data_raw[market] = df

    print("\nFear & Greed ophalen ...")
    fng_data = fetch_fng_history(limit=42)
    fng      = current_fng(fng_data)
    print(f"  FNG: {fng['score']} — {fng['label']} ({fng['trend']})")

    # 2. Indicatoren & regime
    print("\nIndicatoren berekenen ...")
    data_ind = {}
    for market, df in data_raw.items():
        ind = add_indicators(df)
        data_ind[market] = ind
        bars = len(ind) if ind is not None else 0
        print(f"  {market}: {bars} bars")

    p33, p67 = compute_thresholds(data_ind)
    print(f"  Percentiel grenzen: p33={p33:.4f}  p67={p67:.4f}")

    # Regime stats op basis van laatste candle
    bull_regimes    = 0
    bear_regimes    = 0
    atr_vals        = []
    regime_details  = {}

    for market, df in data_ind.items():
        if df is None or df.empty:
            continue
        row = df.iloc[-1]
        sc  = regime_score(row)
        reg = get_regime(sc, p33, p67)
        regime_details[market] = reg
        if reg == "bull":
            bull_regimes += 1
        elif reg == "bear":
            bear_regimes += 1
        if market in ("BTC-EUR", "ETH-EUR"):
            v = float(row.get("atr_ratio", 1.0))
            if pd.notna(v):
                atr_vals.append(v)

    atr_ratio = float(np.mean(atr_vals)) if atr_vals else 1.0
    print(f"  Bull: {bull_regimes}  Bear: {bear_regimes}  ATR ratio: {atr_ratio:.2f}")
    for m, r in regime_details.items():
        print(f"    {m}: {r}")

    # 3. Huidige configuratie laden
    print("\nHuidige configuratie laden ...")
    existing_config, config_sha = gh_read_config()

    now_utc   = datetime.now(timezone.utc)
    now_month = now_utc.strftime("%Y-%m")
    now_iso   = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    if existing_config is None:
        print("  Geen bestaande configuratie — start vers met alle bots aan.")
        regime_history     = []
        prev_active_base   = list(BASE_BOTS)
        last_switch_ts     = None
        wisselingen_maand  = 0
        last_switch_month  = ""
    else:
        regime_history     = existing_config.get("regime_history", [])
        prev_active_base   = existing_config.get("actieve_bots_base", list(BASE_BOTS))
        last_switch_ts     = existing_config.get("laatste_wissel")
        wisselingen_maand  = existing_config.get("wisselingen_deze_maand", 0)
        last_switch_month  = existing_config.get("laatste_wissel_maand", "")
        if last_switch_month != now_month:
            wisselingen_maand = 0

    # Ruwe aanbeveling
    recommended_base, _ = meta_agent_beslissing(fng["score"], bull_regimes, bear_regimes, atr_ratio)
    print(f"  Ruwe aanbeveling: {recommended_base}")
    print(f"  Huidige config:   {prev_active_base}")

    # Regime history bijwerken (bewaar max 50 entries)
    regime_history.append({
        "ts":                 now_iso,
        "recommended_active": list(recommended_base),
        "bear_regimes":       bear_regimes,
        "bull_regimes":       bull_regimes,
        "fng":                fng["score"],
        "atr_ratio":          round(atr_ratio, 3),
    })
    if len(regime_history) > 50:
        regime_history = regime_history[-50:]

    # Fix1: bevestigingscheck
    if should_switch_config(prev_active_base, recommended_base, regime_history):
        new_active_base = list(recommended_base)
        if set(new_active_base) != set(prev_active_base):
            print(f"  Fix1: wissel toegestaan → {new_active_base}")
        else:
            print(f"  Fix1: geen wijziging nodig")
    else:
        new_active_base = list(prev_active_base)
        print(f"  Fix1: bevestiging nog niet compleet ({CONFIRM_CANDLES} candles nodig) — config behouden")

    new_paused_base = [b for b in BASE_BOTS if b not in new_active_base]

    # Fix2: bear bescherming
    new_active_base, new_paused_base = apply_bear_protection(
        new_active_base, new_paused_base, regime_history)

    # Bear percentage berekenen
    if len(regime_history) >= BEAR_LOOKBACK:
        bear_pct_7d = sum(
            1 for r in regime_history[-BEAR_LOOKBACK:]
            if r.get("bear_regimes", 0) >= 3
        ) / BEAR_LOOKBACK
    else:
        bear_pct_7d = 0.0

    bear_bescherming_actief = bear_pct_7d > BEAR_THRESHOLD

    if bear_bescherming_actief and "fear" in new_active_base and "fear" not in (
        list(recommended_base) if should_switch_config(prev_active_base, recommended_base, regime_history)
        else prev_active_base
    ):
        print(f"  Fix2: bear bescherming actief ({bear_pct_7d*100:.0f}% bear) — fear bot geforceerd aan")

    # Configuratiewissel detecteren
    config_switched = set(new_active_base) != set(prev_active_base)
    if config_switched:
        wisselingen_maand += 1
        last_switch_ts     = now_iso
        print(f"  CONFIGURATIEWISSEL: {prev_active_base} → {new_active_base}")

    # Uitbreiden naar v2 varianten
    actieve_bots    = [b for b in ALL_BOTS if b.replace("_v2", "") in new_active_base]
    gepauzeerde_bots = [b for b in ALL_BOTS if b.replace("_v2", "") not in new_active_base]

    # Markt label
    s = fng["score"]
    markt_label = (
        "extreme angst" if s < 20 else
        "angst"         if s < 40 else
        "neutraal"      if s < 60 else
        "hebzucht"      if s < 80 else
        "extreme hebzucht"
    )

    # Redenering bouwen
    reden_parts = [f"FNG {s} {fng['trend']}"]
    if bear_bescherming_actief:
        reden_parts.append(f"bear bescherming actief ({bear_pct_7d*100:.0f}% bear laatste 7d)")
    if bull_regimes >= 3:
        reden_parts.append(f"{bull_regimes} bull regimes")
    elif bear_regimes >= 3:
        reden_parts.append(f"{bear_regimes} bear regimes")
    if atr_ratio > 2.0:
        reden_parts.append(f"hoge volatiliteit (ATR ratio {atr_ratio:.1f})")
    redenering = ", ".join(reden_parts) + "."

    # Nieuw config object
    new_config = {
        "timestamp":               now_iso,
        "markt_label":             markt_label,
        "fng":                     fng,
        "redenering":              redenering,
        "actieve_bots":            actieve_bots,
        "gepauzeerde_bots":        gepauzeerde_bots,
        "actieve_bots_base":       new_active_base,
        "laatste_wissel":          last_switch_ts,
        "wisselingen_deze_maand":  wisselingen_maand,
        "laatste_wissel_maand":    now_month,
        "bear_bescherming_actief": bear_bescherming_actief,
        "regime_details":          {m: regime_details.get(m, "neutraal") for m in REGIME_TICKERS},
        "regime_summary": {
            "bull_count":  bull_regimes,
            "bear_count":  bear_regimes,
            "atr_ratio":   round(atr_ratio, 3),
            "bear_pct_7d": round(bear_pct_7d, 3),
        },
        "regime_history": regime_history,
    }

    # 4. Naar GitHub schrijven
    commit_msg = (
        f"meta_agent: {now_iso[:16]} — "
        f"actief={','.join(new_active_base)} "
        f"{'(wissel)' if config_switched else '(geen wissel)'}"
    )
    gh_write_config(new_config, config_sha, commit_msg)

    # 5. Slack
    print("\nSlack notificatie versturen ...")
    blocks = build_slack_blocks(new_config, config_switched, prev_active_base)
    send_slack(blocks)

    print("\nKlaar.")
    print(f"  Actief:             {actieve_bots}")
    print(f"  Gepauzeerd:         {gepauzeerde_bots}")
    print(f"  Bear bescherming:   {bear_bescherming_actief}")
    print(f"  Wisselingen/maand:  {wisselingen_maand}")
    print("=" * 60)


if __name__ == "__main__":
    main()
