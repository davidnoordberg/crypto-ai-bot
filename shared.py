"""
shared.py — Gedeelde infrastructuur voor v2 trading bots
=========================================================
Bevat:
  - Bitvavo data ophalen
  - Indicator berekening
  - Regime detectie
  - Fear & Greed
  - Nieuws ophalen
  - GitHub state (lezen/schrijven)
  - Supabase logging
  - Slack helpers
  - ask_llm()          — enkelvoudige LLM (voor referentie)
  - multi_agent_decision() — vier-agent pipeline (voor v2 bots)

Vereiste env vars:
  ANTHROPIC_API_KEY  — Anthropic
  GITHUB_TOKEN       — GitHub Contents API
  GITHUB_REPOSITORY  — "owner/repo"
  SLACK_WEBHOOK_URL  — Slack Incoming Webhook
  SUPABASE_URL       — Supabase project URL
  SUPABASE_KEY       — Supabase service role key
"""

from __future__ import annotations
import os, json, re, base64, warnings
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

try:
    from supabase import create_client as _sb_create
    _SUPABASE_CLIENT = None
except ImportError:
    _sb_create       = None
    _SUPABASE_CLIENT = None

# ── Constanten ────────────────────────────────────────────────────────────────
BITVAVO_BASE  = "https://api.bitvavo.com/v2"
CANDLE_LIMIT  = 250
GH_API        = "https://api.github.com"
LLM_MODEL     = "claude-haiku-4-5-20251001"

ATR_WIN  = 14; MA50_WIN = 50; MA200_WIN = 200
MA200_LAG = 10; VOL_WIN = 20
MACD_F, MACD_S, MACD_SIG_W = 12, 26, 9
MR_BB_WIN = 30; MR_BB_SIG = 2.5
FEE_RT    = 0.004

PARAMS_FILE = Path(__file__).parent / "best_params_hybrid.json"


# ── Client initialisatie ───────────────────────────────────────────────────────
def init_clients():
    global _ANTHR_CLIENT, _SUPABASE_CLIENT
    ak = os.environ.get("ANTHROPIC_API_KEY", "")
    if _anthropic and ak:
        _ANTHR_CLIENT = _anthropic.Anthropic(api_key=ak)
    su = os.environ.get("SUPABASE_URL", "")
    sk = os.environ.get("SUPABASE_KEY", "")
    if _sb_create and su and sk:
        try:
            _SUPABASE_CLIENT = _sb_create(su, sk)
        except Exception as e:
            print(f"  [WARN] Supabase init mislukt: {e}")


# ── Bitvavo data ───────────────────────────────────────────────────────────────
def fetch_candles(market: str, interval: str = "4h",
                  limit: int = CANDLE_LIMIT) -> Optional[pd.DataFrame]:
    url = f"{BITVAVO_BASE}/{market}/candles"
    try:
        r = requests.get(url, params={"interval": interval, "limit": limit}, timeout=15)
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


def fetch_all_candles(tickers: list) -> dict:
    result = {}
    for t in tickers:
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


# ── Indicatoren ────────────────────────────────────────────────────────────────
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


def price_change_pct(df: pd.DataFrame, bars: int) -> float:
    """Prijsverandering over de laatste N bars als percentage."""
    if df is None or len(df) < bars + 1:
        return 0.0
    now  = float(df["Close"].iloc[-1])
    then = float(df["Close"].iloc[-1 - bars])
    return (now - then) / then * 100 if then else 0.0


# ── Regime ─────────────────────────────────────────────────────────────────────
def load_thresholds() -> dict:
    if PARAMS_FILE.exists():
        raw = json.loads(PARAMS_FILE.read_text())
        return {k.replace("-EUR", "").replace("-USD", ""): v for k, v in raw.items()}
    print("  [WARN] best_params_hybrid.json niet gevonden — gebruik 0/0 fallback.")
    return {}


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


# ── Nieuws ─────────────────────────────────────────────────────────────────────
_COIN_MAP = {
    "doge": "dogecoin", "sol": "solana", "eth": "ethereum",
    "btc": "bitcoin",   "avax": "avalanche", "ada": "cardano",
}


def fetch_news(ticker: str) -> list[str]:
    coin = ticker.replace("-EUR", "").replace("-USD", "").lower()
    slug = _COIN_MAP.get(coin, coin)
    url  = f"https://cryptopanic.com/news/{slug}/rss/"
    try:
        feed  = feedparser.parse(url)
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


# ── Label helpers ──────────────────────────────────────────────────────────────
def candle_label(row) -> str:
    o = float(row["Open"]); c = float(row["Close"])
    h = float(row["High"]); l = float(row["Low"])
    body   = abs(c - o); range_ = h - l or 1e-9
    dir_   = "bullish" if c > o else "bearish"
    size   = "groot" if body / range_ > 0.6 else ("klein" if body / range_ < 0.2 else "normaal")
    return f"{dir_} {size} lichaam"


def ma200_label(row) -> str:
    ma200 = float(row["ma200"]); lag = float(row["ma200_lag"])
    s = (ma200 - lag) / lag * 100 if lag else 0
    if s > 0.1:  return f"stijgend (+{s:.2f}%/10 bars)"
    if s < -0.1: return f"dalend ({s:.2f}%/10 bars)"
    return "zijwaarts"


def prijs_label(row) -> str:
    close = float(row["Close"]); ma200 = float(row["ma200"])
    pct   = (close - ma200) / ma200 * 100
    if pct > 5:  return f"+{pct:.1f}% boven MA200"
    if pct > 0:  return f"+{pct:.1f}% net boven MA200"
    return f"{pct:.1f}% onder MA200"


# ── GitHub state ───────────────────────────────────────────────────────────────
def _gh_headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN", "")
    h = {"Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def gh_read(path: str) -> tuple[Optional[dict], Optional[str]]:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    url  = f"{GH_API}/repos/{repo}/contents/{path}"
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


def gh_write(path: str, content: dict, sha: Optional[str], message: str):
    repo    = os.environ.get("GITHUB_REPOSITORY", "")
    url     = f"{GH_API}/repos/{repo}/contents/{path}"
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


# ── Supabase logging ───────────────────────────────────────────────────────────
def log_agent_decision(data: dict):
    """
    Logt een multi-agent beslissing naar de agent_decisions tabel.
    Faalt stil als Supabase niet geconfigureerd is.
    """
    if _SUPABASE_CLIENT is None:
        print("  [INFO] Supabase niet geconfigureerd — beslissing niet gelogd.")
        return
    try:
        _SUPABASE_CLIENT.table("agent_decisions").insert(data).execute()
        print(f"  [Supabase] Beslissing gelogd: {data.get('ticker')} → {data.get('finale_beslissing')}")
    except Exception as e:
        print(f"  [WARN] Supabase log mislukt: {e}")


# ── Slack helpers ──────────────────────────────────────────────────────────────
def send_slack_blocks(blocks: list):
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


def portfolio_blocks(bot_label: str, state: dict, fng: dict,
                     start_cap: float) -> list:
    """Bouw de standaard portfolio-samenvatting blocks (gedeeld door alle v2 bots)."""
    now_cet              = datetime.now(timezone.utc) + timedelta(hours=2)
    now_str              = now_cet.strftime("%d-%m-%Y %H:%M")
    gw                   = state["gross_win"]; gl = state["gross_loss"]
    pf                   = round(gw / gl, 2) if gl > 0 else 0.0
    wr                   = state.get("win_rate", 0.0)
    portfolio_total      = state.get("portfolio_total",
                               state["capital_eur"] + state.get("open_positions_value", 0.0))
    open_positions_value = state.get("open_positions_value", 0.0)
    ret_pct              = state.get("total_return_pct",
                               (portfolio_total - start_cap) / start_cap * 100)
    ret_icon             = ":arrow_up_small:" if ret_pct >= 0 else ":arrow_down_small:"

    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f":robot_face: {bot_label} — {now_str} CET"}
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
                         f"WR {wr:.1f}%  |  PF {pf:.2f}"},
                {"type": "mrkdwn",
                 "text": f":scream: *Fear & Greed*\n{fng['score']} — {fng['label']}"},
            ]
        },
        {"type": "divider"},
    ]


def agent_decision_block(decision: dict) -> dict:
    """Bouw een Slack block voor de multi-agent beslissing."""
    sent_s  = decision.get("sentiment_score", 0.0)
    sent_l  = decision.get("sentiment_label", "?")
    tech_s  = decision.get("technical_score", 0.0)
    tech_q  = decision.get("setup_kwaliteit", "?")
    risk_s  = decision.get("risico_score", 0.0)
    risk_l  = decision.get("risico_label", "?")
    bes     = decision.get("beslissing", "?")
    cons    = decision.get("consensus", "?")
    reden   = decision.get("reden", "")
    conf    = decision.get("confidence", 0.0)
    factor  = decision.get("doorslaggevende_factor", "?")
    failed  = decision.get("agents_failed", [])

    failed_txt = f"  ⚠️ Agents gefaald: {', '.join(failed)}\n" if failed else ""
    bes_icon   = ":white_check_mark:" if bes == "KOOP" else ":no_entry_sign:"

    text = (
        f":brain: *Multi-Agent Beslissing*\n"
        f"  :newspaper: Sentiment: *{sent_l}* ({sent_s:+.2f})\n"
        f"  :chart_with_upwards_trend: Technisch: *{tech_q}* ({tech_s:+.2f})\n"
        f"  :warning: Risico: *{risk_l}* ({risk_s:.2f})\n"
        f"  {bes_icon} Beslissing: *{bes}*  |  Consensus: {cons}  |  conf={conf:.2f}\n"
        f"  :pushpin: Factor: _{factor}_\n"
        f"  :memo: _{reden[:160]}_\n"
        f"{failed_txt}"
    )
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


# ── ask_llm — enkelvoudige agent (voor v1 bots, hier als referentie) ───────────
def ask_llm(ticker: str, row, regime: str, strategy: str,
            regime_map: dict, fng: dict, headlines: list,
            valid_sizes: list = None) -> dict:
    """Enkelvoudige Haiku beslissing (zelfde als in v1 bots)."""
    if valid_sizes is None:
        valid_sizes = [7, 10, 15]
    fallback = {"beslissing": "KOOP", "confidence": 0.5,
                "positiegrootte": valid_sizes[len(valid_sizes) // 2],
                "nieuws_sentiment": "neutraal",
                "reden": "LLM niet beschikbaar"}
    if _ANTHR_CLIENT is None:
        return fallback

    hl_txt = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines[:5]))
    sizes_str = ", ".join(str(s) for s in valid_sizes)
    prompt = f"""Crypto: {ticker.replace('-EUR','')}
Datum: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC
Regime: {regime} | Strategie: {strategy}
RSI: {float(row['rsi']):.1f} | MACD hist: {float(row['macd_hist']):.6f}
ATR ratio: {float(row['atr'])/float(row['Close'])*100:.2f}% | Vol ratio: {float(row['vol_ratio']):.2f}x
MA200: {ma200_label(row)} | Prijs: {prijs_label(row)}
Fear & Greed: {fng['score']} ({fng['label']})
Nieuws: {hl_txt}

Geef antwoord als JSON:
{{"beslissing":"KOOP" of "NIETS","confidence":0.0-1.0,"positiegrootte":{sizes_str},"nieuws_sentiment":"bullish/neutraal/bearish","reden":"max 2 zinnen"}}"""

    try:
        resp = _ANTHR_CLIENT.messages.create(
            model=LLM_MODEL, max_tokens=200,
            messages=[{"role": "user", "content": prompt}])
        raw = resp.content[0].text.strip()
        m   = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return fallback
        p   = json.loads(m.group())
        bes = str(p.get("beslissing", "KOOP")).upper()
        if bes not in ("KOOP", "NIETS"):
            bes = "KOOP"
        ps  = int(p.get("positiegrootte", valid_sizes[0]))
        if ps not in valid_sizes:
            ps = valid_sizes[len(valid_sizes) // 2]
        ns  = str(p.get("nieuws_sentiment", "neutraal")).lower()
        if ns not in ("bullish", "neutraal", "bearish"):
            ns = "neutraal"
        return {"beslissing": bes, "confidence": float(p.get("confidence", 0.5)),
                "positiegrootte": ps, "nieuws_sentiment": ns,
                "reden": str(p.get("reden", ""))[:300]}
    except Exception as e:
        print(f"  [WARN] ask_llm {ticker}: {e}")
        return fallback


# ── multi_agent_decision — vier-agent pipeline ────────────────────────────────
def _call_agent(prompt: str, schema_keys: list, fallback: dict,
                agent_name: str) -> dict:
    """Roep een enkel Haiku-agent aan, retourneer dict of fallback bij fout."""
    if _ANTHR_CLIENT is None:
        return {**fallback, "_failed": True}
    try:
        resp = _ANTHR_CLIENT.messages.create(
            model=LLM_MODEL, max_tokens=300,
            messages=[{"role": "user", "content": prompt}])
        raw = resp.content[0].text.strip()
        m   = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            raise ValueError("geen JSON in response")
        parsed = json.loads(m.group())
        # Valideer dat verplichte sleutels aanwezig zijn
        for k in schema_keys:
            if k not in parsed:
                parsed[k] = fallback.get(k)
        parsed["_failed"] = False
        return parsed
    except Exception as e:
        print(f"  [WARN] Agent {agent_name} fout: {e}")
        return {**fallback, "_failed": True}


def multi_agent_decision(
    ticker: str,
    row,
    df: pd.DataFrame,
    regime: str,
    strategy: str,
    regime_map: dict,
    fng: dict,
    headlines: list,
    state: dict,
    proposed_frac: float,
    valid_sizes: list,
    bot_id: str = "",
    max_open: int = 5,
) -> dict:
    """
    Vier-agent beslissingspipeline:
      1. Sentiment Analyst
      2. Technical Analyst
      3. Risk Manager
      4. Decision Maker (gebruikt outputs 1-3)

    Retourneert een volledig beslissingsdict inclusief alle agent-outputs.
    """
    ticker_name   = ticker.replace("-EUR", "").replace("-USD", "")
    hl_txt        = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines[:5]))
    close         = float(row["Close"])
    atr_pct       = float(row["atr"]) / close * 100
    vol_ratio     = float(row["vol_ratio"])
    rsi           = float(row["rsi"])
    macd_hist     = float(row["macd_hist"])
    p24h          = price_change_pct(df, 6)   # 6 × 4h ≈ 24 uur
    p7d           = price_change_pct(df, 42)  # 42 × 4h ≈ 7 dagen
    n_open        = len(state.get("open_positions", {}))
    free_cap      = state.get("capital_eur", 0.0)
    total_val     = state.get("portfolio_total",
                        free_cap + state.get("open_positions_value", 0.0))
    expo_pct      = (state.get("open_positions_value", 0.0) / total_val * 100
                     if total_val > 0 else 0.0)
    proposed_pct  = round(proposed_frac * 100, 1)
    sizes_str     = ", ".join(str(s) for s in valid_sizes)
    agents_failed = []

    # ── Agent 1: Sentiment Analyst ────────────────────────────────────────────
    a1_fallback = {
        "sentiment_score": 0.0, "sentiment_label": "neutraal",
        "nieuws_impact": "neutraal", "belangrijkste_factor": "onbekend",
        "vertrouwen": 0.3,
    }
    a1_prompt = f"""Jij bent een crypto sentiment analyst.

Crypto: {ticker_name}
Fear & Greed Index: {fng['score']} ({fng['label']})
Prijs 24u verandering: {p24h:+.1f}%
Prijs 7d verandering: {p7d:+.1f}%
Recent nieuws (laatste 24u):
{hl_txt}

Beoordeel het marktsentiment voor een mogelijke LONG entry.
Geef je antwoord UITSLUITEND als geldig JSON:
{{
  "sentiment_score": <getal -1.0 tot +1.0>,
  "sentiment_label": "sterk bullish|bullish|neutraal|bearish|sterk bearish",
  "nieuws_impact": "positief|neutraal|negatief",
  "belangrijkste_factor": "<max 10 woorden>",
  "vertrouwen": <getal 0.0 tot 1.0>
}}"""
    a1 = _call_agent(a1_prompt, list(a1_fallback.keys()), a1_fallback, "Sentiment")
    if a1.get("_failed"):
        agents_failed.append("sentiment")

    # ── Agent 2: Technical Analyst ────────────────────────────────────────────
    a2_fallback = {
        "technische_score": 0.0, "setup_kwaliteit": "matig",
        "entry_timing": "onbekend", "belangrijkste_signaal": "onbekend",
        "vertrouwen": 0.3,
    }
    a2_prompt = f"""Jij bent een crypto technisch analist.

Crypto: {ticker_name}
RSI(14): {rsi:.1f}
MACD histogram: {macd_hist:.6f} (vorige: {float(row['prev_hist']):.6f})
ATR ratio: {atr_pct:.2f}% van prijs
Volume ratio: {vol_ratio:.2f}x gemiddelde
MA200 richting: {ma200_label(row)}
Prijspositie: {prijs_label(row)}
Laatste candle: {candle_label(row)}
Regime: {regime}
Strategie: {strategy}

Beoordeel de technische setup voor een LONG entry.
Geef je antwoord UITSLUITEND als geldig JSON:
{{
  "technische_score": <getal -1.0 tot +1.0>,
  "setup_kwaliteit": "uitstekend|goed|matig|slecht",
  "entry_timing": "ideaal|goed|vroeg|laat",
  "belangrijkste_signaal": "<max 10 woorden>",
  "vertrouwen": <getal 0.0 tot 1.0>
}}"""
    a2 = _call_agent(a2_prompt, list(a2_fallback.keys()), a2_fallback, "Technical")
    if a2.get("_failed"):
        agents_failed.append("technisch")

    # ── Agent 3: Risk Manager ─────────────────────────────────────────────────
    a3_fallback = {
        "risico_score": 0.5, "risico_label": "matig",
        "aanbevolen_positiegrootte_pct": valid_sizes[len(valid_sizes) // 2],
        "belangrijkste_risico": "onbekend",
        "blokkeer_trade": False, "blokkeer_reden": "",
    }
    a3_prompt = f"""Jij bent een crypto risk manager.

Crypto: {ticker_name}
Vrij kapitaal: €{free_cap:.2f}
Open posities: {n_open}/{max_open}
Huidige exposure: {expo_pct:.1f}%
Voorgestelde positiegrootte: {proposed_pct}%
Fear & Greed: {fng['score']} ({fng['label']})
ATR ratio: {atr_pct:.2f}% (volatiliteit indicator)
Regime: {regime}

Beoordeel het risico van deze trade en of hij uitgevoerd moet worden.
Beschikbare positiegroottes: {sizes_str} (% van kapitaal)
Geef je antwoord UITSLUITEND als geldig JSON:
{{
  "risico_score": <getal 0.0 tot 1.0, hoger = meer risico>,
  "risico_label": "laag|matig|hoog|zeer hoog",
  "aanbevolen_positiegrootte_pct": <kies uit {sizes_str}>,
  "belangrijkste_risico": "<max 10 woorden>",
  "blokkeer_trade": <true of false>,
  "blokkeer_reden": "<reden als blokkeer_trade=true, anders leeg>"
}}"""
    a3 = _call_agent(a3_prompt, list(a3_fallback.keys()), a3_fallback, "Risk")
    if a3.get("_failed"):
        agents_failed.append("risico")

    # Valideer blokkeer_trade
    if not isinstance(a3.get("blokkeer_trade"), bool):
        a3["blokkeer_trade"] = False

    # ── Agent 4: Decision Maker ────────────────────────────────────────────────
    a4_fallback = {
        "beslissing": "KOOP", "positiegrootte_pct": valid_sizes[len(valid_sizes) // 2],
        "confidence": 0.5, "consensus": "normaal",
        "doorslaggevende_factor": "technisch", "reden": "LLM niet beschikbaar",
    }

    # Als risk manager blokkeert: sla Agent 4 over
    if a3.get("blokkeer_trade") and not a3.get("_failed"):
        a4 = {**a4_fallback,
              "beslissing": "NIETS",
              "confidence": 0.9,
              "consensus": "sterk",
              "doorslaggevende_factor": "risico",
              "reden": a3.get("blokkeer_reden", "Risk manager blokkeert trade."),
              "_failed": False}
        print(f"  [Agent4] Risk manager blokkeert — NIETS")
    else:
        a4_prompt = f"""Jij bent de finale decision maker voor crypto trading.

Je ontvangt de analyses van drie specialisten:

SENTIMENT ANALYST:
- Score: {a1.get('sentiment_score', 0):+.2f} ({a1.get('sentiment_label', '?')})
- Nieuws impact: {a1.get('nieuws_impact', '?')}
- Belangrijkste factor: {a1.get('belangrijkste_factor', '?')}
- Vertrouwen: {a1.get('vertrouwen', 0):.2f}

TECHNICAL ANALYST:
- Score: {a2.get('technische_score', 0):+.2f} ({a2.get('setup_kwaliteit', '?')})
- Entry timing: {a2.get('entry_timing', '?')}
- Belangrijkste signaal: {a2.get('belangrijkste_signaal', '?')}
- Vertrouwen: {a2.get('vertrouwen', 0):.2f}

RISK MANAGER:
- Risico: {a3.get('risico_score', 0.5):.2f} ({a3.get('risico_label', '?')})
- Aanbevolen grootte: {a3.get('aanbevolen_positiegrootte_pct', valid_sizes[0])}%
- Belangrijkste risico: {a3.get('belangrijkste_risico', '?')}
- Trade blokkeren: {a3.get('blokkeer_trade', False)}

REGEL: Als Risk Manager blokkeer_trade=true, dan ALTIJD beslissing=NIETS.
Beschikbare positiegroottes: {sizes_str}%

Neem de finale handelsbeslissing.
Geef je antwoord UITSLUITEND als geldig JSON:
{{
  "beslissing": "KOOP" of "NIETS",
  "positiegrootte_pct": <kies uit {sizes_str}>,
  "confidence": <getal 0.0 tot 1.0>,
  "consensus": "sterk|normaal|verdeeld",
  "doorslaggevende_factor": "sentiment|technisch|risico",
  "reden": "<max 2 zinnen>"
}}"""
        a4 = _call_agent(a4_prompt, list(a4_fallback.keys()), a4_fallback, "Decision")
        if a4.get("_failed"):
            agents_failed.append("decision")

    # Valideer en normaliseer a4 output
    bes = str(a4.get("beslissing", "KOOP")).upper()
    if bes not in ("KOOP", "NIETS"):
        bes = "KOOP"
    a4["beslissing"] = bes

    ps = a4.get("positiegrootte_pct", valid_sizes[len(valid_sizes) // 2])
    try:
        ps = int(ps)
    except (TypeError, ValueError):
        ps = valid_sizes[len(valid_sizes) // 2]
    if ps not in valid_sizes:
        # Kies de dichtstbijzijnde geldige grootte
        ps = min(valid_sizes, key=lambda x: abs(x - ps))
    a4["positiegrootte_pct"] = ps

    # Als alle agents gefaald zijn: fallback op rule-based
    if len(agents_failed) == 4:
        print(f"  [WARN] Alle agents gefaald — rule-based fallback")
        a4["beslissing"]    = "KOOP"
        a4["reden"]         = "Rule-based fallback: alle LLM agents gefaald."
        a4["confidence"]    = 0.3
        a4["consensus"]     = "verdeeld"
        a4["_llm_fallback"] = True

    print(f"  [Multi-Agent] {ticker_name}: "
          f"sent={a1.get('sentiment_score',0):+.2f} "
          f"tech={a2.get('technische_score',0):+.2f} "
          f"risk={a3.get('risico_score',0):.2f} "
          f"→ {a4['beslissing']} {a4['positiegrootte_pct']}% "
          f"(conf={a4.get('confidence',0):.2f})")

    return {
        # Decision Maker output
        "beslissing":            a4["beslissing"],
        "positiegrootte_pct":    a4["positiegrootte_pct"],
        "confidence":            float(a4.get("confidence", 0.5)),
        "consensus":             a4.get("consensus", "normaal"),
        "doorslaggevende_factor": a4.get("doorslaggevende_factor", "?"),
        "reden":                 a4.get("reden", "")[:300],
        # Agent outputs voor Supabase/Slack
        "sentiment_score":       float(a1.get("sentiment_score", 0.0)),
        "sentiment_label":       a1.get("sentiment_label", "?"),
        "technische_score":      float(a2.get("technische_score", 0.0)),
        "setup_kwaliteit":       a2.get("setup_kwaliteit", "?"),
        "risico_score":          float(a3.get("risico_score", 0.5)),
        "risico_label":          a3.get("risico_label", "?"),
        # Meta
        "agents_failed":         agents_failed,
        "llm_used":              len(agents_failed) < 4,
        "_llm_fallback":         a4.get("_llm_fallback", False),
    }
