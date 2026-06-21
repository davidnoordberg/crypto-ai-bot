import os
import base64
import json
import requests
from datetime import datetime


GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

STATE_FILES = {
    "baseline_v1": "state.json",
    "v3_v1": "state_v3.json",
    "allin_v1": "state_allin.json",
    "fear_v1": "state_fear.json",
    "baseline_v2": "state_baseline_v2.json",
    "v3_v2": "state_v3_v2.json",
    "allin_v2": "state_allin_v2.json",
    "fear_v2": "state_fear_v2.json",
}

BOT_EMOJIS = {
    "baseline": "🛡️",
    "v3": "🎯",
    "allin": "🚀",
    "fear": "💀",
}

BOT_LABELS = {
    "baseline": "Baseline",
    "v3": "V3     ",
    "allin": "All-in ",
    "fear": "Fear   ",
}


def read_state_from_github(filename: str) -> dict | None:
    """Read a JSON state file from the GitHub repo via the Contents API."""
    if not GITHUB_TOKEN or not GITHUB_REPOSITORY:
        return None
    url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/contents/{filename}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        content = base64.b64decode(resp.json()["content"]).decode("utf-8")
        return json.loads(content)
    except Exception:
        return None


def extract_metrics(state: dict | None) -> dict:
    """Extract key metrics from a state dict, with safe fallbacks."""
    if state is None:
        return {
            "capital": None,
            "total": None,
            "return_pct": None,
            "trades": None,
            "win_rate": None,
        }
    capital = state.get("capital_eur")
    total = state.get("portfolio_total", capital)
    return {
        "capital": capital,
        "total": total,
        "return_pct": state.get("total_return_pct"),
        "trades": state.get("total_trades"),
        "win_rate": state.get("win_rate"),
    }


def fmt_eur(val) -> str:
    if val is None:
        return "—"
    return f"€{val:.2f}"


def fmt_pct(val, sign=True) -> str:
    if val is None:
        return "—"
    return f"{val:+.1f}%" if sign else f"{val:.1f}%"


def fmt_int(val) -> str:
    if val is None:
        return "—"
    return str(int(val))


def fmt_winrate(val) -> str:
    if val is None:
        return "—"
    return f"{val:.1f}%"


def build_bot_row(key: str, metrics: dict) -> str:
    emoji = BOT_EMOJIS[key]
    label = BOT_LABELS[key]
    total = fmt_eur(metrics["total"])
    ret = fmt_pct(metrics["return_pct"])
    trades = fmt_int(metrics["trades"])
    win = fmt_winrate(metrics["win_rate"])
    return f"{emoji} {label}  {total:<12} {ret:<12} {trades:<10} {win}"


def build_diff_row(key: str, v1: dict, v2: dict) -> str:
    emoji = BOT_EMOJIS[key]
    label = BOT_LABELS[key]
    r1 = v1["return_pct"]
    r2 = v2["return_pct"]
    if r1 is None or r2 is None:
        diff_str = "—"
    else:
        diff = r2 - r1
        diff_str = f"{diff:+.1f}%"
    return f"{emoji} {label}: {diff_str}"


def build_table(title: str, bot_keys: list[str], all_metrics: dict) -> str:
    header = f"*{title}*\n```\nBot          €Kapitaal    Rendement    Trades    Win%\n"
    rows = []
    for key in bot_keys:
        rows.append(build_bot_row(key, all_metrics[key]))
    return header + "\n".join(rows) + "\n```"


def main():
    # Read all state files
    states = {}
    for key, filename in STATE_FILES.items():
        states[key] = read_state_from_github(filename)

    # Extract metrics
    metrics = {key: extract_metrics(states[key]) for key in STATE_FILES}

    today = datetime.utcnow().strftime("%d-%m-%Y")

    v1_keys = ["baseline_v1", "v3_v1", "allin_v1", "fear_v1"]
    v2_keys = ["baseline_v2", "v3_v2", "allin_v2", "fear_v2"]
    bot_names = ["baseline", "v3", "allin", "fear"]

    # Rename metrics keys for table builders
    v1_metrics = {name: metrics[f"{name}_v1"] for name in bot_names}
    v2_metrics = {name: metrics[f"{name}_v2"] for name in bot_names}

    v1_table = build_table("ENKELVOUDIGE LLM (v1)", bot_names, v1_metrics)
    v2_table = build_table("MULTI-AGENT LLM (v2)", bot_names, v2_metrics)

    diff_lines = [build_diff_row(name, v1_metrics[name], v2_metrics[name]) for name in bot_names]
    diff_section = "*V2 vs V1 verschil (rendement)*\n" + "\n".join(diff_lines)

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📊 DAGELIJKSE VERGELIJKING — {today}",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": v1_table},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": v2_table},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": diff_section},
        },
    ]

    payload = {"blocks": blocks}

    if SLACK_WEBHOOK_URL:
        resp = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=15)
        if resp.status_code == 200:
            print("Slack bericht verzonden.")
        else:
            print(f"Slack fout: {resp.status_code} — {resp.text}")
    else:
        print("Geen SLACK_WEBHOOK_URL gevonden. Bericht niet verzonden.")
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
