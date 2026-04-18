"""Precompute HuggingFace market metadata for the viz coverage page.

Writes two JSON files into viz/data/:
- hf_temperature_markets.json: detailed temperature markets (city, end_date,
  conditionId, event_id, question, tokens).
- hf_conditions_index.json: compact {condition_id: [category, q_short, end_date_str]}
  for all HF markets with end_date >= 2026-01-01 (so PMXT overlap is computable
  without shipping the full 700K-market dataset).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "SII-WANGZJ/Polymarket_data"
OUT_DIR = Path(__file__).resolve().parents[1] / "viz" / "data"

# Markets with end_date earlier than this are pre-PMXT era and cannot possibly
# appear in PMXT hourly parquet files. Cuts the index from ~735K to ~480K.
INDEX_MIN_END_DATE = "2026-01-01"

CITY_RE = re.compile(
    r"temperature in ([A-Z][A-Za-z .'\-]+?)(?:\s+be\b|\s+on\b|\?\s*$)",
    re.IGNORECASE,
)


def classify(question: str, event_title: str) -> str:
    text = f"{question or ''} {event_title or ''}".lower()
    if "temperature" in text:
        return "temperature"
    if any(k in text for k in ["bitcoin", "ethereum", " btc ", " eth ", " sol ", "crypto", "solana"]):
        return "crypto"
    if any(k in text for k in ["president", "election", "senate", "congress", "trump", "biden", "harris", "governor", "prime minister"]):
        return "politics"
    if any(k in text for k in [" nfl", " nba", " mlb", " nhl", "super bowl", "world cup", " ufc", "champions league", "tennis", "soccer", "football", "basketball", "baseball", "hockey"]):
        return "sports"
    if any(k in text for k in ["oscar", "grammy", "emmy", "movie", "music"]):
        return "entertainment"
    if any(k in text for k in ["fed", "rate cut", "inflation", "gdp", "unemployment", "recession"]):
        return "economics"
    return "other"


def extract_city(question: str) -> str | None:
    m = CITY_RE.search(question or "")
    if not m:
        return None
    city = m.group(1).strip().rstrip("?").strip()
    # Strip trailing "be" phrases just in case
    city = re.sub(r"\s+be$", "", city, flags=re.IGNORECASE).strip()
    return city or None


def extract_temp_date(event_title: str, end_date: pd.Timestamp) -> str | None:
    # Prefer end_date (YYYY-MM-DD)
    if pd.notna(end_date):
        return pd.Timestamp(end_date).strftime("%Y-%m-%d")
    m = re.search(r"on\s+([A-Za-z]+\s+\d+)", event_title or "")
    return m.group(1) if m else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading markets.parquet from {REPO_ID}...")
    path = hf_hub_download(repo_id=REPO_ID, filename="markets.parquet", repo_type="dataset")
    print(f"  Cached at: {path}")

    cols = [
        "id", "question", "slug", "condition_id", "token1", "token2",
        "event_id", "event_slug", "event_title", "end_date", "closed",
    ]
    df = pd.read_parquet(path, columns=cols)
    print(f"Loaded {len(df):,} markets.")

    df = df[df["condition_id"].notna()].copy()
    df["end_date"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce")

    # === Temperature markets (full detail) ===
    temp_mask = df["question"].str.contains("temperature", case=False, na=False) | df[
        "event_title"
    ].fillna("").str.contains("temperature", case=False, na=False)
    temp = df[temp_mask].copy()
    print(f"Temperature markets: {len(temp):,}")

    temp_records = []
    for _, r in temp.iterrows():
        city = extract_city(r["question"] or "") or extract_city(r["event_title"] or "")
        end_date = r["end_date"]
        temp_records.append({
            "condition_id": r["condition_id"],
            "city": city,
            "end_date": end_date.strftime("%Y-%m-%d") if pd.notna(end_date) else None,
            "end_date_full": end_date.isoformat() if pd.notna(end_date) else None,
            "event_id": r["event_id"],
            "event_title": r["event_title"],
            "event_slug": r["event_slug"],
            "question": r["question"],
            "token1": r["token1"],
            "token2": r["token2"],
            "closed": bool(r["closed"]) if pd.notna(r["closed"]) else None,
        })

    temp_out = OUT_DIR / "hf_temperature_markets.json"
    with temp_out.open("w") as f:
        json.dump({
            "repo": REPO_ID,
            "count": len(temp_records),
            "markets": temp_records,
        }, f, separators=(",", ":"))
    print(f"Wrote {temp_out} ({temp_out.stat().st_size/1048576:.2f} MB)")

    # === Condition index (compact, for overlap) ===
    # Emit single-character category codes to keep transfer size manageable.
    # 483K entries * ~70 bytes per object-pair ≈ 34MB raw, ~12MB gzipped.
    cutoff = pd.Timestamp(INDEX_MIN_END_DATE, tz="UTC")
    idx = df[df["end_date"] >= cutoff].copy()
    print(f"Index markets (end_date >= {INDEX_MIN_END_DATE}): {len(idx):,}")

    cat_code = {
        "temperature": "t",
        "crypto": "c",
        "politics": "p",
        "sports": "s",
        "entertainment": "e",
        "economics": "n",
        "other": "o",
    }

    index = {}
    for _, r in idx.iterrows():
        cat = classify(r["question"] or "", r["event_title"] or "")
        index[r["condition_id"]] = cat_code[cat]

    idx_out = OUT_DIR / "hf_conditions_index.json"
    with idx_out.open("w") as f:
        json.dump({
            "repo": REPO_ID,
            "min_end_date": INDEX_MIN_END_DATE,
            "count": len(index),
            "categories": {v: k for k, v in cat_code.items()},
            "conditions": index,
        }, f, separators=(",", ":"))
    print(f"Wrote {idx_out} ({idx_out.stat().st_size/1048576:.2f} MB)")


if __name__ == "__main__":
    main()
