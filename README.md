# ☠ Dead Internet Observatory

**Tracking and quantifying the synthetic takeover of the public internet — in real time.**

The [Dead Internet Theory](https://en.wikipedia.org/wiki/Dead_Internet_theory) says the internet is increasingly populated by bots, AI-generated content, and automated engagement farms rather than real humans. Whether you think it's a conspiracy theory or an obvious reality, one thing is clear: nobody's actually measuring it. This project does.

We built a living "mission control" observatory that computes an **Internet Aliveness Index (IAI)** — a 0–100 score measuring how much of the sampled public internet still looks authentically human. We pull from six data sources, run a battery of statistical detection signals on every document, aggregate everything into a daily index, and serve it through a cyberpunk Streamlit dashboard.

No LLMs. No paid APIs. No subscriptions. 100% open source, 100% free to run.

---

## The Dashboard

> Live at: **[dead-internet-observatory.streamlit.app](https://dead-internet-observatory.streamlit.app)**

- Animated IAI gauge with live score and delta vs. 2019 baseline
- Two-year timeline with anomaly markers and decay shading
- Source breakdown by corpus (Web, Social, News, Wiki, HN, Wayback)
- Detection signal radar chart
- Anomaly spotlight — statistically significant spikes and crashes
- **What-if Simulator** — project future aliveness under different AI acceleration scenarios

---

## How the Index Works

The IAI is a weighted composite of seven linguistic and behavioural signals:

| Signal | Weight | What it's catching |
|---|---|---|
| **Type-Token Ratio** | 18% | Narrow vocabulary is a classic AI tell |
| **Shannon Entropy** | 15% | AI text has lower information density |
| **Sentence Length Variance** | 15% | AI writes unnaturally uniform sentences |
| **Bigram Repetition** | 15% | Stock phrase reuse, templated content |
| **Temporal Burstiness** | 15% | Bots post on schedules; humans don't |
| **MTLD** | 12% | Length-independent lexical diversity |
| **Zipf Law Alignment** | 10% | Natural language follows a power law; AI text deviates |

All signals are statistical — no external model calls, no GPU required. Everything runs on standard CPU in Python.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA MINIONS                             │
│  Common Crawl · Reddit · News RSS · Wikipedia · HackerNews      │
│  Wayback Machine (longitudinal sentinel URL tracking)           │
└────────────────────────┬────────────────────────────────────────┘
                         │ raw JSONL → GitHub Artifacts
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PIPELINE                                   │
│  Bronze (raw JSONL) → Silver (normalised Parquet)               │
│                     → Gold (scored Parquet)                     │
│                     → SQLite (daily index aggregates)           │
└────────────┬──────────────────────────┬────────────────────────┘
             │ Parquet → HF Datasets    │ observatory.db → git repo
             ▼                          ▼
┌────────────────────┐      ┌──────────────────────────────────────┐
│  HuggingFace       │      │  Streamlit Community Cloud           │
│  Datasets          │      │  (cyberpunk mission control dashboard)│
│  (public, free)    │      │                                      │
└────────────────────┘      └──────────────────────────────────────┘
```

GitHub Actions runs the full cycle autonomously — no server required.

```
03:00 UTC daily  →  Reddit Minion
03:30 UTC daily  →  News Crawler
04:00 UTC daily  →  Wikipedia + HackerNews
06:00 UTC daily  →  Pipeline (bronze→silver→gold→index)
02:00 UTC Sunday →  Wayback Machine + Common Crawl
```

---

## Data Sources

| Minion | Source | What we collect |
|---|---|---|
| `common_crawl_bot` | [Common Crawl](https://commoncrawl.org) | Quarterly WET snapshots — raw web text across 5 crawl dates |
| `reddit_bot` | Reddit public JSON API | Posts + comment trees from 10 subreddits |
| `news_crawler_bot` | 8 RSS feeds | Full article text from major news outlets |
| `wikipedia_bot` | Wikipedia API | 500 random articles + 24h edit pattern tracking |
| `hackernews_bot` | [Algolia HN API](https://hn.algolia.com/api) | Stories + comments, last 7 days |
| `wayback_bot` | [Wayback Machine CDX API](https://web.archive.org/cdx) | 14 sentinel URLs × 5 years (2019–2025) — the longitudinal signal |

**Where data lives:**
- Raw JSONL (bronze) → GitHub Artifacts, 7-day retention
- Processed Parquet (silver/gold) → [HuggingFace Datasets](https://huggingface.co/datasets/jupiternull/dead-internet-observatory)
- Daily index aggregates → `data/observatory.db` in this repo (what the dashboard reads)

---

## Quick Start

```bash
git clone https://github.com/jupiternull/dead-internet-observatory
cd dead-internet-observatory
pip install -r requirements.txt

# Populate with synthetic demo data and launch the dashboard
python3 run_minions.py seed-demo
streamlit run app/app.py
```

That's it. The app runs in demo mode immediately with two years of synthetic historical data.

---

## Running Real Data Collection

No API keys needed for any of these. All public endpoints.

```bash
# Individual minions
python3 run_minions.py reddit
python3 run_minions.py news
python3 run_minions.py wikipedia
python3 run_minions.py hackernews
python3 run_minions.py wayback        # slow — fetches 5 years of snapshots
python3 run_minions.py commoncrawl    # heavy — downloads multi-GB WET files

# Run the full pipeline after harvesting
python3 run_minions.py pipeline

# Or run everything at once (excludes Common Crawl and Wayback)
python3 run_minions.py all
```

---

## Deployment

**Streamlit Community Cloud (free):**
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → point to `app/app.py`
3. Done — auto-deploys on every push

**Activating autonomous data collection (GitHub Actions):**
1. Push to GitHub — the cron schedules activate automatically
2. *(Optional)* Add a `HF_TOKEN` secret in repo Settings → Secrets for Hugging Face dataset sync

---

## Repo Structure

```
dead-internet-observatory/
├── minions/          # Six data collection bots
├── pipeline/         # Bronze → Silver → Gold ETL
├── detection/        # Statistical AI-content detection engine
├── analytics/        # Aliveness Index + anomaly detection
├── app/              # Streamlit dashboard
├── scripts/          # HuggingFace push helper
├── config/           # config.yaml — all tuneable parameters
└── .github/workflows/ # Seven autonomous GitHub Actions workflows
```

---

## Contributing

This is a living project and there's a lot of room to make the detection engine sharper.

A few things that would move the needle:
- **Better signal weights** — calibrated against labeled human/AI datasets (HC3, RAID, M4 on HF)
- **New minions** — Internet Archive bulk data, academic paper feeds, forum scrapers
- **KenLM perplexity scoring** — train a 3-gram model on pre-2022 Common Crawl as a baseline; score new text against it
- **Dashboard improvements** — domain-level leaderboard, country-level breakdowns, embed widget

Open an issue or send a PR. The more data sources and signals we add, the better the index gets.

---

## License

MIT. Do whatever you want with it.
