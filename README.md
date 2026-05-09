# ☠ Dead Internet Observatory

**Tracking and quantifying the synthetic takeover of the public internet**

The [Dead Internet Theory](https://en.wikipedia.org/wiki/Dead_Internet_theory) says the internet is increasingly populated by bots, AI-generated content, and automated engagement farms rather than real humans.

This project is a living observatory that computes an **Internet Aliveness Index (IAI)** — a 0–100 score measuring how much of the sampled public internet still looks authentically human. Fourteen autonomous data minions harvest content continuously across the web, social platforms, forums, code repositories, Q&A sites, and the historical web. A statistical detection engine scores every document and aggregates everything into a daily index served through a research-grade Streamlit dashboard.

No LLMs, no paid APIs, no subscriptions. 100% open source and free to run.

---

## The Dashboard

> Live at: **[dead-internet-observatory-8ryn4twqetgroo5xvrivm5.streamlit.app](https://dead-internet-observatory-8ryn4twqetgroo5xvrivm5.streamlit.app/)**

- Animated IAI gauge with live score and delta vs. 2019 baseline
- Full historical timeline with anomaly markers and decay shading
- Per-platform aliveness health bars across all 14 active sources
- Detection signal radar chart (7 NLP signals vs. 2019 baseline)
- Anomaly spotlight: statistically significant spikes and crashes
- **What-if Simulator** — project future aliveness under different AI acceleration and regulatory scenarios

---

## How the Index Works

The IAI is a weighted composite of eight linguistic and behavioural signals:

| Signal | Weight | What it's catching |
|---|---|---|
| **Type-Token Ratio** | 18% | Narrow vocabulary is a classic AI tell |
| **Shannon Entropy** | 15% | AI text has lower information density |
| **Sentence Length Variance** | 15% | AI writes unnaturally uniform sentences |
| **Bigram Repetition** | 15% | Stock phrase reuse, templated content |
| **Temporal Burstiness** | 15% | Bots post on schedules; humans don't |
| **MTLD** | 12% | Length-independent lexical diversity |
| **Zipf Law Alignment** | 10% | Natural language follows a power law; AI text deviates |
| **GPT-2 Perplexity** | 15% | Low perplexity under distilgpt2 signals predictable, AI-like generation |

The seven classical NLP signals are scaled to 85% of the composite; perplexity fills the remaining 15%. All scoring runs on CPU — no GPU required.

---

## Architecture

```
+------------------------------------------------------------------------+
|                          DATA MINIONS  (14 active)                     |
|  Common Crawl · Reddit · News RSS · Wikipedia · HackerNews             |
|  Wayback Machine · Bluesky · 4chan · Steam · YouTube                   |
|  LinkedIn · Stack Overflow · Mastodon · GitHub Content                 |
+-----------------------------------+------------------------------------+
                                    | raw JSONL → GitHub Artifacts
                                    v
+------------------------------------------------------------------------+
|                             PIPELINE                                   |
|  Bronze  (raw JSONL)        — normalised, deduplicated                 |
|  Silver  (Parquet)          — schema-unified across all sources        |
|  Gold    (scored Parquet)   — incremental, new docs only               |
|  SQLite  (observatory.db)   — daily index aggregates                   |
+--------------------+--------------------------------------------------+
                     | observatory.db → committed to repo
                     v
         +----------------------------------+
         |  Streamlit Community Cloud       |
         |  (research dashboard, live)      |
         +----------------------------------+
```

GitHub Actions orchestrates the full cycle autonomously. No server required. The pipeline workflow triggers on every minion completion (`workflow_run`) with a concurrency lock to prevent race conditions.

```
01:00 UTC daily          →  Wayback Machine Minion
02:00 / 10:00 / 18:00   →  Common Crawl Minion (5 CC dates per run)
03:00 UTC daily          →  Reddit Minion
03:30 UTC daily          →  News Crawler Minion
04:00 UTC daily          →  Wikipedia + HackerNews Minions
05:00 UTC daily          →  Daily Full Sweep (fast minions)
06:00 / 18:00 UTC        →  Stack Overflow Minion
06:30 UTC daily          →  Bluesky Minion
07:00 UTC daily          →  4chan + Steam Minions
07:00 / 13:00 / 21:00   →  Mastodon Minion
08:00 / 14:00 / 20:00   →  YouTube Minion
08:30 UTC daily          →  LinkedIn Minion
10:00 / 22:00 UTC        →  GitHub Content Minion
─────────────────────────────────────────────────
event-driven (+ 06:00 catch-all) → Pipeline + Index Update
```

---

## Data Sources

| Minion | Source | What we collect |
|---|---|---|
| `common_crawl_bot` | [Common Crawl](https://commoncrawl.org) | Quarterly WET snapshots 2012–present, raw web text |
| `reddit_bot` | Reddit public JSON API | Posts + comment trees from 10 subreddits |
| `news_crawler_bot` | RSS feeds | Full article text from major news outlets |
| `wikipedia_bot` | Wikipedia API | Random articles + 24h edit pattern tracking |
| `hackernews_bot` | [Algolia HN API](https://hn.algolia.com/api) | Stories + comments, last 7 days |
| `wayback_bot` | [Wayback Machine CDX API](https://web.archive.org/cdx) | 14 sentinel URLs × 5 years — the longitudinal signal |
| `bluesky_bot` | Bluesky public firehose | Recent posts via the AT Protocol public API |
| `fourchan_bot` | 4chan public API | Threads and replies from /g/, /sci/, /pol/, /biz/ |
| `steam_bot` | Steam Review API | User-written game reviews across top titles |
| `youtube_bot` | YouTube Data API v3 | Video metadata and descriptions, 3× daily |
| `linkedin_bot` | LinkedIn public posts | Professional network public content |
| `stackoverflow_bot` | [StackExchange API v2.3](https://api.stackexchange.com) | Questions + top answers across 10 technical tags |
| `mastodon_bot` | Mastodon public API | Public timelines from 5 federated instances |
| `github_bot` | GitHub REST API | Repository READMEs + open issues from trending repos |

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

# Run a minion to collect real data
python3 -m minions.reddit_bot
python3 -m minions.hackernews_bot

# Process and score collected data
python3 -m pipeline.bronze_ingestion
python3 -m pipeline.silver_processing

# Launch the dashboard
streamlit run app/app.py
```

---

## Running Individual Minions

Most minions need no API keys — all public endpoints.

```bash
python3 -m minions.reddit_bot
python3 -m minions.hackernews_bot
python3 -m minions.wikipedia_bot
python3 -m minions.news_crawler_bot
python3 -m minions.bluesky_bot
python3 -m minions.fourchan_bot
python3 -m minions.steam_bot
python3 -m minions.stackoverflow_bot     # STACKOVERFLOW_API_KEY optional (10K req/day vs 300)
python3 -m minions.mastodon_bot
python3 -m minions.github_bot            # uses GITHUB_TOKEN automatically in Actions
python3 -m minions.youtube_bot           # requires YOUTUBE_API_KEY
python3 -m minions.wayback_bot           # slow — fetches multi-year snapshots
python3 -m minions.commoncrawl_bot       # heavy — downloads multi-GB WET files
```

---

## Deployment

**Streamlit Community Cloud (free):**
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → point to `app/app.py`
3. Done — auto-deploys on every push that updates `data/observatory.db`

**Activating autonomous data collection (GitHub Actions):**
1. Push to GitHub — all cron schedules activate automatically
2. Add a `YOUTUBE_API_KEY` secret in repo Settings → Secrets to enable the YouTube minion
3. *(Optional)* Add a `HF_TOKEN` secret for HuggingFace dataset sync
4. *(Optional)* Add a `STACKOVERFLOW_API_KEY` secret for higher StackExchange API rate limits

---

## Repo Structure

```
dead-internet-observatory/
├── minions/            # 14 data collection bots
├── pipeline/           # Bronze → Silver → Gold ETL
├── detection/          # Statistical AI-content detection engine + perplexity scorer
├── analytics/          # Aliveness Index computation + anomaly detection
├── app/                # Streamlit dashboard
├── scripts/            # HuggingFace push helper
├── config/             # config.yaml + Common Crawl progress tracker
└── .github/workflows/  # 18 autonomous GitHub Actions workflows
```

---

## Contributing

A few things that would move the needle:

- **Better signal weights** — calibrate against labeled human/AI datasets (HC3, RAID, M4 on HuggingFace)
- **New minions** — academic paper feeds (arXiv), Telegram public channels, Discord public servers
- **Domain-level leaderboard** — per-domain IAI breakdown surfaced in the dashboard
- **Anomaly alerts** — webhook or email notification when the IAI drops more than X points in 24h
- **Embeddable widget** — single-number badge for external sites to display the current IAI

Open an issue or send a PR. The more data sources and signals, the sharper the index.

---

## License

MIT. Do whatever you want with it.
