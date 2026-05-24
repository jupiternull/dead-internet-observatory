# Dead Internet Observatory

**Tracking and quantifying the synthetic takeover of the public internet**

The [Dead Internet Theory](https://en.wikipedia.org/wiki/Dead_Internet_theory) holds that the internet is increasingly populated by bots, AI-generated content, and automated engagement farms rather than real humans.

This project is a living observatory that computes an **Internet Aliveness Index (IAI)** — a 0–100 score measuring how much of the sampled public internet still looks authentically human. Bots pull from 14 data sources, run a battery of statistical detection signals on every document, and aggregate everything into a daily index served through a Streamlit dashboard.

No LLMs, no paid APIs, no subscriptions. 100% open source and free to run.

---

## The Dashboard

> Live at: **[deadinternetobservatory.streamlit.app](https://deadinternetobservatory.streamlit.app/)**

- IAI gauge with live score and delta vs. 2019 baseline
- Platform health bars across all active sources
- Methodology and signal weights

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

All signals are statistical. No external model calls, no GPU required. Everything runs on standard CPU in Python.

---

## Architecture

```
+---------------------------------------------------------------------------+
|                             DATA MINIONS (14 active)                      |
|  Common Crawl · Reddit · News RSS · Wikipedia · HackerNews · Wayback      |
|  Bluesky · 4chan · Steam · YouTube · LinkedIn · Stack Overflow             |
|  Mastodon · GitHub Content                                                 |
+-----------------------------------+---------------------------------------+
                                    | raw JSONL -> GitHub Artifacts
                                    v
+---------------------------------------------------------------------------+
|                               PIPELINE                                    |
|  Bronze (raw JSONL) -> Silver (normalised Parquet)                        |
|                     -> Gold (scored Parquet, 3000 docs/run cap)           |
|                     -> Supabase sync (daily index aggregates)             |
+-------------+-----------------------------+-----------------------------+
              | Parquet -> HF Datasets      | Supabase (source of truth)
              v                             v
+--------------------+         +-------------------------------------+
|  HuggingFace       |         |  Streamlit Community Cloud          |
|  Datasets          |         |  (reads Supabase REST API)          |
|  (public, free)    |         |                                     |
+--------------------+         +-------------------------------------+
```

GitHub Actions runs the full cycle autonomously. No server required.

```
01:00 UTC daily   ->  Wayback Machine
02:00 UTC daily   ->  Common Crawl (5 CC dates/run, backfill)
10:00 UTC daily   ->  Common Crawl (repeat)
18:00 UTC daily   ->  Common Crawl (repeat)
03:00 UTC daily   ->  Reddit
03:30 UTC daily   ->  News Crawler
04:00 UTC daily   ->  Wikipedia + HackerNews
05:00 UTC daily   ->  Daily Full Sweep (fast minions)
06:00 UTC daily   ->  Pipeline (bronze->silver->gold->Supabase)
06:00 UTC daily   ->  Stack Overflow
06:30 UTC daily   ->  Bluesky
07:00 UTC daily   ->  4chan + Steam
07:00 UTC daily   ->  Mastodon
08:00 UTC daily   ->  YouTube
08:30 UTC daily   ->  LinkedIn
10:00 UTC daily   ->  GitHub Content
18:00 UTC daily   ->  Stack Overflow (repeat)
18:00 UTC daily   ->  GitHub Content (repeat)
20:00 UTC daily   ->  YouTube (repeat)
21:00 UTC daily   ->  Mastodon (repeat)
```

---

## Data Sources

| Minion | Source | What we collect | Status |
|---|---|---|---|
| `common_crawl_bot` | [Common Crawl](https://commoncrawl.org) | Quarterly WET snapshots, raw web text, multi-year backfill | ✓ |
| `reddit_bot` | Reddit public JSON API | Posts + comment trees from 10 subreddits | ✓ |
| `news_crawler_bot` | 8 RSS feeds | Full article text from major news outlets | ✓ |
| `wikipedia_bot` | Wikipedia API | 500 random articles + 24h edit pattern tracking | ✓ |
| `hackernews_bot` | [Algolia HN API](https://hn.algolia.com/api) | Stories + comments, last 7 days | ✓ |
| `wayback_bot` | [Wayback Machine CDX API](https://web.archive.org/cdx) | 14 sentinel URLs × 5 years, longitudinal signal | ✓ |
| `bluesky_bot` | Bluesky public API | Posts across 5 topic search terms | ✓ |
| `fourchan_bot` | 4chan JSON API | Posts from 8 boards | ✓ |
| `steam_bot` | Steam Reviews API | Reviews across 16 popular games | ✓ |
| `youtube_bot` | YouTube Data API v3 | Comments across 5 topic searches, 3× daily | ✓ |
| `linkedin_bot` | LinkedIn public feeds | Article text | ✓ |
| `stackoverflow_bot` | StackExchange API v2.3 | Q&A across 10 tags | ✓ |
| `mastodon_bot` | 5 Mastodon instances | Public timelines + 5 tag streams | ✓ |
| `github_bot` | GitHub REST API | README + top issues from trending repos | ✓ |
| `twitter_bot` | Twitter/X | — | ✗ Cloudflare-blocked |
| `substack_bot` | Substack | — | ✗ Cloudflare-blocked on GH Actions IPs |

**Where data lives:**
- Raw JSONL (bronze) → GitHub Artifacts, 7-day retention
- Processed Parquet (silver/gold) → [HuggingFace Datasets](https://huggingface.co/datasets/jupiternull/dead-internet-observatory)
- Daily index aggregates → Supabase (`composite_index`, `daily_index`, `meta`)
- Scored doc registry → Supabase `doc_registry` (dedup, all-time count)

---

## Quick Start

```bash
git clone https://github.com/jupiternull/dead-internet-observatory
cd dead-internet-observatory
pip install -r requirements.txt
streamlit run app/app.py
```

The dashboard reads from Supabase by default. To run data collection locally, see below.

---

## Running Real Data Collection

```bash
# Individual minions
python3 -m minions.reddit_bot
python3 -m minions.news_crawler_bot
python3 -m minions.wikipedia_bot
python3 -m minions.hackernews_bot
python3 -m minions.bluesky_bot
python3 -m minions.fourchan_bot
python3 -m minions.steam_bot
python3 -m minions.youtube_bot
python3 -m minions.linkedin_bot
python3 -m minions.stackoverflow_bot
python3 -m minions.mastodon_bot
python3 -m minions.github_bot
python3 -m minions.wayback_bot        # slow — fetches 5 years of snapshots
python3 -m minions.common_crawl_bot   # heavy — downloads WET files

# Run the full pipeline after harvesting
python3 -m pipeline.silver_processing
```

---

## Deployment

**Streamlit Community Cloud (free):**
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → point to `app/app.py`
3. No secrets needed — dashboard reads Supabase via public REST API
4. Done — auto-deploys on every push

**Activating autonomous data collection (GitHub Actions):**
1. Push to GitHub — cron schedules activate automatically
2. Add `DATABASE_URL` secret in repo Settings → Secrets (Supabase connection string)
3. *(Optional)* Add `HF_TOKEN` for HuggingFace dataset sync
4. *(Optional)* Add `YOUTUBE_API_KEY`, `STACKOVERFLOW_API_KEY`, `GITHUB_TOKEN` for those minions

---

## Repo Structure

```
dead-internet-observatory/
├── minions/           # 14 data collection bots
├── pipeline/          # Bronze → Silver → Gold ETL + Supabase sync
├── detection/         # Statistical AI-content detection engine
├── analytics/         # Aliveness Index computation + anomaly detection
├── app/               # Streamlit dashboard
├── scripts/           # HuggingFace push, Supabase migration helpers
├── config/            # config.yaml — all tuneable parameters
└── .github/workflows/ # Autonomous GitHub Actions workflows
```

---

## Contributing

This is a living project. A few things that would move the needle:

- **Better signal weights** — calibrate against labeled human/AI datasets (HC3, RAID, M4 on HF)
- **New minions** — Internet Archive bulk data, academic paper feeds, forum scrapers
- **KenLM perplexity scoring** — train a 3-gram model on pre-2022 Common Crawl; score new text against it
- **Anomaly alerts** — webhook or email when the IAI drops more than X points in a day

Open an issue or send a PR.

---

## License

MIT. Do whatever you want with it.
