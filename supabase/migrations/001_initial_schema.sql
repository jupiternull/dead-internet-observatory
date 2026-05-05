-- Dead Internet Observatory — Initial Schema
-- Run this in the Supabase SQL editor:
-- https://supabase.com/dashboard/project/qwuabrmpudlqngfcxezh/sql/new

-- ── Core document store ──────────────────────────────────────────────────────
-- Every scored document from every minion lives here permanently.
-- This is the primary aggregate — text capped at 10k chars to manage storage.

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT        PRIMARY KEY,
    source          TEXT        NOT NULL,
    category        TEXT,
    domain          TEXT,
    url             TEXT,
    title           TEXT,
    text            TEXT,
    text_length     INTEGER,
    author          TEXT,
    created_dt      TIMESTAMPTZ,
    crawl_partition TEXT,
    ingested_at     TIMESTAMPTZ,
    content_hash    TEXT,
    aliveness_score REAL,
    scored_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_docs_source      ON documents(source);
CREATE INDEX IF NOT EXISTS idx_docs_created_dt  ON documents(created_dt);
CREATE INDEX IF NOT EXISTS idx_docs_aliveness   ON documents(aliveness_score);
CREATE INDEX IF NOT EXISTS idx_docs_domain      ON documents(domain);
CREATE INDEX IF NOT EXISTS idx_docs_source_date ON documents(source, created_dt);

-- ── Daily aggregated index ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS daily_index (
    date            DATE    NOT NULL,
    source          TEXT    NOT NULL,
    category        TEXT    NOT NULL,
    n_docs          INTEGER,
    mean_score      REAL,
    median_score    REAL,
    std_score       REAL,
    pct_below_50    REAL,
    bot_fraction    REAL,
    aliveness_index REAL,
    PRIMARY KEY (date, source, category)
);

CREATE INDEX IF NOT EXISTS idx_daily_date   ON daily_index(date);
CREATE INDEX IF NOT EXISTS idx_daily_source ON daily_index(source);

-- ── Composite IAI time series ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS composite_index (
    date            DATE    PRIMARY KEY,
    aliveness_index REAL,
    smoothed_index  REAL,
    n_docs          INTEGER,
    anomaly_flag    INTEGER DEFAULT 0,
    anomaly_reason  TEXT,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Domain-level scores ──────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS domain_scores (
    date        DATE    NOT NULL,
    domain      TEXT    NOT NULL,
    category    TEXT,
    mean_score  REAL,
    n_docs      INTEGER,
    PRIMARY KEY (date, domain)
);

CREATE INDEX IF NOT EXISTS idx_domain_scores_domain ON domain_scores(domain);

-- ── Meta key-value store ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS meta (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Convenience views ────────────────────────────────────────────────────────

-- Latest IAI score per source
CREATE OR REPLACE VIEW latest_source_scores AS
SELECT DISTINCT ON (source)
    source, category, date, mean_score, n_docs
FROM daily_index
ORDER BY source, date DESC;

-- Weekly aggregate (useful for long-horizon charts)
CREATE OR REPLACE VIEW weekly_index AS
SELECT
    DATE_TRUNC('week', date::TIMESTAMPTZ)::DATE AS week,
    AVG(aliveness_index)                         AS mean_iai,
    AVG(smoothed_index)                          AS smoothed_iai,
    SUM(n_docs)                                  AS total_docs
FROM composite_index
GROUP BY 1
ORDER BY 1;

-- Source aliveness by year (for historical baseline comparison)
CREATE OR REPLACE VIEW yearly_source_scores AS
SELECT
    EXTRACT(YEAR FROM date)::INTEGER AS year,
    source,
    AVG(mean_score)                  AS mean_aliveness,
    SUM(n_docs)                      AS total_docs
FROM daily_index
GROUP BY 1, 2
ORDER BY 1, 2;
