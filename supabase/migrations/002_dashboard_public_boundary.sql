-- Dashboard reliability/public boundary hardening.
-- Forward-only and conservative: create missing metadata table, replace
-- aggregate views, enable RLS on aggregate tables, and grant read access only
-- to the public dashboard surface.

BEGIN;

CREATE TABLE IF NOT EXISTS public.doc_registry (
    doc_id     TEXT PRIMARY KEY,
    scored_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE VIEW public.latest_source_scores AS
WITH latest_dates AS (
    SELECT
        source,
        MAX(date) AS date
    FROM public.daily_index
    WHERE mean_score IS NOT NULL
    GROUP BY source
),
latest_rows AS (
    SELECT d.source, d.date, d.mean_score, COALESCE(d.n_docs, 0) AS n_docs
    FROM public.daily_index d
    JOIN latest_dates ld
      ON ld.source = d.source
     AND ld.date = d.date
    WHERE d.mean_score IS NOT NULL
)
SELECT
    source,
    'all'::TEXT AS category,
    date,
    (CASE
        WHEN SUM(n_docs) > 0 THEN SUM(mean_score * n_docs)::REAL / SUM(n_docs)
        ELSE AVG(mean_score)::REAL
    END)::REAL AS mean_score,
    SUM(n_docs)::INTEGER AS n_docs
FROM latest_rows
GROUP BY source, date
ORDER BY source;

ALTER VIEW public.latest_source_scores SET (security_invoker = true);
ALTER VIEW public.weekly_index SET (security_invoker = true);
ALTER VIEW public.yearly_source_scores SET (security_invoker = true);

ALTER TABLE public.composite_index ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.daily_index ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.domain_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.meta ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.doc_registry ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'composite_index'
          AND policyname = 'dashboard_read_composite_index'
    ) THEN
        CREATE POLICY dashboard_read_composite_index
        ON public.composite_index
        FOR SELECT
        TO anon, authenticated
        USING (true);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'daily_index'
          AND policyname = 'dashboard_read_daily_index'
    ) THEN
        CREATE POLICY dashboard_read_daily_index
        ON public.daily_index
        FOR SELECT
        TO anon, authenticated
        USING (true);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'domain_scores'
          AND policyname = 'dashboard_read_domain_scores'
    ) THEN
        CREATE POLICY dashboard_read_domain_scores
        ON public.domain_scores
        FOR SELECT
        TO anon, authenticated
        USING (true);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'meta'
          AND policyname = 'dashboard_read_meta'
    ) THEN
        CREATE POLICY dashboard_read_meta
        ON public.meta
        FOR SELECT
        TO anon, authenticated
        USING (key IN ('total_scored_count', 'last_scored_at', 'last_updated_at'));
    END IF;

END $$;

REVOKE ALL ON TABLE public.documents FROM anon, authenticated;
REVOKE ALL ON TABLE public.doc_registry FROM anon, authenticated;
REVOKE ALL ON TABLE public.composite_index FROM anon, authenticated;
REVOKE ALL ON TABLE public.daily_index FROM anon, authenticated;
REVOKE ALL ON TABLE public.domain_scores FROM anon, authenticated;
REVOKE ALL ON TABLE public.meta FROM anon, authenticated;

GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT SELECT ON TABLE public.composite_index TO anon, authenticated;
GRANT SELECT ON TABLE public.daily_index TO anon, authenticated;
GRANT SELECT ON TABLE public.domain_scores TO anon, authenticated;
GRANT SELECT ON TABLE public.meta TO anon, authenticated;
GRANT SELECT ON TABLE public.latest_source_scores TO anon, authenticated;
GRANT SELECT ON TABLE public.weekly_index TO anon, authenticated;
GRANT SELECT ON TABLE public.yearly_source_scores TO anon, authenticated;

COMMIT;
