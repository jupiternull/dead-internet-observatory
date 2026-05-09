"""
One-time migration: slim down the Supabase documents table.

What this does:
  1. Creates a lightweight doc_registry table (doc_id only)
  2. Migrates all existing doc_ids from documents → doc_registry
  3. Saves the current document count to a meta table
  4. Truncates the documents table (removes all text/feature data)
  5. Runs VACUUM ANALYZE to reclaim storage

Run once from your local machine or any machine with DATABASE_URL set:
  DATABASE_URL=<your-supabase-url> python3 scripts/migrate_supabase_slim.py

After this runs, supabase_sync.py will write only to doc_registry going forward.
The dashboard document count and dedup both continue working normally.
"""

import os
import sys

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable not set.")
    sys.exit(1)


def run():
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=30)
    conn.autocommit = False
    cur = conn.cursor()

    print("── Step 1: Create doc_registry table ───────────────────────────────")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS doc_registry (
            doc_id     TEXT PRIMARY KEY,
            scored_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    conn.commit()
    print("  ✓ doc_registry ready")

    print("── Step 2: Create meta table ────────────────────────────────────────")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()
    print("  ✓ meta ready")

    print("── Step 3: Check documents table ────────────────────────────────────")
    cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'documents')")
    documents_exists = cur.fetchone()[0]

    if not documents_exists:
        print("  ! documents table does not exist — nothing to migrate")
        # Still seed the counter from doc_registry if it has data
        cur.execute("SELECT COUNT(*) FROM doc_registry")
        count = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO meta (key, value) VALUES ('total_scored_count', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """, (str(count),))
        conn.commit()
        print(f"  ✓ meta.total_scored_count set to {count:,}")
        return

    print("── Step 4: Get current document count ───────────────────────────────")
    cur.execute("SELECT COUNT(*) FROM documents")
    total = cur.fetchone()[0]
    print(f"  {total:,} documents found")

    print("── Step 5: Migrate doc_ids → doc_registry ───────────────────────────")
    cur.execute("""
        INSERT INTO doc_registry (doc_id, scored_at)
        SELECT doc_id, COALESCE(scored_at, ingested_at, NOW())
        FROM documents
        WHERE doc_id IS NOT NULL
        ON CONFLICT (doc_id) DO NOTHING
    """)
    migrated = cur.rowcount
    conn.commit()
    print(f"  ✓ {migrated:,} doc_ids migrated")

    print("── Step 6: Save total count to meta ─────────────────────────────────")
    # Count from doc_registry (authoritative after migration)
    cur.execute("SELECT COUNT(*) FROM doc_registry")
    registry_count = cur.fetchone()[0]
    cur.execute("""
        INSERT INTO meta (key, value) VALUES ('total_scored_count', %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
    """, (str(registry_count),))
    conn.commit()
    print(f"  ✓ meta.total_scored_count = {registry_count:,}")

    print("── Step 7: Truncate documents table ─────────────────────────────────")
    cur.execute("TRUNCATE TABLE documents")
    conn.commit()
    print("  ✓ documents table truncated (schema preserved, data cleared)")

    print("── Step 8: VACUUM ANALYZE ───────────────────────────────────────────")
    conn.autocommit = True
    cur.execute("VACUUM ANALYZE documents")
    cur.execute("VACUUM ANALYZE doc_registry")
    print("  ✓ VACUUM complete — Supabase will reclaim storage shortly")

    cur.close()
    conn.close()
    print("\n✓ Migration complete. Storage will update in Supabase within ~1 hour.")
    print("  Dashboard count and dedup continue working normally.")


if __name__ == "__main__":
    run()
