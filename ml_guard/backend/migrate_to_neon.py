"""
Data Migration Script: SQLite → Neon PostgreSQL.

This script extracts all records from a local SQLite database
and inserts them into a Neon PostgreSQL database, preserving
foreign key relationships and scan history.

Usage:
    python migrate_to_neon.py

Environment variables required:
    DATABASE_URL  — Neon PostgreSQL connection string
                    (e.g. postgresql://user:pass@ep-xxxxx.neon.tech/neondb?sslmode=require)

The script auto-detects the local SQLite file at ./ml_guard.db
"""
import os
import sys
import sqlite3
import json

# Ensure app modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker


# ─── Configuration ──────────────────────────────────────────────────────────
SQLITE_PATH = os.path.join(os.path.dirname(__file__), "ml_guard.db")

# The Neon URL must be set in the environment
NEON_URL = os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")

# Tables in dependency order (parents before children)
TABLE_ORDER = [
    "organizations",
    "users",
    "projects",
    "api_keys",
    "models",
    "datasets",
    "nlp_intents",
    "jobs",
    "policy_versions",
    "policy_rules",
    "scan_records",
    "audit_logs",
    "alert_rules",
    "alert_events",
    "preflight_results",
    "drift_results",
    "performance_results",
    "fairness_results",
    "llm_results",
    "governance_results",
    "llm_scan_records",
    "stream_drift_records",
    "ci_integrations",
]


def get_sqlite_tables(sqlite_conn) -> list:
    """Get list of tables that exist in the SQLite database."""
    cursor = sqlite_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return [row[0] for row in cursor.fetchall()]


def get_table_columns(sqlite_conn, table_name: str) -> list:
    """Get column names for a SQLite table."""
    cursor = sqlite_conn.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]


def get_postgres_columns(pg_engine, table_name: str) -> list:
    """Get column names for a PostgreSQL table."""
    inspector = inspect(pg_engine)
    try:
        columns = inspector.get_columns(table_name)
        return [col["name"] for col in columns]
    except Exception:
        return []


def migrate_table(sqlite_conn, pg_engine, table_name: str, pg_session):
    """Migrate a single table from SQLite to PostgreSQL."""
    sqlite_cols = get_table_columns(sqlite_conn, table_name)
    pg_cols = get_postgres_columns(pg_engine, table_name)

    if not pg_cols:
        print(f"  ⚠ Table '{table_name}' does not exist in PostgreSQL. Skipping.")
        return 0

    # Only migrate columns that exist in both databases
    common_cols = [c for c in sqlite_cols if c in pg_cols]
    if not common_cols:
        print(f"  ⚠ No common columns for '{table_name}'. Skipping.")
        return 0

    # Read all rows from SQLite
    col_list = ", ".join(common_cols)
    cursor = sqlite_conn.execute(f"SELECT {col_list} FROM {table_name}")
    rows = cursor.fetchall()

    if not rows:
        print(f"  ○ Table '{table_name}' is empty. Skipping.")
        return 0

    # Insert into PostgreSQL
    placeholders = ", ".join([f":{c}" for c in common_cols])
    insert_sql = f"INSERT INTO {table_name} ({col_list}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

    inserted = 0
    for row in rows:
        row_dict = {}
        for i, col in enumerate(common_cols):
            val = row[i]
            # Handle JSON strings: if a column stores JSON in SQLite,
            # parse it so PostgreSQL JSONB receives the proper type
            if isinstance(val, str) and col.endswith("_json") or col in (
                "settings", "scopes", "config", "condition", "channels",
                "details", "rules_json", "checks_run", "results_json",
                "computed_metrics_json", "severity_counts",
                "top_drifted_features", "fairness_metrics",
                "parsed_constraints", "metadata_json", "complexity",
            ):
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
            row_dict[col] = val

        try:
            pg_session.execute(text(insert_sql), row_dict)
            inserted += 1
        except Exception as e:
            print(f"  ⚠ Row insert failed in '{table_name}': {e}")
            pg_session.rollback()
            continue

    pg_session.commit()
    return inserted


def main():
    print("=" * 60)
    print("ML Guard — SQLite → Neon PostgreSQL Migration")
    print("=" * 60)

    # Validate SQLite
    if not os.path.exists(SQLITE_PATH):
        print(f"\n✗ SQLite database not found at: {SQLITE_PATH}")
        print("  Nothing to migrate. Exiting.")
        return

    # Validate Neon URL
    if not NEON_URL:
        print("\n✗ DATABASE_URL environment variable is not set.")
        print("  Set DATABASE_URL to your Neon PostgreSQL connection string.")
        print("  Example: postgresql://user:pass@ep-xxxx.neon.tech/neondb?sslmode=require")
        return

    neon_url = NEON_URL
    if neon_url.startswith("postgres://"):
        neon_url = neon_url.replace("postgres://", "postgresql://", 1)

    print(f"\n📂 Source: {SQLITE_PATH}")
    print(f"🌐 Target: {neon_url[:50]}...")

    # Connect to SQLite
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    existing_tables = get_sqlite_tables(sqlite_conn)
    print(f"\n📋 Found {len(existing_tables)} tables in SQLite: {', '.join(existing_tables)}")

    # Connect to Neon PostgreSQL
    pg_engine = create_engine(
        neon_url,
        pool_pre_ping=True,
        pool_size=5,
        connect_args={"sslmode": "require"} if "neon.tech" in neon_url else {},
    )

    # Create all tables in PostgreSQL first
    print("\n🔨 Ensuring PostgreSQL schema exists...")
    from app.db.session import Base
    from app.db import models  # noqa: F401 — import all models
    Base.metadata.create_all(bind=pg_engine)
    print("  ✓ Schema created / verified.")

    # Migrate tables in dependency order
    PgSession = sessionmaker(bind=pg_engine)
    pg_session = PgSession()

    total_migrated = 0
    print("\n📦 Migrating data...\n")

    for table_name in TABLE_ORDER:
        if table_name not in existing_tables:
            continue
        print(f"  → {table_name}...", end=" ")
        count = migrate_table(sqlite_conn, pg_engine, table_name, pg_session)
        print(f"✓ {count} rows")
        total_migrated += count

    # Also migrate any tables not in our predefined order
    for table_name in existing_tables:
        if table_name not in TABLE_ORDER:
            print(f"  → {table_name} (extra)...", end=" ")
            count = migrate_table(sqlite_conn, pg_engine, table_name, pg_session)
            print(f"✓ {count} rows")
            total_migrated += count

    pg_session.close()
    sqlite_conn.close()

    print(f"\n{'=' * 60}")
    print(f"✅ Migration complete! {total_migrated} total rows migrated.")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print("  1. Set DATABASE_URL in your .env file")
    print("  2. Restart the backend server")
    print("  3. Verify data at /health/database")


if __name__ == "__main__":
    main()
