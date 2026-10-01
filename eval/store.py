"""Writes one evaluation summary row into Postgres `eval_runs` (Grafana's Model performance).

    uv run python -m eval.run_eval store     # or: make eval-store (uses .env)

The row comes from eval/results/routing.json, plus the answer-quality score from
quality.json when that run used the same dataset and threshold (never a mock run).
The DSN is --database-url, else DATABASE_URL, else built from POSTGRES_USER,
POSTGRES_PASSWORD and POSTGRES_DB on 127.0.0.1:5432; it is never printed.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import asyncpg

COLUMNS = (
    "threshold",
    "dataset_size",
    "router_accuracy",
    "under_routed_pct",
    "over_routed_pct",
    "local_share",
    "savings_pct",
    "quality_score",
    "notes",
)
INSERT = (
    f"INSERT INTO eval_runs ({', '.join(COLUMNS)}) "
    f"VALUES ({', '.join(f'${i}' for i in range(1, len(COLUMNS) + 1))}) RETURNING id, created_at"
)


class StoreError(Exception):
    """The row could not be built or written."""


def matching_quality(
    routing: dict[str, Any], quality: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The quality run that belongs with this routing run, if there is one."""
    if quality is None or quality.get("mock"):
        return None
    same_data = quality["dataset"]["sha256"] == routing["dataset"]["sha256"]
    if not same_data or quality.get("threshold") != routing["threshold"]:
        return None
    return quality


def eval_run_row(routing: dict[str, Any], quality: dict[str, Any] | None) -> dict[str, Any]:
    """The `eval_runs` values for one run. Percentages are 0-100, as the schema says."""
    m = routing["metrics"]
    q = matching_quality(routing, quality)
    quality_score = q["summary"]["quality_score_pct"] if q else None
    notes = (
        f"routing dry run of {m['n']} labelled prompts (no model called); "
        f"version {routing['version']}; dataset sha256 {routing['dataset']['sha256'][:12]}; "
        f"under-routed % is of all prompts (miss rate {m['miss_rate_pct']}% of premium-labelled); "
        f"savings estimated on input tokens ({m['savings_basis']}); "
        + (
            f"quality from {q['summary']['graded']} model-graded answers ({q['generated_at'][:10]})"
            if q
            else "answer quality not measured yet"
        )
    )
    return {
        "threshold": routing["threshold"],
        "dataset_size": m["n"],
        "router_accuracy": m["accuracy_pct"],
        "under_routed_pct": m["under_pct"],
        "over_routed_pct": m["over_pct"],
        "local_share": m["local_share_pct"],
        "savings_pct": m["savings_pct"],
        "quality_score": quality_score,
        "notes": notes,
    }


def database_url(explicit: str | None, env: dict[str, str] | None = None) -> str:
    env = dict(os.environ) if env is None else env
    if explicit:
        return explicit
    if env.get("DATABASE_URL"):
        return env["DATABASE_URL"]
    user, password = env.get("POSTGRES_USER", "router"), env.get("POSTGRES_PASSWORD")
    if not password:
        raise StoreError("set DATABASE_URL, or POSTGRES_PASSWORD (as in .env), or --database-url")
    host = env.get("POSTGRES_HOST", "127.0.0.1")
    port = env.get("POSTGRES_PORT", "5432")
    db = env.get("POSTGRES_DB", "router")
    return f"postgresql://{quote(user)}:{quote(password)}@{host}:{port}/{quote(db)}"


async def insert(dsn: str, row: dict[str, Any]) -> tuple[int, Any]:
    try:
        conn = await asyncpg.connect(dsn, timeout=10)
    except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
        # Never echo the DSN: it carries the password.
        raise StoreError(
            f"cannot connect to Postgres ({type(exc).__name__}); is the stack up?"
        ) from exc
    try:
        record = await conn.fetchrow(INSERT, *(row[c] for c in COLUMNS))
    except asyncpg.PostgresError as exc:
        raise StoreError(f"insert into eval_runs failed: {exc}") from exc
    finally:
        await conn.close()
    assert record is not None
    return record["id"], record["created_at"]


def main(results: Path, explicit_url: str | None) -> int:
    try:
        routing = json.loads((results / "routing.json").read_text(encoding="utf-8"))
        quality_path = results / "quality.json"
        quality = json.loads(quality_path.read_text("utf-8")) if quality_path.exists() else None
        row = eval_run_row(routing, quality)
        run_id, created = asyncio.run(insert(database_url(explicit_url), row))
    except FileNotFoundError:
        print("error: no eval/results/routing.json; run `make eval` first", file=sys.stderr)
        return 2
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"eval_runs id {run_id} at {created:%Y-%m-%d %H:%M:%S}: threshold {row['threshold']}, "
        f"{row['dataset_size']} prompts, accuracy {row['router_accuracy']}%, under-routed "
        f"{row['under_routed_pct']}%, local share {row['local_share']}%, savings "
        f"{row['savings_pct']}% (est.), quality "
        + ("not measured" if row["quality_score"] is None else f"{row['quality_score']}%")
    )
    return 0
