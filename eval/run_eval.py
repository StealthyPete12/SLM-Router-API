"""Router evaluation: accuracy, under- and over-routing, savings, the threshold sweep.

Free commands (no model is called; the router runs in-process, exactly as
POST /v1/route would, so no stack is needed):

    uv run python -m eval.run_eval route            # evaluate the threshold in routing.yaml
    uv run python -m eval.run_eval sweep            # thresholds 20-45, plus the selection rule
    uv run python -m eval.run_eval report           # docs/results.md and the sweep chart
    uv run python -m eval.run_eval store            # one summary row into Postgres eval_runs
    uv run python -m eval.run_eval spot-check       # agreement of your hand grades with the model's

Paid command (calls the premium model through a running router; opt-in, capped):

    uv run python -m eval.run_eval quality --estimate-only
    uv run python -m eval.run_eval quality --confirm-paid --max-cost-usd 1.00

Artifacts go to eval/results/ (routing.json, sweep.json, quality.json). The
dataset is synthetic, so they are safe to commit. Metric definitions live in
eval/metrics.py; the Makefile's eval-* targets wrap these commands.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.aggregator.costs import price
from app.config import DEFAULT_CONFIG_DIR, AppConfig, load_config
from app.router.engine import Router
from app.router.policy import RoutingError
from eval.dataset import (
    PROMPTS,
    DatasetError,
    EvalPrompt,
    composition,
    dataset_sha256,
    load_dataset,
)
from eval.metrics import (
    MISS_RATE_TOLERANCE_PP,
    UNDER_ROUTING_WEIGHT,
    Outcome,
    RoutingMetrics,
    Selection,
    select_threshold,
    summarise,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
SWEEP_RANGE = (20, 45)


class EvalError(Exception):
    """The evaluation cannot run; the message says why."""


class RoutingEvaluator:
    """Routes dataset rows through the real Router with a chosen threshold.

    Rows with `privacy_mode: true` are routed with routing.yaml privacy_mode on;
    every other setting is the configured one, so only the threshold varies.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._routers: dict[tuple[int, bool], Router] = {}

    def _router(self, threshold: int, privacy_mode: bool) -> Router:
        key = (threshold, privacy_mode)
        if key not in self._routers:
            routing = self.config.routing.model_copy(
                update={"threshold": threshold, "privacy_mode": privacy_mode}
            )
            self._routers[key] = Router(self.config.model_copy(update={"routing": routing}))
        return self._routers[key]

    def route(self, prompt: EvalPrompt, threshold: int) -> Outcome:
        privacy = prompt.privacy_mode or self.config.routing.privacy_mode
        router = self._router(threshold, privacy)
        try:
            result = router.route(prompt.request(), prompt.forced_tier)
        except RoutingError as exc:
            raise EvalError(f"{prompt.id}: the router refused it ({exc.message})") from exc
        model = result.decision.model
        return Outcome(
            id=prompt.id,
            task=prompt.task,
            expected=prompt.expected_tier,
            routed=result.decision.tier,
            input_tokens=result.input_tokens,
            score=result.score.total,
            reason=result.decision.reason,
            classified_task=result.classification.task_type,
            tags=tuple(sorted(prompt.tags)),
            model_id=model.id,
            cost_usd=price(model, result.input_tokens),
            baseline_cost_usd=price(self.config.baseline_model, result.input_tokens),
        )

    def evaluate(self, prompts: list[EvalPrompt], threshold: int) -> list[Outcome]:
        return [self.route(p, threshold) for p in prompts]


# -------------------------------------------------------------------- provenance


def git_version(root: Path = ROOT) -> str:
    """Short commit hash, with -dirty when the working tree has changes; or 'unknown'."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(
    config: AppConfig, dataset_path: Path, config_dir: Path, now: datetime, version: str
) -> dict[str, Any]:
    baseline = config.baseline_model
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "version": version,
        "dataset": {"path": dataset_path.name, "sha256": dataset_sha256(dataset_path)},
        "config": {
            "routing_yaml_sha256": file_sha256(config_dir / "routing.yaml"),
            "configured_threshold": config.routing.threshold,
            "baseline_model": baseline.id,
            "baseline_priced": baseline.priced,
        },
    }


def outcome_row(o: Outcome) -> dict[str, Any]:
    return {
        "id": o.id,
        "task": o.task,
        "classified_task": o.classified_task,
        "expected": o.expected,
        "routed": o.routed,
        "error": o.error,
        "score": o.score,
        "reason": o.reason,
        "model_id": o.model_id,
        "input_tokens": o.input_tokens,
        "tags": list(o.tags),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# -------------------------------------------------------------------- runs


def run_routing(
    prompts: list[EvalPrompt], config: AppConfig, threshold: int, meta: dict[str, Any]
) -> dict[str, Any]:
    outcomes = RoutingEvaluator(config).evaluate(prompts, threshold)
    return {
        "kind": "routing_eval",
        **meta,
        "threshold": threshold,
        "composition": composition(prompts),
        "metrics": summarise(outcomes, threshold).as_dict(),
        "outcomes": [outcome_row(o) for o in outcomes],
    }


def run_sweep(
    prompts: list[EvalPrompt],
    config: AppConfig,
    thresholds: range,
    meta: dict[str, Any],
) -> tuple[dict[str, Any], list[RoutingMetrics], Selection]:
    evaluator = RoutingEvaluator(config)
    sweep: list[RoutingMetrics] = []
    decisions: dict[int, tuple[str, ...]] = {}
    for threshold in thresholds:
        outcomes = evaluator.evaluate(prompts, threshold)
        sweep.append(summarise(outcomes, threshold))
        decisions[threshold] = tuple(o.routed for o in outcomes)
    selection = select_threshold(sweep, decisions, current=config.routing.threshold)
    data = {
        "kind": "threshold_sweep",
        **meta,
        "rule": {
            "miss_rate_tolerance_pp": MISS_RATE_TOLERANCE_PP,
            "under_routing_weight": UNDER_ROUTING_WEIGHT,
        },
        "thresholds": [m.as_dict() for m in sweep],
        "selection": {
            "threshold": selection.threshold,
            "eligible": selection.eligible,
            "plateau": selection.plateau,
            "miss_rate_floor_pct": selection.miss_rate_floor,
            "miss_rate_ceiling_pct": selection.miss_rate_ceiling,
            "explanation": selection.explanation,
        },
    }
    return data, sweep, selection


# -------------------------------------------------------------------- printing


def _fmt(value: float | None, suffix: str = "%") -> str:
    return "n/a" if value is None else f"{value:.1f}{suffix}"


def print_routing(data: dict[str, Any], out: Callable[[str], None] = print) -> None:
    m = data["metrics"]
    savings_note = "input tokens" + (
        "" if m["savings_basis"] == "priced" else ", prices unset: token share"
    )
    out(f"Routing evaluation at threshold {m['threshold']} on {m['n']} prompts (no model called)")
    out(f"  accuracy        {_fmt(m['accuracy_pct'])}  ({m['correct']}/{m['n']})")
    out(
        f"  under-routed    {m['under_count']}  ({_fmt(m['under_pct'])} of all, "
        f"{_fmt(m['miss_rate_pct'])} of premium-labelled)  <- quality risk"
    )
    out(
        f"  over-routed     {m['over_count']}  ({_fmt(m['over_pct'])} of all, "
        f"{_fmt(m['waste_rate_pct'])} of local-labelled)   <- cost only"
    )
    out(f"  local share     {_fmt(m['local_share_pct'])}")
    out(f"  est. savings    {_fmt(m['savings_pct'])}  ({savings_note})")
    out(f"  classifier task agreement {_fmt(m['classifier_agreement_pct'])}")
    out(f"  {'task':<10} {'n':>3} {'acc.':>7} {'under':>5} {'over':>5}")
    for task, g in m["by_task"].items():
        out(f"  {task:<10} {g['n']:>3} {_fmt(g['accuracy_pct']):>7} {g['under']:>5} {g['over']:>5}")
    errors = [o for o in data["outcomes"] if o["error"]]
    if errors:
        out("  misrouted (expected -> routed, score, classified task):")
        for o in errors:
            out(
                f"    {o['error']:<5} {o['id']:<15} {o['expected']}->{o['routed']:<8} "
                f"score {o['score']:>3}  classified {o['classified_task']}"
            )


def print_sweep(
    sweep: list[RoutingMetrics], selection: Selection, out: Callable[[str], None] = print
) -> None:
    out(
        f"{'thr':>4} {'acc%':>6} {'under':>5} {'miss%':>6} {'over':>5} {'waste%':>7} "
        f"{'local%':>7} {'prem%':>6} {'save%':>6} {'w.err':>5}"
    )
    for m in sweep:
        mark = " <" if m.threshold == selection.threshold else ""
        out(
            f"{m.threshold:>4} {_fmt(m.accuracy_pct, ''):>6} {m.under_count:>5} "
            f"{_fmt(m.miss_rate_pct, ''):>6} {m.over_count:>5} {_fmt(m.waste_rate_pct, ''):>7} "
            f"{_fmt(m.local_share_pct, ''):>7} {_fmt(m.premium_share_pct, ''):>6} "
            f"{_fmt(m.savings_pct, ''):>6} {m.weighted_errors:>5}{mark}"
        )
    out(f"Selected threshold: {selection.threshold}")
    for line in selection.explanation:
        out(f"  - {line}")


# -------------------------------------------------------------------- CLI


def _load(args: argparse.Namespace) -> tuple[list[EvalPrompt], AppConfig]:
    config = load_config(args.config_dir)
    try:
        prompts = load_dataset(args.dataset, config)
    except DatasetError as exc:
        raise EvalError(
            f"the dataset is invalid; run `make eval-validate`:\n  {exc.problems[0]}"
        ) from exc
    return prompts, config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=PROMPTS)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS)
    commands = parser.add_subparsers(dest="command", required=True)

    route = commands.add_parser("route", help="evaluate one threshold (free)")
    route.add_argument("--threshold", type=int, help="default: routing.yaml threshold")
    sweep = commands.add_parser("sweep", help="evaluate thresholds 20-45 (free)")
    sweep.add_argument("--start", type=int, default=SWEEP_RANGE[0])
    sweep.add_argument("--stop", type=int, default=SWEEP_RANGE[1])
    commands.add_parser("report", help="write docs/results.md and the sweep chart (free)")
    store = commands.add_parser("store", help="write one summary row to eval_runs (free)")
    store.add_argument("--database-url", help="default: DATABASE_URL, else built from .env")
    quality = commands.add_parser("quality", help="answer-quality check (PAID, opt-in)")
    quality.add_argument("rest", nargs=argparse.REMAINDER)
    commands.add_parser("spot-check", help="compare hand grades with the model's (free)")
    args = parser.parse_args(argv)

    try:
        if args.command == "quality":
            from eval import quality as quality_module

            return quality_module.main(args.rest, results=args.results, dataset=args.dataset)
        if args.command == "spot-check":
            from eval import quality as quality_module

            return quality_module.spot_check_main(args.results)
        if args.command == "report":
            from eval import report

            return report.main(args.results)
        if args.command == "store":
            from eval import store as store_module

            return store_module.main(args.results, args.database_url)

        prompts, config = _load(args)
        meta = provenance(config, args.dataset, args.config_dir, datetime.now(UTC), git_version())
        if args.command == "route":
            threshold = config.routing.threshold if args.threshold is None else args.threshold
            data = run_routing(prompts, config, threshold, meta)
            write_json(args.results / "routing.json", data)
            print_routing(data)
            print(f"wrote {args.results / 'routing.json'}")
        else:
            if not 0 <= args.start <= args.stop <= config.routing.max_score:
                raise EvalError(f"sweep range must be within 0-{config.routing.max_score}")
            data, results, selection = run_sweep(
                prompts, config, range(args.start, args.stop + 1), meta
            )
            write_json(args.results / "sweep.json", data)
            print_sweep(results, selection)
            print(f"wrote {args.results / 'sweep.json'}")
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
