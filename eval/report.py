"""Writes docs/results.md and docs/img/threshold-sweep.svg from eval/results/*.json.

    uv run python -m eval.run_eval report     # or: make eval-report

Every number in the output comes from the artifacts: routing.json (one threshold),
sweep.json (20-45), decision.yaml (the person's decision), and, once the paid check
has run, quality.json and spot_check_summary.json. Missing pieces are reported as
pending, never filled in. The same inputs always give the same files.
"""

import json
import sys
from html import escape
from pathlib import Path
from typing import Any

import yaml

from eval.metrics import MISS_RATE_TOLERANCE_PP, SCORE_STEP, UNDER_ROUTING_WEIGHT

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
RESULTS_MD = DOCS / "results.md"
CHART = DOCS / "img" / "threshold-sweep.svg"
DECISION = Path(__file__).resolve().parent / "decision.yaml"


class ReportError(Exception):
    """Inputs for the report are missing or inconsistent."""


# -------------------------------------------------------------------- chart

# Categorical slots 1-5 of the validated reference palette (light / dark steps);
# colour follows the series, the same in both panels.
SERIES = {
    "accuracy_pct": ("Accuracy", "#2a78d6", "#3987e5"),
    "miss_rate_pct": ("Miss rate (premium prompts kept local)", "#eb6834", "#d95926"),
    "waste_rate_pct": ("Waste rate (local prompts sent premium)", "#1baf7a", "#199e70"),
    "local_share_pct": ("Local share", "#eda100", "#c98500"),
    "savings_pct": ("Estimated savings (input tokens)", "#e87ba4", "#d55181"),
}
PANELS = [
    ("Routing quality", ["accuracy_pct", "miss_rate_pct", "waste_rate_pct"]),
    ("Cost", ["local_share_pct", "savings_pct"]),
]
W, PANEL_H, TOP, GAP = 760, 210, 120, 100
LEFT, RIGHT = 56, 120


def _x(t: float, lo: int, hi: int) -> float:
    return LEFT + (t - (lo - 0.5)) / ((hi + 0.5) - (lo - 0.5)) * (W - LEFT - RIGHT)


def _y(v: float, top: float) -> float:
    return top + PANEL_H - v / 100 * PANEL_H


def sweep_chart_svg(sweep: dict[str, Any], kept: int, rule_pick: int) -> str:
    """Step lines (decisions only change between whole thresholds) on one 0-100% axis."""
    rows = sweep["thresholds"]
    lo, hi = rows[0]["threshold"], rows[-1]["threshold"]
    n = rows[0]["n"]
    height = TOP + 2 * PANEL_H + GAP + 60
    slot = {key: i for i, key in enumerate(SERIES, 1)}
    # Literal colours, not CSS variables: some SVG viewers ignore var(). The light
    # rules come first; the dark media query overrides the same selectors.
    tokens = {
        "light": {
            "surface": "#fcfcfb",
            "ink": "#0b0b0b",
            "ink2": "#52514e",
            "muted": "#898781",
            "grid": "#e1e0d9",
            "axis": "#c3c2b7",
        }
        | {f"s{i}": light for i, (_, light, _) in enumerate(SERIES.values(), 1)},
        "dark": {
            "surface": "#1a1a19",
            "ink": "#ffffff",
            "ink2": "#c3c2b7",
            "muted": "#898781",
            "grid": "#2c2c2a",
            "axis": "#383835",
        }
        | {f"s{i}": dark for i, (_, _, dark) in enumerate(SERIES.values(), 1)},
    }

    def colour_rules(c: dict[str, str]) -> str:
        return (
            f".bg{{fill:{c['surface']}}} .t,.val{{fill:{c['ink']}}} .st,.lab{{fill:{c['ink2']}}} "
            f".ax{{fill:{c['muted']}}} .grid{{stroke:{c['grid']}}} .base{{stroke:{c['axis']}}} "
            f".rule{{stroke:{c['ink2']}}} .soft{{stroke:{c['muted']}}} "
            f".dot,.halo{{stroke:{c['surface']}}} .halo{{fill:{c['surface']}}} "
            + " ".join(
                f".c{i}{{stroke:{c[f's{i}']}}} .f{i}{{fill:{c[f's{i}']}}}" for i in slot.values()
            )
        )

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" '
        f'height="{height}" role="img" aria-labelledby="t d" '
        'font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
        "<style>",
        ".t{font-size:15px;font-weight:600} .st{font-size:12px} .lab,.ax{font-size:11px}",
        ".ax{font-variant-numeric:tabular-nums} .val,.halo{font-size:11px;font-weight:600}",
        ".halo{stroke-width:4px;stroke-linejoin:round}",
        ".grid,.base,.rule,.soft{stroke-width:1} .dot{stroke-width:2} .hit{fill:transparent}",
        ".ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}",
        colour_rules(tokens["light"]),
        f"@media (prefers-color-scheme: dark){{{colour_rules(tokens['dark'])}}}",
        "</style>",
        f'<rect class="bg" width="{W}" height="{height}"/>',
        f'<title id="t">Threshold sweep {lo}-{hi}: routing quality and cost</title>',
        f'<desc id="d">Router accuracy, under- and over-routing rates, local share and '
        f"estimated savings for thresholds {lo} to {hi} on {n} labelled prompts. Kept "
        f"threshold {kept}; the automated rule picked {rule_pick}.</desc>",
        f'<text class="t" x="{LEFT}" y="24">Threshold sweep: routing quality vs cost</text>',
        f'<text class="st" x="{LEFT}" y="42">{n} labelled prompts, dry-run routing (no model '
        "called). Savings are estimates on input tokens.</text>",
    ]
    for p, (title, keys) in enumerate(PANELS):
        top = TOP + p * (PANEL_H + GAP)
        out.append(
            f'<text class="t" x="{LEFT}" y="{top - 56}" style="font-size:13px">{title}</text>'
        )
        lx = LEFT
        for key in keys:  # legend: a line key beside each name, text in ink
            label = SERIES[key][0]
            out.append(
                f'<line class="ln c{slot[key]}" x1="{lx}" y1="{top - 36}" x2="{lx + 16}" '
                f'y2="{top - 36}"/><text class="lab" x="{lx + 21}" y="{top - 32}">'
                f"{escape(label)}</text>"
            )
            lx += 30 + len(label) * 5.6
        for v in (0, 25, 50, 75, 100):
            y = _y(v, top)
            cls = "base" if v == 0 else "grid"
            out.append(f'<line class="{cls}" x1="{LEFT}" y1="{y}" x2="{W - RIGHT}" y2="{y}"/>')
            out.append(f'<text class="ax" x="{LEFT - 8}" y="{y + 4}" text-anchor="end">{v}%</text>')
        for t in range(lo, hi + 1):
            if t % SCORE_STEP == 0:
                out.append(
                    f'<text class="ax" x="{_x(t, lo, hi)}" y="{top + PANEL_H + 16}" '
                    f'text-anchor="middle">{t}</text>'
                )
        for t, cls, label in (
            (rule_pick, "soft", f"rule's pick {rule_pick}"),
            (kept, "rule", f"kept {kept}"),
        ):
            x = _x(t, lo, hi)
            out.append(
                f'<line class="{cls}" x1="{x}" y1="{top - 6}" x2="{x}" y2="{top + PANEL_H}"/>'
            )
            if p == 0:
                out.append(
                    f'<text class="lab" x="{x}" y="{top - 10}" text-anchor="middle">{label}</text>'
                )
        for key in keys:
            points = [(r["threshold"], r[key]) for r in rows if r[key] is not None]
            d = f"M{_x(points[0][0] - 0.5, lo, hi):.1f},{_y(points[0][1], top):.1f}"
            for t, v in points:
                d += f" V{_y(v, top):.1f} H{_x(t + 0.5, lo, hi):.1f}"
            out.append(f'<path class="ln c{slot[key]}" d="{d}"/>')
        # Selective direct labels: each series' value at the kept threshold.
        kept_row = next(r for r in rows if r["threshold"] == kept)
        labels = sorted(((kept_row[k], k) for k in keys if kept_row[k] is not None), reverse=True)
        last_y = -100.0
        for value, key in labels:
            y = _y(value, top)
            ly = max(y, last_y + 13)
            last_y = ly
            x = _x(kept, lo, hi)
            out.append(f'<circle class="dot f{slot[key]}" cx="{x}" cy="{y}" r="4"/>')
            for cls in ("halo", "val"):  # a surface-coloured copy keeps the label off the line
                out.append(f'<text class="{cls}" x="{x + 8}" y="{ly + 4}">{value:.1f}%</text>')
        for r in rows:  # native tooltips when the SVG is opened on its own
            x0 = _x(r["threshold"] - 0.5, lo, hi)
            tip = f"Threshold {r['threshold']}: " + ", ".join(
                f"{SERIES[k][0]} {r[k]:.1f}%" for k in keys if r[k] is not None
            )
            out.append(
                f'<rect class="hit" x="{x0:.1f}" y="{top}" width="{_x(1, 0, 1) - _x(0, 0, 1):.1f}" '
                f'height="{PANEL_H}"><title>{escape(tip)}</title></rect>'
            )
    axis_y = TOP + 2 * PANEL_H + GAP + 42
    out.append(
        f'<text class="st" x="{(LEFT + W - RIGHT) / 2}" y="{axis_y}" text-anchor="middle">'
        "Threshold (a score at or above it goes premium)</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# -------------------------------------------------------------------- markdown


def _p(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}%"


def plateaus(rows: list[dict[str, Any]]) -> list[tuple[int, int, dict[str, Any]]]:
    """Group consecutive thresholds with identical routing (scores move in steps of 5).

    Each group is shown with its last row, the multiple of SCORE_STEP for full
    plateaus, so the stability count matches "a score of X or more goes premium".
    """
    keys = ("correct", "under_count", "over_count", "local_count")
    groups: list[tuple[int, int, dict[str, Any]]] = []
    for r in rows:
        if groups and all(groups[-1][2][k] == r[k] for k in keys):
            groups[-1] = (groups[-1][0], r["threshold"], r)
        else:
            groups.append((r["threshold"], r["threshold"], r))
    return groups


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _decision_status(
    decision: dict[str, Any], routing: dict[str, Any], sweep: dict[str, Any]
) -> list[str]:
    problems = []
    if decision["dataset_sha256"] != sweep["dataset"]["sha256"]:
        problems.append("it was made on a different version of the dataset")
    if decision["rule_selected"] != sweep["selection"]["threshold"]:
        problems.append(
            f"the rule now picks {sweep['selection']['threshold']}, not {decision['rule_selected']}"
        )
    if decision["selected_threshold"] != routing["config"]["configured_threshold"]:
        problems.append("routing.yaml has a different threshold")
    return problems


def render_markdown(
    routing: dict[str, Any],
    sweep: dict[str, Any],
    decision: dict[str, Any],
    quality: dict[str, Any] | None,
    spot: dict[str, Any] | None,
    chart_path: str,
) -> str:
    m = routing["metrics"]
    comp = routing["composition"]
    sel = sweep["selection"]
    rows = sweep["thresholds"]
    by_t = {r["threshold"]: r for r in rows}
    kept = decision["selected_threshold"]
    k = by_t[kept]
    rule = by_t[sel["threshold"]]
    stale = _decision_status(decision, routing, sweep)
    has_quality = quality is not None and not quality.get("mock")
    q = quality["summary"] if has_quality and quality else None
    errors = [o for o in routing["outcomes"] if o["error"]]
    under = [o for o in errors if o["error"] == "under"]
    unreachable = [o for o in under if o["score"] < rows[0]["threshold"]]
    misclassified_under = [o for o in under if o["classified_task"] != o["task"]]
    savings_basis = (
        "priced at the baseline model's input price in config/models.yaml"
        if m["savings_basis"] == "priced"
        else "premium prices are not configured, so this is the share of input tokens kept "
        "local, which equals the input-cost saving because local models cost 0 and the "
        "baseline's input price cancels out"
    )

    lines = [
        "# Evaluation results",
        "",
        f"Evaluated {routing['generated_at'][:10]} at version `{routing['version']}`, dataset "
        f"`eval/prompts.jsonl` (sha256 `{routing['dataset']['sha256'][:12]}`). Generated by "
        "`make eval-report` from `eval/results/`; do not edit by hand.",
        "",
        "| Part | Status |",
        "| --- | --- |",
        f"| Router evaluation (dry run, {m['n']} prompts) | **Done.** Deterministic; "
        "no model called |",
        f"| Threshold sweep {rows[0]['threshold']}-{rows[-1]['threshold']} | **Done.** |",
        "| Answer-quality check (premium grades local answers) | "
        + (
            f"**Done** on {q['graded']} prompts"
            if q
            else "**Pending.** Needs real local models and a premium key (`make eval-quality-paid`)"
        )
        + " |",
        "| Latency (p95 per tier) | "
        + ("**Done**, from the quality run" if q else "**Pending.** Measured by the quality run")
        + " |",
        "",
        "No answer-quality, latency or USD savings figure is claimed below unless its row "
        "above says done. Savings are estimates.",
        "",
        "## Headline",
        "",
        f"At the threshold in `config/routing.yaml` (**{kept}**):",
        "",
        *_table(
            ["Metric", "Value"],
            [
                ["Router accuracy", f"{_p(m['accuracy_pct'])} ({m['correct']} of {m['n']})"],
                [
                    "Under-routed (premium prompt sent local)",
                    f"{m['under_count']} prompts: {_p(m['under_pct'])} of all, "
                    f"**{_p(m['miss_rate_pct'])} of premium-labelled prompts**",
                ],
                [
                    "Over-routed (local prompt sent premium)",
                    f"{m['over_count']} prompts: {_p(m['over_pct'])} of all, "
                    f"{_p(m['waste_rate_pct'])} of local-labelled prompts",
                ],
                ["Local share", _p(m["local_share_pct"])],
                ["Estimated savings vs all-premium (input tokens)", _p(m["savings_pct"])],
                [
                    "Answer quality (local judged same or better)",
                    _p(q["quality_score_pct"]) if q else "pending",
                ],
                [
                    "p95 latency local / premium",
                    f"{q['p95_local_latency_ms']} ms / {q['p95_premium_latency_ms']} ms"
                    if q
                    else "pending",
                ],
            ],
        ),
        "",
        f"Under-routing is the weak spot: the router keeps {_p(m['miss_rate_pct'])} of the "
        f"prompts that need the premium model on a local model. {len(unreachable)} of those "
        f"{len(under)} score below {rows[0]['threshold']}, so no threshold in the sweep "
        f"reaches them; {len(misclassified_under)} of the {len(under)} were given the wrong "
        "task type by the keyword classifier. The fix is the classifier, not the threshold "
        "(see Errors).",
        "",
        "## Dataset",
        "",
        f"{comp['rows']} hand-written prompts in `eval/prompts.jsonl`: {comp['local']} labelled "
        f"local and {comp['premium']} premium, {comp['tags'].get('borderline', 0)} marked "
        "borderline. Every row has an id, the task, the expected tier, a rationale and tags; "
        "`make eval-validate` checks the schema, unique ids, labels, distribution, "
        "duplicates, secrets and that any personal data is synthetic (example.com, 555-01xx "
        "numbers, published test cards).",
        "",
        *_table(
            ["Task", "Prompts", "Local", "Premium", "Borderline", "Quality subset"],
            [
                [
                    t,
                    str(c["total"]),
                    str(c["local"]),
                    str(c["premium"]),
                    str(c["borderline"]),
                    str(c["quality"]),
                ]
                for t, c in comp["by_task"].items()
            ],
        ),
        "",
        "Edge cases (rows can carry several tags): "
        + ", ".join(f"{t.replace('_', ' ')} {n}" for t, n in comp["tags"].items())
        + ".",
        "",
        "**How labels were set.** `local` means a 7-8B local model (Phi-3 Mini, Mistral 7B, "
        "Llama 3.1 8B) would likely answer adequately; `premium` means a small model would "
        "likely be noticeably worse: multi-step reasoning, non-trivial code, judgement-heavy "
        "analysis, long high-quality output or precise language work. Rows decided by a hard "
        "rule carry the tier the policy must give (context window: premium; privacy: local; "
        "override: the forced tier). Labels were written before the router was run on the "
        "set and were not changed afterwards. They are one person's judgement, not ground "
        "truth, and several prompts deliberately avoid the classifier's keywords.",
        "",
        "## Method",
        "",
        "- Each prompt is routed in-process by the same `Router` that serves `POST /v1/route` "
        "(a test checks they agree), with `config/routing.yaml` as committed. No model is "
        "called, so the run is free and deterministic.",
        "- Rows marked `privacy_mode` are routed with privacy mode on; rows with "
        "`forced_tier` send it as `X-Router-Tier`.",
        f"- The sweep re-routes every prompt at each threshold from {rows[0]['threshold']} to "
        f"{rows[-1]['threshold']}; every other weight is unchanged.",
        "- Savings are estimated against the premium default model (the configured "
        f"baseline, `{routing['config']['baseline_model']}`), not a flagship. A dry run has no "
        f"answers, so only input tokens are priced: {savings_basis}.",
        "",
        "**Definitions.** *Under-routed*: labelled premium, routed local (the answer may be "
        "worse). *Over-routed*: labelled local, routed premium (only costs money). Accuracy + "
        "under-routed % + over-routed % = 100. *Miss rate*: under-routed / premium-labelled "
        "prompts. *Waste rate*: over-routed / local-labelled prompts. *Local share*: prompts "
        "routed local / all. *Savings %*: (baseline cost - routed cost) / baseline cost.",
        "",
        "## Threshold sweep",
        "",
        f"Every base score and modifier is a multiple of {SCORE_STEP}, so routing only changes "
        f"between plateaus; thresholds on one row route all {m['n']} prompts identically. "
        "Per-threshold numbers are in `eval/results/sweep.json`.",
        "",
        *_table(
            [
                "Threshold",
                "Accuracy",
                "Under-routed",
                "Miss rate",
                "Over-routed",
                "Waste rate",
                "Local share",
                "Premium share",
                "Est. savings",
                "Within one step",
            ],
            [
                [
                    (f"{a}" if a == b else f"{a}-{b}")
                    + (" **(kept)**" if a <= kept <= b else "")
                    + (" (rule)" if a <= sel["threshold"] <= b else ""),
                    _p(r["accuracy_pct"]),
                    str(r["under_count"]),
                    _p(r["miss_rate_pct"]),
                    str(r["over_count"]),
                    _p(r["waste_rate_pct"]),
                    _p(r["local_share_pct"]),
                    _p(r["premium_share_pct"]),
                    _p(r["savings_pct"]),
                    str(r["near_threshold"]),
                ]
                for a, b, r in plateaus(rows)
            ],
        ),
        "",
        f"*Within one step*: prompts scoring between threshold - {SCORE_STEP} and the threshold, "
        "which one modifier more or less would flip. Fewer is more stable.",
        "",
        "![Threshold sweep: accuracy, miss rate and waste rate (top); local share and estimated "
        f"savings (bottom), thresholds {rows[0]['threshold']}-{rows[-1]['threshold']}]"
        f"({chart_path})",
        "",
        "## Selected threshold",
        "",
        f"**Previous {decision['previous_threshold']}, selected {kept}.** {decision['decision']}",
        "",
        f"The automated rule (`select_threshold` in `eval/metrics.py`, fixed before the first "
        f"sweep) treats under-routing as a hard limit: a threshold is eligible only if its miss "
        f"rate is within {MISS_RATE_TOLERANCE_PP:g} points of the sweep's lowest; among those it "
        f"takes the fewest weighted errors ({UNDER_ROUTING_WEIGHT} x under + over), then higher "
        "savings, then stability. Its output:",
        "",
        *[f"- {line}" for line in sel["explanation"]],
        "",
        f"At the rule's pick ({rule['threshold']}) versus {kept}: miss rate "
        f"{_p(rule['miss_rate_pct'])} vs {_p(k['miss_rate_pct'])}, over-routed "
        f"{rule['over_count']} vs {k['over_count']}, local share {_p(rule['local_share_pct'])} vs "
        f"{_p(k['local_share_pct'])}, estimated savings {_p(rule['savings_pct'])} vs "
        f"{_p(k['savings_pct'])}, weighted errors {rule['weighted_errors']} vs "
        f"{k['weighted_errors']}.",
        "",
        f"Why (from `eval/decision.yaml`, decided {decision['decided_on']}):",
        "",
        *[f"- {reason}" for reason in decision["reasons"]],
        "",
        f"Revisit when: {decision['revisit_when']}",
        "",
    ]
    if stale:
        lines += [
            f"> **Warning: the decision record is stale**: {'; '.join(stale)}. Re-read the "
            "sweep and update `eval/decision.yaml`.",
            "",
        ]
    lines += [
        "## Errors",
        "",
        f"At threshold {kept}. Confusion (rows: labelled tier, columns: routed tier):",
        "",
        *_table(
            ["", "Routed local", "Routed premium"],
            [
                [f"Labelled {e}", str(c["local"]), str(c["premium"])]
                for e, c in m["confusion"].items()
            ],
        ),
        "",
        *_table(
            ["Task", "Prompts", "Accuracy", "Under-routed", "Over-routed"],
            [
                [t, str(g["n"]), _p(g["accuracy_pct"]), str(g["under"]), str(g["over"])]
                for t, g in m["by_task"].items()
            ],
        ),
        "",
        f"Decided by the threshold: {_p(m['by_decider']['threshold']['accuracy_pct'])} accurate "
        f"over {m['by_decider']['threshold']['n']} prompts; by a hard rule (override, privacy, "
        f"context window): {_p(m['by_decider']['hard_rule']['accuracy_pct'])} over "
        f"{m['by_decider']['hard_rule']['n']}. Borderline prompts: "
        f"{_p(m['borderline']['accuracy_pct'])} over {m['borderline']['n']}. The classifier's "
        f"task type matches the label on {_p(m['classifier_agreement_pct'])} of prompts.",
        "",
        "Every misrouted prompt (texts in `eval/prompts.jsonl`):",
        "",
        *_table(
            ["Prompt", "Error", "Task", "Classified as", "Score"],
            [
                [f"`{o['id']}`", o["error"], o["task"], o["classified_task"], str(o["score"])]
                for o in sorted(errors, key=lambda o: (o["error"] != "under", o["id"]))
            ],
        ),
        "",
        f"Patterns (from `eval/decision.yaml`): {decision['error_patterns']}",
        "",
        "## Savings",
        "",
        f"Estimated savings at {kept}: **{_p(m['savings_pct'])}** of the all-premium baseline "
        f"cost, on input tokens only ({savings_basis}).",
    ]
    if m["savings_usd"] is not None:
        lines.append(
            f"In USD for this dataset's input: baseline ${m['baseline_usd']:.6f}, saved "
            f"${m['savings_usd']:.6f}."
        )
    lines += [
        "",
        (
            "This is lower than the local share because the prompts kept local are the shorter "
            "ones: long prompts score more and go premium. "
            if (m["savings_pct"] or 0) < (m["local_share_pct"] or 0)
            else ""
        )
        + "Output tokens, usually the larger part of a bill, are not in a dry run; real "
        "per-request savings (provider-reported tokens) appear on the dashboard once real "
        "traffic flows. All savings figures are estimates: a premium model would have "
        "written a different number of output tokens.",
        "",
        "## Answer quality and latency",
        "",
    ]
    if q and quality:
        lines += [
            f"Run {quality['generated_at'][:10]} against `{quality['premium_model']}` at "
            f"threshold {quality['threshold']}; {q['selected']} quality-subset prompts, "
            f"{q['routed_local']} routed local and graded. Cost ${q['cost_usd']}.",
            "",
            *_table(
                ["Better", "Same", "Worse", "Invalid", "Failed", "Quality score"],
                [
                    [
                        str(q["better"]),
                        str(q["same"]),
                        str(q["worse"]),
                        str(q["invalid"]),
                        str(q["failed"]),
                        _p(q["quality_score_pct"]),
                    ]
                ],
            ),
            "",
            f"p95 latency over {q['latency_samples']} prompts: local "
            f"{q['p95_local_latency_ms']} ms, premium {q['p95_premium_latency_ms']} ms "
            "(router-measured, cache bypassed).",
            "",
        ]
        if spot and spot.get("checked"):
            lines += [
                f"Hand spot-check: {spot['checked']} blind prompts; the grader agreed on "
                f"{spot['agree']} ({spot['agreement_pct']}%).",
                "",
            ]
        else:
            lines += ["Hand spot-check: pending (`eval/results/spot_check.csv`).", ""]
    else:
        lines += [
            "**Not run yet.** It needs the real local models and a premium API key, neither of "
            "which was available where this evaluation ran; no quality or latency number is "
            "reported until it runs.",
            "",
        ]
    lines += [
        "How it works (`eval/quality.py`): for each of the quality-subset prompts (all task "
        "types, mostly local-labelled and borderline), the router's own dry run decides the "
        "tier. For prompts it keeps local, the router fetches the local answer (forced local, "
        "so a failure cannot hide behind a cloud fallback) and a premium answer, then asks the "
        'premium model to compare them as "A" and "B" in a seeded random order and reply '
        "in JSON (`A`, `B` or `same`, plus a reason). Every call goes through the router, so "
        "keys stay in the API container and each call is logged. A blind sample of 12 is "
        "written to `eval/results/spot_check.md` for a person to grade.",
        "",
        "Limits of model grading: the grader is one of the two contestants and may prefer its "
        "own style; it sees no reference answer; one grade per prompt is noisy. Read the "
        'quality score as "how often the premium model saw no loss", not as ground truth, '
        "and check it against the hand spot-check.",
        "",
        "## Limitations",
        "",
        "- One person wrote and labelled all prompts; labels are judgements, and 'needs "
        "premium' depends on which local models and which premium model you run.",
        f"- {m['n']} prompts: one prompt moves the miss rate by "
        f"{100 / max(1, comp['premium']):.1f} points; differences of one or two prompts "
        "between thresholds are noise.",
        "- The set is synthetic and English-first. It is not a sample of real traffic, so "
        "local share and savings will differ in production.",
        "- The threshold was chosen on the same prompts it is reported on. Weight or keyword "
        "changes made from these errors need a separate held-out set to be measured fairly.",
        "- Savings cover input tokens only and are estimates; premium prices may be unset.",
        "- Quality and latency are pending until a run with real models (see above).",
        "",
        "## Reproduce",
        "",
        "```bash",
        "make eval-validate          # dataset checks (free)",
        "make eval                   # route + sweep + report + chart (free, no stack needed)",
        "make eval-store             # write the summary row to Postgres eval_runs (stack up)",
        "make eval-quality-estimate  # cost cap of the paid check (free, stack up)",
        "make eval-quality-paid CONFIRM_PAID=yes   # PAID: real models and a premium key",
        "make eval-spot-check        # after filling eval/results/spot_check.csv",
        "```",
        "",
    ]
    return "\n".join(lines)


def _read(path: Path, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise ReportError(f"{path} is missing; run `make eval` first")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build(results: Path, decision_path: Path = DECISION) -> tuple[str, str]:
    """(markdown, svg) from the artifacts in `results`."""
    routing = _read(results / "routing.json")
    sweep = _read(results / "sweep.json")
    assert routing is not None and sweep is not None
    if routing["dataset"]["sha256"] != sweep["dataset"]["sha256"]:
        raise ReportError("routing.json and sweep.json come from different datasets; rerun both")
    decision = yaml.safe_load(decision_path.read_text(encoding="utf-8"))
    quality = _read(results / "quality.json", required=False)
    spot = _read(results / "spot_check_summary.json", required=False)
    if routing["threshold"] != decision["selected_threshold"]:
        raise ReportError(
            f"routing.json is for threshold {routing['threshold']} but the decision is "
            f"{decision['selected_threshold']}; rerun `make eval`"
        )
    svg = sweep_chart_svg(sweep, decision["selected_threshold"], sweep["selection"]["threshold"])
    markdown = render_markdown(routing, sweep, decision, quality, spot, "img/threshold-sweep.svg")
    return markdown, svg


def main(results: Path) -> int:
    try:
        markdown, svg = build(results)
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    CHART.parent.mkdir(parents=True, exist_ok=True)
    CHART.write_text(svg, encoding="utf-8")
    RESULTS_MD.write_text(markdown, encoding="utf-8")
    print(f"wrote {RESULTS_MD.relative_to(ROOT)} and {CHART.relative_to(ROOT)}")
    return 0
