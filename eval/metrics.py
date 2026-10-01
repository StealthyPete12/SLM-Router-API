"""Router evaluation metrics and the threshold-selection rule. Pure functions, no I/O.

Definitions (N = prompts evaluated):

    correct         routed tier == expected tier
    under-routed    expected premium, routed local: a quality risk (the answer may be worse)
    over-routed     expected local, routed premium: only a cost (the answer is fine, just dearer)
    accuracy        correct / N
    under/over %    count / N, so accuracy + under % + over % = 100
    miss rate       under-routed / prompts expected premium (how often a complex prompt
                    is kept local); the number the threshold is tuned to keep low
    waste rate      over-routed / prompts expected local
    local share     routed local / N; premium share = 1 - local share
    savings %       estimated share of the all-premium baseline cost avoided, on input
                    tokens only (see `savings`); a dry run has no output tokens to price

Under-routing is reported on its own everywhere and never only inside accuracy.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from app.config import TASK_TYPES, Tier

# Threshold-selection rule. Fixed before the first sweep was run; changing these
# values changes the rule, so do it in a reviewed commit, not to fit a result.
MISS_RATE_TOLERANCE_PP = 5.0  # eligible: miss rate <= lowest miss rate in the sweep + this
UNDER_ROUTING_WEIGHT = 2  # an under-routed prompt costs twice an over-routed one
SCORE_STEP = 5  # every base point and modifier is a multiple of 5 in routing.yaml


@dataclass(frozen=True)
class Outcome:
    """One routed prompt."""

    id: str
    task: str
    expected: Tier
    routed: Tier
    input_tokens: int
    score: int
    reason: str
    classified_task: str
    tags: tuple[str, ...] = ()
    model_id: str = ""
    cost_usd: float | None = None  # input tokens priced at the routed model
    baseline_cost_usd: float | None = None  # input tokens priced at the baseline model

    @property
    def correct(self) -> bool:
        return self.routed == self.expected

    @property
    def error(self) -> str | None:
        if self.expected == "premium" and self.routed == "local":
            return "under"
        if self.expected == "local" and self.routed == "premium":
            return "over"
        return None

    @property
    def decided_by_rule(self) -> bool:
        """True when a hard rule, not the threshold, chose the tier."""
        return self.reason not in ("score_below_threshold", "score_at_or_above_threshold")


def pct(part: float, whole: float) -> float | None:
    return round(100 * part / whole, 2) if whole else None


@dataclass(frozen=True)
class Savings:
    """Estimated savings against the all-premium baseline, on input tokens."""

    pct: float | None
    basis: str  # "priced" (USD from models.yaml) or "token_share" (prices unset)
    baseline_usd: float | None = None
    cost_usd: float | None = None
    savings_usd: float | None = None


def savings(outcomes: list[Outcome]) -> Savings:
    """Savings % = (baseline cost - routed cost) / baseline cost, input tokens only.

    With the baseline priced, USD figures come from models.yaml. Unpriced, the
    percentage is still exact for input tokens when local models cost 0: the
    baseline's input price cancels out, leaving local input tokens / all input tokens.
    """
    priced = all(o.baseline_cost_usd is not None and o.cost_usd is not None for o in outcomes)
    if outcomes and priced:
        baseline = sum(o.baseline_cost_usd or 0 for o in outcomes)
        cost = sum(o.cost_usd or 0 for o in outcomes)
        return Savings(
            pct(baseline - cost, baseline),
            "priced",
            round(baseline, 8),
            round(cost, 8),
            round(baseline - cost, 8),
        )
    local_free = all(o.cost_usd in (None, 0) for o in outcomes if o.routed == "local")
    if not local_free:
        return Savings(None, "unknown")
    total = sum(o.input_tokens for o in outcomes)
    local = sum(o.input_tokens for o in outcomes if o.routed == "local")
    return Savings(pct(local, total), "token_share")


@dataclass(frozen=True)
class GroupStats:
    n: int
    correct: int
    under: int
    over: int
    accuracy_pct: float | None


def _group(outcomes: list[Outcome]) -> GroupStats:
    correct = sum(o.correct for o in outcomes)
    return GroupStats(
        n=len(outcomes),
        correct=correct,
        under=sum(o.error == "under" for o in outcomes),
        over=sum(o.error == "over" for o in outcomes),
        accuracy_pct=pct(correct, len(outcomes)),
    )


@dataclass(frozen=True)
class RoutingMetrics:
    threshold: int
    n: int
    correct: int
    accuracy_pct: float | None
    under_count: int
    under_pct: float | None  # of all prompts
    miss_rate_pct: float | None  # of prompts expected premium
    over_count: int
    over_pct: float | None  # of all prompts
    waste_rate_pct: float | None  # of prompts expected local
    local_count: int
    local_share_pct: float | None
    premium_share_pct: float | None
    savings_pct: float | None
    savings_basis: str
    savings_usd: float | None
    baseline_usd: float | None
    weighted_errors: int  # UNDER_ROUTING_WEIGHT * under + over
    near_threshold: int  # prompts scoring exactly one step below or at the threshold
    by_task: dict[str, GroupStats] = field(default_factory=dict)
    by_decider: dict[str, GroupStats] = field(default_factory=dict)  # threshold vs hard rule
    borderline: GroupStats | None = None
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)  # expected -> routed
    classifier_agreement_pct: float | None = None  # classified task == labelled task

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarise(outcomes: list[Outcome], threshold: int) -> RoutingMetrics:
    n = len(outcomes)
    premium_expected = sum(o.expected == "premium" for o in outcomes)
    local_expected = n - premium_expected
    under = sum(o.error == "under" for o in outcomes)
    over = sum(o.error == "over" for o in outcomes)
    local = sum(o.routed == "local" for o in outcomes)
    correct = n - under - over
    s = savings(outcomes)
    confusion = {
        expected: dict(Counter(o.routed for o in outcomes if o.expected == expected))
        for expected in ("local", "premium")
    }
    # Fixed key order so the output files are deterministic.
    confusion = {e: {r: confusion[e].get(r, 0) for r in ("local", "premium")} for e in confusion}
    return RoutingMetrics(
        threshold=threshold,
        n=n,
        correct=correct,
        accuracy_pct=pct(correct, n),
        under_count=under,
        under_pct=pct(under, n),
        miss_rate_pct=pct(under, premium_expected),
        over_count=over,
        over_pct=pct(over, n),
        waste_rate_pct=pct(over, local_expected),
        local_count=local,
        local_share_pct=pct(local, n),
        premium_share_pct=pct(n - local, n),
        savings_pct=s.pct,
        savings_basis=s.basis,
        savings_usd=s.savings_usd,
        baseline_usd=s.baseline_usd,
        weighted_errors=UNDER_ROUTING_WEIGHT * under + over,
        near_threshold=sum(threshold - SCORE_STEP <= o.score <= threshold for o in outcomes),
        by_task={
            task: _group([o for o in outcomes if o.task == task])
            for task in TASK_TYPES
            if any(o.task == task for o in outcomes)
        },
        by_decider={
            "threshold": _group([o for o in outcomes if not o.decided_by_rule]),
            "hard_rule": _group([o for o in outcomes if o.decided_by_rule]),
        },
        borderline=_group([o for o in outcomes if "borderline" in o.tags]),
        confusion=confusion,
        classifier_agreement_pct=pct(sum(o.classified_task == o.task for o in outcomes), n),
    )


@dataclass(frozen=True)
class Selection:
    threshold: int
    eligible: list[int]
    plateau: list[int]  # thresholds that route every prompt exactly like the chosen one
    miss_rate_floor: float
    miss_rate_ceiling: float
    explanation: list[str]


def select_threshold(
    sweep: list[RoutingMetrics], decisions: dict[int, tuple[str, ...]], current: int
) -> Selection:
    """Pick a threshold from a sweep with a fixed, explainable rule.

    `decisions[t]` is the routed tier of every prompt at threshold t (same order),
    used to find plateaus: thresholds that make identical decisions.

    1. Eligible: miss rate <= the sweep's lowest miss rate + MISS_RATE_TOLERANCE_PP.
       Under-routing is the quality risk, so it is a hard limit, not a trade-off.
    2. Among eligible thresholds, fewest weighted errors
       (UNDER_ROUTING_WEIGHT x under-routed + over-routed), not plain accuracy.
    3. Ties: higher estimated savings, then fewer prompts within one score step of
       the threshold (stability), then the threshold closest to the current one.
    4. Report the chosen plateau's multiple of SCORE_STEP when the plateau has one,
       so the rule reads "a score of X or more goes premium".
    """
    if not sweep:
        raise ValueError("empty sweep")
    floor = min(m.miss_rate_pct or 0.0 for m in sweep)
    ceiling = round(floor + MISS_RATE_TOLERANCE_PP, 2)
    eligible = [m for m in sweep if (m.miss_rate_pct or 0.0) <= ceiling]

    def key(m: RoutingMetrics) -> tuple[float, ...]:
        return (
            m.weighted_errors,
            -(m.savings_pct or 0.0),
            m.near_threshold,
            abs(m.threshold - current),
        )

    best = min(eligible, key=key)
    plateau = sorted(t for t, d in decisions.items() if d == decisions[best.threshold])
    canonical = [t for t in plateau if t % SCORE_STEP == 0]
    chosen = best.threshold
    if canonical:
        # Prefer the current threshold when it is on the plateau, else the plateau's multiple.
        chosen = current if current in canonical else canonical[-1]
    explanation = [
        f"Lowest miss rate in the sweep: {floor:.1f}% of premium prompts; "
        f"eligible thresholds keep it at or below {ceiling:.1f}% "
        f"(+{MISS_RATE_TOLERANCE_PP:g} percentage points).",
        f"Eligible: {', '.join(str(m.threshold) for m in eligible)}.",
        f"Fewest weighted errors ({UNDER_ROUTING_WEIGHT} x under + over) among them: "
        f"{best.weighted_errors} at threshold {best.threshold} "
        f"({best.under_count} under-routed, {best.over_count} over-routed).",
        f"Thresholds {plateau[0]}-{plateau[-1]} route every prompt identically; "
        f"{chosen} is reported for that plateau.",
    ]
    return Selection(chosen, [m.threshold for m in eligible], plateau, floor, ceiling, explanation)
