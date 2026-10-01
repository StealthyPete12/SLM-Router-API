"""Cost, baseline cost and savings, in USD, from token counts and models.yaml prices.

cost = (tokens_in * usd_per_1m_input + tokens_out * usd_per_1m_output) / 1e6
savings = baseline_cost - cost

The baseline prices the same token counts at the baseline model (the premium
default). All three are estimates: a cloud model would have written a different
number of output tokens. An unpriced model gives None, never a guessed number.
"""

from dataclasses import dataclass

from app.config import ModelConfig

# Enough decimals for one short local answer priced at a cheap cloud model.
_DECIMALS = 8


@dataclass(frozen=True)
class Costs:
    cost_usd: float | None
    baseline_cost_usd: float | None
    savings_usd: float | None


def price(model: ModelConfig, input_tokens: int, output_tokens: int | None = 0) -> float | None:
    if model.usd_per_1m_input is None or model.usd_per_1m_output is None:
        return None
    total = input_tokens * model.usd_per_1m_input + (output_tokens or 0) * model.usd_per_1m_output
    return round(total / 1_000_000, _DECIMALS)


def compute_costs(
    model: ModelConfig, baseline: ModelConfig, input_tokens: int, output_tokens: int
) -> Costs:
    cost = price(model, input_tokens, output_tokens)
    baseline_cost = price(baseline, input_tokens, output_tokens)
    savings = None
    if cost is not None and baseline_cost is not None:
        savings = round(baseline_cost - cost, _DECIMALS)
    return Costs(cost, baseline_cost, savings)
