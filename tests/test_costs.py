"""Cost, baseline cost and savings from token counts and models.yaml prices."""

from app.aggregator.costs import compute_costs, price
from app.config import DEFAULT_CONFIG_DIR, load_config


def test_unpriced_models_give_unknown_not_zero(premium_env):
    config = load_config(DEFAULT_CONFIG_DIR)
    premium = config.default_model("premium")
    local = config.default_local_model

    assert price(premium, 1000, 100) is None
    costs = compute_costs(local, config.baseline_model, 1000, 100)
    assert costs.cost_usd == 0
    assert costs.baseline_cost_usd is None
    assert costs.savings_usd is None


def test_local_answer_saves_the_baseline_cost(premium_env, priced_env):
    config = load_config(DEFAULT_CONFIG_DIR)
    local = config.models.get("mistral-7b")

    costs = compute_costs(local, config.baseline_model, 812, 64)

    # (812 * 2.0 + 64 * 8.0) / 1e6
    assert costs.baseline_cost_usd == 0.002136
    assert costs.cost_usd == 0
    assert costs.savings_usd == 0.002136


def test_premium_answer_saves_nothing(premium_env, priced_env):
    config = load_config(DEFAULT_CONFIG_DIR)
    premium = config.default_model("premium")

    costs = compute_costs(premium, config.baseline_model, 1000, 500)

    assert costs.cost_usd == costs.baseline_cost_usd == 0.006
    assert costs.savings_usd == 0


def test_prices_come_from_the_environment(premium_env, priced_env):
    premium = load_config(DEFAULT_CONFIG_DIR).default_model("premium")

    assert premium.usd_per_1m_input == 2.0
    assert premium.usd_per_1m_output == 8.0
    assert str(premium.priced_on) == "2026-10-01"
    assert price(premium, 1_000_000, 0) == 2.0
