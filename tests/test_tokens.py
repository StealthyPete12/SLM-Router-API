from app.config import AppConfig
from app.router.tokens import TokenEstimator


def test_message_estimate_adds_fixed_overheads(config: AppConfig):
    estimator = TokenEstimator(config.routing.tokens)
    tokens = config.routing.tokens

    one = estimator.estimate_messages(["Hi! What can you do?"])
    two = estimator.estimate_messages(["Hi! What can you do?", "Hi!"])

    assert one == estimator.count("Hi! What can you do?") + tokens.per_message_overhead + 3
    assert two - one == estimator.count("Hi!") + tokens.per_message_overhead


def test_estimate_grows_with_input(config: AppConfig):
    estimator = TokenEstimator(config.routing.tokens)

    assert estimator.count("word " * 1000) > estimator.count("word " * 100) > 0


def test_special_token_text_is_counted_not_rejected(config: AppConfig):
    assert TokenEstimator(config.routing.tokens).count("<|endoftext|>") > 0
