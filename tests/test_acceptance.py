"""Phase 1 gate: the hand-picked prompts land on their expected route (dry run only)."""

import pytest

from scripts.acceptance import CASES, check
from tests.conftest import no_network


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_acceptance_prompt(make_client, case):
    response = make_client(no_network).post(
        "/v1/route",
        json={"model": case.model, "messages": [{"role": "user", "content": case.prompt}]},
        headers=case.headers,
    )

    assert response.status_code == 200
    assert check(case, response.json()["router"]) == []
