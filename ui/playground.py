"""Streamlit playground: chat with the router and see why every answer went where it did.

    make playground          # or: uv run streamlit run ui/playground.py
    open http://localhost:8501

Settings come from the sidebar or the environment: ROUTER_API_URL (default
http://localhost:8000), ROUTER_API_KEY and GRAFANA_URL. The key is never shown.
"""

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# `streamlit run` puts ui/ on sys.path, not the repo root that `ui.client` lives under.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.client import (
    DEFAULT_API_URL,
    ApiError,
    FeedbackLedger,
    Health,
    RouterClient,
    RoutingCard,
    RoutingMode,
    conversation_messages,
    format_ms,
    format_share,
    format_usd,
)

# Same tier colours as the Grafana dashboard.
TIER_COLOURS = {"local": "#3987e5", "premium": "#d95926"}
TIER_BADGES = {"local": "blue", "premium": "orange"}
MODES: dict[str, RoutingMode] = {
    "Automatic": "auto",
    "Force local": "local",
    "Force premium": "premium",
}

st.set_page_config(
    page_title="SLM Router Playground", page_icon=":material/alt_route:", layout="wide"
)

state = st.session_state
state.setdefault("history", [])
state.setdefault("ledger", FeedbackLedger())


@st.cache_data(ttl=10, show_spinner=False)
def cached_health(base_url: str, api_key: str | None) -> Health:
    return RouterClient(base_url, api_key, timeout=5).health()


# Sidebar: connection, routing controls, totals ------------------------------------
env_key = os.environ.get("ROUTER_API_KEY") or None
with st.sidebar:
    st.header("Router")
    api_url = st.text_input(
        "API URL",
        os.environ.get("ROUTER_API_URL", DEFAULT_API_URL),
        help="Where this playground's server reaches the router: http://api:8000 inside "
        "Docker Compose, http://localhost:8000 when run on the host.",
    )
    typed_key = st.text_input(
        "API key",
        type="password",
        placeholder="Using ROUTER_API_KEY from the environment" if env_key else "ROUTER_API_KEY",
        help="Sent as Authorization: Bearer. Leave empty to use ROUTER_API_KEY from the env.",
    )
    api_key = typed_key.strip() or env_key

    try:
        client = RouterClient(api_url, api_key)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    health = cached_health(client.base_url, api_key)
    icon = {"ok": ":material/check_circle:", "degraded": ":material/warning:"}.get(
        health.state, ":material/error:"
    )
    message = f"API **{health.state}**: {health.detail}"
    if health.state == "ok":
        st.success(message, icon=icon)
    elif health.state == "degraded":
        st.warning(message, icon=icon)
    else:
        st.error(f"{message}. Is the stack up? `make up`", icon=icon)
    if health.premium_provider == "mock":
        st.info("Premium is the **mock** model: simulated answers, illustrative prices.")
    if st.button("Re-check", icon=":material/refresh:"):
        cached_health.clear()
        st.rerun()
    if not api_key:
        st.warning("No API key: set one above or ROUTER_API_KEY in .env.")

    st.divider()
    mode_label = st.radio(
        "Routing",
        list(MODES),
        help="Forcing sends X-Router-Tier. Privacy, capability and context rules still apply.",
    )
    mode = MODES[mode_label]
    dry_run = st.toggle(
        "Dry run",
        help="POST /v1/route: shows the routing decision without calling any model. Free.",
    )
    max_tokens = st.slider(
        "Max answer tokens",
        min_value=32,
        max_value=1024,
        value=256,
        step=32,
        help="Kept low to save time and cost. 1,000 or more adds output-demand points.",
    )
    if st.button("Clear conversation", icon=":material/delete:", disabled=not state.history):
        state.history = []
        state.ledger = FeedbackLedger()
        st.rerun()

    st.divider()
    st.subheader("Totals")
    try:
        totals = client.stats() if api_key else None
    except ApiError as exc:
        totals = None
        st.caption(f"Stats unavailable: {exc}")
    if totals:
        left, right = st.columns(2)
        left.metric("Requests", totals["requests"])
        right.metric("Local share", format_share(totals["local_share"]))
        left.metric("p95 latency", format_ms(totals["p95_latency_ms"]["all"]))
        right.metric("Fallbacks", format_share(totals["fallback_share"]))
        st.metric("Saved (est.)", format_usd(totals["savings_usd"]))
        st.caption("All requests stored by the API, from any client. Money values are estimates.")
    grafana = os.environ.get("GRAFANA_URL", "http://localhost:3000")
    st.markdown(f"[Open the Grafana dashboard]({grafana})")


# Routing card ------------------------------------------------------------------------
def score_bar(card: RoutingCard) -> str:
    """A score bar with a tick at the threshold. Only numbers are interpolated."""
    colour = TIER_COLOURS.get(card.tier, "#888888")
    return f"""
<div style="position:relative;height:12px;border-radius:6px;background:rgba(128,128,128,0.25);
            margin:4px 0 2px" role="img"
     aria-label="Complexity score {card.score} of 100, threshold {card.threshold}">
  <div style="width:{card.score_fraction * 100:.1f}%;height:100%;border-radius:6px;
              background:{colour}"></div>
  <div style="position:absolute;top:-4px;left:{card.threshold_fraction * 100:.1f}%;width:2px;
              height:20px;background:currentColor;opacity:0.8"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:0.8em;opacity:0.75">
  <span>0</span><span>threshold {card.threshold}</span><span>100</span>
</div>"""


def show_card(card: RoutingCard) -> None:
    with st.container(border=True):
        badge = TIER_BADGES.get(card.tier, "gray")
        flags = "".join(f" :gray-badge[{flag}]" for flag in card.flags)
        st.markdown(
            f":{badge}-badge[{card.tier.upper()}] **{card.model_id}**"
            f"{f' ({card.model})' if card.model else ''} · task **{card.task_type}**{flags}"
        )
        st.markdown(
            f"Complexity **{card.score}** / 100 against threshold **{card.threshold}**: "
            f"{card.reason_label}."
        )
        st.markdown(score_bar(card), unsafe_allow_html=True)

        latency, cost, baseline, savings = st.columns(4)
        if card.dry_run:
            latency.metric("Latency", "n/a", help="Dry run: no model was called.")
            cost.metric(
                "Input cost (est.)",
                format_usd(card.estimated_input_cost_usd),
                help="Input tokens only; the answer has not been written.",
            )
            baseline.metric(
                "Baseline input (est.)", format_usd(card.estimated_baseline_input_cost_usd)
            )
            savings.metric("Savings", "n/a", help="Known once an answer exists.")
        else:
            latency.metric("Latency", format_ms(card.latency_ms))
            cost.metric("Cost (est.)", format_usd(card.cost_usd))
            baseline.metric("Baseline (est.)", format_usd(card.baseline_cost_usd))
            savings.metric("Savings (est.)", format_usd(card.savings_usd))

        st.caption(card.summary)
        with st.expander("Signals and details"):
            rows = "\n".join(f"| {name} | {points} |" for name, points in card.signal_rows)
            st.markdown(
                f"| Signal | Points |\n|---|---:|\n{rows}\n| **total** | **{card.score}** |"
            )
            details = {
                "request_id": card.request_id,
                "fallback": card.fallback_from
                and f"from {card.fallback_from} ({card.fallback_cause})",
                "cache_hit": card.cache_hit,
                "pii_detected": card.pii_detected,
                "pii_kinds": card.pii_kinds,
                "tokens": None
                if card.dry_run
                else {"input": card.input_tokens, "output": card.output_tokens},
                "input_tokens_estimate": card.input_tokens_estimate,
            }
            st.json(details, expanded=True)
            if card.rules:
                st.markdown("**Hard rules, in order**")
                st.markdown(
                    "\n".join(f"- `{r['rule']}` {r['outcome']}: {r['detail']}" for r in card.rules)
                )


def show_feedback(entry: dict[str, Any], index: int) -> None:
    card: RoutingCard = entry["card"]
    ledger: FeedbackLedger = state.ledger
    if not ledger.can_rate(card.request_id):
        rating = ledger.ratings[card.request_id]
        st.caption(
            f"Thanks, rated {':material/thumb_up:' if rating == 1 else ':material/thumb_down:'}"
        )
        return
    comment = st.text_input(
        "Comment (optional)", key=f"comment-{index}", placeholder="What was good or wrong?"
    )
    up, down, _ = st.columns([1, 1, 6])
    for column, rating, label in (
        (up, 1, ":material/thumb_up:"),
        (down, -1, ":material/thumb_down:"),
    ):
        if column.button(label, key=f"rate-{rating}-{index}", help="Rate this answer"):
            # Recorded before the call, so a fast double click cannot send twice.
            ledger.record(card.request_id, rating)
            try:
                client.feedback(card.request_id, rating, comment)
            except ApiError as exc:
                del ledger.ratings[card.request_id]
                st.error(f"Feedback not saved: {exc}")
            else:
                st.rerun()


# Conversation --------------------------------------------------------------------------
st.title("SLM Router Playground")
st.caption(
    "Each prompt goes to the cheapest model that can handle it. The card under every answer "
    "explains the decision."
)

for index, entry in enumerate(state.history):
    with st.chat_message(entry["role"]):
        if entry.get("error"):
            st.error(entry["error"])
            continue
        if entry["role"] == "user":
            if entry.get("kind") == "dry_run":
                st.caption("Dry run")
            st.markdown(entry["content"])
            continue
        if entry.get("kind") == "dry_run":
            st.markdown("_Dry run: this is where the prompt would go. No model was called._")
        else:
            st.markdown(entry["content"])
        show_card(entry["card"])
        if entry.get("kind") == "chat":
            show_feedback(entry, index)

prompt = st.chat_input("Ask something, e.g. 'Hi! What can you do?'", disabled=not api_key)
if prompt:
    messages = [*conversation_messages(state.history), {"role": "user", "content": prompt}]
    user_entry: dict[str, Any] = {
        "role": "user",
        "content": prompt,
        "kind": "dry_run" if dry_run else "chat",
    }
    state.history.append(user_entry)
    try:
        with st.spinner("Routing…" if dry_run else "Waiting for the model…"):
            if dry_run:
                card = client.dry_run(messages, mode, max_tokens)
                state.history.append(
                    {"role": "assistant", "content": "", "card": card, "kind": "dry_run"}
                )
            else:
                result = client.chat(messages, mode, max_tokens)
                state.history.append(
                    {
                        "role": "assistant",
                        "content": result.content,
                        "card": result.card,
                        "kind": "chat",
                    }
                )
    except (ApiError, ValueError) as exc:
        user_entry["kind"] = "unanswered"
        state.history.append({"role": "assistant", "error": str(exc), "kind": "unanswered"})
    st.rerun()
