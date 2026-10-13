import requests
import streamlit as st


CHAT_URL = "http://127.0.0.1:8000/chat"


st.set_page_config(page_title="EuroTrip Agent", page_icon="EU")
st.title("EuroTrip Agent")


def _render_sources(dest: dict) -> None:
    citations = dest.get("citations") or []
    if citations:
        st.markdown(
            "**Sources**  "
            + " · ".join(
                f"[{citation.get('section') or citation.get('title')}]({citation['url']})"
                for citation in citations
                if citation.get("url")
            )
        )


def render_fixed_itinerary(plan: dict) -> None:
    destination = (plan.get("ranked_destinations") or [{}])[0]
    title = destination.get("name", "Destination")
    country = destination.get("country", "")
    cost = destination.get("estimated_cost")

    st.caption(f"Planner mode: {plan.get('planner_mode', 'fixed_destination_itinerary')}")
    st.success(plan["final_recommendation"])

    if plan.get("explanation"):
        st.info(plan["explanation"])

    metric_cols = st.columns(2)
    metric_cols[0].metric("Destination", f"{title}, {country}" if country else title)
    if cost is not None:
        metric_cols[1].metric("Estimated trip cost", f"€{cost}")

    if plan.get("tradeoff_summary"):
        with st.expander("How this was built", expanded=True):
            for item in plan["tradeoff_summary"]:
                st.markdown(f"- {item}")

    st.subheader("Itinerary")
    for day in plan["itinerary"]:
        st.markdown(f"**Day {day['day']}**")
        st.write(day["plan"])

    if destination.get("pros") or destination.get("cons"):
        st.subheader("Evidence notes")
        if destination.get("pros"):
            st.markdown("**Useful retrieved signals**")
            for item in destination["pros"]:
                st.markdown(f"- {item}")
        if destination.get("cons"):
            st.markdown("**Limitations**")
            for item in destination["cons"]:
                st.markdown(f"- {item}")

    _render_sources(destination)

    st.subheader("Risks & caveats")
    for risk in plan["risks"]:
        st.warning(risk, icon="⚠️")

    with st.expander("Raw API response (debug)"):
        st.json(plan)


def render_plan(plan: dict) -> None:
    planner_mode = plan.get("planner_mode", "unknown")
    if planner_mode.startswith("fixed_destination_itinerary"):
        render_fixed_itinerary(plan)
        return

    fallback_reason = plan.get("fallback_reason")
    if planner_mode == "llm_agent":
        st.caption("Planner mode: LLM agent")
    elif planner_mode == "deterministic_fallback":
        st.warning(
            f"Planner mode: deterministic fallback"
            + (f" ({fallback_reason})" if fallback_reason else ""),
            icon="⚠️",
        )
    else:
        st.caption(f"Planner mode: {planner_mode}")

    st.success(plan["final_recommendation"])

    if plan.get("explanation"):
        st.info(plan["explanation"])

    if plan.get("tradeoff_summary"):
        with st.expander("Why this ranking?", expanded=True):
            for item in plan["tradeoff_summary"]:
                st.markdown(f"- {item}")

    if plan.get("filtered_candidates"):
        with st.expander("Filtered candidates"):
            for candidate in plan["filtered_candidates"]:
                reason = candidate.get("filtered_reason") or "Filtered by ranking constraints"
                st.markdown(f"- **{candidate['name']}**: {reason}")

    if planner_mode == "fixed_destination_itinerary":
        st.subheader("Destination")
    else:
        st.subheader("Top destinations")

    for dest in plan["ranked_destinations"]:
        with st.container(border=True):
            col_name, col_score, col_cost, col_home = st.columns([3, 1, 1, 1])
            col_name.markdown(f"### {dest['name']}, {dest['country']}")
            col_score.metric("Score", dest["score"])
            col_cost.metric("Est. cost", f"€{dest['estimated_cost']}")
            if dest.get("estimated_cost_home") and dest.get("home_currency"):
                col_home.metric(
                    dest["home_currency"],
                    f"{dest['estimated_cost_home']:,.0f}",
                )

            col_pros, col_cons = st.columns(2)
            col_pros.markdown("**Pros**\n" + "\n".join(f"- {p}" for p in dest["pros"]))
            col_cons.markdown("**Cons**\n" + "\n".join(f"- {c}" for c in dest["cons"]))
            _render_sources(dest)
            if dest.get("score_breakdown"):
                with st.expander("Score breakdown"):
                    st.json(dest["score_breakdown"])

    st.subheader("Itinerary")

    for day in plan["itinerary"]:
        st.markdown(f"**Day {day['day']} — {day['destination']}**  \n{day['plan']}")

    st.subheader("Risks & caveats")
    for risk in plan["risks"]:
        st.warning(risk, icon="⚠️")

    with st.expander("Raw API response (debug)"):
        st.json(plan)


if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Tell me what you want to plan. For example: "
                "'3 days in Paris for food and art' or "
                "'4 days from Munich in July, budget 1000, nature history, train, medium'."
            ),
        }
    ]
if "chat_extracted_fields" not in st.session_state:
    st.session_state.chat_extracted_fields = {}
if "chat_conversation_id" not in st.session_state:
    st.session_state.chat_conversation_id = None
if "chat_latest_plan" not in st.session_state:
    st.session_state.chat_latest_plan = None
if "chat_latest_trace" not in st.session_state:
    st.session_state.chat_latest_trace = []
if "chat_latest_llm_usage" not in st.session_state:
    st.session_state.chat_latest_llm_usage = []

if st.button("Reset chat"):
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Tell me what you want to plan. For example: "
                "'3 days in Paris for food and art' or "
                "'4 days from Munich in July, budget 1000, nature history, train, medium'."
            ),
        }
    ]
    st.session_state.chat_extracted_fields = {}
    st.session_state.chat_conversation_id = None
    st.session_state.chat_latest_plan = None
    st.session_state.chat_latest_trace = []
    st.session_state.chat_latest_llm_usage = []
    st.rerun()

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_message = st.chat_input("e.g. 3 days in Paris for food and art")
if user_message:
    st.session_state.chat_messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    payload = {
        "message": user_message,
        "conversation_id": st.session_state.chat_conversation_id,
        "extracted_fields": st.session_state.chat_extracted_fields,
    }
    with st.status("Agent is working...", expanded=True) as status:
        st.write("Reading your message and extracting trip fields...")
        st.write("Routing the request to the right planning workflow...")
        st.write("Retrieving travel evidence and building a structured plan...")
        try:
            response = requests.post(CHAT_URL, json=payload, timeout=90)
            response.raise_for_status()
            chat_response = response.json()
        except requests.RequestException as error:
            status.update(label="Agent request failed", state="error")
            st.error(f"Could not reach the chat backend: {error}")
            st.stop()
        status.update(label="Agent finished", state="complete")

    st.session_state.chat_conversation_id = chat_response["conversation_id"]
    st.session_state.chat_extracted_fields = chat_response.get("extracted_fields", {})
    st.session_state.chat_latest_plan = chat_response.get("plan")
    st.session_state.chat_latest_trace = chat_response.get("trace", [])
    st.session_state.chat_latest_llm_usage = chat_response.get("llm_usage", [])
    assistant_message = chat_response["assistant_message"]
    st.session_state.chat_messages.append({"role": "assistant", "content": assistant_message})
    with st.chat_message("assistant"):
        st.markdown(assistant_message)

if st.session_state.chat_extracted_fields:
    with st.expander("Extracted trip details"):
        st.json(st.session_state.chat_extracted_fields)

if st.session_state.chat_latest_trace:
    with st.expander("Execution trace", expanded=True):
        for index, step in enumerate(st.session_state.chat_latest_trace, start=1):
            status = step.get("status", "ok")
            label = step.get("stage", "step")
            detail = step.get("detail", "")
            st.markdown(f"**{index}. {label}** · `{status}`")
            if detail:
                st.caption(detail)

if st.session_state.chat_latest_llm_usage:
    with st.expander("LLM usage and cost"):
        total_cost = sum(item.get("estimated_cost_usd", 0.0) for item in st.session_state.chat_latest_llm_usage)
        total_input = sum(item.get("input_tokens", 0) for item in st.session_state.chat_latest_llm_usage)
        total_output = sum(item.get("output_tokens", 0) for item in st.session_state.chat_latest_llm_usage)
        col_in, col_out, col_cost = st.columns(3)
        col_in.metric("Input tokens", total_input)
        col_out.metric("Output tokens", total_output)
        col_cost.metric("Estimated cost", f"${total_cost:.6f}")
        st.json(st.session_state.chat_latest_llm_usage)

if st.session_state.chat_latest_plan:
    render_plan(st.session_state.chat_latest_plan)
