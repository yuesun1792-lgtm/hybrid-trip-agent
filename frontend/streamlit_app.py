import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/plan"


st.set_page_config(page_title="EuroTrip Agent", page_icon="EU")
st.title("EuroTrip Agent")

with st.form("trip_form"):
    origin_city = st.text_input("Origin city", value="Munich")
    date_range = st.text_input("Date range", value="June 10-15")
    trip_length = st.number_input("Trip length", min_value=1, max_value=30, value=4)
    budget = st.number_input("Budget", min_value=1, value=700)
    interests = st.multiselect(
        "Interests",
        options=["food", "history", "art", "nature", "nightlife", "architecture"],
        default=["food", "history"],
    )
    transport_preference = st.selectbox("Transport preference", ["train", "flight", "bus"])
    physical_level = st.selectbox("Physical level", ["low", "medium", "high"])

    submitted = st.form_submit_button("Plan trip")

if submitted:
    payload = {
        "origin_city": origin_city,
        "date_range": date_range,
        "trip_length": int(trip_length),
        "budget": int(budget),
        "interests": interests,
        "transport_preference": transport_preference,
        "physical_level": physical_level,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as error:
        st.error(f"Could not reach the backend: {error}")
        st.stop()

    plan = response.json()

    # ── Recommendation ──────────────────────────────────────────────────────
    st.success(plan["final_recommendation"])

    # ── LLM explanation (present only when ANTHROPIC_API_KEY is configured) ─
    if plan.get("explanation"):
        st.info(plan["explanation"])

    # ── Ranked destinations ──────────────────────────────────────────────────
    st.subheader("Top 3 destinations")

    for dest in plan["ranked_destinations"]:
        with st.container(border=True):
            col_name, col_score, col_cost = st.columns([3, 1, 1])
            col_name.markdown(f"### {dest['name']}, {dest['country']}")
            col_score.metric("Score", dest["score"])
            col_cost.metric("Est. cost", f"€{dest['estimated_cost']}")

            col_pros, col_cons = st.columns(2)
            col_pros.markdown("**Pros**\n" + "\n".join(f"- {p}" for p in dest["pros"]))
            col_cons.markdown("**Cons**\n" + "\n".join(f"- {c}" for c in dest["cons"]))

    # ── Itinerary ────────────────────────────────────────────────────────────
    st.subheader("Itinerary")

    for day in plan["itinerary"]:
        st.markdown(f"**Day {day['day']} — {day['destination']}**  \n{day['plan']}")

    # ── Risks ────────────────────────────────────────────────────────────────
    st.subheader("Risks & caveats")
    for risk in plan["risks"]:
        st.warning(risk, icon="⚠️")

    # ── Raw JSON for debugging ───────────────────────────────────────────────
    with st.expander("Raw API response (debug)"):
        st.json(plan)
