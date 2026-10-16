"""Controlled scheduler shared by discovery, confirmed destinations and revisions.

Nodes are ordinary functions with typed inputs/outputs. Dynamic decisions are bounded
inside discovery and itinerary generation; validation and commits stay deterministic.
"""
import json

from backend.app.config import llm_rag_enhancements_enabled, research_max_candidates, research_max_rounds, research_queries_per_round
from backend.app.planning.destination_analysis import analyze_destination
from backend.app.domain.destination_names import canonical_destination
from backend.app.runtime.execution import RUN_CONTEXT, RunContext, BudgetExceeded
from backend.app.domain.destination_names import COUNTRY_BY_DESTINATION
from backend.app.llm.llm import generate_text
from backend.app.domain.models import CandidateDestination, Citation, DestinationAnalysis, DestinationSuggestion, EvidenceChunk, TripPlan
from backend.app.planning.plan_builder import build_itinerary, BuildResult
from backend.app.planning.plan_renderer import render_plan
from backend.app.planning.plan_store import PLAN_STORE, PlanVersionConflict
from backend.app.planning.plan_validation import validate_plan
from backend.app.places.place_resolution import resolve_plan_places
from backend.app.domain.planning_models import PlanningTask
from backend.app.planning.revision import merge_revision, RevisionNeedsInput
from backend.app.tool_calling.tool_executor import default_executor
from backend.app.runtime.trajectory import observed


@observed("candidate_resolution", "discovery")
def resolve_candidates(task: PlanningTask, seed_candidates=None, allow_research=True):
    if task.candidate_destinations:
        names = list(dict.fromkeys(canonical_destination(name) for name in task.candidate_destinations))
        seeds = {candidate.name: candidate for candidate in seed_candidates or []}
        return [seeds.get(name, CandidateDestination(name=name, source="user")) for name in names[:research_max_candidates()]]
    if seed_candidates:
        return [c for c in seed_candidates if c.name not in task.rejected_destinations][:research_max_candidates()]
    executor = default_executor()
    constraints = task.constraints
    interests = " ".join(term for term in constraints.interests if term.isascii()) or "local highlights"
    queries = [f"Europe destinations {interests} {constraints.trip_length or ''} days",
               f"{constraints.origin_city or 'Europe'} {constraints.transport_preference or 'travel'} {interests} destinations"]
    grouped = {}
    searched = set()
    for round_number in range(research_max_rounds()):
        for query in queries[:research_queries_per_round()]:
            if not query.isascii() or query in searched:
                continue
            searched.add(query)
            result = executor.execute("search_travel_knowledge", {"query": query, "k": 8})
            for raw in result.data or []:
                name = canonical_destination(raw.get("destination", ""))
                if not name or name in task.rejected_destinations:
                    continue
                chunk = EvidenceChunk.model_validate({**raw, "destination": name})
                grouped.setdefault(name, {})[chunk.evidence_id] = chunk
        if len(grouped) >= 3 or not allow_research or not llm_rag_enhancements_enabled() or round_number == research_max_rounds() - 1:
            break
        try:
            prompt = f"""Propose at most {research_queries_per_round()} NEW ENGLISH search queries for an English travel corpus.
Return only JSON {{"queries": [...]}}. Do not answer travel facts or choose a destination.
Constraints: {constraints.model_dump_json()}
Already searched: {json.dumps(sorted(searched))}
Found destinations: {json.dumps(list(grouped))}"""
            response = generate_text(prompt, stage="research_controller", max_tokens=400, prompt_version="candidate_research_v1")
            queries = json.loads(response.text).get("queries", [])
            if not isinstance(queries, list) or any(not isinstance(q, str) for q in queries):
                raise ValueError("Research queries must be strings")
        except Exception:
            break
    candidates = [CandidateDestination(name=name, source="rag", evidence=list(chunks.values()),
        score=max((c.score or 0 for c in chunks.values()), default=0)) for name, chunks in grouped.items()]
    candidates.sort(key=lambda c: c.score or 0, reverse=True)
    if not candidates:
        # Curated names seed the same analysis pipeline; no independent fallback planner.
        from backend.app.legacy.planner import load_destination_notes
        candidates = [CandidateDestination(name=note["name"], country=note["country"], source="curated_fallback")
                      for note in load_destination_notes() if note["name"] not in task.rejected_destinations]
    return candidates[:research_max_candidates()]


@observed("candidate_comparison", "selection")
def compare_candidates(analyses, constraints, confirmed=False):
    ranked = sorted(analyses, key=lambda a: a.score, reverse=True)
    rejected = []
    if not confirmed and constraints.budget is not None:
        rejected = [a for a in ranked if a.estimated_cost is not None and a.estimated_cost > constraints.budget]
        ranked = [a for a in ranked if a not in rejected]
    return ranked, rejected


def citations_for(evidence, refs=None):
    selected = set(refs) if refs is not None else None
    citations = {}
    for chunk in evidence:
        if not chunk.url or (selected is not None and chunk.evidence_id not in selected):
            continue
        key = (chunk.url, chunk.section)
        if key not in citations:
            citations[key] = Citation(title=f"{chunk.destination} - {chunk.section}", url=chunk.url,
                section=chunk.section, snippet=chunk.text.split("\n", 1)[-1][:250], evidence_ids=[])
        citations[key].evidence_ids.append(chunk.evidence_id)
    return list(citations.values())


def suggestion_for(analysis, build=None, refs=None):
    return DestinationSuggestion(name=analysis.destination, country=COUNTRY_BY_DESTINATION.get(analysis.destination, "Unknown"),
        score=round(analysis.score), score_breakdown=analysis.score_breakdown,
        estimated_cost=analysis.estimated_cost, tool_signals=analysis.tool_signals,
        pros=(build.pros if build and build.pros else [f"Retrieved themes: {', '.join(analysis.activity_themes) or 'evidence unavailable'}"]),
        cons=(build.cons if build and build.cons else []) + analysis.gaps + ["Verify live logistics before booking."],
        citations=citations_for(analysis.evidence, refs))


@observed("unified_planning_workflow", "workflow")
def execute_task(task: PlanningTask, base_plan: TripPlan | None = None, seed_candidates=None, store=None) -> TripPlan:
    context_token = None
    if RUN_CONTEXT.get() is None:
        context_token = RUN_CONTEXT.set(RunContext())
    store = store or PLAN_STORE
    try:
        if task.operation == "revise":
            return execute_revision(task, base_plan, store)
        candidates = resolve_candidates(task, seed_candidates)
        if not candidates:
            raise RevisionNeedsInput("No eligible destinations remain. Adjust constraints or rejected destinations.", "destination")
        analyses = [analyze_destination(c.name, task.constraints, "overview", c.evidence) for c in candidates]
        ranked, rejected = compare_candidates(analyses, task.constraints, bool(task.constraints.destination))
        if not ranked:
            raise RevisionNeedsInput("No candidate fits the estimated budget; adjust budget or trip length.", "budget")
        build = None
        evidence = []
        itinerary = []
        detailed = None
        if task.requested_output == "itinerary":
            selected = ranked[0]
            detailed = analyze_destination(selected.destination, task.constraints, "detailed", selected.evidence,
                                           task.queries if task.constraints.destination else None)
            build = build_itinerary(detailed, task.constraints, "en", requirements=task.user_message)
            itinerary = build.days
            evidence = detailed.evidence
            analyses = [detailed if a.destination == detailed.destination else a for a in analyses]
        else:
            evidence = [c for a in ranked for c in a.evidence]
        refs = {ref for day in itinerary for activity in day.activities for ref in activity.evidence_ids}
        suggestions = [suggestion_for(detailed if detailed and a.destination == detailed.destination else a,
                                      build if i == 0 else None, refs if i == 0 and itinerary else None)
                       for i, a in enumerate(ranked[:3])]
        fallback = build.fallback_reason if build else None
        if any(c.source == "curated_fallback" for c in candidates):
            fallback = fallback or "rag_discovery_empty_using_curated_names"
        output = TripPlan(constraints=task.constraints, requested_output=task.requested_output,
            ranked_destinations=suggestions, selected_destinations=[ranked[0].destination] if itinerary else [], itinerary=itinerary, evidence=evidence, candidate_analyses=analyses,
            risks=["Opening hours and travel connections require live verification.", "Costs are curated estimates, not bookings."],
            final_recommendation=f"{'Compare destinations including' if task.requested_output == 'options' else 'Use the itinerary for'} {ranked[0].destination}.",
            explanation=build.explanation if build and build.explanation else "Candidates were analyzed under the same constraints; only selected destinations receive a full itinerary.",
            planner_mode="unified_options" if not itinerary else ("unified_itinerary_llm" if build.llm_used else "unified_itinerary_partial" if build.generation_mode == "mixed" else "unified_itinerary_rule"),
            fallback_reason=fallback,
            tradeoff_summary=["Candidate overviews share the same analysis capability as detailed itinerary research.",
                              "Unselected candidates receive overview analysis, not complete daily schedules."],
            filtered_candidates=[CandidateDestination(name=a.destination, score=a.score, evidence=a.evidence,
                                                     filtered_reason="Estimated cost exceeds budget") for a in rejected])
        RUN_CONTEXT.get().remaining()
        output = resolve_plan_places(output)
        output = validate_plan(render_plan(output))
        return store.commit(output)
    finally:
        if context_token is not None:
            RUN_CONTEXT.reset(context_token)


@observed("revision_workflow", "workflow")
def execute_revision(task, base, store):
    if base is None or task.revision is None or task.affected_scope is None:
        raise RevisionNeedsInput("A base plan and a validated revision scope are required.")
    current = store.latest(base.plan_id)
    if current is None or current.version != base.version:
        raise PlanVersionConflict("Base plan is not the current committed version.")
    scope = task.affected_scope
    constraints = task.constraints
    revision = task.revision
    # Reuse surviving evidence and immutable records. Reanalyze only when changes need new activities.
    needs_generation = scope.full_replan or any(change.action in {"replace", "add"} for change in revision.changes)
    analyses = base.candidate_analyses
    evidence = {c.evidence_id: c for c in base.evidence}
    build = None
    replacements = []
    selected_destination = constraints.destination or (base.itinerary[0].destination if base.itinerary else None)
    if "destination" in revision.constraints_updates and revision.constraints_updates["destination"] is None:
        candidates = resolve_candidates(task)
        overviews = [analyze_destination(c.name, constraints, "overview", c.evidence) for c in candidates]
        ranked, _ = compare_candidates(overviews, constraints)
        if not ranked:
            raise RevisionNeedsInput("No replacement destination fits the constraints.", "destination")
        selected_destination = ranked[0].destination
        analyses = overviews
    if not selected_destination:
        raise RevisionNeedsInput("Choose a destination before revising an itinerary.", "destination")
    if needs_generation:
        categories = [c.category for c in revision.changes if c.category]
        research_constraints = constraints.model_copy(update={"interests": list(dict.fromkeys([*constraints.interests, *categories]))})
        previous = [c for c in base.evidence if c.destination == selected_destination]
        analysis = analyze_destination(selected_destination, research_constraints, "detailed", previous)
        if scope.full_replan:
            numbers = list(range(1, (constraints.trip_length or len(base.itinerary)) + 1))
            build = build_itinerary(analysis, constraints, "en", day_numbers=numbers,
                                   requirements=json.dumps([c.model_dump() for c in revision.changes], ensure_ascii=False))
            replacements = build.days
        else:
            builds = []
            for number in scope.day_numbers:
                day = next(d for d in base.itinerary if d.day == number)
                day_ids = {a.activity_id for a in day.activities}
                changes = [c for c in revision.changes if c.action in {"replace", "add"} and
                           (number in c.day_numbers or bool(day_ids & set(c.activity_ids)))]
                if not changes:
                    continue
                slots = sorted({a.time_slot for a in day.activities if a.activity_id in scope.activity_ids and
                                any((not c.activity_ids or a.activity_id in c.activity_ids) and
                                    (c.time_slot is None or c.time_slot == a.time_slot) for c in changes)})
                slots = list(dict.fromkeys([*slots, *(c.time_slot or "flexible" for c in changes if c.action == "add")]))
                requested_categories = {c.category for c in changes if c.category}
                # Replacing N scoped records cannot regenerate protected records as
                # additional replacements. Extra activities use the explicit add operation.
                activity_limit = len(day_ids & set(scope.activity_ids)) + sum(c.action == "add" for c in changes)
                result = build_itinerary(analysis, constraints, "en", day_numbers=[number], time_slots=slots,
                    requirements=json.dumps([c.model_dump() for c in changes], ensure_ascii=False),
                    category=next(iter(requested_categories)) if len(requested_categories) == 1 else None,
                    activity_limit=activity_limit)
                builds.append(result)
                replacements.extend(result.days)
            build = BuildResult(days=replacements, llm_used=bool(builds) and all(b.llm_used for b in builds),
                fallback_reason="; ".join(dict.fromkeys(b.fallback_reason for b in builds if b.fallback_reason)) or None,
                pros=[p for b in builds for p in b.pros][:3], cons=[c for b in builds for c in b.cons][:3],
                explanation=" ".join(b.explanation for b in builds))
        evidence.update({c.evidence_id: c for c in analysis.evidence})
        analyses = [a for a in analyses if a.destination != analysis.destination] + [analysis]
    days = merge_revision(base, replacements, revision, scope, constraints)
    refs = {ref for day in days for activity in day.activities for ref in activity.evidence_ids}
    retained = [chunk for identity, chunk in evidence.items() if identity in refs]
    from backend.app.planning.costs import estimate_trip_cost, load_daily_costs
    suggestions = []
    for index, suggestion in enumerate(base.ranked_destinations[:1]):
        cost = estimate_trip_cost(selected_destination, constraints.trip_length or len(days), load_daily_costs())
        suggestions.append(suggestion.model_copy(update={"name": selected_destination,
            "country": COUNTRY_BY_DESTINATION.get(selected_destination, "Unknown"), "estimated_cost": cost,
            "citations": citations_for(retained), "pros": build.pros if build and build.pros else suggestion.pros,
            "cons": build.cons if build and build.cons else suggestion.cons}))
    updated = base.model_copy(deep=True, update={"constraints": constraints, "itinerary": days,
        "evidence": retained, "candidate_analyses": analyses, "ranked_destinations": suggestions, "selected_destinations": [selected_destination],
        "explanation": build.explanation if build and build.explanation else "Updated only the affected scope and revalidated the whole plan.",
        "final_recommendation": "Updated itinerary; unaffected activities were preserved.",
        "planner_mode": "unified_itinerary_revised", "fallback_reason": build.fallback_reason if build else None,
        "tradeoff_summary": [*base.tradeoff_summary, *scope.reasons]})
    automatically_preserved = [a.activity_id for day in base.itinerary for a in day.activities
                               if a.activity_id not in scope.activity_ids and not scope.full_replan]
    RUN_CONTEXT.get().remaining()
    updated = resolve_plan_places(updated, base)
    updated = validate_plan(render_plan(updated), base, list(set(scope.preserve_activity_ids + automatically_preserved)))
    return store.commit(updated, expected_version=base.version)
