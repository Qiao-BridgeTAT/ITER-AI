"""Read-only plan recovery from the owner's committed conversation history.

This is a presentation projection, never the authoritative plan for mutations.
Changing the task book still invalidates trip_state.published_plan normally.
"""

from backend.contracts.v4.conversation import ConversationSnapshotV4
from backend.contracts.v4.planner_publication import PlannerPublishedPlan


def visible_published_plan(snapshot: ConversationSnapshotV4) -> PlannerPublishedPlan | None:
    if snapshot.trip_state.published_plan is not None:
        return snapshot.trip_state.published_plan
    state = snapshot.trip_state.semantic_state
    candidates = (
        (message.state_version, plan.published_at, str(plan.plan_version_id), plan)
        for message in snapshot.messages
        if message.role == "assistant"
        and message.status == "committed"
        and str(message.trip_id) == str(state.trip_id)
        and message.state_version <= state.state_version
        for attachment in message.attachments
        if isinstance(plan := attachment.root, PlannerPublishedPlan)
        and str(plan.trip_id) == str(state.trip_id)
        and plan.based_on_state_version <= message.state_version
    )
    latest = max(candidates, key=lambda item: item[:3], default=None)
    return latest[3] if latest else None
