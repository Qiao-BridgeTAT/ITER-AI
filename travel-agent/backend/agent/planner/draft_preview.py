"""Project the latest saved draft without blessing its feasibility or publishing it."""

from backend.agent.planner.react_review import evidence_digest
from backend.contracts.v4.enums import PlannerStatus
from backend.contracts.v4.planner_preview import (
    PlannerDraftDayPreview,
    PlannerDraftPreview,
    PlannerDraftStopPreview,
)
from backend.contracts.v4.planner_refs import CandidateRef, candidate_ref_key
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState

_PERIODS = {
    "morning": "上午",
    "midday": "中午",
    "afternoon": "下午",
    "evening": "晚上",
    "anytime": "时段待定",
}
_MEALS = {"breakfast": "早餐", "lunch": "午餐", "dinner": "晚餐", "snack": "加餐"}
_MODES = {"public_transit": "公共交通", "taxi": "打车", "walking": "步行", "driving": "自驾"}


def build_draft_preview(workspace: PlannerWorkspaceState) -> PlannerDraftPreview | None:
    draft = workspace.working_itinerary
    if draft is None or workspace.status is PlannerStatus.STALE:
        return None
    names = {
        candidate_ref_key(item.candidate_ref): item.display_name
        for item in workspace.candidate_pool.candidates
    }
    days = []
    for day in draft.days:
        stops = []
        for item in day.ordered_items:
            if isinstance(item.object_ref, CandidateRef):
                title = names[candidate_ref_key(item.object_ref)]
            else:
                title = {"arrival": "已确认的抵达安排", "departure": "已确认的返程安排"}.get(
                    item.item_kind, "已确认的预订安排"
                )
            hint = _PERIODS[item.expected_window.part_of_day]
            if item.meal_slot:
                hint += f" · {_MEALS[item.meal_slot]}"
            if item.onsite_lunch:
                hint += " · 含园内午餐意向"
            stops.append(
                PlannerDraftStopPreview(
                    draft_item_id=item.draft_item_id, title=title, time_hint=hint
                )
            )
        days.append(
            PlannerDraftDayPreview(
                service_date=day.service_date,
                theme=day.day_theme,
                transport_summary="、".join(_MODES[mode] for mode in day.transport_preferences),
                stops=tuple(stops),
            )
        )
    lodging = {
        "not_applicable": "本次不安排住宿",
        "fixed": "沿用任务书中已确认的住宿",
        "unresolved": "住宿尚待确定",
        "selected_offer": "所选住宿尚待核对",
    }[draft.lodging_baseline.mode]
    if workspace.hotel_observation and draft.lodging_baseline.selected_offer_ref:
        lodging = next(
            (
                offer.property_name
                for offer in workspace.hotel_observation.offers
                if offer.offer_ref == draft.lodging_baseline.selected_offer_ref
            ),
            lodging,
        )
    validation = workspace.validation_observation
    current_validation = bool(
        validation
        and validation.draft_id == draft.draft_id
        and validation.draft_revision == draft.draft_revision
    )
    review = workspace.react_state.review if workspace.react_state else None
    current_review = bool(
        review
        and review.draft_revision == draft.draft_revision
        and review.draft_digest == draft.content_digest
        and review.evidence_digest == evidence_digest(workspace)
        and current_validation
        and validation
        and review.validation_fingerprint == validation.validation_fingerprint
    )
    issues: list[str] = []
    if review and current_review:
        issues.extend(
            f"{issue.target}：{issue.description} 建议：{issue.suggestion}"[:2000]
            for issue in review.verdict.issues
        )
    elif workspace.react_state:
        issues.append("当前草稿尚未完成有效的独立评审。")
    if validation and current_validation:
        issues.extend(issue.message_summary for issue in validation.issues)
    else:
        issues.append("当前草稿的路线、时间与费用尚未完成核对。")
    if draft.unassigned_intents:
        issues.append("仍有已表达的旅行意愿未安排进这份草稿。")
    notice = "先保留最近一版安排供你参考，具体时间、路线和费用仍需核对。"
    if workspace.status is PlannerStatus.CANCELLED:
        notice = "规划已暂停，以下是暂停前保存的安排，仍需继续核对。"
    elif review and current_review and not review.verdict.accepted:
        notice = "已生成一版安排，检查仍有待修改的问题，先保留给你参考。"
    return PlannerDraftPreview(
        draft_id=draft.draft_id,
        draft_revision=draft.draft_revision,
        content_digest=draft.content_digest,
        notice=notice,
        days=tuple(days),
        lodging_summary=lodging,
        unresolved_issues=tuple(dict.fromkeys(issues)),
    )


def draft_preview_response(workspace: PlannerWorkspaceState) -> str:
    """No model/Provider call is needed to retain an already saved result."""
    if build_draft_preview(workspace) is None:
        raise ValueError("draft preview requires a saved itinerary")
    return (
        "已保留一版待确认草稿，可以先查看下方每天的安排。"
        "部分问题尚未解决，还没有形成正式行程；你可以继续规划，或补充希望调整的内容。"
    )
