"""Shared lodging evidence status for model feedback and honest result notices."""

from backend.agent.planner.proposals import PlannerReferenceCatalog
from backend.agent.planner.workspace import PlannerGuardError
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.contracts.v4.task_book import TaskBookV4


def hotel_query_status(workspace: PlannerWorkspaceState) -> str:
    observation = workspace.hotel_observation
    if observation is None:
        return "not_queried"
    if observation.query_status is not None:
        return observation.query_status
    if observation.offers:
        return "available"
    if "provider_city_binding" in observation.missing_fact_kinds:
        return "failed"
    # Old checkpoints lack query receipts: do not invent a provider failure or empty response.
    return "unknown"


def hotel_gap_message(workspace: PlannerWorkspaceState) -> str:
    status = hotel_query_status(workspace)
    messages = {
        "not_queried": "本次尚未查询和选择酒店，行程中的住宿未安排。",
        "empty": "已按本次住宿条件查询酒店，查询正常返回但没有结果，住宿未安排。",
        "failed": "本次酒店查询或位置核验失败，尚未选定酒店，住宿未安排。",
        "unverified": "酒店查询返回了候选，但尚无符合约束且通过位置核验的酒店，住宿未安排。",
        "available": "已有酒店候选，但本次尚未选择酒店，住宿未安排。",
        "unknown": "历史酒店记录缺少查询明细，无法确认是无结果还是查询失败，住宿尚待补充。",
    }
    message = messages[status]
    observation = workspace.hotel_observation
    if status == "failed" and observation:
        codes = sorted({a.error_code for a in observation.query_attempts if a.error_code})
        if codes:
            message += "错误类型：" + "、".join(codes) + "。"
    return message


def require_hotel_action(workspace: PlannerWorkspaceState, book: TaskBookV4) -> None:
    """Correct normal completion; emergency result delivery can still disclose gaps."""
    if book.lodging_direction.not_applicable or book.lodging_direction.existing_booking:
        return
    if hotel_query_status(workspace) == "not_queried":
        raise PlannerGuardError(
            "planner_hotel_not_queried:next=search_hotels:"
            "尚未查询酒店；请查询并选择一家酒店后再完成，参考价可用，不要求房态。"
        )
    catalog = PlannerReferenceCatalog(workspace)
    if catalog.selectable_hotels and catalog.current_hotel_key is None:
        raise PlannerGuardError(
            "planner_plan_hotel_selection_required:path=selected_hotel_key:"
            f"allowed_values={','.join(catalog.selectable_hotels)}:"
            "已有可选酒店；请用 edit_plan 选择一家并重新评审。"
        )
