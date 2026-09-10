"""Compile raw Planner failures into narrow, privacy-safe repair instructions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from backend.agent.planner.proposals import PlannerReferenceCatalog
from backend.agent.planner.workspace import server_id
from backend.contracts.v4.planner_evidence import PlannerGuardViolation
from backend.contracts.v4.planner_workspace import PlannerWorkspaceState
from backend.persistence.outbox_repository import canonical_json_hash

GuardStage = Literal[
    "schema",
    "reference",
    "business",
    "materialize",
    "validate",
    "final",
]


@dataclass(frozen=True)
class GuardRule:
    field_path: str
    expected_rule: str
    option_source: Literal[
        "none",
        "candidates",
        "plan_candidates",
        "clusters",
        "routes",
        "hotels",
        "alternative_hotels",
        "evidence",
        "issues",
        "validation_issues",
        "draft_objects",
        "dates",
    ] = "none"
    allowed_next_actions: tuple[str, ...] = ("retry_current_action",)
    preserve_paths: tuple[str, ...] = (
        "confirmed_task_book",
        "current_strategy",
        "accepted_observations",
    )
    minimal_safe_fragment: dict[str, JsonValue] | None = None


_RULES: tuple[tuple[str, GuardRule], ...] = (
    (
        "planner_repair_outside_local_change_dates",
        GuardRule(
            field_path="choices",
            expected_rule="只修改本轮局部请求已涉及的日期及相邻路线；其他日期的地点、顺序和时段保留。",
            allowed_next_actions=("revise_draft",),
        ),
    ),
    (
        "planner_repair_replacement_kind_mismatch",
        GuardRule(
            field_path="choices[].replacement_key",
            expected_rule="景点替换为景点，餐厅替换为餐厅，保留原餐次；区域可以不同，不需要跨簇字段。",
            option_source="plan_candidates",
            allowed_next_actions=("revise_draft",),
        ),
    ),
    (
        "planner_repair_target_outside_current_issue",
        GuardRule(
            field_path="choice.target_key",
            expected_rule="只修改本次问题明确关联的对象，不要删除其他已正确安排的景点来尝试绕过冲突。",
            allowed_next_actions=("revise_draft",),
        ),
    ),
    (
        "planner_unknown_validation_issue_key",
        GuardRule(
            field_path="issue_keys",
            expected_rule=(
                "只复制当前待修复问题的完整 v 短键，例如 v1；不能只填 v、旧编号或内部 ID。"
            ),
            option_source="validation_issues",
            allowed_next_actions=("revise_draft", "request_evidence"),
        ),
    ),
    (
        "planner_repair_target_not_in_current_draft",
        GuardRule(
            field_path="choice.target_key",
            expected_rule="从当前已安排对象中复制完整 c/f 短键，保留数字编号；不得猜测对象。",
            option_source="draft_objects",
            allowed_next_actions=("revise_draft",),
        ),
    ),
    (
        "planner_repair_omit_requires_optional_candidate",
        GuardRule(
            field_path="choice.target_key",
            expected_rule=(
                "omit_soft 仅可移除 soft/filler/neutral 候选；"
                "不得删除 strong/immutable。优先保留用户想去项。"
            ),
            allowed_next_actions=("revise_draft",),
        ),
    ),
    (
        "planner_transport_selection_conflicts_with_schedule",
        GuardRule(
            field_path="transport_mode",
            expected_rule=(
                "该交通选择与当天时间或硬约束冲突，旧行程保持不变。"
                "说明具体冲突路段和时间，请用户选择其他交通方式或明确授权调整日程。"
            ),
            allowed_next_actions=("ask_user",),
        ),
    ),
    (
        "planner_repair_no_effective_change",
        GuardRule(
            field_path="choice.hotel_offer_key",
            expected_rule=(
                "change_hotel 必须选择与当前住宿不同的可用 h 短键；没有合法备选时不得原样重试。"
            ),
            option_source="alternative_hotels",
            allowed_next_actions=("revise_draft", "ask_user"),
            minimal_safe_fragment={
                "choice": {
                    "operation": "change_hotel",
                    "hotel_offer_key": "<allowed_h_key>",
                }
            },
        ),
    ),
    (
        "planner_plan_hotel_selection_forbidden",
        GuardRule(
            field_path="selected_hotel_key",
            expected_rule="当前住宿模式不从酒店候选中选择，selected_hotel_key 必须为 null。",
            allowed_next_actions=("materialize_draft",),
            minimal_safe_fragment={"selected_hotel_key": None},
        ),
    ),
    (
        "planner_plan_hotel_selection_required",
        GuardRule(
            field_path="selected_hotel_key",
            expected_rule="当前有已核验酒店候选时，PlanIntent 必须恰好选择一个 h 短键。",
            option_source="hotels",
            allowed_next_actions=("materialize_draft",),
            minimal_safe_fragment={"selected_hotel_key": "<allowed_h_key>"},
        ),
    ),
    (
        "planner_plan_missing_required_restaurant",
        GuardRule(
            field_path="days[].stops",
            expected_rule=(
                "不能遗漏任务书中的必吃餐厅；为指出的 c 键选择可营业的日期和 lunch 或 dinner，"
                "可以替换普通餐厅，不得补到晚餐后作为额外下午茶。"
            ),
            option_source="plan_candidates",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
    (
        "planner_plan_duplicate_candidate",
        GuardRule(
            field_path="days[].stops[].candidate_key",
            expected_rule="同一候选 c 键在整份 PlanIntent 中最多出现一次。",
            option_source="plan_candidates",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
    (
        "planner_plan_candidate_key_invalid",
        GuardRule(
            field_path="days[].stops[].candidate_key",
            expected_rule="PlanIntent 只能选择本轮允许且可行的 c 候选短键；固定事项由程序插入。",
            option_source="plan_candidates",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
    (
        "planner_plan_duplicate_day",
        GuardRule(
            field_path="days[].day_index",
            expected_rule="每个 day_index 最多出现一次，缺少的日期由程序补成休息日。",
            option_source="dates",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
    (
        "planner_plan_requires_activity",
        GuardRule(
            field_path="days[].stops",
            expected_rule="存在可选候选时，整份 PlanIntent 至少安排一个 c 键；单独休息日仍然允许。",
            option_source="plan_candidates",
            allowed_next_actions=("materialize_draft",),
            minimal_safe_fragment={"days": [{"stops": [{"candidate_key": "<allowed_c_key>"}]}]},
        ),
    ),
    (
        "planner_plan_day_outside_trip",
        GuardRule(
            field_path="days[].day_index",
            expected_rule="day_index 必须位于本次旅行的 1 到总天数范围内。",
            option_source="dates",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
    (
        "planner_plan_day_capacity_exceeded",
        GuardRule(
            field_path="days[].stops",
            expected_rule="当天非餐饮活动数不得超过输入给出的每日 major activity 上限。",
            option_source="plan_candidates",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
    (
        "planner_candidate_date_infeasible",
        GuardRule(
            field_path="days[].stops[].candidate_key",
            expected_rule="候选不能安排在已有明确不可行证据的日期。",
            option_source="dates",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
    (
        "planner_draft_hotel_selection_required",
        GuardRule(
            field_path="draft.lodging_baseline.selected_offer_key",
            expected_rule="住宿策略为 search 时，必须从当前可用酒店短键中选择一个真实报价。",
            option_source="hotels",
            allowed_next_actions=("materialize_draft", "request_evidence"),
            minimal_safe_fragment={"lodging_baseline": {"selected_offer_key": "<allowed_h_key>"}},
        ),
    ),
    (
        "planner_unknown_hotel_key",
        GuardRule(
            field_path="draft.lodging_baseline.selected_offer_key",
            expected_rule="酒店只能引用当前 HotelObservation 暴露的 h 短键。",
            option_source="hotels",
            allowed_next_actions=("materialize_draft", "request_evidence"),
            minimal_safe_fragment={"selected_offer_key": "<allowed_h_key>"},
        ),
    ),
    (
        "planner_required_hotel_evidence_unavailable",
        GuardRule(
            field_path="hotel_observation.offers",
            expected_rule="Provider 未返回可绑定报价时不得虚构酒店或原样重试。",
            allowed_next_actions=("resume_after_provider_recovery",),
        ),
    ),
    (
        "planner_opening_hours_already_observed",
        GuardRule(
            field_path="requests[].arguments",
            expected_rule="同一 generation 中已观测的候选和日期组合不得重复查询。",
            option_source="dates",
            allowed_next_actions=("request_evidence", "materialize_draft"),
        ),
    ),
    (
        "planner_hotel_search_already_observed",
        GuardRule(
            field_path="requests[].arguments",
            expected_rule="相同住宿日期与活动簇的酒店请求只执行一次，后续复用 Observation。",
            option_source="hotels",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
    (
        "planner_unknown_route_endpoint",
        GuardRule(
            field_path="requests[].arguments.endpoint_pairs[]",
            expected_rule="路线端点只能引用当前 c、f、g 或 h 短键。",
            option_source="candidates",
            allowed_next_actions=("request_evidence",),
        ),
    ),
    (
        "planner_unknown_or_forbidden_candidate_key",
        GuardRule(
            field_path="draft.days[].ordered_items[].object_key",
            expected_rule="活动只能引用当前允许选择的候选 c 键或固定承诺 f 键。",
            option_source="candidates",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
    (
        "planner_unknown_cluster_key",
        GuardRule(
            field_path="draft.days[].primary_cluster_key",
            expected_rule="活动簇只能引用当前 SpatialObservation 的 g 短键。",
            option_source="clusters",
            allowed_next_actions=("materialize_draft", "request_evidence"),
        ),
    ),
    (
        "planner_unknown_cross_cluster_route_key",
        GuardRule(
            field_path="draft.days[].cross_cluster_segments[].route_edge_keys[]",
            expected_rule="跨簇路线只能引用当前可用且端点匹配的 r 短键。",
            option_source="routes",
            allowed_next_actions=("materialize_draft", "request_evidence"),
        ),
    ),
    (
        "planner_unknown_evidence_key",
        GuardRule(
            field_path="draft.*.observation_keys[]",
            expected_rule="证据引用只能复制当前 evidence_keys 中的短键。",
            option_source="evidence",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
    (
        "planner_cross_cluster_coverage",
        GuardRule(
            field_path="draft.days[].cross_cluster_segments[].covered_object_keys",
            expected_rule="非主簇活动必须且只能各被一个跨簇段覆盖一次。",
            option_source="candidates",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
    (
        "planner_day_exceeds_declared_capacity",
        GuardRule(
            field_path="draft.days[].ordered_items",
            expected_rule="当天 visit 数不得超过已接受策略的上限；移动或取舍后必须真正减少超量。",
            option_source="dates",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
    (
        "planner_draft_duplicate_objects",
        GuardRule(
            field_path="draft.days[].ordered_items[].object_key",
            expected_rule="同一对象在整份草稿中最多出现一次。",
            option_source="candidates",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
    (
        "planner_candidate_assigned_and_unassigned",
        GuardRule(
            field_path="draft.unassigned_intents",
            expected_rule="同一候选不能同时已安排和未安排。",
            option_source="candidates",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
    (
        "planner_unassigned_candidate_not_protected",
        GuardRule(
            field_path="draft.unassigned_intents[].candidate_key",
            expected_rule="只有 strong 或 soft 候选可进入 unassigned_intents。",
            option_source="candidates",
            allowed_next_actions=("materialize_draft",),
        ),
    ),
)


def compile_guard_violation(
    workspace: PlannerWorkspaceState,
    *,
    action: str,
    stage: GuardStage,
    raw_code: str,
) -> PlannerGuardViolation:
    """Return one root violation; dynamic rejected values are never echoed."""

    stable_code, schema_path = _stable_code_and_path(raw_code)
    rule = next((value for prefix, value in _RULES if stable_code.startswith(prefix)), None)
    if (
        stable_code == "planner_repair_no_effective_change"
        and schema_path != "choice.hotel_offer_key"
    ):
        rule = GuardRule(
            field_path="choice",
            expected_rule=(
                "本次操作没有改变当前日程。对照已安排日期、顺序和 expected_window，选择"
                "当前问题允许的有效修改；前序活动造成的延迟需重排或换日，不能重复原窗口。"
            ),
            allowed_next_actions=("revise_draft",),
        )
    if rule is None and action == "revise_draft" and schema_path:
        reference_code = None
        if schema_path.startswith("issue_keys"):
            reference_code = "planner_unknown_validation_issue_key"
        elif schema_path.startswith("choice.") and schema_path.endswith(
            ("target_key", "relative_to_key", "from_key", "to_key")
        ):
            reference_code = "planner_repair_target_not_in_current_draft"
        if reference_code is not None:
            rule = next(value for prefix, value in _RULES if prefix == reference_code)
    if rule is None:
        rule = GuardRule(
            field_path=schema_path or "$",
            expected_rule=_generic_expected_rule(stable_code),
            allowed_next_actions=_default_next_actions(workspace, action),
        )
    path = schema_path or _indexed_path(rule.field_path, raw_code)
    options = _explicit_legal_options(raw_code) or _legal_options(workspace, rule.option_source)
    no_hotel_replacement = (
        stable_code.startswith("planner_repair_no_effective_change")
        and path == "choice.hotel_offer_key"
        and not options
    )
    allowed_next_actions = ("ask_user",) if no_hotel_replacement else rule.allowed_next_actions
    minimal_safe_fragment = None if no_hotel_replacement else rule.minimal_safe_fragment
    if rule.option_source == "validation_issues" and len(options) == 1:
        minimal_safe_fragment = {"issue_keys": [options[0]]}
    related_issue_ids = _related_issue_ids(workspace, raw_code)
    fingerprint = canonical_json_hash(
        {
            "generation_id": workspace.generation_id,
            "action": action,
            "stage": stage,
            "code": stable_code,
            "field_path": path,
            "candidate_pool_revision": workspace.candidate_pool.revision,
            "draft_revision": (
                workspace.working_itinerary.draft_revision
                if workspace.working_itinerary is not None
                else None
            ),
            "validation_observation_id": (
                workspace.validation_observation.observation_id
                if workspace.validation_observation is not None
                else None
            ),
            "related_issue_ids": related_issue_ids,
        }
    )
    return PlannerGuardViolation(
        violation_id=server_id(workspace.generation_id, "guard", fingerprint),
        attempted_action=action,
        stage=stage,
        code=stable_code[:200],
        field_path=path[:200],
        object_index=_object_index(raw_code),
        expected_rule=rule.expected_rule,
        legal_option_keys=options[:40],
        related_issue_ids=related_issue_ids[:4],
        preserve_field_paths=rule.preserve_paths,
        allowed_next_actions=allowed_next_actions,
        failure_fingerprint=fingerprint,
        minimal_valid_fragment=minimal_safe_fragment,
    )


def repeated_guard_failure(workspace: PlannerWorkspaceState) -> bool:
    """One targeted retry is allowed; the second identical root failure stops."""

    if len(workspace.guard_observations) < 2:
        return False
    previous, current = workspace.guard_observations[-2:]
    return bool(
        previous.failure_fingerprint
        and current.failure_fingerprint
        and previous.failure_fingerprint == current.failure_fingerprint
    )


def _stable_code_and_path(raw_code: str) -> tuple[str, str | None]:
    base, separator, detail = raw_code.partition(":")
    if base in {"planner_contract_invalid", "planner_model_schema_invalid"} and separator:
        first = detail.split(";", 1)[0]
        path, _, error_type_and_message = first.partition(":")
        error_type = error_type_and_message.split(":", 1)[0]
        suffix = re.sub(r"[^a-zA-Z0-9_.-]", "_", error_type)[:60]
        return f"{base}.{suffix}" if suffix else base, _normalize_schema_path(path)
    explicit_path = re.search(r"(?:^|:)path=([^:]+)", raw_code)
    return base[:200], explicit_path.group(1)[:200] if explicit_path else None


def _normalize_schema_path(path: str) -> str:
    value = path.removeprefix("root.").strip()
    return value[:200] or "$"


def _indexed_path(template: str, raw_code: str) -> str:
    result = template
    for name in ("day_index", "segment_index", "request_index", "candidate_index"):
        match = re.search(rf"(?:^|:){name}=([0-9]+)(?::|$)", raw_code)
        if match is not None:
            result = result.replace("[]", f"[{match.group(1)}]", 1)
    return result


def _object_index(raw_code: str) -> int | None:
    for name in ("candidate_index", "segment_index", "request_index", "day_index"):
        match = re.search(rf"(?:^|:){name}=([0-9]+)(?::|$)", raw_code)
        if match is not None:
            return int(match.group(1))
    return None


def _explicit_legal_options(raw_code: str) -> tuple[str, ...]:
    """Read only server-authored compact options embedded in a Guard code."""

    match = re.search(r"(?:^|:)allowed_values=([^:]+)", raw_code)
    if match is None:
        return ()
    return tuple(value.strip() for value in match.group(1).split(",") if value.strip())[:40]


def _legal_options(workspace: PlannerWorkspaceState, source: str) -> tuple[str, ...]:
    catalog = PlannerReferenceCatalog(workspace)
    if source == "candidates":
        return tuple(
            key
            for key, entry in catalog.candidates.items()
            if entry.selection_permission != "forbidden"
        ) + tuple(catalog.fixed)
    if source == "plan_candidates":
        return tuple(
            key
            for key, entry in catalog.candidates.items()
            if entry.selection_permission != "forbidden"
            and entry.eligibility not in {"unavailable", "excluded"}
        )
    if source == "clusters":
        return tuple(catalog.clusters)
    if source == "routes":
        return tuple(
            key
            for key, edge in catalog.routes.items()
            if edge.status == "available" and edge.duration_minutes is not None
        )
    if source == "hotels":
        return tuple(
            key
            for key, offer in catalog.hotels.items()
            if offer.availability_status != "unavailable"
        )
    if source == "alternative_hotels":
        return tuple(catalog.alternative_hotels)
    if source == "evidence":
        return tuple(catalog.evidence)
    if source == "issues":
        return tuple(catalog.issues)
    if source == "validation_issues":
        return tuple(
            key
            for key, issue in catalog.validation_issues.items()
            if issue.severity in {"error", "blocking"}
        )
    if source == "draft_objects":
        if workspace.working_itinerary is None:
            return ()
        scheduled_refs = {
            item.object_ref.model_dump_json()
            for day in workspace.working_itinerary.days
            for item in day.ordered_items
        }
        return tuple(
            key
            for key, entry in catalog.candidates.items()
            if entry.candidate_ref.model_dump_json() in scheduled_refs
        ) + tuple(
            key for key, ref in catalog.fixed.items() if ref.model_dump_json() in scheduled_refs
        )
    if source == "dates":
        return (
            tuple(day.service_date.isoformat() for day in workspace.working_itinerary.days)
            if workspace.working_itinerary
            else ()
        )
    return ()


def _related_issue_ids(workspace: PlannerWorkspaceState, raw_code: str) -> tuple[str, ...]:
    validation = workspace.validation_observation
    if validation is None:
        return ()
    explicitly_named = {
        value for value in re.findall(r"(?:issue_id|issue)=([^:,;]+)", raw_code) if value
    }
    matching = [item.issue_id for item in validation.issues if item.issue_id in explicitly_named]
    if matching:
        return tuple(matching)
    return tuple(item.issue_id for item in validation.issues[:4])


def _default_next_actions(workspace: PlannerWorkspaceState, action: str) -> tuple[str, ...]:
    if workspace.validation_observation is not None:
        issue_actions = {
            value
            for issue in workspace.validation_observation.issues
            for value in issue.allowed_actions
        }
        result = []
        if issue_actions & {
            "move_item",
            "reorder_item",
            "replace_item",
            "remove_item",
            "change_transport",
            "change_hotel",
        }:
            result.append("revise_draft")
        if "request_evidence" in issue_actions:
            result.append("request_evidence")
        if "ask_user" in issue_actions:
            result.append("ask_user")
        if result:
            return tuple(result)
    if not action or action == "planner_decide":
        return ("retry_current_action",)
    return (action,)


def _generic_expected_rule(stable_code: str) -> str:
    if stable_code.startswith("planner_model_schema_invalid"):
        return "只修正所指字段，使其符合当前窄化输出 Schema；保留未被指出的选择。"
    if stable_code.startswith("planner_contract_invalid"):
        return "所指字段必须满足正式合同；不要补写未要求的服务端字段。"
    return "只修正当前根因并保持已接受的任务书、策略和 Observation 不变。"
