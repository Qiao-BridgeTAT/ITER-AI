"""Deterministic published-plan routing and server-owned change authority."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from backend.agent.planner.workspace import server_id
from backend.contracts.v4.enums import ConfidenceLevel
from backend.contracts.v4.plan_change import PlanChangeRequest
from backend.contracts.v4.planner_publication import PlannerPublishedPlan
from backend.contracts.v4.prepare import PublishedPlanIntent
from backend.contracts.v4.semantic_operations import (
    AddConditionalRequirementOperation,
    ConfirmFinalSupplementOperation,
    ConfirmTaskBookOperation,
    ExcludeConcreteEntityOperation,
    ResolveConflictOperation,
    RevokePriorIntentOperation,
    SelectConcreteEntityOperation,
    SemanticDomainV4,
    SemanticOperationProposal,
    SemanticTargetV4,
    SetExistingBookingOperation,
    SetNotApplicableOperation,
    SetTripBasicsOperation,
)
from backend.domain.discovery.state_merge import AcceptedV4Operation, stable_operation_id

PlanChangeScope = PublishedPlanIntent

_CORE_TEXT = re.compile(
    r"(?:目的地|日期|出发|返程|同行|带(?:孩子|老人)|必去|一定要去|不要去|别去|排除|"
    r"过敏|无障碍|已经订|已订)"
)
_LOCAL_EDIT_TEXT = re.compile(
    r"(?:换成|替换|改到|调到|移到|挪到|提前到|延后到|调整顺序|交换顺序|改住|换酒店|"
    r"重新安排|重新规划|重排|删掉|去掉|少安排|轻松一点)"
)
_HOTEL_REPLACEMENT_TEXT = re.compile(
    r"(?:换(?:一?家|个)?(?:下)?酒店|更换酒店|改住|换到(?:另一|别的|其他)?酒店)"
)
_FULL_REPLAN_TEXT = re.compile(
    r"重新(?:生成|规划|安排)(?:全部|整个|完整)?(?:行程|计划)|重跑(?:全部|整个)?行程|"
    r"重新(?:生成|规划)(?:这份|这次|本次)"
    r"(?:(?!第[一二三四五1-5]天|周[一二三四五六日天]|上午|下午|午餐|晚餐)[^，。！？；\n]){0,24}"
    r"(?:行程|计划)(?!的?(?:第[一二三四五1-5]天|周[一二三四五六日天]|上午|下午|午餐|晚餐))"
)


@dataclass(frozen=True, slots=True)
class PublishedPlanRoute:
    scope: PlanChangeScope
    operations: tuple[AcceptedV4Operation, ...]
    reason_code: str


def route_published_plan_message(
    *,
    user_text: str,
    user_message_id: UUID,
    turn_id: UUID,
    accepted_operations: tuple[AcceptedV4Operation, ...],
    semantic_scope: PlanChangeScope | None = None,
) -> PublishedPlanRoute:
    """Route from validated semantics; text is only a conservative plan-edit fallback."""

    if accepted_operations:
        scope = _scope_for_operations(user_text, accepted_operations)
        if scope != "task_book_change" and semantic_scope in {"local_replan", "full_replan"}:
            scope = semantic_scope
        return PublishedPlanRoute(
            scope=scope,
            operations=accepted_operations,
            reason_code=f"semantic_operations:{scope}",
        )
    if semantic_scope in {"reply_only", "task_book_change"}:
        return PublishedPlanRoute(
            scope=semantic_scope,
            operations=(),
            reason_code=f"qwen_plan_intent:{semantic_scope}",
        )
    if semantic_scope is None and _CORE_TEXT.search(user_text):
        # Prepare owns task-book semantics. If its guarded interpretation found no
        # operation, keep its clarification/reply rather than inventing one here.
        return PublishedPlanRoute(
            scope="task_book_change",
            operations=(),
            reason_code="core_text_requires_prepare",
        )
    if (
        semantic_scope in {"local_replan", "full_replan"}
        or _LOCAL_EDIT_TEXT.search(user_text)
        or _FULL_REPLAN_TEXT.search(user_text)
        or requests_hotel_replacement(user_text)
    ):
        local_key = server_id(user_message_id, "published-plan-directive")
        proposal = SemanticOperationProposal(
            root=AddConditionalRequirementOperation(
                operation_type="add_conditional_requirement",
                local_operation_key=local_key,
                target=SemanticTargetV4.GENERAL_CONSTRAINT,
                source_refs=[f"message:{user_message_id}"],
                confidence=ConfidenceLevel.HIGH,
                domain=SemanticDomainV4.GENERAL,
                condition="已发布正式行程",
                required_outcome=user_text,
            )
        )
        return PublishedPlanRoute(
            scope=(
                semantic_scope
                if semantic_scope in {"local_replan", "full_replan"}
                else "full_replan"
                if _FULL_REPLAN_TEXT.search(user_text)
                else "local_replan"
            ),
            operations=(
                AcceptedV4Operation(
                    operation_id=stable_operation_id(turn_id, local_key),
                    proposal=proposal,
                ),
            ),
            reason_code=f"qwen_plan_intent:{semantic_scope}"
            if semantic_scope
            else "explicit_plan_edit_text",
        )
    return PublishedPlanRoute(
        scope="reply_only",
        operations=(),
        reason_code="no_semantic_plan_change",
    )


def build_plan_change_request(
    *,
    plan: PlannerPublishedPlan,
    base_plan_version: int,
    user_message_id: UUID,
    route: PublishedPlanRoute,
) -> PlanChangeRequest:
    if route.scope not in {"local_replan", "full_replan"}:
        raise ValueError("only replanning routes produce a PlanChangeRequest")
    if not route.operations:
        raise ValueError("replanning requires at least one validated semantic operation")
    return PlanChangeRequest(
        plan_change_request_id=server_id(
            plan.plan_version_id,
            user_message_id,
            *(item.operation_id for item in route.operations),
        ),
        trip_id=str(plan.trip_id),
        base_plan_id=str(plan.plan_version_id),
        base_plan_version=base_plan_version,
        user_message_id=str(user_message_id),
        proposed_semantic_operations=[item.proposal for item in route.operations],
        semantic_operation_ids=tuple(str(item.operation_id) for item in route.operations),
        requested_scope=route.scope,
    )


def requests_hotel_replacement(user_text: str) -> bool:
    """Recognize an explicit request to replace the current published-plan hotel."""

    return _HOTEL_REPLACEMENT_TEXT.search(user_text) is not None


def _scope_for_operations(
    user_text: str,
    operations: tuple[AcceptedV4Operation, ...],
) -> PlanChangeScope:
    proposals = tuple(item.proposal.root for item in operations)
    if any(
        isinstance(
            proposal,
            (
                SetTripBasicsOperation,
                SetExistingBookingOperation,
                SetNotApplicableOperation,
                ExcludeConcreteEntityOperation,
                ResolveConflictOperation,
                RevokePriorIntentOperation,
            ),
        )
        for proposal in proposals
    ):
        return "task_book_change"
    if any(
        isinstance(proposal, AddConditionalRequirementOperation)
        and proposal.target
        in {
            SemanticTargetV4.DINING_REQUIREMENT,
            SemanticTargetV4.GENERAL_CONSTRAINT,
        }
        for proposal in proposals
    ):
        return "task_book_change"
    if any(
        isinstance(proposal, SelectConcreteEntityOperation)
        and proposal.disposition in {"must", "destination", "avoid"}
        for proposal in proposals
    ):
        return "task_book_change"
    if _CORE_TEXT.search(user_text) and any(
        isinstance(proposal, AddConditionalRequirementOperation) for proposal in proposals
    ):
        return "task_book_change"
    if all(
        isinstance(proposal, (ConfirmFinalSupplementOperation, ConfirmTaskBookOperation))
        for proposal in proposals
    ):
        return "reply_only"
    if any(
        proposal.target
        in {
            SemanticTargetV4.ATTRACTION_PREFERENCE,
            SemanticTargetV4.DINING_PREFERENCE,
            SemanticTargetV4.LODGING_AREA,
            SemanticTargetV4.LODGING_CLASS,
            SemanticTargetV4.TRANSPORT_AND_PACE,
            SemanticTargetV4.GENERAL_CONSTRAINT,
        }
        for proposal in proposals
    ):
        return "full_replan"
    return "local_replan"


__all__ = [
    "PlanChangeScope",
    "PublishedPlanRoute",
    "build_plan_change_request",
    "requests_hotel_replacement",
    "route_published_plan_message",
]
