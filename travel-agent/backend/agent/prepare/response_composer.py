"""Qwen response buffering and post-generation grounding checks."""

from __future__ import annotations

import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from typing import TypedDict

from backend.agent.model_audit import record_model_call_annotation
from backend.agent.model_gateway import (
    ModelCancellation,
    ModelFailureCode,
    ModelGateway,
    ModelGatewayError,
)
from backend.agent.prepare.prompts import build_prepare_response_request
from backend.domain.party_size import parse_party_size

_MAX_RESPONSE_CHARACTERS = 2_000
_INTERNAL_TERMS = {
    "TripSemanticState",
    "DiscoveryRuntimeState",
    "PrepareDecision",
    "CardActionObservation",
    "card_action_observations",
    "TaskBookActionObservation",
    "task_book_action_observations",
    "final_supplement_incomplete",
    "section_proposal",
    "semantic_operations",
}


class ResponseGroundingError(ValueError):
    pass


class _DateRangeCandidate(TypedDict):
    start_date: str
    end_date: str
    duration_days: int


@dataclass(frozen=True, slots=True)
class ComposedPrepareResponse:
    text: str
    generation_mode: str
    failure_code: str | None = None


async def compose_prepare_response(
    gateway: ModelGateway,
    grounding_context: dict[str, object],
    *,
    cancellation: ModelCancellation,
) -> ComposedPrepareResponse:
    """Consume model deltas into a private buffer; nothing is sent from here."""

    repair_instruction: str | None = None
    failure_code = "response_grounding_failed"
    last_call_id: str | None = None
    for attempt in range(2):
        call_id: str | None = None
        try:
            chunks: list[str] = []
            length = 0
            async for item in gateway.stream_text(
                build_prepare_response_request(
                    grounding_context, repair_instruction=repair_instruction
                ),
                cancellation=cancellation,
            ):
                call_id = item.audit_call_id or call_id
                last_call_id = call_id or last_call_id
                cancellation.raise_if_cancelled("compose_response")
                if not item.delta:
                    continue
                length += len(item.delta)
                if length > _MAX_RESPONSE_CHARACTERS:
                    raise ResponseGroundingError("composed response exceeds the public limit")
                chunks.append(item.delta)
            text = "".join(chunks).strip()
            validate_composed_response(text, grounding_context)
            await record_model_call_annotation(
                gateway,
                call_id,
                "llm_business_guard",
                {
                    "business_guard_result": {
                        "status": "accepted",
                        "guard": "prepare_response_grounding",
                    },
                    "accepted_or_rejected": "accepted",
                    "agent_output_full": text,
                    "materialized_output": {"text": text},
                },
            )
            return ComposedPrepareResponse(text=text, generation_mode="qwen")
        except ModelGatewayError as error:
            last_call_id = error.audit_call_id or call_id or last_call_id
            if error.code is ModelFailureCode.CANCELLED:
                raise
            failure_code = f"response_{error.code.value}"
            break
        except ResponseGroundingError as error:
            if str(error).startswith("trip party size mismatch:"):
                recovery_rule = (
                    "人数只使用 trusted_state.party_size，不得把旅行天数或年龄当作人数；"
                    "修正出行人数表述即可，不修改已经保存的同行人或任务书。"
                )
            elif str(error).startswith("published plan status mismatch:"):
                recovery_rule = (
                    "只回答当前问题，使用 trusted_state.published_plan 中的正式行程和所选酒店；"
                    "没有修改需求时，不要说任务书待确认、尚未选酒店或要求再次启动规划。"
                )
            elif _task_book_action_observed(grounding_context):
                recovery_rule = (
                    "任务书预检已经回到最终补充；只继续询问是否还有补充，"
                    "不得声称任务书已生成或正在生成。"
                )
            elif _card_action_observed(grounding_context):
                recovery_rule = (
                    "结构化卡片观察已经触发最终再决策；按最终 next_action 聚焦提问或"
                    "说明当前边界，不得要求重新生成，也不得承诺卡片正在生成。"
                )
            elif _card_generation_failed(grounding_context):
                recovery_rule = "旧内部卡片降级须说明‘卡片暂未生成’，并保留‘重新生成卡片’入口。"
            elif _is_date_range_followup(grounding_context):
                recovery_rule = (
                    "只用一句简短确认加一句问题；已有旅行天数时只需问出发日，"
                    "有待确认候选时复述其两个端点并询问是否正确；"
                    "不举例、不解释用途、不承诺后续资源。"
                )
            else:
                recovery_rule = (
                    "如有成功附件，用一两句亲切的话引导选择，不枚举或复述卡片选项；"
                    "用户主动追问细节时只回答相关内容，必要确认和事实边界不能省略。"
                )
            repair_feedback = (
                f"回复未通过校验：{error}。依据实际执行结果重写，不可重复未实现的行动承诺。"
                + recovery_rule
            )
            await record_model_call_annotation(
                gateway,
                call_id,
                "llm_business_guard",
                {
                    "business_guard_result": {
                        "status": "rejected",
                        "guard": "prepare_response_grounding",
                        "reason": str(error),
                    },
                    "accepted_or_rejected": "rejected",
                    "failure_stage": "business_guard",
                    "failure_code": "response_grounding_failed",
                    "request_guard_feedback_full": repair_feedback,
                    "agent_output_full": "".join(chunks).strip(),
                },
            )
            if attempt == 0:
                # Only our constant guard description is returned to Qwen;
                # rejected output never becomes visible or tool authority.
                repair_instruction = repair_feedback
    fallback_text = fallback_prepare_text(grounding_context)
    await record_model_call_annotation(
        gateway,
        last_call_id,
        "llm_fallback_selected",
        {
            "accepted_or_rejected": "fallback",
            "failure_stage": "response_composition",
            "failure_code": failure_code,
            "generation_mode": "fallback",
            "agent_output_full": fallback_text,
            "materialized_output": {"text": fallback_text},
        },
    )
    return ComposedPrepareResponse(
        text=fallback_text,
        generation_mode="fallback",
        failure_code=failure_code,
    )


def validate_composed_response(text: str, grounding_context: dict[str, object]) -> None:
    if not text:
        raise ResponseGroundingError("composed response is empty")
    if len(text) > _MAX_RESPONSE_CHARACTERS:
        raise ResponseGroundingError("composed response is too long")
    if any(term in text for term in _INTERNAL_TERMS):
        raise ResponseGroundingError("composed response leaks internal contracts")
    trusted = grounding_context.get("trusted_state")
    party_size = trusted.get("party_size") if isinstance(trusted, dict) else None
    if isinstance(party_size, int) and not isinstance(party_size, bool):
        count = r"[0-9一二两三四五六七八九十]+"
        claims = re.findall(
            rf"(?:适合|为|共|一共|合计|同行人数为)({count})人(?:的|安排|规划|旅行|出行|[，。；])",
            text,
        ) + re.findall(rf"({count})人(?:同行|出行|出游|成行|旅程|旅行)", text)
        if any(parse_party_size((f"{claim}人",)) != party_size for claim in claims):
            raise ResponseGroundingError(
                f"trip party size mismatch: trusted_state.party_size={party_size}"
            )

    published = trusted.get("published_plan") if isinstance(trusted, dict) else None
    if (
        isinstance(published, dict)
        and grounding_context.get("next_action") == "reply_only"
        and not grounding_context.get("verified_changes")
    ):
        if re.search(
            r"任务书[^。！？\n]{0,15}(?:待确认|尚未确认|未确认)"
            r"|(?:正式|最终)(?:的)?(?:逐日)?行程(?:文件|计划)?[^。！？\n]{0,8}"
            r"(?:尚未|暂未|没有|未)(?:正式)?(?:生成|发布)"
            r"|(?:尚未|暂未|还没)(?:生成|发布)[^。！？\n]{0,8}(?:正式|最终)行程",
            text,
        ):
            raise ResponseGroundingError("published plan status mismatch: plan already published")
        if published.get("selected_hotel_name") and re.search(
            r"(?:尚未|暂未|还未|还没有|没有)(?:为您|为你)?(?:选定|选择|安排|确定)(?:具体)?酒店",
            text,
        ):
            raise ResponseGroundingError("published plan status mismatch: hotel already selected")

    if _is_date_range_followup(grounding_context):
        if len(text) > 100:
            raise ResponseGroundingError("date range followup is not concise")
        # The structured follow-up owns the question's target. A regex cannot
        # judge every natural phrasing; asking only for a start day is sufficient
        # when duration is known. An actual candidate still needs both dates.
        candidate = _date_range_candidate(grounding_context)
        if candidate is not None and not (
            _mentions_candidate_date(text, candidate["start_date"])
            and _mentions_candidate_date(text, candidate["end_date"])
        ):
            raise ResponseGroundingError("date range followup omits the assessed candidate")
        if re.search(r"匹配|景点开放|住宿资源|机票|生成任务书", text):
            raise ResponseGroundingError("date range followup promises unrelated downstream work")

    if _card_action_observed(grounding_context):
        if "重新生成卡片" in text:
            raise ResponseGroundingError("card observation response cannot request a card retry")
        if _contains_unissued_card_promise(text):
            raise ResponseGroundingError("card observation response promises an unissued card")

    if _card_generation_failed(grounding_context):
        if "卡片暂未生成" not in text or "重新生成卡片" not in text:
            raise ResponseGroundingError("failed card requires an explicit failure and retry entry")
        # A retry invitation is actionable; an unconditional promise after a
        # finished turn is not. Inspect clauses so a later retry sentence cannot
        # excuse an earlier false promise.
        if _contains_unissued_card_promise(text):
            raise ResponseGroundingError("failed card response promises an unissued card")

    if _task_book_action_observed(grounding_context) and _contains_unissued_task_book_promise(text):
        raise ResponseGroundingError("task-book recovery response promises an unissued task book")

    if (
        grounding_context.get("next_action") == "generate_task_book"
        and not _task_book_attachment_issued(grounding_context)
        and _contains_unissued_task_book_promise(text)
    ):
        raise ResponseGroundingError("response promises an unissued task book")

    observations = grounding_context.get("tool_observations")
    fact_required = bool(grounding_context.get("fact_required"))
    if not observations and re.search(r"余票|售罄|可预[约订]|可订状态", text):
        raise ResponseGroundingError("booking availability claim has no tool observation")
    if fact_required and not observations:
        raise ResponseGroundingError("fact answer has no tool observation")
    if not isinstance(observations, list):
        return
    fact_values = [
        str(fact.get("value_summary", ""))
        for observation in observations
        if isinstance(observation, dict)
        for fact in observation.get("facts", [])
        if isinstance(fact, dict)
    ]
    hour_values = [value for value in fact_values if re.search(r"\d{1,2}[:：]\d{2}", value)]
    if hour_values:
        expected_tokens = {
            token for value in hour_values for token in re.findall(r"\d{1,2}[:：]\d{2}", value)
        }
        if expected_tokens and not any(token in text for token in expected_tokens):
            raise ResponseGroundingError("hours response is not grounded in the observation")


def fallback_prepare_text(grounding_context: dict[str, object]) -> str:
    if _is_date_range_followup(grounding_context):
        candidate = _date_range_candidate(grounding_context)
        if candidate is not None:
            start = _short_date(candidate["start_date"])
            end = _short_date(candidate["end_date"])
            duration = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五"}.get(
                candidate["duration_days"], str(candidate["duration_days"])
            )
            return f"按{duration}天计算，你准备{start}出发、{end}结束，对吗？"
        return "准备哪天出发、哪天结束？"
    if _date_confirmation_write_failed(grounding_context):
        trusted_state = grounding_context.get("trusted_state")
        basics = trusted_state.get("trip_basics") if isinstance(trusted_state, dict) else None
        if (
            isinstance(basics, dict)
            and basics.get("destination_name")
            and basics.get("duration_days")
        ):
            return (
                "日期这次没确认成功，目的地和旅行天数还记着。再告诉我一下哪天出发、哪天结束，好吗？"
            )
        return "这次没能保存完整的旅行信息。请再告诉我目的地、哪天出发、哪天结束，好吗？"
    if _task_book_action_observed(grounding_context):
        return "还有其他补充吗？没有的话，告诉我一声就好。"
    observation = _latest_card_action_observation(grounding_context)
    if observation is not None:
        missing_fields = observation.get("missing_fields")
        requested_targets = grounding_context.get("requested_targets")
        targets = {
            value
            for values in (missing_fields, requested_targets)
            if isinstance(values, list)
            for value in values
            if isinstance(value, str)
        }
        if "duration_days" in targets:
            return "这次计划玩几天？目前支持一到五天。"
        if "destination" in targets:
            return "这次想去哪个城市？确认目的地后我再继续筛选。"
        section = str(observation.get("section", ""))
        if section.startswith("attraction"):
            return "这轮暂时没找到合适的景点。有没有特别想去的地方，或想体验的玩法？"
        if section.startswith("dining"):
            return "这轮暂时没找到合适的餐饮选择。有没有特别想吃的口味或店？"
        return "这轮暂时没找到合适的住宿选择。你对住的位置或舒适程度有什么想法？"
    if _card_generation_failed(grounding_context):
        return (
            "卡片暂未生成，之前的选择还在。点一下‘重新生成卡片’就能再试，也可以直接告诉我你的想法。"
        )
    if grounding_context.get("fact_required"):
        return "这条信息我暂时没能准确整理出来，先不误导你。可以稍后再试一下。"
    changes = grounding_context.get("verified_changes")
    if isinstance(changes, list) and changes:
        return "你刚补充的信息我记下了，但这次没能顺利回复。你可以再说一次，我们接着聊。"
    return "我在呢。刚才没能顺利回复，你可以再说一次，我们接着聊。"


def _card_generation_failed(grounding_context: dict[str, object]) -> bool:
    result = grounding_context.get("card_generation")
    return isinstance(result, dict) and result.get("status") == "unavailable"


def _is_date_range_followup(grounding_context: dict[str, object]) -> bool:
    return bool(
        grounding_context.get("next_action") == "ask_clarification"
        and grounding_context.get("requested_targets") == ["date_range"]
    )


def _date_confirmation_write_failed(grounding_context: dict[str, object]) -> bool:
    failure_code = grounding_context.get("decision_failure_code")
    resolution = grounding_context.get("date_range_resolution")
    trusted = grounding_context.get("trusted_state")
    basics = trusted.get("trip_basics") if isinstance(trusted, dict) else None
    if (
        isinstance(basics, dict)
        and isinstance(resolution, dict)
        and resolution.get("status") == "ready"
        and basics.get("start_date") == resolution.get("start_date")
        and basics.get("end_date") == resolution.get("end_date")
        and basics.get("start_date")
        and basics.get("end_date")
    ):
        # A later action failure does not undo already validated date writes.
        return False
    if failure_code == "prepare_intake_assessment_invalid":
        # Intake may fail before there is a parsed date resolution. Do not lose
        # the active question or imply that the user's dates were saved.
        return grounding_context.get("date_followup_active") is True
    return bool(
        isinstance(failure_code, str)
        and failure_code.startswith("decision_")
        and isinstance(resolution, dict)
        and resolution.get("status") == "ready"
    )


def _date_range_candidate(
    grounding_context: dict[str, object],
) -> _DateRangeCandidate | None:
    candidate = grounding_context.get("date_range_candidate")
    if not isinstance(candidate, dict) or candidate.get("status") != "needs_confirmation":
        return None
    start_date = candidate.get("start_date")
    end_date = candidate.get("end_date")
    duration_days = candidate.get("duration_days")
    if (
        not isinstance(start_date, str)
        or not isinstance(end_date, str)
        or not isinstance(duration_days, int)
    ):
        return None
    return {
        "start_date": start_date,
        "end_date": end_date,
        "duration_days": duration_days,
    }


def _mentions_candidate_date(text: str, value: str) -> bool:
    parsed = date.fromisoformat(value)
    compact = re.sub(r"\s+", "", text)
    # Compare calendar values, not presentation substrings. Whitespace, zero
    # padding and a shared month in a range do not change the proposed dates.
    found: set[date] = set()

    def add(year: int, month: int, day: int) -> None:
        with suppress(ValueError):
            found.add(date(year, month, day))

    for match in re.finditer(r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)", compact):
        add(*(int(part) for part in match.groups()))
    for match in re.finditer(
        r"(?<!\d)(?:(\d{4})年)?(\d{1,2})月(\d{1,2})[日号]"
        r"(?:(?:至|到|[-—–~～])(?:(\d{4})年)?(?:(\d{1,2})月)?(\d{1,2})[日号])?",
        compact,
    ):
        year, month, day, end_year, end_month, end_day = match.groups()
        effective_year = int(year) if year else parsed.year
        add(effective_year, int(month), int(day))
        if end_day:
            add(
                int(end_year) if end_year else effective_year,
                int(end_month) if end_month else int(month),
                int(end_day),
            )
    return parsed in found


def _short_date(value: str) -> str:
    parsed = date.fromisoformat(value)
    return f"{parsed.month}月{parsed.day}日"


def _latest_card_action_observation(
    grounding_context: dict[str, object],
) -> dict[str, object] | None:
    observations = grounding_context.get("card_action_observations")
    if not isinstance(observations, list) or not observations:
        return None
    latest = observations[-1]
    return latest if isinstance(latest, dict) else None


def _card_action_observed(grounding_context: dict[str, object]) -> bool:
    return _latest_card_action_observation(grounding_context) is not None


def _task_book_action_observed(grounding_context: dict[str, object]) -> bool:
    observations = grounding_context.get("task_book_action_observations")
    return isinstance(observations, list) and bool(observations)


def _contains_unissued_card_promise(text: str) -> bool:
    return any(
        re.search(
            r"(?:正在|马上|即将|稍后|接下来|将为|将会).{0,18}(?:生成|展示|提供|呈现|准备)"
            r"|(?:卡片|区域卡|偏好卡|候选卡).{0,8}已.{0,8}(?:生成|展示|准备好)"
            r"|已.{0,10}(?:生成|展示|准备好).{0,12}(?:卡片|区域卡|偏好卡|候选卡)"
            r"|请.{0,3}(?:稍等|等待)",
            clause,
        )
        for clause in re.split(r"[。！？!?；;，,\n]", text)
    )


def _task_book_attachment_issued(grounding_context: dict[str, object]) -> bool:
    attachments = grounding_context.get("attachments")
    return bool(
        grounding_context.get("outcome") == "task_book_ready"
        and isinstance(attachments, list)
        and any(
            isinstance(attachment, dict) and attachment.get("kind") == "task_book"
            for attachment in attachments
        )
    )


def _contains_unissued_task_book_promise(text: str) -> bool:
    return any(
        "任务书" in clause
        and re.search(
            r"(?:正在|马上|即将|稍后|接下来|将为|将会).{0,18}(?:生成|展示|提供|呈现|准备)"
            r"|(?:已经|已).{0,8}(?:生成|准备好|整理好)"
            r"|请.{0,3}(?:稍等|等待)",
            clause,
        )
        for clause in re.split(r"[。！？!?；;，,\n]", text)
    )


__all__ = [
    "ComposedPrepareResponse",
    "ResponseGroundingError",
    "compose_prepare_response",
    "fallback_prepare_text",
    "validate_composed_response",
]
