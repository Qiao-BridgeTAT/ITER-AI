"""Actionable, bounded feedback shared by Prepare retries and audit annotations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from backend.agent.model_gateway import ModelGatewayError
from backend.agent.prepare.semantic_transaction import SemanticCompilationError


@dataclass(frozen=True)
class PrepareGuardFeedback:
    code: str
    path: str
    recovery: str
    instruction: str
    allowed_next_action: dict[str, object] | None = None
    preserve: tuple[str, ...] = ()
    problem_key: str = ""

    def payload(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self, context_key: str) -> str:
        value = json.dumps(
            [
                self.code,
                self.path,
                self.recovery,
                self.allowed_next_action,
                self.problem_key,
                context_key,
            ],
            sort_keys=True,
        )
        return hashlib.sha256(value.encode()).hexdigest()[:16]


def is_action_conflict(error: Exception) -> bool:
    return str(error).startswith(
        (
            "card action is not executable after semantic merge:",
            "concrete entity continuation requires next action ",
            "trip intake requires the ",
            "card text answer requires an executable next interaction",
        )
    )


_DATE_SCHEMA_REPAIRS = {
    "date_resolution_empty": (
        "date_range_resolution",
        "status=none 时 basis=none，start_date/end_date/duration_days 均为 null。",
    ),
    "date_resolution_incomplete": (
        "date_range_resolution.start_date/end_date/basis",
        "非 none 状态必须有完整起止日期及来源；不能唯一确定两个端点时返回 none，"
        "不要猜测或保存单端点。天数由程序计算。",
    ),
    "date_resolution_reversed": (
        "date_range_resolution.end_date",
        "结束日早于开始日。按用户原话重新核对两个端点，不得自行交换或改写用户日期。",
    ),
    "date_resolution_too_long": (
        "date_range_resolution.start_date/end_date",
        "日期范围超过五天。不得截短用户行程；无法形成范围内的明确日期时返回 none 并继续澄清。",
    ),
    "date_resolution_confirmation_basis": (
        "date_range_resolution.status/basis",
        "needs_confirmation 只用于 start_plus_duration 推算候选；"
        "明确或已确认的完整日期才可 ready。不要把推算候选升级为已确认。",
    ),
    "date_resolution_unconfirmed": (
        "date_range_resolution.status",
        "basis=start_plus_duration 是推算而非确认，status 必须为 needs_confirmation；"
        "保留候选起止日期，不能改成已确认来源。",
    ),
    "date_resolution_explicit_conflict": (
        "explicit_date_range/date_range_resolution.status/basis",
        "explicit_date_range 只描述本句是否给出完整日期；若仍需推算，设为 false，"
        "保留 needs_confirmation 候选，不得为了通过校验把候选改成 ready。",
    ),
    "date_resolution_explicit_required": (
        "explicit_date_range/date_range_resolution.basis",
        "current_explicit_range 表示本句给出完整起止日期，explicit_date_range 应为 true；"
        "若仅补齐或确认当前有效追问，保留 contextual_completion/contextual_confirmation。",
    ),
}

_BASICS_SCHEMA_REPAIRS = {
    "trip_basics_empty": "至少保留一个用户明确字段；不要凭空补日期、同行人或目标。",
    "trip_basics_date_pair": (
        "start_date/end_date 必须同时提供。只有一个端点时保持日期为空，"
        "保留已明确的 duration_days；待确认候选只用于追问。"
    ),
    "trip_basics_dates_reversed": "end_date 早于 start_date；核对用户原话，不得自行交换日期。",
    "trip_basics_dates_too_long": "start_date/end_date 超过五天；不得截短或伪造用户日期。",
}


def decision_schema_repair_instruction(error: ModelGatewayError) -> str | None:
    details = []
    for issue in error.validation_issues[:6]:
        parts = issue.split(":", 2)
        if len(parts) >= 2 and parts[1] in _BASICS_SCHEMA_REPAIRS:
            details.append(f"{parts[0]}：{_BASICS_SCHEMA_REPAIRS[parts[1]]}")
    return " ".join(details) + " 保留其他有依据的字段，只修复这些位置。" if details else None


def intake_schema_repair_instruction(error: ModelGatewayError) -> str:
    """Return safe, field-specific guidance without exposing validation inputs."""

    details: list[str] = []
    paths: list[str] = []
    for issue in error.validation_issues[:6]:
        parts = issue.split(":", 2)
        path, kind = parts[0], parts[1] if len(parts) > 1 else "invalid"
        paths.append(path)
        repair = _DATE_SCHEMA_REPAIRS.get(kind)
        if repair is not None:
            details.append(f"{repair[0]}：{repair[1]}")
        else:
            details.append(f"{path}（{kind}）：按该字段的 Schema 修正类型、枚举或必填项。")
    if any(path.startswith("requirement_facts") for path in paths):
        details.append(
            "requirement_facts 的 target 仅允许 general_constraint、transport_and_pace、"
            "lodging_class、dining_requirement（target+quote），或 attraction_preference、"
            "dining_preference、lodging_area（还需 disposition=select/exclude）。"
            "餐厅委托仅在 required_additional_targets 标记 dining_entity；"
            "具体地点放 named_entity_intents。quote/query 只复制当前 user_text 原文。"
        )
    return (
        "TripBasicsAssessment 结构错误："
        + (" ".join(details) or "输出不是完整的合同 JSON 对象，请按 Schema 重新输出。")
        + " 只修复上述字段，保留其他有依据的信息；不要输出主决策或解释。"
    )


def guard_feedback(error: Exception, instruction: str) -> PrepareGuardFeedback:
    if isinstance(error, ModelGatewayError) and error.validation_issues:
        issues = [item.split(":", 2) for item in error.validation_issues]
        details = [f"{item[0]} ({item[1]})" for item in issues if len(item) >= 2]
        if details:
            return PrepareGuardFeedback(
                "prepare_model_schema_invalid",
                issues[0][0],
                "repair_fields",
                instruction + " 具体出错字段：" + "；".join(details[:6]) + "。保留其他正确字段。",
                problem_key="|".join(details),
            )
    if str(error).startswith("compound intake omitted semantic targets: "):
        return PrepareGuardFeedback(
            "prepare_semantic_targets_missing",
            "semantic_operations",
            "repair_fields",
            instruction,
            problem_key=str(error).partition(": ")[2],
        )
    if isinstance(error, SemanticCompilationError):
        return PrepareGuardFeedback(
            error.code,
            error.path,
            error.recovery,
            instruction,
        )
    if is_action_conflict(error):
        return PrepareGuardFeedback(
            "prepare_next_action_conflict",
            "next_action",
            "redecide_action",
            instruction,
            preserve=("semantic_operations",),
        )
    return PrepareGuardFeedback(
        "prepare_decision_invalid",
        "decision",
        "repair_fields",
        instruction,
        problem_key=hashlib.sha256(str(error).encode()).hexdigest()[:16],
    )
