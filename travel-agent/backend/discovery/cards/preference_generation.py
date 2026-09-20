"""Qwen-generated, program-validated discovery direction plans."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from backend.agent.model_audit import record_model_call_annotation
from backend.agent.model_gateway import (
    ModelAuditMetadata,
    ModelCancellation,
    ModelFailureCode,
    ModelGateway,
    ModelGatewayError,
    ModelMessage,
    ModelRequest,
    ModelRole,
)
from backend.agent.prepare.progress import (
    AttractionProgress,
    DiningProgress,
    report_attraction_progress,
    report_dining_progress,
)
from backend.contracts.v4.attraction_search import AttractionSearchHints, AttractionSearchPlace
from backend.contracts.v4.base import Identifier, V4ContractModel, require_unique
from backend.contracts.v4.content_quality import (
    attraction_direction_quality_issue,
    normalized_visible_text,
    require_distinct_visible_labels,
    require_meaningful_description,
    require_meaningful_label,
    require_visible_text,
)
from backend.contracts.v4.dining_search import DiningSearchHints
from backend.contracts.v4.enums import CompositionRole, DiscoverySection
from backend.contracts.v4.lodging_preferences import LodgingAreaPlan
from backend.contracts.v4.state import TripSemanticState
from backend.discovery.cards.attraction_preference_prompt import (
    ATTRACTION_PREFERENCE_PROMPT_VERSION,
    ATTRACTION_PREFERENCE_SYSTEM_PROMPT,
)
from backend.discovery.cards.attraction_schema import attraction_schema
from backend.discovery.cards.dining.prompts import (
    DINING_PREFERENCE_PROMPT_VERSION,
    MAIN_MEAL_DIRECTION_RULES,
)
from backend.discovery.cards.lodging_prompt import LODGING_AREA_SYSTEM_PROMPT
from backend.domain.discovery.cold_start import cold_start_default_notes

PREFERENCE_DIRECTION_PROMPT_VERSION = "prepare-card-directions-v4-03-12"
logger = logging.getLogger(__name__)
GeneratedLabelText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=2, max_length=80, pattern=r"\S"),
]
GeneratedDescriptionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=6, max_length=240, pattern=r"\S"),
]
GeneratedDetailText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=4, max_length=160, pattern=r"\S"),
]
LodgingClassDirectionId = Literal[
    "economy",
    "comfort",
    "upscale",
    "luxury",
    "boutique_resort",
]


class DirectionDraft(V4ContractModel):
    direction_id: Identifier
    label: GeneratedLabelText = Field(
        description="用户可见的完整中文体验方向短语，至少两个汉字，例如‘六朝古都文脉’。",
    )
    description: GeneratedDescriptionText = Field(
        description="用户可见的完整中文说明句，至少六个汉字，解释该方向的旅行体验。",
    )
    composition_role: CompositionRole
    tags: list[Identifier] = Field(default_factory=list, max_length=8)
    search_query: GeneratedLabelText | None = Field(
        default=None,
        description="可用于地点召回的完整中文搜索短语；没有必要时填 null。",
    )

    @model_validator(mode="after")
    def copy_is_complete_and_readable(self) -> DirectionDraft:
        require_meaningful_label(self.label, "direction label")
        require_meaningful_description(self.description, "direction description")
        if self.search_query is not None:
            require_visible_text(
                self.search_query,
                "direction search query",
                minimum_units=2,
            )
        return self


class AttractionDirectionDraft(V4ContractModel):
    direction_id: Identifier
    label: GeneratedLabelText = Field(
        description="约4至12字，以体验方向为主，尽量少用具体景点名；有助于理解时可自然点到。"
    )
    description: GeneratedDescriptionText = Field(
        description="约20至45字，具体自然且有适度文采的体验说明；少量地点名可酌情出现，不罗列或重复强调。"
    )
    composition_role: CompositionRole
    tags: list[Identifier] = Field(default_factory=list, max_length=8)
    representative_places: list[AttractionSearchPlace] = Field(max_length=4)
    search_queries: list[GeneratedLabelText] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def valid_search_hints(self) -> AttractionDirectionDraft:
        require_meaningful_label(self.label, "direction label")
        require_meaningful_description(self.description, "direction description")
        require_unique((word.casefold().strip() for word in self.search_queries), "search query")
        require_unique(
            (place.name.casefold().strip() for place in self.representative_places),
            "representative place",
        )
        return self

    @property
    def search_query(self) -> str:
        return self.search_queries[0]

    @property
    def attraction_search_hints(self) -> AttractionSearchHints:
        return AttractionSearchHints(
            representative_places=self.representative_places,
            search_queries=self.search_queries,
        )


class AttractionDirectionPlan(V4ContractModel):
    directions: list[AttractionDirectionDraft] = Field(min_length=6, max_length=7)

    @model_validator(mode="after")
    def has_city_representative_quota(self) -> AttractionDirectionPlan:
        _validate_directions(self.directions)
        for item in self.directions:
            issue = attraction_direction_quality_issue(item.label)
            if issue is not None:
                raise ValueError(issue)
        count = sum(
            item.composition_role is CompositionRole.REPRESENTATIVE_EXTRA
            for item in self.directions
        )
        if count not in {3, 4} or len(self.directions) - count not in {2, 3}:
            raise ValueError("attraction directions require 3-4 city and 2-3 personalized roles")
        return self


class DiningDirectionPlan(V4ContractModel):
    directions: list[DirectionDraft] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def has_two_local_directions(self) -> DiningDirectionPlan:
        _validate_directions(self.directions)
        count = sum(
            item.composition_role is CompositionRole.REPRESENTATIVE_EXTRA
            for item in self.directions
        )
        if count != 2:
            raise ValueError("dining direction plan requires exactly two local directions")
        return self


class LodgingAreaDirectionDraft(V4ContractModel):
    direction_id: Identifier
    label: GeneratedLabelText = Field(
        description="完整的中文住宿区位方向短语，不是酒店名称。",
    )
    description: GeneratedDescriptionText = Field(
        description="面向用户的一句区位特点与必要取舍，约20到40字；不逐项复述内部分析。",
    )
    search_query: GeneratedLabelText = Field(
        description="目的地内可被地图解析的真实区域、商圈或交通节点短名称。",
    )
    city_atmosphere: GeneratedDetailText = Field(
        description="至少四个汉字，说明该区位真实的城市氛围。",
    )
    anchor_fit: GeneratedDetailText = Field(
        description="至少四个汉字，说明该区位与已知景点餐饮锚点的适配。",
    )
    transport_characteristics: GeneratedDetailText = Field(
        description="至少四个汉字，说明该区位的交通特点。",
    )
    main_tradeoff: GeneratedDetailText = Field(
        description="至少四个汉字，说明选择该区位的主要取舍。",
    )

    @model_validator(mode="after")
    def copy_is_complete_and_readable(self) -> LodgingAreaDirectionDraft:
        require_meaningful_label(self.label, "lodging area label")
        require_meaningful_description(self.description, "lodging area description")
        require_visible_text(
            self.search_query,
            "lodging area search query",
            minimum_units=2,
        )
        for field_name, value in (
            ("city_atmosphere", self.city_atmosphere),
            ("anchor_fit", self.anchor_fit),
            ("transport_characteristics", self.transport_characteristics),
            ("main_tradeoff", self.main_tradeoff),
        ):
            require_visible_text(value, field_name, minimum_units=4)
        return self


class LodgingAreaDirectionPlan(V4ContractModel):
    directions: list[LodgingAreaDirectionDraft] = Field(min_length=5, max_length=6)

    @model_validator(mode="after")
    def directions_are_unique(self) -> LodgingAreaDirectionPlan:
        require_unique((item.direction_id for item in self.directions), "direction_id")
        require_distinct_visible_labels(
            (item.label for item in self.directions),
            "lodging direction labels",
        )
        require_unique(
            (item.search_query.casefold() for item in self.directions),
            "area search query",
        )
        return self


class FlexibleLodgingAreaDirectionPlan(V4ContractModel):
    directions: list[LodgingAreaDirectionDraft] = Field(
        min_length=5,
        max_length=6,
        description="五到六个真实、可解析且彼此不同的住宿区位方向。",
    )


class LodgingClassDirectionDraft(V4ContractModel):
    direction_id: LodgingClassDirectionId
    label: GeneratedLabelText
    description: GeneratedDescriptionText

    @model_validator(mode="after")
    def copy_is_complete_and_readable(self) -> LodgingClassDirectionDraft:
        require_meaningful_label(self.label, "lodging class label")
        require_meaningful_description(self.description, "lodging class description")
        _require_lodging_class_label(self.label)
        return self


class LodgingClassDirectionPlan(V4ContractModel):
    directions: list[LodgingClassDirectionDraft] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def contains_each_supported_direction(self) -> LodgingClassDirectionPlan:
        expected = {"economy", "comfort", "upscale", "luxury", "boutique_resort"}
        if {item.direction_id for item in self.directions} != expected:
            raise ValueError("lodging class plan requires the five frozen direction families")
        require_distinct_visible_labels(
            (item.label for item in self.directions),
            "lodging class direction labels",
        )
        _require_lodging_class_families(self.directions)
        return self


class LodgingClassCopyDraft(V4ContractModel):
    label: GeneratedLabelText = Field(
        description="至少两个汉字的完整中文住宿档次或类型短语，不含具体酒店名。",
    )
    description: GeneratedDescriptionText = Field(
        description="面向用户的一句品质或体验区别，约15到30字，不复述标题或用户条件。",
    )

    @model_validator(mode="after")
    def copy_is_complete_and_readable(self) -> LodgingClassCopyDraft:
        require_meaningful_label(self.label, "lodging class label")
        require_meaningful_description(self.description, "lodging class description")
        _require_lodging_class_label(self.label)
        return self


class LodgingClassContentPlan(V4ContractModel):
    """Qwen-owned copy in five code-owned category slots."""

    economy: LodgingClassCopyDraft
    comfort: LodgingClassCopyDraft
    upscale: LodgingClassCopyDraft
    luxury: LodgingClassCopyDraft
    boutique_resort: LodgingClassCopyDraft


def _require_lodging_class_label(label: str) -> None:
    normalized = normalized_visible_text(label)
    if any(token in normalized for token in ("酒店", "宾馆", "饭店", "客栈", "民宿", "hotel")):
        raise ValueError("lodging class label must describe a tier, not a lodging property")


def _require_lodging_class_families(
    directions: list[LodgingClassDirectionDraft],
) -> None:
    signals = {
        "economy": ("经济", "实惠", "预算"),
        "comfort": ("舒适", "中档", "均衡"),
        "upscale": ("高档", "高端", "品质"),
        "luxury": ("豪华", "奢华"),
        "boutique_resort": ("精品", "度假", "特色"),
    }
    for item in directions:
        label = normalized_visible_text(item.label)
        if not any(token in label for token in signals[item.direction_id]):
            raise ValueError("lodging class label does not match its frozen direction family")


class FlexibleAttractionDirectionPlan(V4ContractModel):
    """Qwen-owned attraction content and roles before code-owned IDs."""

    directions: list[AttractionDirectionDraft] = Field(
        min_length=6,
        max_length=7,
        description="六到七个内容完整、语义彼此不同的中文景点体验方向。",
    )


class FlexibleDiningDirectionPlan(V4ContractModel):
    """Qwen-owned dining content with the frozen five-direction count."""

    directions: list[DirectionDraft] = Field(
        min_length=5,
        max_length=5,
        description="正好五个内容完整、语义彼此不同的中文餐饮体验方向。",
    )


class DiningDirectionDraft(DirectionDraft):
    dining_search_hints: DiningSearchHints


class DiningPreferencePlan(V4ContractModel):
    directions: list[DiningDirectionDraft] = Field(min_length=1)


class PreferenceDirectionGenerator:
    """Let Qwen choose city-specific content while code owns composition rules."""

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        attraction_gateway: ModelGateway | None = None,
        dining_gateway: ModelGateway | None = None,
    ) -> None:
        self._gateway = gateway
        self._attraction_gateway = attraction_gateway or gateway
        self._dining_gateway = dining_gateway

    async def generate_attraction(
        self,
        state: TripSemanticState,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> tuple[AttractionDirectionPlan, Literal["qwen", "safe_seed_fallback"]]:
        result, mode = await self._generate(
            state,
            section=DiscoverySection.ATTRACTION_PREFERENCE,
            output_type=FlexibleAttractionDirectionPlan,
            rules=ATTRACTION_PREFERENCE_SYSTEM_PROMPT,
            fallback=None,
            normalize=lambda value: _normalize_attraction_directions(
                value,
                destination_name=state.trip_basics.destination_name,
            ),
            cancellation=cancellation,
        )
        assert isinstance(result, AttractionDirectionPlan)
        return result, mode

    async def generate_dining(
        self,
        state: TripSemanticState,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> tuple[DiningDirectionPlan | DiningPreferencePlan, Literal["qwen", "safe_seed_fallback"]]:
        await report_dining_progress(DiningProgress.PREFERENCE_DISCOVERY)
        if self._dining_gateway is not None:
            from backend.discovery.cards.dining.model import (
                preference_context,
                preference_errors,
                run_task,
            )

            output = await run_task(
                self._dining_gateway,
                "B",
                preference_context(state),
                cancellation=cancellation,
                validator=preference_errors,
            )
            await report_dining_progress(DiningProgress.PREFERENCE_READY)
            return DiningPreferencePlan(
                directions=[
                    DiningDirectionDraft(
                        direction_id="dining_"
                        + sha256(
                            f"{state.trip_basics.destination_canonical_id}:{row['kind']}:{row['label']}".encode()
                        ).hexdigest()[:16],
                        label=row["label"],
                        description=row["description"],
                        composition_role=CompositionRole.REPRESENTATIVE_EXTRA
                        if row["kind"] == "local_specialty"
                        else CompositionRole.PERSONALIZED_TOP,
                        tags=[],
                        search_query=row["search_keywords"][0],
                        dining_search_hints=DiningSearchHints(
                            kind=row["kind"],
                            representative_restaurants=row["representative_restaurants"],
                            search_keywords=row["search_keywords"],
                        ),
                    )
                    for row in output["directions"]
                ]
            ), "qwen"
        result, mode = await self._generate(
            state,
            section=DiscoverySection.DINING_PREFERENCE,
            output_type=FlexibleDiningDirectionPlan,
            rules=(
                MAIN_MEAL_DIRECTION_RULES + "\n"
                "生成正好5个餐饮方向。正好2个是该城市代表性饮食，标为 representative_extra；"
                "另外3个结合用户偏好和常规菜系，标为 personalized_top。"
                "每个 representative_extra 的标签或说明中必须明确写出目的地城市名，"
                "以证明不是通用模板。"
                "先尊重过敏、宗教、医疗和明确禁忌。"
                "方向不是具体餐厅，不能编造价格、地址或营业信息。"
                "说明用一句亲切自然的话介绍口味或用餐体验，约15到35字，不重述标题。"
                "不写用户尚未选择、匹配偏好、探索候选、结合同行人等内部过程；必要的饮食风险仍须说清。"
                "每个标签必须是完整、可理解的短语，每段说明必须是完整句子；不得输出单字、残句、占位词或内部枚举。"
            ),
            fallback=FlexibleDiningDirectionPlan(directions=_fallback_dining(state).directions),
            normalize=lambda value: _normalize_dining_directions(
                value,
                destination_name=state.trip_basics.destination_name,
            ),
            cancellation=cancellation,
        )
        assert isinstance(result, DiningDirectionPlan)
        return result, mode

    async def generate_lodging_area(
        self,
        state: TripSemanticState,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> tuple[LodgingAreaPlan, Literal["qwen", "safe_seed_fallback"]]:
        result, mode = await self._generate(
            state,
            section=DiscoverySection.LODGING_AREA_PREFERENCE,
            output_type=LodgingAreaPlan,
            rules=LODGING_AREA_SYSTEM_PROMPT,
            fallback=None,
            normalize=lambda value: value,
            cancellation=cancellation,
        )
        assert isinstance(result, LodgingAreaPlan)
        return result, mode

    async def generate_lodging_class(
        self,
        state: TripSemanticState,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> tuple[LodgingClassDirectionPlan, Literal["qwen", "safe_seed_fallback"]]:
        result, mode = await self._generate(
            state,
            section=DiscoverySection.LODGING_CLASS_PREFERENCE,
            output_type=LodgingClassContentPlan,
            rules=(
                "分别填写固定的 economy、comfort、upscale、luxury、boutique_resort 五个字段，"
                "五个中文标签必须彼此不同；"
                "根据目的地、同行人、节奏和限制生成自然中文标签与说明。标签只描述档次或类型，"
                "不得包含酒店、宾馆、饭店、客栈、民宿、hotel等住宿实体词，也不得输出具体酒店名称；"
                "不得自行编造价格；实时价格将由程序另行附加。"
                "description只用一句话讲品质或体验区别，约15到30字，不重述标签或用户条件，"
                "不写结合同行人、参与筛选、供应样本等内部说明；不承诺未核实的具体设施或服务。"
            ),
            fallback=_fallback_lodging_class(state),
            normalize=_normalize_lodging_class_directions,
            cancellation=cancellation,
        )
        assert isinstance(result, LodgingClassDirectionPlan)
        return result, mode

    async def _generate(
        self,
        state: TripSemanticState,
        *,
        section: DiscoverySection,
        output_type: type[V4ContractModel],
        rules: str,
        fallback: V4ContractModel | None,
        normalize: Callable[[V4ContractModel], V4ContractModel],
        cancellation: ModelCancellation | None,
        provider_feedback: dict[str, object] | None = None,
    ) -> tuple[V4ContractModel, Literal["qwen", "safe_seed_fallback"]]:
        is_attraction = section is DiscoverySection.ATTRACTION_PREFERENCE
        is_lodging = section is DiscoverySection.LODGING_AREA_PREFERENCE
        structured = is_attraction or is_lodging
        if is_attraction:
            await report_attraction_progress(AttractionProgress.PREFERENCE_DISCOVERY)
        gateway = self._attraction_gateway if structured else self._gateway
        prompt_version = (
            ATTRACTION_PREFERENCE_PROMPT_VERSION
            if is_attraction
            else "lodging-area-v8"
            if is_lodging
            else DINING_PREFERENCE_PROMPT_VERSION
            if section is DiscoverySection.DINING_PREFERENCE
            else PREFERENCE_DIRECTION_PROMPT_VERSION
        )
        payload: dict[str, object] = {
            "prompt_version": prompt_version,
            "section": section.value,
            "trip_context": _state_excerpt(state),
        }
        if is_lodging:
            payload = {
                "destination": state.trip_basics.destination_name,
                "attractions": [
                    {"name": item.display_name, "intention": item.disposition}
                    for item in state.attractions.concrete_intents
                ],
                "preferences": state.transport_and_pace.model_dump(mode="json"),
                "travelers": state.trip_basics.travelers,
            }
        if structured:
            system_prompt = rules
        else:
            payload = {
                "prompt_version": prompt_version,
                "section": section.value,
                "rules": rules,
                "visible_text_quality_contract": _visible_text_quality_contract(state, section),
                "required_output_json_schema": output_type.model_json_schema(),
                "trip_context": payload["trip_context"],
            }
            system_prompt = (
                "你为旅行 Prepare Agent 生成城市化偏好方向。只输出请求的结构；"
                "不输出思维过程，不把模型记忆中的动态事实当作已核验事实。"
                "只输出一个 JSON 对象，不要 Markdown。"
                "所有字符串字段都必须填写有语义的完整中文内容，"
                "绝不能用单字或最短占位符填充结构。"
                "必须逐字段遵守 required_output_json_schema，"
                "不得改名、漏字段或增加 schema 之外的字段。"
            )
        if provider_feedback is not None:
            payload["provider_feedback"] = provider_feedback
        last_call_id: str | None = None
        for attempt in range(2):
            result_call_id: str | None = None
            if attempt:
                payload["repair_instruction"] = (
                    "请按原 Schema 修正上一轮不合规字段，保留简洁的展示文案。"
                    if is_lodging
                    else _repair_instruction(section)
                )
            try:
                result = await gateway.generate_structured(
                    ModelRequest(
                        audit=ModelAuditMetadata(
                            stage=f"prepare_{section.value}_direction_generation",
                            node="prepare_main_action",
                            contract_version=prompt_version,
                            repair=attempt > 0,
                            attempt=attempt + 1,
                        ),
                        messages=[
                            ModelMessage(
                                role=ModelRole.SYSTEM,
                                content=system_prompt,
                            ),
                            ModelMessage(
                                role=ModelRole.USER,
                                content=json.dumps(
                                    payload, ensure_ascii=False, separators=(",", ":")
                                ),
                            ),
                        ],
                        max_output_tokens=(
                            6_144 if section is DiscoverySection.ATTRACTION_PREFERENCE else 4_000
                        ),
                        structured_output_mode="json_schema" if structured else "json_object",
                        output_schema_override=attraction_schema(output_type)
                        if structured
                        else None,
                        temperature_override=0.65,
                        request_timeout_seconds=90 if structured else None,
                    ),
                    output_type,
                    cancellation=cancellation,
                )
                result_call_id = result.audit_call_id
                last_call_id = result_call_id or last_call_id
                if hasattr(result.value, "directions"):
                    payload["rejected_direction_labels"] = [
                        item.label for item in result.value.directions
                    ]
                normalized = normalize(result.value)
                await record_model_call_annotation(
                    gateway,
                    result_call_id,
                    "llm_business_guard",
                    {
                        "business_guard_result": {
                            "status": "accepted",
                            "guard": "preference_direction_quality",
                        },
                        "accepted_or_rejected": "accepted",
                        "materialized_output": normalized.model_dump(mode="json"),
                    },
                )
                return normalized, "qwen"
            except ModelGatewayError as error:
                if error.code is ModelFailureCode.CANCELLED:
                    raise
                last_call_id = error.audit_call_id or result_call_id or last_call_id
                payload["repair_validation_code"] = f"model_{error.code.value}"
                payload["repair_contract_issues"] = list(error.validation_issues)
                logger.warning(
                    "preference direction model attempt rejected section=%s attempt=%d "
                    "failure_code=%s validation_issues=%s",
                    section.value,
                    attempt + 1,
                    f"model_{error.code.value}",
                    ",".join(error.validation_issues) or "none",
                )
                if error.retryable and attempt < 1:
                    await asyncio.sleep(0.75 * (attempt + 1))
                continue
            except ValueError as error:
                last_call_id = result_call_id or last_call_id
                await record_model_call_annotation(
                    gateway,
                    result_call_id,
                    "llm_business_guard",
                    {
                        "business_guard_result": {
                            "status": "rejected",
                            "guard": "preference_direction_quality",
                            "reason": _direction_failure_code(error),
                        },
                        "accepted_or_rejected": "rejected",
                        "failure_stage": "business_guard",
                        "failure_code": _direction_failure_code(error),
                        "request_guard_feedback_full": _repair_instruction(section),
                    },
                )
                payload["repair_validation_code"] = _direction_failure_code(error)
                logger.warning(
                    "preference direction quality attempt rejected section=%s attempt=%d "
                    "failure_code=%s",
                    section.value,
                    attempt + 1,
                    _direction_failure_code(error),
                )
                continue
        if fallback is None:
            from backend.discovery.cards.candidate_composition import CardGenerationError

            raise CardGenerationError(
                "attraction direction generation did not complete",
                code="candidate_direction_unavailable",
                recoverable=True,
            )
        normalized_fallback = normalize(fallback)
        await record_model_call_annotation(
            gateway,
            last_call_id,
            "llm_fallback_selected",
            {
                "accepted_or_rejected": "fallback",
                "failure_stage": "preference_direction_generation",
                "failure_code": str(payload.get("repair_validation_code") or "attempts_exhausted"),
                "generation_mode": "safe_seed_fallback",
                "section": section.value,
                "materialized_output": normalized_fallback.model_dump(mode="json"),
            },
        )
        return normalized_fallback, "safe_seed_fallback"


def _repair_instruction(section: DiscoverySection) -> str:
    common = (
        "上一次结果未通过展示前Guard；只修复结构和内容质量，不改变用户事实。"
        "所有用户可见标签必须是至少两个有效字符的完整短语，说明必须是完整句子；"
        "不得输出单字、残句、占位编号、JSON片段或内部英文枚举。"
    )
    section_rules = {
        DiscoverySection.ATTRACTION_PREFERENCE: (
            "输出6到7个体验方向，3到4个城市代表、2到3个个性化，默认4+2或4+3；"
            "标题尽量少用具体景点名，描述也少量酌情点到；合理点名不是错误。避免整组句式雷同，检查语义差异和排除要求。"
            "不得出现以美食、小吃、茶社茶馆、探店、餐厅或住宿为主题的景点标签，"
            "必须将这些错误方向重写为公园、街区、人文历史、自然或艺术空间的体验。"
            "程序只规范内部ID，不替你决定城市代表角色。"
        ),
        DiscoverySection.DINING_PREFERENCE: (
            MAIN_MEAL_DIRECTION_RULES + "\n"
            "输出正好5个语义不同的方向，标签不得同义重复；每个代表性方向的标签或说明必须写出目的地城市名；"
            "先尊重饮食硬约束。description只用一句简短的口味或体验介绍，不复述标题或生成过程。"
        ),
        DiscoverySection.LODGING_AREA_PREFERENCE: (
            "输出5到6个语义不同的区位方向；label与search_query分别唯一。"
            "search_query必须是目的地内可由地图解析的真实区域、商圈或交通节点短名称，"
            "不得包含酒店、宾馆、饭店、客栈、民宿、hotel等住宿词，也不得是具体酒店名称；"
            "description、city_atmosphere、anchor_fit、transport_characteristics和main_tradeoff均不能为空。"
            "description只用一句区位特点与必要取舍，其余四项是内部分析，不逐项拼入description。"
        ),
        DiscoverySection.LODGING_CLASS_PREFERENCE: (
            "economy、comfort、upscale、luxury、boutique_resort五个命名字段必须全部填写；"
            "五个中文标签彼此不同且只描述档次或类型；标签不得包含酒店、宾馆、饭店、客栈、民宿、"
            "hotel等住宿实体词，也不得包含具体酒店名称。"
            "description只用一句简短的品质或体验区别，不重复标签、用户条件或生成过程，不编造价格。"
        ),
    }
    return f"{common}{section_rules.get(section, '')}"


def _visible_text_quality_contract(
    state: TripSemanticState,
    section: DiscoverySection,
) -> dict[str, object]:
    city = state.trip_basics.destination_name or "目的地"
    common: dict[str, object] = {
        "invalid_examples": [
            {"label": city[:1], "description": "在"},
            {"label": "方向1", "description": "说明1"},
        ],
        "instruction": "正例只示范完整度，不得照抄；必须根据 trip_context 重新生成。",
    }
    examples: dict[DiscoverySection, dict[str, str]] = {
        DiscoverySection.ATTRACTION_PREFERENCE: {
            "label": f"{city}历史文化脉络",
            "description": f"沿着{city}具有代表性的历史层次，安排可理解且节奏合适的人文体验。",
        },
        DiscoverySection.DINING_PREFERENCE: {
            "label": f"{city}地方风味体验",
            "description": "尝尝当地家常菜，感受日常餐桌上的地道味道。",
        },
        DiscoverySection.LODGING_AREA_PREFERENCE: {
            "label": "城市核心交通便利",
            "description": "出门逛街、乘车方便，繁忙时段也会更热闹。",
        },
        DiscoverySection.LODGING_CLASS_PREFERENCE: {
            "label": "舒适型品质住宿",
            "description": "更注重睡得舒服，兼顾房间品质和预算。",
        },
    }
    common["valid_style_example"] = examples.get(
        section,
        {
            "label": "完整中文方向",
            "description": "这里必须填写能够直接展示给用户的完整中文说明句。",
        },
    )
    return common


def _direction_failure_code(error: ValueError) -> str:
    """Reduce validation failures to value-free diagnostics."""

    message = str(error).casefold()
    patterns = (
        ("attraction_direction_domain_mismatch", "attraction_direction_domain_mismatch"),
        ("must name the destination", "missing_city_grounding"),
        ("near-duplicate", "near_duplicate_label"),
        ("unique values", "duplicate_label"),
        ("too_short", "text_too_short"),
        ("residual_fragment", "residual_fragment"),
        ("placeholder_text", "placeholder_text"),
        ("raw_internal_enum", "raw_internal_enum"),
        ("representative", "representative_quota"),
        ("validation error", "structured_validation"),
    )
    return next((code for token, code in patterns if token in message), "normalization_rejected")


def _normalize_lodging_class_directions(value: V4ContractModel) -> LodgingClassDirectionPlan:
    if isinstance(value, LodgingClassDirectionPlan):
        return value
    if not isinstance(value, LodgingClassContentPlan):
        raise ValueError("lodging class directions use an invalid content contract")
    content: tuple[tuple[LodgingClassDirectionId, LodgingClassCopyDraft], ...] = (
        ("economy", value.economy),
        ("comfort", value.comfort),
        ("upscale", value.upscale),
        ("luxury", value.luxury),
        ("boutique_resort", value.boutique_resort),
    )
    directions = [
        LodgingClassDirectionDraft(
            direction_id=direction_id,
            label=copy.label,
            description=copy.description,
        )
        for direction_id, copy in content
    ]
    return LodgingClassDirectionPlan(directions=directions)


def _normalize_attraction_directions(
    value: V4ContractModel,
    *,
    destination_name: str | None,
) -> AttractionDirectionPlan:
    if not isinstance(value, FlexibleAttractionDirectionPlan):
        raise ValueError("attraction directions require six or seven generated directions")
    plan = AttractionDirectionPlan(
        directions=[
            item.model_copy(update={"direction_id": _canonical_direction_id(item.label)})
            for item in value.directions
        ]
    )
    return plan


def _normalize_dining_directions(
    value: V4ContractModel,
    *,
    destination_name: str | None,
) -> DiningDirectionPlan:
    if not isinstance(value, FlexibleDiningDirectionPlan):
        raise ValueError("dining directions require five generated directions")
    plan = DiningDirectionPlan(
        directions=_normalize_composition_roles(
            value.directions,
            minimum_representative_count=2,
            maximum_representative_count=2,
        )
    )
    _require_city_grounded_representatives(plan.directions, destination_name)
    return plan


def _normalize_lodging_area_directions(
    value: V4ContractModel,
) -> LodgingAreaDirectionPlan:
    if not isinstance(value, FlexibleLodgingAreaDirectionPlan):
        raise ValueError("lodging area directions use an invalid draft contract")
    require_distinct_visible_labels(
        (item.label for item in value.directions),
        "lodging direction labels",
    )
    require_unique(
        (item.search_query.casefold() for item in value.directions),
        "lodging area search query",
    )
    if any(
        any(
            token in "".join(item.search_query.casefold().split())
            for token in ("酒店", "宾馆", "客栈", "民宿", "hotel")
        )
        for item in value.directions
    ):
        raise ValueError("lodging area query cannot target a named hotel")
    return LodgingAreaDirectionPlan(
        directions=[
            item.model_copy(update={"direction_id": _canonical_direction_id(item.label)})
            for item in value.directions
        ]
    )


def _normalize_composition_roles(
    directions: list[DirectionDraft],
    *,
    minimum_representative_count: int,
    maximum_representative_count: int,
) -> list[DirectionDraft]:
    require_distinct_visible_labels(
        (item.label for item in directions),
        "direction labels",
    )
    representative_indexes = [
        index
        for index, item in enumerate(directions)
        if item.composition_role is CompositionRole.REPRESENTATIVE_EXTRA
    ]
    personalized_indexes = [
        index
        for index, item in enumerate(directions)
        if item.composition_role is CompositionRole.PERSONALIZED_TOP
    ]
    representative_count = (
        len(representative_indexes)
        if minimum_representative_count
        <= len(representative_indexes)
        <= maximum_representative_count
        else minimum_representative_count
    )
    selected = [
        *representative_indexes[:representative_count],
        *personalized_indexes[: max(0, representative_count - len(representative_indexes))],
    ]
    selected_set = set(selected)
    return [
        item.model_copy(
            update={
                "direction_id": _canonical_direction_id(item.label),
                "composition_role": (
                    CompositionRole.REPRESENTATIVE_EXTRA
                    if index in selected_set
                    else CompositionRole.PERSONALIZED_TOP
                ),
            }
        )
        for index, item in enumerate(directions)
    ]


def _canonical_direction_id(label: str) -> str:
    normalized = "".join(label.casefold().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"direction-{digest}"


def _validate_directions(directions: Sequence[DirectionDraft | AttractionDirectionDraft]) -> None:
    require_unique((item.direction_id for item in directions), "direction_id")
    require_distinct_visible_labels(
        (item.label for item in directions),
        "direction labels",
    )


def _require_city_grounded_representatives(
    directions: list[DirectionDraft],
    destination_name: str | None,
) -> None:
    if destination_name is None:
        raise ValueError("city representative directions require a destination")
    city = normalized_visible_text(destination_name)
    if not city:
        raise ValueError("city representative directions require a readable destination")
    for item in directions:
        if item.composition_role is not CompositionRole.REPRESENTATIVE_EXTRA:
            continue
        visible_copy = normalized_visible_text(f"{item.label}{item.description}")
        if city not in visible_copy:
            raise ValueError("city representative direction must name the destination")


def _state_excerpt(state: TripSemanticState) -> dict[str, object]:
    return {
        "state_version": state.state_version,
        "destination": state.trip_basics.destination_name,
        "dates": {
            "start": (
                state.trip_basics.start_date.isoformat()
                if state.trip_basics.start_date is not None
                else None
            ),
            "end": (
                state.trip_basics.end_date.isoformat()
                if state.trip_basics.end_date is not None
                else None
            ),
            "duration_days": state.trip_basics.duration_days,
        },
        "travelers": state.trip_basics.travelers,
        "trip_goals": state.trip_basics.trip_goals,
        "attraction_preferences": [
            item.model_dump(mode="json") for item in state.attractions.preference_directions
        ],
        "attraction_anchors": [
            item.model_dump(mode="json") for item in state.attractions.concrete_intents
        ],
        "dining_preferences": [
            item.model_dump(mode="json") for item in state.dining.preference_directions
        ],
        "dining_anchors": [
            item.model_dump(mode="json") for item in state.dining.concrete_restaurant_intents
        ],
        "dining_requirements": [
            *state.dining.requirements,
            *state.dining.allergies,
            *state.dining.avoidances,
        ],
        "pace_and_transport": state.transport_and_pace.model_dump(mode="json"),
        "cold_start_defaults": (
            state.cold_start_profile_snapshot.model_dump(mode="json")
            if state.cold_start_profile_snapshot is not None
            else None
        ),
        "cold_start_readable_defaults": [
            item.value for item in cold_start_default_notes(state.cold_start_profile_snapshot)
        ],
        "preference_priority": "本次用户明确要求优先于冷启长期默认；不得用默认值覆盖本次选择。",
        "constraints": state.constraints,
    }


def _city(state: TripSemanticState) -> str:
    return state.trip_basics.destination_name or "目的地"


def _fallback_dining(state: TripSemanticState) -> DiningDirectionPlan:
    city = _city(state)
    values = (
        (
            "local_signature",
            f"{city}特色正餐",
            "以本地代表菜肴搭配主食，品尝一顿完整的午餐或晚餐。",
            "representative_extra",
        ),
        (
            "local_daily",
            f"{city}家常正餐",
            "用当地家常菜搭配主食，吃一顿踏实的午餐或晚餐。",
            "representative_extra",
        ),
        ("light_balanced", "清淡均衡", "减少重油重辣，兼顾同行人舒适度。", "personalized_top"),
        ("quality_dinner", "品质正餐", "为一顿完整、有记忆点的晚餐留出空间。", "personalized_top"),
        (
            "flexible_familiar",
            "熟悉菜系与灵活选择",
            "在城市特色之外保留稳妥选项。",
            "personalized_top",
        ),
    )
    return DiningDirectionPlan(
        directions=[
            DirectionDraft(
                direction_id=key,
                label=label,
                description=description,
                composition_role=CompositionRole(role),
            )
            for key, label, description, role in values
        ]
    )


def _fallback_lodging_area(state: TripSemanticState) -> LodgingAreaDirectionPlan:
    city = _city(state)
    values = (
        (
            "central",
            "市中心综合便利",
            f"{city}市中心",
            "城市核心",
            "兼顾多个方向",
            "公共交通密集",
            "高峰更拥挤",
        ),
        (
            "shopping",
            "成熟商圈附近",
            f"{city}商圈",
            "热闹便利",
            "方便餐饮和夜间活动",
            "地铁与出租车便利",
            "环境可能偏商业化",
        ),
        (
            "transit",
            "交通枢纽附近",
            f"{city}地铁枢纽",
            "高效通勤",
            "适合跨区域安排",
            "换乘便利",
            "度假氛围较弱",
        ),
        (
            "scenic",
            "景区与历史街区附近",
            f"{city}历史街区",
            "城市特色浓",
            "靠近代表性体验",
            "步行友好度视区域而定",
            "价格与客流波动较大",
        ),
        (
            "waterfront",
            "滨水或公园环境",
            f"{city}滨水公园",
            "安静舒展",
            "适合轻松节奏",
            "通常需换乘到核心景点",
            "通勤距离可能增加",
        ),
    )
    return LodgingAreaDirectionPlan(
        directions=[
            LodgingAreaDirectionDraft(
                direction_id=key,
                label=label,
                description=f"{atmosphere}；{tradeoff}。",
                search_query=query,
                city_atmosphere=atmosphere,
                anchor_fit=fit,
                transport_characteristics=transport,
                main_tradeoff=tradeoff,
            )
            for key, label, query, atmosphere, fit, transport, tradeoff in values
        ]
    )


def _fallback_lodging_class(state: TripSemanticState) -> LodgingClassDirectionPlan:
    del state
    return LodgingClassDirectionPlan(
        directions=[
            LodgingClassDirectionDraft(
                direction_id="economy", label="经济实用", description="优先基本舒适与预算效率。"
            ),
            LodgingClassDirectionDraft(
                direction_id="comfort",
                label="舒适中档",
                description="在房间品质、位置和价格之间取平衡。",
            ),
            LodgingClassDirectionDraft(
                direction_id="upscale", label="高档品质", description="更重视服务、设施和稳定体验。"
            ),
            LodgingClassDirectionDraft(
                direction_id="luxury",
                label="豪华享受",
                description="把住宿本身作为旅行体验的一部分。",
            ),
            LodgingClassDirectionDraft(
                direction_id="boutique_resort",
                label="特色精品或度假型",
                description="偏好有设计感、在地感或度假氛围的物业类型。",
            ),
        ]
    )


__all__ = [
    "AttractionDirectionPlan",
    "DiningDirectionPlan",
    "LodgingAreaDirectionPlan",
    "LodgingClassDirectionPlan",
    "PreferenceDirectionGenerator",
]
