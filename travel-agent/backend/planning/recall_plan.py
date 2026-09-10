"""One-call model planning plus deterministic validation for V3-30 recall."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from types import GenericAlias
from typing import Literal, Protocol, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, JsonValue, ValidationError, create_model

from backend.agent.model_audit import record_model_call_annotation
from backend.agent.model_gateway import (
    ModelAuditMetadata,
    ModelCancellation,
    ModelContract,
    ModelFailureCode,
    ModelGateway,
    ModelGatewayError,
    ModelMessage,
    ModelRequest,
    ModelRole,
)
from backend.contracts.candidate_recall import (
    CandidateDomain,
    CandidateRecallRequest,
    ContentSeedRecallQuery,
    LandmarkRecallPolicy,
    ProviderRecallQuery,
    RecallChannel,
    RecallPlan,
    RecallThemeMode,
)
from backend.contracts.city_content import RegisteredCityContentPackage
from backend.contracts.city_registry import CityRegistration

RECALL_PROMPT_VERSION = "candidate-recall-plan-v1.0"
DISCOVERY_RECALL_PROMPT_VERSION = "discovery-recall-plan-v4-03-11"

# Legal broad-search arguments, not city seeds or a preselected query sequence.
# Qwen chooses the categories; every named-place hypothesis remains free text.
_ATTRACTION_CATEGORY_KEYWORDS = frozenset(
    {
        "博物馆",
        "美术馆",
        "科技馆",
        "纪念馆",
        "天文馆",
        "公园",
        "湿地公园",
        "森林公园",
        "植物园",
        "动物园",
        "历史街区",
        "历史建筑",
        "名人故居",
        "古镇",
        "古村落",
        "寺庙",
        "道观",
        "教堂",
        "遗址公园",
        "园林",
        "王府",
        "宫殿",
        "城墙",
        "购物中心",
        "商场",
    }
)


class RecallPlanError(ValueError):
    pass


class ContentSeedRecallQueryDraft(ModelContract):
    channel: RecallChannel
    theme_ids: tuple[str, ...] = ()
    max_results: int = Field(ge=1, le=20, strict=True)
    reason: str = Field(min_length=1, max_length=500)


class ProviderRecallQueryDraft(ModelContract):
    channel: RecallChannel
    domain: CandidateDomain
    keyword: str = Field(min_length=1, max_length=200)
    search_scope: Literal["named_place", "category"] = "category"
    theme_ids: tuple[str, ...] = ()
    typecodes: tuple[str, ...] = ()
    named_clue_id: UUID | None = None
    anchor_id: UUID | None = None
    radius_m: int | None = Field(default=None, ge=100, le=50_000, strict=True)
    max_results: int = Field(ge=1, le=25, strict=True)
    reason: str = Field(min_length=1, max_length=500)


class RecallPlanDraft(ModelContract):
    """Model-owned search angles; all protocol and query IDs are server-owned."""

    content_queries: tuple[ContentSeedRecallQueryDraft, ...] = ()
    provider_queries: tuple[ProviderRecallQueryDraft, ...] = ()


class ProviderOnlyRecallQueryDraft(ProviderRecallQueryDraft):
    """No guessed user-named or reviewed-content provenance in live card plans."""

    channel: Literal[
        RecallChannel.CITY_LANDMARK,
        RecallChannel.EXPLORATION,
        RecallChannel.NEARBY_ANCHOR,
    ]
    keyword: str = Field(min_length=2, max_length=80)
    search_scope: Literal["named_place", "category"] = Field(
        description=(
            "Must be explicit. named_place means one proper place/brand name, "
            "even for exploration; "
            "category means a generic searchable venue class or cuisine, never one named site."
        )
    )
    reason: str = Field(min_length=6, max_length=160)
    named_clue_id: None = None


class ProviderOnlyRecallPlanDraft1(RecallPlanDraft):
    provider_queries: tuple[ProviderOnlyRecallQueryDraft, ...] = Field(
        min_length=1,
        max_length=12,
    )


class ProviderOnlyRecallPlanDraft2(RecallPlanDraft):
    provider_queries: tuple[ProviderOnlyRecallQueryDraft, ...] = Field(
        min_length=2,
        max_length=12,
    )


class ProviderOnlyRecallPlanDraft3(RecallPlanDraft):
    provider_queries: tuple[ProviderOnlyRecallQueryDraft, ...] = Field(
        min_length=3,
        max_length=12,
    )


class ProviderOnlyRecallPlanDraft4(RecallPlanDraft):
    provider_queries: tuple[ProviderOnlyRecallQueryDraft, ...] = Field(
        min_length=4,
        max_length=12,
    )


class ProviderOnlyRecallPlanDraft5(RecallPlanDraft):
    provider_queries: tuple[ProviderOnlyRecallQueryDraft, ...] = Field(
        min_length=5,
        max_length=12,
    )


class ProviderOnlyRecallPlanDraft6(RecallPlanDraft):
    provider_queries: tuple[ProviderOnlyRecallQueryDraft, ...] = Field(
        min_length=6,
        max_length=12,
    )


class ProviderOnlyRecallPlanDraft7(RecallPlanDraft):
    provider_queries: tuple[ProviderOnlyRecallQueryDraft, ...] = Field(
        min_length=7,
        max_length=12,
    )


class ProviderOnlyRecallPlanDraft8(RecallPlanDraft):
    provider_queries: tuple[ProviderOnlyRecallQueryDraft, ...] = Field(
        min_length=8,
        max_length=12,
    )


_PROVIDER_ONLY_DRAFT_CONTRACTS = {
    1: ProviderOnlyRecallPlanDraft1,
    2: ProviderOnlyRecallPlanDraft2,
    3: ProviderOnlyRecallPlanDraft3,
    4: ProviderOnlyRecallPlanDraft4,
    5: ProviderOnlyRecallPlanDraft5,
    6: ProviderOnlyRecallPlanDraft6,
    7: ProviderOnlyRecallPlanDraft7,
    8: ProviderOnlyRecallPlanDraft8,
}


def _provider_only_output_contract(
    request: CandidateRecallRequest,
) -> type[RecallPlanDraft]:
    """Expose only current stable preference references to constrained generation.

    The final plan validator remains authoritative. JSON Schema enumeration
    prevents the model from inventing or truncating IDs in the first place;
    it does not invent any relationship between a query and a preference.
    """

    theme_ids: list[JsonValue] = [item.theme_id for item in request.themes]
    theme_field = (
        Field(
            default=(),
            max_length=len(theme_ids),
            json_schema_extra={"items": {"type": "string", "enum": theme_ids}},
        )
        if theme_ids
        else Field(
            default=(),
            max_length=12,
            description=(
                "No legal theme IDs exist for this request. Leave this empty; "
                "the server discards any supplied values instead of treating them as references."
            ),
        )
    )
    count = max(1, _required_provider_query_count(request))
    domain_field = Field(
        json_schema_extra={"enum": [item.domain.value for item in request.budget.domains]},
    )
    if (
        request.composition_feedback is not None
        and _required_representative_query_count(request) == count
    ):
        query_contract = create_model(
            "ProviderOnlyRecallQueryForTurn",
            __base__=ProviderOnlyRecallQueryDraft,
            theme_ids=(tuple[str, ...], theme_field),
            domain=(CandidateDomain, domain_field),
            channel=(Literal[RecallChannel.CITY_LANDMARK], Field(...)),
        )
    else:
        query_contract = create_model(
            "ProviderOnlyRecallQueryForTurn",
            __base__=ProviderOnlyRecallQueryDraft,
            theme_ids=(tuple[str, ...], theme_field),
            domain=(CandidateDomain, domain_field),
        )
    return cast(
        type[RecallPlanDraft],
        create_model(
            "ProviderOnlyRecallPlanForTurn",
            __base__=_PROVIDER_ONLY_DRAFT_CONTRACTS[count],
            provider_queries=(
                GenericAlias(tuple, (query_contract, Ellipsis)),
                Field(min_length=count, max_length=count),
            ),
        ),
    )


class RecallPlanGenerator(Protocol):
    async def create_plan(
        self,
        request: CandidateRecallRequest,
        city: CityRegistration,
        content: RegisteredCityContentPackage | None,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> RecallPlan: ...


class ModelRecallPlanGenerator:
    """Ask the model for search intent; code owns IDs, validation, and repair bounds."""

    def __init__(self, gateway: ModelGateway, *, provider_only: bool = False) -> None:
        self._gateway = gateway
        self._provider_only = provider_only

    async def create_plan(
        self,
        request: CandidateRecallRequest,
        city: CityRegistration,
        content: RegisteredCityContentPackage | None,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> RecallPlan:
        payload = _normalized_planning_context(request, city, content)
        payload["provider_only_selectable_entities"] = self._provider_only
        if self._provider_only:
            coarse_attraction = request.attraction_discovery is not None
            payload["provider_only_contract"] = {
                "prompt_version": DISCOVERY_RECALL_PROMPT_VERSION,
                "content_queries_must_be_empty": True,
                "distinct_provider_queries_per_domain": _required_provider_query_count(request),
                "provider_query_count_is_exact": True,
                "query_angles_must_be_semantically_distinct": not coarse_attraction,
                "cover_representative_preference_and_exploration_angles": (
                    not coarse_attraction and request.composition_feedback is None
                ),
                "minimum_city_landmark_queries": (
                    0 if coarse_attraction else _required_representative_query_count(request)
                ),
                "minimum_category_queries": (
                    0 if coarse_attraction else _required_category_query_count(request)
                ),
                "representative_reason_must_explain_city_specificity": True,
                "user_named_clue_queries_are_server_owned": True,
                "agent_named_search_hypotheses_are_allowed": True,
                "do_not_use_excluded_displayed_entities_or_other_branches_of_their_brands": True,
                "query_ids_are_server_owned": True,
                "provider_result_limits_are_server_owned": True,
                "provider_typecodes_are_server_owned": True,
                "search_scope_must_be_explicit": True,
                "category_keyword_is_one_searchable_noun_phrase": True,
                "allowed_domains": [item.domain.value for item in request.budget.domains],
                "attraction_category_keywords": sorted(_ATTRACTION_CATEGORY_KEYWORDS),
                "named_place_keywords_are_not_restricted_to_category_keywords": True,
                "attraction_queries_must_not_target_restaurants_cafes_or_hotels": True,
                "use_only_supplied_named_clue_and_anchor_ids": True,
            }
        output_type = (
            _provider_only_output_contract(request) if self._provider_only else RecallPlanDraft
        )
        last_model_error: ModelGatewayError | None = None
        last_validation_code = "invalid_plan"
        max_output_tokens = 2_048
        for attempt in range(3):
            result_call_id: str | None = None
            if attempt:
                payload["repair_instruction"] = (
                    "上一次搜索计划未通过结构、引用或预算校验。只修复查询角度；"
                    "生成足够多且语义不同的真实搜索关键词，保留所需城市代表性查询。"
                    "可以基于城市知识提出待查询的地点名，但不得编造地点 ID、主题 ID、"
                    "锚点或已核验事实；不得用通用句式或同一品牌分店凑数。"
                    "如果 repair_uncovered_themes 非空，必须为其中每个偏好生成相关的查询，"
                    "或把其真实 ID 标注到已有且确实相关的查询；不得遗漏或无关地贴标签。"
                    "仍须输出完整计划而非局部 Patch，保持精确查询条数，不要超预算。"
                )
                payload["repair_validation_code"] = last_validation_code
                if request.attraction_discovery is not None:
                    payload["repair_instruction"] = (
                        "只修复指出的结构、来源引用或预算问题，输出完整计划。"
                        "优先补充尚缺的城市代表和偏好方向；类别搜索与具体地名可以并存。"
                        "不强制主题全覆盖、类别配额或查询角度语义不同，不重做精筛。"
                        "已选项不重复搜索，未选候选不是新增禁忌，用户明确排除仍须遵守。"
                    )
            model_request = ModelRequest(
                audit=ModelAuditMetadata(
                    stage="candidate_recall_plan",
                    node="prepare_main_action",
                    contract_version=(
                        DISCOVERY_RECALL_PROMPT_VERSION
                        if self._provider_only
                        else RECALL_PROMPT_VERSION
                    ),
                    repair=attempt > 0,
                    attempt=attempt + 1,
                ),
                messages=[
                    ModelMessage(
                        role=ModelRole.SYSTEM,
                        content=(
                            _discovery_system_instruction()
                            if self._provider_only
                            else _system_instruction()
                        ),
                    ),
                    ModelMessage(
                        role=ModelRole.USER,
                        content=json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    ),
                ],
                max_output_tokens=max_output_tokens,
                # Live compatibility endpoints can satisfy a strict JSON shape
                # while repeatedly dropping theme links. Explicit schema context
                # is more reliable here; Pydantic and all plan guards still run.
                structured_output_mode="json_object" if self._provider_only else "json_schema",
            )
            try:
                result = await self._gateway.generate_structured(
                    model_request,
                    output_type,
                    cancellation=cancellation,
                )
                result_call_id = result.audit_call_id
                last_model_error = None
                payload["rejected_query_angles"] = [
                    {
                        "keyword": item.keyword,
                        "channel": item.channel.value,
                        "search_scope": item.search_scope,
                        "theme_ids": list(item.theme_ids),
                    }
                    for item in result.value.provider_queries
                ]
                covered_themes = {
                    theme_id
                    for query in (() if self._provider_only else result.value.content_queries)
                    for theme_id in query.theme_ids
                } | {
                    theme_id
                    for query in result.value.provider_queries
                    for theme_id in query.theme_ids
                }
                payload["repair_uncovered_themes"] = [
                    {"theme_id": theme.theme_id, "label": theme.label}
                    for theme in request.themes
                    if theme.theme_id not in covered_themes and request.composition_feedback is None
                ]
                payload["repair_invalid_category_keywords"] = [
                    query.keyword
                    for query in result.value.provider_queries
                    if self._provider_only
                    and query.domain is CandidateDomain.ATTRACTION
                    and query.search_scope == "category"
                    and query.keyword not in _ATTRACTION_CATEGORY_KEYWORDS
                ]
                plan = _materialize_recall_plan(
                    request,
                    result.value,
                    provider_only=self._provider_only,
                )
                validated = validate_recall_plan(request, plan, content)
                await record_model_call_annotation(
                    self._gateway,
                    result_call_id,
                    "llm_business_guard",
                    {
                        "business_guard_result": {
                            "status": "accepted",
                            "guard": "candidate_recall_plan_guard",
                        },
                        "accepted_or_rejected": "accepted",
                        "materialized_output": validated.model_dump(mode="json"),
                    },
                )
                return validated
            except ModelGatewayError as error:
                if error.code is ModelFailureCode.CANCELLED or error.requires_runtime_recovery:
                    raise
                last_model_error = error
                last_validation_code = "model_contract_validation_failed"
                payload["repair_contract_issues"] = list(error.validation_issues)
                if "completion:truncated" in error.validation_issues:
                    # Retry inside the existing three-call ceiling. Increase
                    # only the output allowance; do not alter or fill content.
                    max_output_tokens = 4_096
            except (RecallPlanError, ValueError) as error:
                last_validation_code = _safe_plan_validation_code(error)
                await record_model_call_annotation(
                    self._gateway,
                    result_call_id,
                    "llm_business_guard",
                    {
                        "business_guard_result": {
                            "status": "rejected",
                            "guard": "candidate_recall_plan_guard",
                            "reason": last_validation_code,
                        },
                        "accepted_or_rejected": "rejected",
                        "failure_stage": "business_guard",
                        "failure_code": last_validation_code,
                        "request_guard_feedback_full": payload.get("repair_instruction"),
                    },
                )
                continue
        if last_model_error is not None:
            raise ModelGatewayError(
                last_model_error.code,
                "candidate_recall_plan",
                retryable=last_model_error.retryable,
                validation_issues=last_model_error.validation_issues,
                audit_call_id=last_model_error.audit_call_id,
            ) from last_model_error
        raise RecallPlanError(f"model_repair_failed:{last_validation_code}")


def _materialize_recall_plan(
    request: CandidateRecallRequest,
    draft: RecallPlanDraft,
    *,
    provider_only: bool,
) -> RecallPlan:
    if provider_only:
        content_queries: tuple[ContentSeedRecallQuery, ...] = ()
        provider_queries = _materialize_provider_only_queries(request, draft)
    else:
        content_queries = tuple(
            ContentSeedRecallQuery(
                query_id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"recall-query:{request.request_id}:content:{index}",
                    )
                ),
                channel=item.channel,
                theme_ids=item.theme_ids,
                max_results=item.max_results,
                reason=item.reason,
            )
            for index, item in enumerate(draft.content_queries)
        )
        provider_queries = tuple(
            ProviderRecallQuery(
                query_id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"recall-query:{request.request_id}:provider:{index}",
                    )
                ),
                channel=item.channel,
                domain=item.domain,
                keyword=item.keyword,
                theme_ids=item.theme_ids,
                typecodes=item.typecodes,
                named_clue_id=item.named_clue_id,
                anchor_id=item.anchor_id,
                radius_m=item.radius_m,
                max_results=item.max_results,
                reason=item.reason,
            )
            for index, item in enumerate(draft.provider_queries)
        )
    return RecallPlan(
        plan_version="1.0.0",
        request_id=request.request_id,
        city_id=request.city_id,
        content_queries=content_queries,
        provider_queries=provider_queries,
    )


def _materialize_provider_only_queries(
    request: CandidateRecallRequest,
    draft: RecallPlanDraft,
) -> tuple[ProviderRecallQuery, ...]:
    domain_budgets = {item.domain: item for item in request.budget.domains}
    remaining_calls = {
        domain: budget.max_provider_calls for domain, budget in domain_budgets.items()
    }
    remaining_candidates = {
        domain: budget.max_candidates for domain, budget in domain_budgets.items()
    }
    remaining_total_calls = request.budget.max_total_provider_calls
    remaining_total_candidates = request.budget.max_total_candidates
    allowed_theme_ids = {item.theme_id for item in request.themes}
    queries: list[ProviderRecallQuery] = []

    def append_query(
        *,
        channel: RecallChannel,
        domain: CandidateDomain,
        keyword: str,
        reason: str,
        named_clue_id: UUID | None = None,
        anchor_id: UUID | None = None,
        radius_m: int | None = None,
        typecodes: tuple[str, ...] = (),
        requested_limit: int | None = None,
        theme_ids: tuple[str, ...] = (),
        search_scope: Literal["named_place", "category"] = "category",
    ) -> None:
        nonlocal remaining_total_calls, remaining_total_candidates
        if (
            domain not in domain_budgets
            or remaining_calls[domain] <= 0
            or remaining_total_calls <= 0
            or remaining_candidates[domain] <= 0
            or remaining_total_candidates <= 0
        ):
            return
        limit = min(
            requested_limit or 25,
            remaining_candidates[domain],
            remaining_total_candidates,
        )
        index = len(queries)
        queries.append(
            ProviderRecallQuery(
                query_id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"recall-query:{request.request_id}:provider:{index}",
                    )
                ),
                channel=channel,
                domain=domain,
                keyword=keyword,
                exact_name_match=channel is RecallChannel.CITY_LANDMARK
                or search_scope == "named_place",
                named_clue_id=named_clue_id,
                anchor_id=anchor_id,
                radius_m=radius_m,
                typecodes=typecodes,
                max_results=limit,
                reason=reason,
                theme_ids=theme_ids,
            )
        )
        remaining_calls[domain] -= 1
        remaining_candidates[domain] -= limit
        remaining_total_calls -= 1
        remaining_total_candidates -= limit

    for clue in request.named_places:
        append_query(
            channel=RecallChannel.USER_NAMED,
            domain=clue.domain,
            keyword=clue.name,
            reason="用户点名地点的精确解析",
            named_clue_id=clue.clue_id,
            requested_limit=1,
        )

    anchors = {item.anchor_id: item for item in request.anchors}
    requested_domains = tuple(domain_budgets)
    raw_items = [item for item in draft.provider_queries if item.named_clue_id is None]
    if len(requested_domains) == 1:
        sole_domain = requested_domains[0]
        unique_items: list[ProviderRecallQueryDraft] = []
        for item in raw_items:
            if item.domain is not sole_domain:
                raise RecallPlanError("provider-only query uses an unbudgeted candidate domain")
            if item.search_scope == "category" and re.search(r"[\s,，、|;；]", item.keyword):
                raise RecallPlanError("category search requires one keyword without separators")
            if (
                sole_domain is CandidateDomain.ATTRACTION
                and item.search_scope == "category"
                and item.keyword not in _ATTRACTION_CATEGORY_KEYWORDS
            ):
                raise RecallPlanError(
                    "attraction category keyword is not a legal broad venue class"
                )
            feedback = request.composition_feedback
            if feedback is not None and _normalize(item.keyword) in {
                _normalize(query) for query in feedback.failed_named_queries
            }:
                raise RecallPlanError("supplemental query repeats a failed named search")
            if feedback is not None and feedback.domain is CandidateDomain.RESTAURANT:
                keyword = _normalize(item.keyword)
                if any(
                    _normalize(brand) == keyword
                    or (len(_normalize(brand)) >= 3 and _normalize(brand) in keyword)
                    for brand in feedback.excluded_restaurant_brands
                ):
                    raise RecallPlanError("supplemental query repeats an observed restaurant brand")
            # For the coarse attraction flow, a broad category and a named
            # landmark are complementary retrievals, not duplicate entities.
            # Preserve the legacy heuristic for dining/older callers only.
            duplicate = (
                any(_normalize(item.keyword) == _normalize(other.keyword) for other in unique_items)
                if request.attraction_discovery is not None
                else _query_angle_is_covered(item.keyword, unique_items)
            )
            if duplicate:
                continue
            unique_items.append(item)
        required_count = min(
            _required_provider_query_count(request),
            remaining_calls[sole_domain],
            remaining_total_calls,
        )
        if len(unique_items) < required_count and request.attraction_discovery is None:
            raise RecallPlanError("provider-only plan requires more distinct query angles")
        # Prioritize model-proposed representative angles within the call budget;
        # never relabel an unrelated exploration to manufacture provenance.
        unique_items.sort(key=lambda item: item.channel is not RecallChannel.CITY_LANDMARK)
        raw_items = unique_items[:required_count]
        if request.composition_feedback is None and request.attraction_discovery is None:
            minimum_categories = _required_category_query_count(
                request, available_queries=required_count
            )
            category_count = sum(
                item.search_scope == "category" and item.channel is not RecallChannel.CITY_LANDMARK
                for item in raw_items
            )
            if category_count < minimum_categories:
                raise RecallPlanError("provider-only plan requires more broad category queries")

    for raw_index, item in enumerate(raw_items):
        if item.named_clue_id is not None:
            continue
        domain = item.domain
        if domain not in domain_budgets:
            if len(requested_domains) != 1:
                continue
            domain = requested_domains[0]
        anchor = anchors.get(item.anchor_id) if item.anchor_id is not None else None
        if anchor is not None and item.radius_m is not None and anchor.domain is domain:
            channel = RecallChannel.NEARBY_ANCHOR
            anchor_id = anchor.anchor_id
            radius_m = item.radius_m
        else:
            channel = (
                RecallChannel.EXPLORATION
                if item.channel in {RecallChannel.USER_NAMED, RecallChannel.NEARBY_ANCHOR}
                else item.channel
            )
            if channel is RecallChannel.CITY_FEATURE:
                channel = RecallChannel.EXPLORATION
            anchor_id = None
            radius_m = None
        append_query(
            channel=channel,
            domain=domain,
            keyword=item.keyword,
            reason=item.reason,
            anchor_id=anchor_id,
            radius_m=radius_m,
            # Broad AMap types can displace the actual keyword (observed even
            # for exact landmarks). Keep keyword retrieval precise; downstream
            # source-taxonomy/domain guards still reject wrong/unknown types.
            typecodes=(),
            # Some compatible endpoints insist on populating this optional
            # field even when the request exposes no legal theme reference.
            # Such strings are not Agent facts or permissions: ignore them
            # deterministically and never let them enter the authoritative plan.
            theme_ids=item.theme_ids if allowed_theme_ids else (),
            search_scope=item.search_scope,
            requested_limit=(
                min(
                    3
                    if channel is RecallChannel.CITY_LANDMARK
                    and request.composition_feedback is None
                    else 25,
                    max(
                        1,
                        min(
                            remaining_candidates[domain],
                            remaining_total_candidates,
                        )
                        // max(1, len(raw_items) - raw_index),
                    ),
                )
                if len(requested_domains) == 1
                else item.max_results
            ),
        )

    for domain in requested_domains:
        eligible_indexes = [
            index
            for index, item in enumerate(queries)
            if item.domain is domain and item.channel is not RecallChannel.USER_NAMED
        ]
        required_representative_queries = _required_representative_query_count(request)
        if domain is CandidateDomain.ATTRACTION and (
            request.landmark_policy is LandmarkRecallPolicy.EXCLUDE
        ):
            required_representative_queries = 0
        representative_indexes = [
            index
            for index in eligible_indexes
            if queries[index].channel in {RecallChannel.CITY_LANDMARK, RecallChannel.CITY_FEATURE}
        ]
        if (
            len(representative_indexes) < required_representative_queries
            and request.attraction_discovery is None
        ):
            raise RecallPlanError("provider-only plan requires explicit representative queries")
    queries.sort(
        key=lambda item: (
            item.channel is not RecallChannel.CITY_LANDMARK,
            item.channel is RecallChannel.USER_NAMED,
        )
    )
    return tuple(
        item.model_copy(
            update={
                "query_id": str(
                    uuid5(
                        NAMESPACE_URL,
                        f"recall-query:{request.request_id}:provider:{index}",
                    )
                )
            }
        )
        for index, item in enumerate(queries)
    )


def _required_provider_query_count(request: CandidateRecallRequest) -> int:
    target = 8 if request.day_count >= 4 else 6
    return min(target, request.budget.max_total_provider_calls)


def _required_representative_query_count(request: CandidateRecallRequest) -> int:
    if request.composition_feedback is not None:
        return min(
            request.composition_feedback.missing_representative_count,
            request.budget.max_total_provider_calls,
        )
    if request.landmark_policy is LandmarkRecallPolicy.EXCLUDE:
        return 0
    if request.dining_city_target is not None:
        # Leave room for broad cuisine queries; a single bounded supplement may
        # fill remaining city places for longer trips. Never invent candidates.
        return min(request.dining_city_target, 4, request.budget.max_total_provider_calls)
    if request.attraction_discovery is not None:
        return min(
            4 if request.day_count >= 3 else 3,
            max(
                0,
                request.budget.max_total_provider_calls - len(request.named_places) - 2,
            ),
        )
    return min(1 if request.day_count == 1 else 2, request.budget.max_total_provider_calls)


def _required_category_query_count(
    request: CandidateRecallRequest, *, available_queries: int | None = None
) -> int:
    if request.composition_feedback is not None:
        return 0
    count = (
        min(
            _required_provider_query_count(request),
            max(0, request.budget.max_total_provider_calls - len(request.named_places)),
        )
        if available_queries is None
        else available_queries
    )
    return min(
        3 if request.day_count >= 3 else 2,
        max(0, count - _required_representative_query_count(request)),
    )


def _query_angle_is_covered(
    keyword: str,
    existing: list[ProviderRecallQueryDraft],
) -> bool:
    normalized = _normalize(keyword)
    for item in existing:
        other = _normalize(item.keyword)
        if normalized == other:
            return True
        if min(len(normalized), len(other)) >= 4 and (
            normalized in other
            or other in normalized
            or SequenceMatcher(None, normalized, other).ratio() >= 0.86
        ):
            return True
    return False


def _safe_plan_validation_code(error: Exception) -> str:
    if isinstance(error, ValidationError):
        issues = error.errors(include_input=False, include_url=False)
        if issues:
            first = issues[0]
            detail = str(first["msg"]).casefold()
            detail_classifications = (
                ("typecodes must contain", "invalid_typecodes"),
                ("user-named queries require", "user_named_reference_mismatch"),
                ("only user-named queries", "unexpected_named_reference"),
                ("nearby queries require", "nearby_reference_missing"),
                ("only nearby queries", "unexpected_nearby_reference"),
                ("recall query ids", "duplicate_query_ids"),
                ("requires at least one query", "empty_plan"),
            )
            classified = next(
                (code for marker, code in detail_classifications if marker in detail),
                None,
            )
            if classified is not None:
                return classified
            location = "_".join(str(item) for item in first["loc"]) or "root"
            issue_type = str(first["type"])
            safe = "_".join(item for item in (location, issue_type) if item)
            return (
                "contract_"
                + "".join(
                    character if character.isalnum() or character == "_" else "_"
                    for character in safe.casefold()
                )[:72]
            )
    message = str(error).casefold()
    classifications = (
        ("does not belong to its request and city", "request_or_city_mismatch"),
        ("cannot use an unavailable city content package", "content_unavailable"),
        ("content seeds require an attraction recall budget", "content_without_attraction"),
        ("provider-only plan requires more distinct", "insufficient_provider_query_angles"),
        ("requires explicit representative queries", "missing_representative_queries"),
        ("requires more broad category queries", "insufficient_category_queries"),
        ("requires one keyword without separators", "compound_category_keyword"),
        ("not a legal broad venue class", "invalid_attraction_category_keyword"),
        ("repeats an observed restaurant brand", "supplement_repeats_restaurant_brand"),
        ("repeats a failed named search", "supplement_repeats_failed_named_search"),
        ("unknown city-content theme", "content_unknown_theme"),
        ("city-feature content queries require reviewed themes", "content_feature_without_theme"),
        ("landmark recall conflicts with an explicit exclusion", "landmark_excluded"),
        ("requires a landmark channel", "missing_landmark_channel"),
        ("exceeds the total provider-call budget", "total_provider_budget"),
        ("exceeds the total candidate budget", "total_candidate_budget"),
        ("exceeds attraction provider-call budget", "attraction_provider_budget"),
        ("exceeds restaurant provider-call budget", "restaurant_provider_budget"),
        ("exceeds attraction candidate budget", "attraction_candidate_budget"),
        ("exceeds restaurant candidate budget", "restaurant_candidate_budget"),
        ("unbudgeted candidate domain", "unbudgeted_domain"),
        ("unknown theme", "unknown_theme"),
        ("requires reviewed city-content themes", "city_feature_without_theme"),
        ("explicitly excluded place", "explicit_exclusion_queried"),
        ("unknown or wrong-domain clue", "unknown_named_clue"),
        ("preserve the user's exact place name", "named_place_changed"),
        ("unknown anchor", "unknown_anchor"),
        ("every named place requires", "named_place_not_covered"),
        ("every selected theme requires", "selected_theme_not_covered"),
        ("open-to-any recall requires", "insufficient_channel_diversity"),
        ("recall plan requires at least one query", "empty_plan"),
        ("unavailable city content", "content_unavailable"),
    )
    return next((code for marker, code in classifications if marker in message), "invalid_plan")


class StaticRecallPlanGenerator:
    """Deterministic Replay boundary using the same validated plan contract."""

    def __init__(self, plan: RecallPlan) -> None:
        self._plan = plan
        self.call_count = 0

    async def create_plan(
        self,
        request: CandidateRecallRequest,
        city: CityRegistration,
        content: RegisteredCityContentPackage | None,
        *,
        cancellation: ModelCancellation | None = None,
    ) -> RecallPlan:
        del city
        if cancellation is not None:
            cancellation.raise_if_cancelled("candidate_recall_plan")
        self.call_count += 1
        return validate_recall_plan(request, self._plan, content)


def validate_recall_plan(
    request: CandidateRecallRequest,
    plan: RecallPlan,
    content: RegisteredCityContentPackage | None,
) -> RecallPlan:
    plan = RecallPlan.model_validate(plan.model_dump(mode="json"))
    if plan.request_id != request.request_id or plan.city_id != request.city_id:
        raise RecallPlanError("recall plan does not belong to its request and city")
    if plan.content_queries and content is None:
        raise RecallPlanError("recall plan cannot use an unavailable city content package")

    budgets = {item.domain: item for item in request.budget.domains}
    requested_domains = set(budgets)
    provider_calls: dict[CandidateDomain, int] = {domain: 0 for domain in requested_domains}
    potential_candidates: dict[CandidateDomain, int] = {domain: 0 for domain in requested_domains}
    potential_candidates[CandidateDomain.ATTRACTION] = sum(
        query.max_results for query in plan.content_queries
    )

    allowed_theme_ids = {theme.theme_id for theme in request.themes}
    content_theme_ids = {theme.theme_id for theme in content.themes} if content else set()
    all_theme_ids = allowed_theme_ids | content_theme_ids
    covered_theme_ids: set[str] = set()
    named_by_id = {clue.clue_id: clue for clue in request.named_places}
    covered_named_ids: list[object] = []
    anchors = {anchor.anchor_id for anchor in request.anchors}
    excluded_names = {_normalize(clue.name) for clue in request.excluded_places}

    for query in plan.content_queries:
        if CandidateDomain.ATTRACTION not in requested_domains:
            raise RecallPlanError("content seeds require an attraction recall budget")
        if not set(query.theme_ids) <= content_theme_ids:
            raise RecallPlanError("content query references an unknown city-content theme")
        if query.channel is RecallChannel.CITY_FEATURE and not query.theme_ids:
            raise RecallPlanError("city-feature content queries require reviewed themes")
        if (
            request.landmark_policy is LandmarkRecallPolicy.EXCLUDE
            and query.channel is RecallChannel.CITY_LANDMARK
        ):
            raise RecallPlanError("landmark recall conflicts with an explicit exclusion")
        covered_theme_ids.update(query.theme_ids)

    for provider_query in plan.provider_queries:
        if provider_query.domain not in requested_domains:
            raise RecallPlanError("provider query uses an unbudgeted candidate domain")
        if not set(provider_query.theme_ids) <= all_theme_ids:
            raise RecallPlanError("provider query references an unknown theme")
        if provider_query.channel is RecallChannel.CITY_FEATURE and (
            content is None or not provider_query.theme_ids
        ):
            raise RecallPlanError("city-feature queries require reviewed city-content themes")
        if request.landmark_policy is LandmarkRecallPolicy.EXCLUDE and (
            provider_query.channel is RecallChannel.CITY_LANDMARK
        ):
            raise RecallPlanError("landmark recall conflicts with an explicit exclusion")
        if _normalize(provider_query.keyword) in excluded_names:
            raise RecallPlanError("recall plan queries an explicitly excluded place")
        if provider_query.named_clue_id is not None:
            clue = named_by_id.get(provider_query.named_clue_id)
            if clue is None or clue.domain is not provider_query.domain:
                raise RecallPlanError("named query references an unknown or wrong-domain clue")
            if _normalize(provider_query.keyword) != _normalize(clue.name):
                raise RecallPlanError("named-place query must preserve the user's exact place name")
            covered_named_ids.append(clue.clue_id)
        if provider_query.anchor_id is not None and provider_query.anchor_id not in anchors:
            raise RecallPlanError("nearby query references an unknown anchor")
        provider_calls[provider_query.domain] += 1
        potential_candidates[provider_query.domain] += provider_query.max_results
        covered_theme_ids.update(provider_query.theme_ids)

    if set(named_by_id) != set(covered_named_ids) or len(covered_named_ids) != len(named_by_id):
        raise RecallPlanError("every named place requires one exact provider query")
    if (
        request.theme_mode is RecallThemeMode.SELECTED
        and request.composition_feedback is None
        and request.attraction_discovery is None
        and not allowed_theme_ids <= covered_theme_ids
    ):
        raise RecallPlanError("every selected theme requires at least one recall query")
    channels = {
        *(query.channel for query in plan.content_queries),
        *(query.channel for query in plan.provider_queries),
    }
    if (
        request.theme_mode is RecallThemeMode.OPEN_TO_ANY
        and len(channels) < 2
        and request.attraction_discovery is None
    ):
        raise RecallPlanError("open-to-any recall requires more than one recall channel")
    if (
        CandidateDomain.ATTRACTION in requested_domains
        and request.attraction_discovery is None
        and request.landmark_policy is not LandmarkRecallPolicy.EXCLUDE
        and RecallChannel.CITY_LANDMARK not in channels
    ):
        raise RecallPlanError("attraction recall requires a landmark channel unless excluded")
    if sum(provider_calls.values()) > request.budget.max_total_provider_calls:
        raise RecallPlanError("recall plan exceeds the total provider-call budget")
    if sum(potential_candidates.values()) > request.budget.max_total_candidates:
        raise RecallPlanError("recall plan exceeds the total candidate budget")
    for domain, budget in budgets.items():
        if provider_calls[domain] > budget.max_provider_calls:
            raise RecallPlanError(f"recall plan exceeds {domain.value} provider-call budget")
        if potential_candidates[domain] > budget.max_candidates:
            raise RecallPlanError(f"recall plan exceeds {domain.value} candidate budget")
    return plan


def _discovery_system_instruction() -> str:
    return (
        f"你是 ITER 任务书前的候选搜索规划能力，版本 {DISCOVERY_RECALL_PROMPT_VERSION}。"
        "只输出要求的 JSON 搜索计划，不直接输出推荐结果。基于用户所在城市、已选偏好和"
        "城市知识自主选择真实可搜索的方向，所有地点假设之后必须由 Provider 核验。\n"
        "1. 输出 provider_only_contract 指定的精确查询条数与唯一业务 domain。"
        "content_queries=[]。查询 ID、类型码和结果配额由服务端管理；typecodes=[]，"
        "named_clue_id=null，无已提供锚点时 anchor_id/radius_m=null。"
        "用户点名 clue 由程序另行绑定；你完全可以自行提出未在输入中出现的具体地点搜索假设。\n"
        "2. search_scope=named_place 表示一个具体景点或餐厅/品牌；"
        "search_scope=category 表示可返回多个独立场所的通用类别。"
        "景点类别关键词必须从 attraction_category_keywords 原样选择，不能加城市前缀。"
        "具体地点名、景区区段或诗意标签不能冒充 category。餐饮类别选一个菜系或菜品词，"
        "不能用空格或逗号拼多个关键词；其他偏好写入 reason 与 theme_ids。\n"
        "3. 保留 minimum_city_landmark_queries 个 city_landmark 精确查询，"
        "必须是该城市真正有代表性的具体景点或当地特色餐厅/品牌，search_scope=named_place。"
        "不能用泛菜名、全国通用快餐或咖啡连锁充当城市特色代表。其他查询用 exploration；"
        "至少 minimum_category_queries 个 exploration 查询是不同的宽类别，"
        "为行程提供足够多独立场所，不能全部点名搜索一两个景区。\n"
        "4. 首次计划必须覆盖 request.themes 中每个已选主题。每个查询的 theme_ids 填"
        "所有确实相关的已提供 theme_id，原样复制，不能创造 ID，也不能全部留空。"
        "一个相关查询可以覆盖多个主题，但不能为了通过校验给无关地点贴标签。"
        "未选偏好不是排除项；明确排除项优先。\n"
        "5. 搜景点只搜完整可游览的景点、场馆、公园或历史街区；"
        "不搜餐饮、酒店、普通校园、馆内展项、入口、办公服务设施或同景区多个区段。"
        "搜餐饮要覆盖不同当地菜品、菜系或品牌，不能用同品牌分店凑查询数。"
        "当 dining_city_target 非空时，餐厅城市代表约占一半并优先展示；"
        "目标只影响真实召回组成，不允许虚构特色身份、评分或招牌菜。"
        "reason 用完整短句解释与用户和城市的关联；不编造地点 ID、营业时间、价格或坐标。\n"
        "6. composition_feedback 表示补召回，只处理仍缺的 TOP 和代表项。"
        "frozen_top_names 已固定，新增代表必须在其外；避开 excluded_places、"
        "excluded_restaurant_brands 的所有分店，以及已经失败的 failed_named_queries。"
        "observed_alternative_names 是本轮已核验、未展示且与 TOP 不重叠的真实候选；"
        "优先从中判断合适的城市代表，再用其原名称做 city_landmark/named_place 精确核验。"
        "它们不是排除项，也还没有被认定为代表；只有确有城市代表性的对象才可选。"
        "如果备选池没有合适对象，可以提出新的具体地点；但不能与 TOP 包含或重叠，"
        "例如已经展示整座景区，就不能再把其中的城门、展项或分区当新代表。"
        "补召回不必重新覆盖所有主题，但保留真实相关的主题 ID。\n"
        "7. 收到修复反馈时一次修正全部问题：非法类别词、缺失主题、查询数或角度重复。"
        "保留已经正确的主题关联，输出完整新计划；不得机械复制 rejected_query_angles。\n"
        "8. 当 request.attraction_discovery 非空时，采用新版景点策略，优先于上述旧TOP规则："
        "directions 包含喜欢与排除的完整语义，selected=false 是明确排除，必须避免任何语义相关的"
        "搜索结果，城市代表也不能例外；不要只看标签文字。explicit_place_intents 是用户已点名意愿。"
        "城市代表为主，建议60%-70%，其余个性化；先搜当地知名主要景点，再补不同风格。"
        "不要因宽类别或具体地点同名片段而把整套城市代表换掉："
        "‘历史街区’与‘清河坊历史街区’可同时搜索，搜索覆盖不等于同一景点重复。"
        "这里的主题覆盖、代表/类别查询比例是搜索建议，不是严格筛选；不同地点可以同类。"
        "博物馆和普通城市公园各最多3个，西湖等大型综合景区不算普通公园，"
        "露天遗址不自动当博物馆，未排除宗教体验时灵隐寺等应正常考虑。"
        "不要扩大用户的排除范围；不同寺庙、园林、街区允许同时召回。"
        "购物中心整体允许作为购物体验，普通商店或内部展项不允许。"
        "努力达到5/7/9/10至12个展示目标，不足不是日常结果。"
        "screening_feedback是粗筛后缺口，补召回优先补城市代表和新的互补地点；"
        "已通过不重复搜索，先前未选不等于用户排除。仅相近体验、远郊或知名度低不能列为禁搜。"
        "本轮最多6次补查，要充分利用不同的真实具体景点查询，避免只搜一两个泛类别。"
        "最终由千问粗筛合池，仍不足会回看已核验候补一次。"
    )


def _system_instruction() -> str:
    return (
        f"You create {RECALL_PROMPT_VERSION} search-plan drafts for a travel "
        "candidate recall layer. Return only the requested RecallPlanDraft schema. "
        "Query and protocol IDs are server-owned; "
        "do not emit them. Use only the supplied city, themes, "
        "named places, anchors and budgets. Split broad themes into useful bounded search "
        "queries. Preserve every named place as an exact user_named query. Never output a "
        "real place ID, URL, coordinate, opening hour, price or claim that the user will like "
        "a place. Provider-only queries must use substantively different search angles across "
        "city representatives, confirmed preferences and broader exploration; changing only "
        "word order or adding a near-synonym is not a different angle. Unselected themes are "
        "neutral, not exclusions. Explicit exclusions win. "
        "City-feature queries require supplied reviewed city content. In provider-only mode, "
        "use city_landmark for genuinely city-specific representative search targets and "
        "explain their local significance; use exploration for the user's other preferences. "
        "A specific place name is a search hypothesis, not a user_named claim. "
        "Use precise POI-search keywords, not sentences such as 'best places for a quiet trip'. "
        "Restaurant queries must cover distinct local dishes, cuisines or brands, not repeated "
        "branches of one brand. If composition_feedback is supplied, target those observed "
        "shortfalls and search for new entities outside excluded_places. frozen_top_names are "
        "already fixed; extra representatives must be distinct from every one. Do not repeat "
        "any excluded_restaurant_brands, even with another branch address, city prefix or cuisine "
        "tagline. A restaurant named in excluded_places excludes ALL branches of that brand. "
        "failed_named_queries already returned no matching independent entity; do not spend "
        "the supplement repeating them. Choose other genuine city representatives instead. "
        "Retain relevant supplied "
        "theme_ids on each query so ranking can measure confirmed preference matches. Never "
        "invent theme_ids or tag an unrelated query just to satisfy coverage. In the initial "
        "plan, cover every selected theme at least once. Keep reasons concise."
        " Aim for many independently visitable places across several search angles. "
        "Museum exhibits, offices, entry gates and multiple sections of one scenic area "
        "are not independent candidates. Mix genuinely broad category searches with "
        "precise representative searches so long trips have enough distinct choices."
        " Respect the ONE requested domain on each query. For attraction discovery, "
        "search independently visitable attractions, museums, parks or historic streets; "
        "never spend attraction queries on restaurants, teahouses, cafes, hotels or food "
        "brands, even if a selected experience mentions local food/tea. Such an experience "
        "can instead lead to a relevant heritage museum or historic neighborhood. Dining "
        "venues will be discovered separately. An ordinary campus is not a tourist attraction."
        " In provider-only mode, city_landmark keyword MUST be the proper name of one specific "
        "city-representative attraction or genuinely local restaurant/brand. Do not use a food "
        "category, a broad search phrase, a national generic chain or another city's specialty "
        "as a city_landmark keyword. Broad cuisines/categories belong to exploration. Returned "
        "landmark entities will be rejected unless their real names match that precise target."
        " Declare search_scope=named_place when keyword is one named venue, or category for "
        "a searchable class of venues. Named searches may not return that venue's sub-sites. "
        "search_scope is required on EVERY query: do not omit it or classify a proper place "
        "name as category to meet quotas. category keyword must be ONE searchable noun phrase, "
        "For attraction category queries choose from the supplied attraction_category_keywords; "
        "these are legal tool arguments, not a fixed plan. A city name, one venue's proper name "
        "or a poetic style label is not a broad category. Keep specific city knowledge in "
        "named_place queries, reason and theme_ids. Do not add a city prefix to category keywords. "
        "without spaces, commas or a list of foods, neighborhoods and adjectives. Put the "
        "additional preference context in reason/theme_ids, not in the Provider keyword. "
        "For the initial plan, keep at least three citywide category searches for 3-5 day trips "
        "(two for shorter trips, within the call budget). Use standard searchable nouns such "
        "as 博物馆、公园、名人故居、历史街区, chosen to fit this user's city and interests; "
        "do not copy a poetic preference label into keywords or search one scenic area's "
        "internal facilities. A broad category should yield several independent places, "
        "not multiple rooms or former residences in one compound."
    )


def _normalized_planning_context(
    request: CandidateRecallRequest,
    city: CityRegistration,
    content: RegisteredCityContentPackage | None,
) -> dict[str, object]:
    return {
        "request": request.model_dump(mode="json"),
        "city": {
            "city_id": city.city_id,
            "display_name": city.display_name,
            "provider_capabilities": [
                capability.value for capability in city.enabled_provider_capabilities
            ],
        },
        "reviewed_city_themes": [
            {
                "theme_id": theme.theme_id,
                "label": theme.label,
                "summary": theme.summary,
                "source_ids": theme.source_ids,
            }
            for theme in content.themes
        ]
        if content
        else [],
        "content_package_available": content is not None,
    }


def _normalize(value: str) -> str:
    return "".join(value.casefold().split())
