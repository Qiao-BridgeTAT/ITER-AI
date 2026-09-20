"""V3-32 projection from ranked attraction candidates to a sourced conversation attachment."""

from __future__ import annotations

from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import model_validator

from backend.contracts.base import ContractModel
from backend.contracts.candidate_ranking import (
    CandidateRankingResult,
    RankedCandidate,
)
from backend.contracts.candidate_recall import CandidateDomain, RecallChannel, RecallSourceKind
from backend.contracts.conversation import (
    ExternalFactReference,
    RecommendationItem,
    RecommendationSetAttachment,
)
from backend.contracts.enums import DataAvailability, ProviderCode


class AttractionRecommendationProjection(ContractModel):
    attachment: RecommendationSetAttachment
    external_facts: tuple[ExternalFactReference, ...] = ()

    @model_validator(mode="after")
    def facts_match_attachment(self) -> AttractionRecommendationProjection:
        fact_ids = {fact.fact_id for fact in self.external_facts}
        if fact_ids != set(self.attachment.external_fact_ids):
            raise ValueError("recommendation projection facts must match attachment references")
        return self


class AttractionRecommendationProjector:
    """Build only UI-safe, source-backed recommendation data from V3-31 output."""

    def project(
        self,
        ranking: CandidateRankingResult,
        *,
        attachment_id: UUID,
        source_message_id: UUID,
        state_version: int,
        generation_id: UUID,
    ) -> AttractionRecommendationProjection:
        ranking = CandidateRankingResult.model_validate(ranking.model_dump(mode="json"))
        selected = [
            item
            for item in ranking.candidates
            if item.candidate.domain is CandidateDomain.ATTRACTION
        ][: min(10, ranking.recommendation_limit)]
        facts_by_id: dict[UUID, ExternalFactReference] = {}
        recommendation_items: list[RecommendationItem] = []
        for item in selected:
            item_fact_ids: list[UUID] = []
            for source in item.candidate.sources:
                provider = _source_provider(source.kind, source.provider)
                fact_id = uuid5(
                    NAMESPACE_URL,
                    "iter:v3-attraction-fact:"
                    f"{provider.value}:{source.source_record_id}:{item.candidate.place.place_id}",
                )
                facts_by_id.setdefault(
                    fact_id,
                    ExternalFactReference(
                        fact_id=fact_id,
                        provider=provider,
                        source_record_id=source.source_record_id,
                        retrieved_at=source.fetched_at or ranking.generated_at,
                    ),
                )
                item_fact_ids.append(fact_id)
            if not item_fact_ids:
                continue
            recommendation_items.append(_project_item(item, tuple(item_fact_ids)))

        if not recommendation_items:
            return AttractionRecommendationProjection(
                attachment=RecommendationSetAttachment(
                    kind="recommendation_set",
                    recommendation_domain="attraction",
                    attachment_id=attachment_id,
                    source_message_id=source_message_id,
                    created_at=ranking.generated_at,
                    state_version=state_version,
                    generation_id=generation_id,
                    prompt="目前没有足够可靠的景点候选，请继续告诉我你想体验什么。",
                    editable=True,
                    availability=DataAvailability.MISSING,
                    missing_reason="没有可追溯来源的景点候选",
                    minimum_selections=0,
                    maximum_selections=0,
                    items=[],
                )
            )

        availability = (
            DataAvailability.AVAILABLE
            if len(recommendation_items) >= 5
            and ranking.status is DataAvailability.AVAILABLE
            and all(
                item.candidate.availability is DataAvailability.AVAILABLE
                for item in selected[: len(recommendation_items)]
            )
            else DataAvailability.PARTIAL
        )
        missing_reason = None
        if availability is DataAvailability.PARTIAL:
            missing_reason = "当前仅有部分可追溯候选或部分候选资料不完整，已保留可用结果"
        external_fact_ids = list(facts_by_id)
        attachment = RecommendationSetAttachment(
            kind="recommendation_set",
            recommendation_domain="attraction",
            attachment_id=attachment_id,
            source_message_id=source_message_id,
            created_at=ranking.generated_at,
            state_version=state_version,
            generation_id=generation_id,
            prompt="这些景点你分别有多想去？不必全部回答。",
            editable=True,
            availability=availability,
            missing_reason=missing_reason,
            external_fact_ids=external_fact_ids,
            minimum_selections=0,
            maximum_selections=len(recommendation_items),
            items=recommendation_items,
        )
        return AttractionRecommendationProjection(
            attachment=attachment,
            external_facts=tuple(facts_by_id.values()),
        )


def _project_item(item: RankedCandidate, source_fact_ids: tuple[UUID, ...]) -> RecommendationItem:
    candidate = item.candidate
    significance = _city_significance(item)
    experience = "；".join(candidate.reasons[:2])
    time_cost = next(
        (
            explanation
            for explanation in item.explanations
            if "交通约" in explanation or "停留约" in explanation or "时长" in explanation
        ),
        "暂无可靠的交通或游玩时长",
    )
    physical_cost = next(
        (risk for risk in item.soft_risks if "physical" in risk.lower()),
        "暂无可靠的体力消耗数据",
    )
    return RecommendationItem(
        recommendation_id=candidate.candidate_id,
        place_id=candidate.place.place_id,
        title=candidate.place.name,
        summary=item.explanations[0],
        image_url=candidate.place.image_url,
        city_significance=significance,
        experience_summary=experience,
        time_cost=time_cost,
        physical_cost=physical_cost,
        source_fact_ids=list(source_fact_ids),
    )


def _city_significance(item: RankedCandidate) -> str:
    channels = set(item.candidate.channels)
    if RecallChannel.CITY_LANDMARK in channels:
        return "城市代表地点，适合纳入经典体验取舍"
    if RecallChannel.CITY_FEATURE in channels:
        return "体现本地特色，补充目的地独有体验"
    if RecallChannel.USER_NAMED in channels:
        return "由你主动点名，优先保留进入取舍"
    return item.candidate.reasons[0]


def _source_provider(
    kind: RecallSourceKind,
    provider: ProviderCode | None,
) -> ProviderCode:
    if kind is RecallSourceKind.PROVIDER:
        assert provider is not None
        return provider
    if kind is RecallSourceKind.CITY_CONTENT:
        return ProviderCode.CITY_CONTENT
    return ProviderCode.MANUAL
