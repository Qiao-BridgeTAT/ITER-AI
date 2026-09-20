"""Source-backed city experience directions for the formal conversation path."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import ConfigDict, model_validator

from backend.agent.semantic_operations import (
    ExperiencePreferenceKind,
    ExperiencePreferenceOperation,
    ExperiencePreferenceValue,
    OperationEvidence,
    SemanticOperationBatch,
    SemanticOperationKind,
    SemanticTarget,
    whole_trip_impact,
)
from backend.contracts.base import ContractModel
from backend.contracts.city_content import RegisteredCityContentPackage
from backend.contracts.commands import MultiChoiceAnswer
from backend.contracts.conversation import (
    ChoiceOption,
    ChoiceSemanticValue,
    ExternalFactReference,
    TextMultiChoiceAttachment,
)
from backend.contracts.enums import Confidence, EvidenceSource, ProviderCode


class CityThemeProjection(ContractModel):
    model_config = ConfigDict(frozen=True)

    attachment: TextMultiChoiceAttachment
    external_facts: tuple[ExternalFactReference, ...]

    @model_validator(mode="after")
    def sources_match_attachment(self) -> CityThemeProjection:
        if self.attachment.interaction_domain != "city_theme":
            raise ValueError("city theme projection requires a city-theme attachment")
        if {fact.fact_id for fact in self.external_facts} != set(self.attachment.external_fact_ids):
            raise ValueError("city theme facts must match attachment references")
        return self


class CityThemeProjector:
    def project(
        self,
        content: RegisteredCityContentPackage,
        *,
        city_name: str,
        attachment_id: UUID,
        source_message_id: UUID,
        created_at: datetime,
        state_version: int,
        generation_id: UUID,
    ) -> CityThemeProjection:
        source_by_id = {source.source_id: source for source in content.sources}
        themes = content.themes[:6]
        facts: dict[UUID, ExternalFactReference] = {}
        options: list[ChoiceOption] = []
        for theme in themes:
            fact_ids: list[UUID] = []
            for source_id in theme.source_ids:
                source = source_by_id[source_id]
                fact_id = uuid5(
                    NAMESPACE_URL,
                    f"iter:city-theme:{content.city_id}:{source.source_id}",
                )
                facts.setdefault(
                    fact_id,
                    ExternalFactReference(
                        fact_id=fact_id,
                        provider=ProviderCode.CITY_CONTENT,
                        source_record_id=source.source_id,
                        retrieved_at=source.retrieved_at,
                    ),
                )
                fact_ids.append(fact_id)
            options.append(
                ChoiceOption(
                    option_id=theme.theme_id,
                    label=theme.label,
                    description=theme.summary,
                    semantic_value=ChoiceSemanticValue(
                        domain="city_theme",
                        kind="theme",
                        value=theme.theme_id,
                        source_fact_ids=fact_ids,
                    ),
                )
            )
        open_option_id = "city-theme:open-to-any"
        options.append(
            ChoiceOption(
                option_id=open_option_id,
                label="都可以，你来安排",
                description="保留城市代表体验，再根据具体景点反馈收敛。",
                semantic_value=ChoiceSemanticValue(
                    domain="city_theme",
                    kind="open_to_any",
                ),
            )
        )
        attachment = TextMultiChoiceAttachment(
            kind="text_multi_choice",
            interaction_domain="city_theme",
            context_label=city_name,
            attachment_id=attachment_id,
            source_message_id=source_message_id,
            created_at=created_at,
            state_version=state_version,
            generation_id=generation_id,
            prompt=f"来到{city_name}，哪些体验更吸引你？",
            external_fact_ids=list(facts),
            minimum_selections=1,
            maximum_selections=len(themes),
            exclusive_option_id=open_option_id,
            options=options,
        )
        return CityThemeProjection(attachment=attachment, external_facts=tuple(facts.values()))


def city_theme_answer_operations(
    *,
    trip_id: UUID,
    request_id: UUID,
    source_message_id: UUID,
    attachment: TextMultiChoiceAttachment,
    answer: MultiChoiceAnswer,
    business_date: date,
) -> SemanticOperationBatch:
    if attachment.interaction_domain != "city_theme":
        raise ValueError("only city-theme attachments use city-theme semantic mapping")
    options = {option.option_id: option for option in attachment.options}
    operations: list[ExperiencePreferenceOperation] = []
    for option_id in answer.option_ids:
        option = options.get(option_id)
        if option is None or option.semantic_value is None:
            raise ValueError("city theme answer references an unknown option")
        semantic = option.semantic_value
        theme_id = "*" if semantic.kind == "open_to_any" else semantic.value
        if theme_id is None:
            raise ValueError("city theme answer is missing a theme identity")
        operations.append(
            _theme_operation(
                trip_id=trip_id,
                request_id=request_id,
                source_message_id=source_message_id,
                attachment_id=attachment.attachment_id,
                theme_id=theme_id,
                label=option.label,
            )
        )
    if answer.free_text is not None:
        note = answer.free_text.strip()
        if note:
            theme_id = "free:" + hashlib.sha256(note.encode("utf-8")).hexdigest()[:16]
            operations.append(
                _theme_operation(
                    trip_id=trip_id,
                    request_id=request_id,
                    source_message_id=source_message_id,
                    attachment_id=attachment.attachment_id,
                    theme_id=theme_id,
                    label=note,
                )
            )
    return SemanticOperationBatch.model_validate(
        {"trip_id": str(trip_id), "operations": operations},
        context={"today": business_date},
    )


def _theme_operation(
    *,
    trip_id: UUID,
    request_id: UUID,
    source_message_id: UUID,
    attachment_id: UUID,
    theme_id: str,
    label: str,
) -> ExperiencePreferenceOperation:
    return ExperiencePreferenceOperation(
        operation_id=uuid5(
            NAMESPACE_URL,
            f"iter:city-theme-answer:{request_id}:{attachment_id}:{theme_id}",
        ),
        trip_id=trip_id,
        operation=SemanticOperationKind.OVERRIDE,
        target=SemanticTarget.EXPERIENCE_PREFERENCES,
        value=ExperiencePreferenceValue(
            kind=ExperiencePreferenceKind.CITY_THEME,
            theme_id=theme_id,
            note=label,
        ),
        evidence=OperationEvidence(
            source=EvidenceSource.CARD,
            source_trip_id=trip_id,
            source_message_id=source_message_id,
            source_attachment_id=attachment_id,
            source_claim_id=f"city-theme:{theme_id}",
        ),
        confidence=Confidence.HIGH,
        impact_scope=whole_trip_impact(),
    )
