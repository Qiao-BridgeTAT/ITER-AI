"""Deterministic quality guards for user-visible V4 semantic content.

These checks validate readability and obvious chapter boundaries, not travel truth.
Dynamic facts and city/entity correctness still belong to Provider observations and
their domain guards.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher

_MEANINGFUL_UNIT = re.compile(r"[\u3400-\u9fffA-Za-z0-9]")
_INTERNAL_ENUM = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_PLACEHOLDER = re.compile(
    r"^(?:[\u3400-\u9fff]{0,8})?(?:方向|选项|偏好|标签|内容|说明|候选)[-_ ]*\d+$",
    re.IGNORECASE,
)
_FRAGMENTS = {
    "在",
    "沿",
    "参",
    "适",
    "和",
    "与",
    "的",
    "去",
    "吃",
    "住",
    "方向",
    "选项",
    "偏好",
    "景点",
    "餐饮",
    "住宿",
    "城市",
    "特色",
    "体验",
    "适合",
    "附近",
    "方便",
}
_RAW_ENUM_VALUES = {
    "representative_extra",
    "personalized_top",
    "local_representative",
    "area_strategy",
    "hotel_class",
    "classic_landmarks",
    "local_history",
    "local_signature",
    "local_daily",
    "economy",
    "comfort",
    "upscale",
    "luxury",
    "boutique_resort",
    "must",
    "want",
    "destination",
    "if_convenient",
    "avoid",
}


def normalized_visible_text(value: str) -> str:
    """Normalize only presentation noise; never translate or infer meaning."""

    return re.sub(r"[\s\W_]+", "", value.casefold(), flags=re.UNICODE)


def attraction_direction_quality_issue(value: str) -> str | None:
    """Reject obvious dining/lodging subjects before attraction exploration.

    Cultural atmosphere (tea culture, historic street life) is still valid;
    choosing food venues and accommodation belongs to its own chapter.
    """

    if re.search(r"美食|小吃|餐厅|餐馆|茶社|茶馆|探店|吃货|品尝|口味|酒店|住宿|民宿", value):
        return "attraction_direction_domain_mismatch"
    return None


def visible_text_quality_issue(
    value: str,
    *,
    minimum_units: int,
    reject_fragment: bool = True,
) -> str | None:
    """Return a stable, value-free failure code for visibly unusable text."""

    stripped = value.strip()
    units = _MEANINGFUL_UNIT.findall(stripped)
    normalized = normalized_visible_text(stripped)
    casefolded = stripped.casefold()
    if len(units) < minimum_units:
        return "too_short"
    if len(set(unit.casefold() for unit in units)) == 1:
        return "repeated_character"
    if reject_fragment and normalized in _FRAGMENTS:
        return "residual_fragment"
    if casefolded in _RAW_ENUM_VALUES or _INTERNAL_ENUM.fullmatch(casefolded):
        return "raw_internal_enum"
    if _PLACEHOLDER.fullmatch(stripped):
        return "placeholder_text"
    if any(token in stripped for token in ("{", "}", "[", "]")):
        return "raw_structured_text"
    return None


def require_visible_text(
    value: str,
    field_name: str,
    *,
    minimum_units: int,
    reject_fragment: bool = True,
) -> str:
    issue = visible_text_quality_issue(
        value,
        minimum_units=minimum_units,
        reject_fragment=reject_fragment,
    )
    if issue is not None:
        raise ValueError(f"{field_name} is not meaningful: {issue}")
    return value


def require_meaningful_label(value: str, field_name: str = "label") -> str:
    return require_visible_text(value, field_name, minimum_units=2)


def require_meaningful_description(value: str, field_name: str = "description") -> str:
    return require_visible_text(value, field_name, minimum_units=6)


def require_meaningful_trip_goal(value: str, field_name: str = "trip_goal") -> str:
    # Compact goals such as 摄影 / 休闲 are valid; split characters are not.
    return require_visible_text(value, field_name, minimum_units=2)


def require_distinct_visible_labels(values: Iterable[str], field_name: str) -> None:
    """Reject exact and obvious near-duplicate labels without semantic invention."""

    normalized = [normalized_visible_text(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique values")
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            shorter, longer = sorted((left, right), key=len)
            if len(shorter) >= 3 and shorter in longer:
                raise ValueError(f"{field_name} contains a near-duplicate label")
            if (
                min(len(left), len(right)) >= 4
                and SequenceMatcher(None, left, right).ratio() >= 0.88
            ):
                raise ValueError(f"{field_name} contains a near-duplicate label")


__all__ = [
    "normalized_visible_text",
    "require_distinct_visible_labels",
    "require_meaningful_description",
    "require_meaningful_label",
    "require_meaningful_trip_goal",
    "require_visible_text",
    "visible_text_quality_issue",
]
