"""Interpret complete original AMap type evidence, never a caller's category hint."""

from __future__ import annotations

import re

from backend.contracts.enums import PlaceCategory


def original_typecodes(value: str | None) -> tuple[str, ...]:
    """Keep all pipe-separated codes; malformed evidence is unusable."""

    if value is None:
        return ()
    codes = tuple(value.split("|"))
    if not codes or not all(re.fullmatch(r"[0-9]{6}", code) for code in codes):
        return ()
    return tuple(dict.fromkeys(codes))


def category_from_original_typecodes(value: str | None) -> PlaceCategory:
    categories: set[PlaceCategory] = set()
    for code in original_typecodes(value):
        if code.startswith("05"):
            categories.add(PlaceCategory.RESTAURANT)
        elif code.startswith("10"):
            categories.add(PlaceCategory.HOTEL)
        elif code.startswith("11") or code[:4] in {"1401", "1404", "1406", "1407"}:
            categories.add(PlaceCategory.ATTRACTION)
        elif code.startswith("15"):
            categories.add(PlaceCategory.TRANSPORT)
    # A shopping street can also have a real scenic tag (061000|110000).
    # Contradictory known domains remain ambiguous; a hint must not resolve them.
    return next(iter(categories)) if len(categories) == 1 else PlaceCategory.OTHER


def is_shopping_complex(value: str | None) -> bool:
    """A whole mall/department store, not every business in shopping category 06."""
    codes = original_typecodes(value)
    return bool(codes) and all(code in {"060100", "060101", "060102"} for code in codes)
