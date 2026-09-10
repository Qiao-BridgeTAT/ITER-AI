"""Count people in travel party descriptions, never ages or naked numbers."""

from __future__ import annotations

import re
from collections.abc import Iterable

_NUMBER = r"(?:\d{1,2}|[一二两三四五六七八九十]+)"
_ROLE = r"(?:成人|大人|儿童|孩子|小孩|老人|长者|宝宝|婴儿|大|小)"
_DIGITS = dict(zip("一二两三四五六七八九", (1, 2, 2, 3, 4, 5, 6, 7, 8, 9), strict=True))


def _number(text: str) -> int:
    if text.isdigit():
        return int(text)
    if "十" in text:
        left, right = text.split("十", 1)
        return _DIGITS.get(left, 1) * 10 + _DIGITS.get(right, 0)
    return _DIGITS.get(text, 0)


def parse_party_size(descriptions: Iterable[str]) -> int:
    texts = tuple(dict.fromkeys(text.strip() for text in descriptions if text.strip()))
    text = "、".join(texts)
    # Remove the whole age, including ranges, before matching count + role.
    text = re.sub(rf"{_NUMBER}(?:\s*[-—至到]\s*{_NUMBER})?\s*(?:周岁|岁|个月大)", "", text)
    totals = [
        _number(m[1])
        for m in re.finditer(
            rf"(?:共计|总共|一共|合计|共|一家)\s*({_NUMBER})\s*(?:个)?[人位口]", text
        )
    ]
    components = [
        _number(m[1])
        for m in re.finditer(
            rf"(?<![\d一二两三四五六七八九十])({_NUMBER})\s*(?:个|位|名)?\s*{_ROLE}", text
        )
    ]
    declared = [
        _number(m[1]) for m in re.finditer(rf"(?<!\d)({_NUMBER})\s*(?:个)?[人位口](?![均民])", text)
    ]
    if totals:
        count = max(totals)
    elif components:
        count = max(sum(components), max(declared, default=0))
        # A leading uncounted self plus a counted companion is additive.
        if re.search(r"(?:^|、)(?:我|本人)(?:和|带|与|跟)", text) and not declared:
            count += 1
    elif declared:
        count = max(declared)
    else:
        people = re.findall(
            r"本人|妈妈|母亲|爸爸|父亲|丈夫|妻子|老公|老婆|孩子|小孩|朋友|我(?!们)", text
        )
        count = len(people) or len(texts) or 1
    return max(1, min(20, count))
