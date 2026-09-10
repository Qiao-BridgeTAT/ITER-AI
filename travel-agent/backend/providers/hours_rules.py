"""Conservative, shared date interpretation; no model or city-specific opening rules."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import date, time
from typing import Literal

from backend.providers.contracts import (
    HoursDayStatus,
    HoursInterval,
    ProviderDateHours,
    ProviderRegularHours,
)

PARSER_VERSION = "date-hours-v6"
_WEEKDAY = r"(?:周|星期|礼拜)[一二三四五六日天1-7]"
_WEEKDAYS = re.compile(rf"(?P<first>{_WEEKDAY})(?:\s*[-~至到]\s*(?P<last>{_WEEKDAY}))?")
_DATE = r"\d{4}(?:-\d{1,2}-\d{1,2}|年\d{1,2}月\d{1,2}日)"
_DATE_PREFIX = re.compile(rf"^(?P<first>{_DATE})(?:\s*(?:至|到|~)\s*(?P<last>{_DATE}))?")
_MONTH_PREFIX = re.compile(
    r"^(?P<first>\d{1,2})月?\s*[-~至到—–]\s*(?P<next_year>次年)?(?P<last>\d{1,2})月\s*[:：]?"
)
_ANNUAL_RANGE_PREFIX = re.compile(
    r"^(?P<first_month>\d{1,2})月(?P<first_day>\d{1,2})日\s*[-~至到—–]\s*"
    r"(?P<next_year>次年)?(?P<last_month>\d{1,2})月(?P<last_day>\d{1,2})日\s*[:：]?"
)
_ANNUAL_SLASH_PREFIX = re.compile(
    r"^(?P<first_month>\d{1,2})/(?P<first_day>\d{1,2})\s*[-~至到—–]\s*"
    r"(?P<last_month>\d{1,2})/(?P<last_day>\d{1,2})\s*"
)
_HOLIDAY_PREFIX = re.compile(
    r"^(?:(?:元旦节?|春节|清明节?|劳动节|五一|端午节?|中秋节?|国庆节?)\s*[,、]?\s*)+"
)
_TIME = r"\d{1,2}:\d{2}"
_INTERVAL = re.compile(rf"({_TIME})\s*[-~至到—–]\s*({_TIME})")
_LAST_ENTRY = re.compile(
    rf"(?P<before>{_TIME})\s*(?:停止入园|停止入馆|停止入场|停止检票|停止售票|止检)"
    rf"|最晚进入\s*:?\s*(?P<after>{_TIME})"
)
_CLOSED = re.compile(r"(?:全天)?(?:闭馆|闭园|不开放|不营业|休息|暂停营业|暂停开放|歇业)")
_CONDITION = re.compile(r"节假日|节日|调休|除外|临时|另行|另见|另定|为准|视情况|可能|调整")


@dataclass(frozen=True)
class _Rule:
    weekdays: frozenset[int] | None
    start_date: date | None
    end_date: date | None
    status: HoursDayStatus
    intervals: tuple[HoursInterval, ...]
    conditional: bool = False
    months: frozenset[int] | None = None
    annual_range: tuple[tuple[int, int], tuple[int, int]] | None = None
    possible_exception: bool = False

    def applies(self, service_date: date) -> bool:
        if self.months is not None and service_date.month not in self.months:
            return False
        if self.annual_range is not None:
            first, last = self.annual_range
            current = (service_date.month, service_date.day)
            if not (
                first <= current <= last if first <= last else current >= first or current <= last
            ):
                return False
        if (
            self.start_date is not None
            and self.end_date is not None
            and not self.start_date <= service_date <= self.end_date
        ):
            return False
        return self.weekdays is None or service_date.isoweekday() in self.weekdays


def evaluate_regular_hours(
    regular: ProviderRegularHours, service_dates: tuple[date, ...] | list[date]
) -> list[ProviderDateHours]:
    """Re-evaluate evidence, never trust caller-injected date conclusions or generic text."""
    rules = _parse_weekly(regular.weekly_text) if regular.weekly_text else []
    if not regular.weekly_text and regular.weekly_periods:
        for period in regular.weekly_periods:
            try:
                interval = HoursInterval(opens_at=period.opens_at, closes_at=period.closes_at)
            except ValueError:
                rules.append(
                    _Rule(frozenset({period.weekday}), None, None, HoursDayStatus.UNKNOWN, ())
                )
            else:
                rules.append(
                    _Rule(frozenset({period.weekday}), None, None, HoursDayStatus.OPEN, (interval,))
                )
    result = []
    for service_date in service_dates:
        matching = [rule for rule in rules if rule.applies(service_date)]
        dated = [rule for rule in matching if rule.start_date is not None]
        selected = dated or matching
        basis: Literal["weekly", "dated_exception", "today", "unverified", "conflicting"] = (
            "dated_exception" if dated else "weekly"
        )
        status, intervals, reason = _combine(selected)
        today = None
        if regular.today_text and service_date == regular.today_date:
            today = _parse_body(_normalize(regular.today_text))
        if today is not None:
            if not rules and not regular.weekly_text:
                status, intervals, reason = _combine([today])
                basis = "today"
            elif today.status is HoursDayStatus.UNKNOWN or today.conditional:
                status, intervals = HoursDayStatus.UNKNOWN, ()
                reason = "今日字段含无法核实的临时/例外信息，不能只按常规规则确认今日开放。"
            elif today.status in (HoursDayStatus.OPEN, HoursDayStatus.CLOSED):
                baseline_status, baseline_intervals, _ = _combine(selected, ignore_condition=True)
                if baseline_status in (HoursDayStatus.OPEN, HoursDayStatus.CLOSED) and (
                    baseline_status != today.status
                    or _signature(baseline_intervals) != _signature(today.intervals)
                    or _cutoff_disagrees(baseline_intervals, today.intervals)
                ):
                    status, intervals = HoursDayStatus.CONFLICT, ()
                    reason = "今日字段与适用的每周/日期规则冲突；未核实例外，不能确认开放。"
                elif status is HoursDayStatus.OPEN:
                    today_entries = {item.opens_at: item.last_entry_at for item in today.intervals}
                    intervals = tuple(
                        item.model_copy(
                            update={
                                "last_entry_at": item.last_entry_at
                                or today_entries.get(item.opens_at)
                            }
                        )
                        for item in intervals
                    )
        if status in (HoursDayStatus.UNKNOWN, HoursDayStatus.CONFLICT):
            basis = "conflicting" if status is HoursDayStatus.CONFLICT else "unverified"
        result.append(
            ProviderDateHours(
                service_date=service_date,
                status=status,
                intervals=list(intervals),
                basis=basis,
                reason=reason,
            )
        )
    return result


def describe_date_hours(item: ProviderDateHours) -> str:
    label = {
        HoursDayStatus.OPEN: "开放",
        HoursDayStatus.CLOSED: "关闭/闭馆",
        HoursDayStatus.UNKNOWN: "待核实",
        HoursDayStatus.CONFLICT: "信息冲突，待核实",
    }[item.status]
    intervals = "、".join(
        f"{period.opens_at:%H:%M}–{period.closes_at:%H:%M}"
        + (f"（{period.last_entry_at:%H:%M}停止入园/接待）" if period.last_entry_at else "")
        for period in item.intervals
    )
    detail = f" {intervals}" if intervals else ""
    return f"{item.service_date.isoformat()} {label}{detail}：{item.reason}"


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().replace("～", "~")


def _clauses(value: str) -> list[str]:
    result: list[str] = []
    start, depth = 0, 0
    for index, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and (
            char in ";；\n"
            or (
                char == ","
                and (
                    _DATE_PREFIX.match(value[index + 1 :].lstrip())
                    or _MONTH_PREFIX.match(value[index + 1 :].lstrip())
                    or _ANNUAL_RANGE_PREFIX.match(value[index + 1 :].lstrip())
                )
            )
        ):
            result.append(value[start:index].strip())
            start = index + 1
    result.append(value[start:].strip())
    return [part for part in result if part]


def _parse_weekly(text: str) -> list[_Rule]:
    clauses = [
        re.sub(r"\(数据来源:[^()]+\)\s*$", "", clause).strip()
        for clause in _clauses(_normalize(text))
    ]
    clauses = [clause for clause in clauses if clause]
    extension = _default_weekly_extension(clauses)
    if extension is not None:
        return extension
    rules: list[_Rule] = []
    for clause in clauses:
        # A trailing attribution is not a condition on the opening times.
        # Other annotations remain subject to the conservative body parser.
        clause = re.sub(r"\(数据来源:[^()]+\)\s*$", "", clause).strip()
        # A hypothetical notice is not an actual closure announcement. Retain
        # the preceding weekday rule (including its unresolved holiday caveat).
        clause = re.sub(r"^温馨提示\s*:\s*", "", clause)
        clause = re.sub(
            r"如遇临时调整或延时开放[,，]请以[^;；]+官方最新公告为准[。.]?$", "", clause
        ).strip()
        annual_match = _ANNUAL_RANGE_PREFIX.match(clause) or _ANNUAL_SLASH_PREFIX.match(clause)
        if annual_match:
            try:
                first_day = (int(annual_match["first_month"]), int(annual_match["first_day"]))
                last_day = (int(annual_match["last_month"]), int(annual_match["last_day"]))
                # A leap-year anchor validates month/day without inventing a
                # source year; comparison below uses the actual requested day.
                date(2000, *first_day)
                date(2000, *last_day)
                if annual_match.groupdict().get("next_year") and first_day <= last_day:
                    raise ValueError("ambiguous multi-year range")
            except ValueError:
                rules.append(_unknown_rule())
                continue
            body_text = clause[annual_match.end() :].strip()
            scoped = (
                _parse_weekly(body_text) if _WEEKDAYS.match(body_text) else [_parse_body(body_text)]
            )
            rules.extend(replace(body, annual_range=(first_day, last_day)) for body in scoped)
            continue
        holiday_match = _HOLIDAY_PREFIX.match(clause)
        if holiday_match:
            # Without a verified holiday calendar, do not assert that the
            # requested date is a holiday. Only identical normal/holiday hours
            # may resolve below; changed hours or closures remain unknown.
            body = _parse_body(clause[holiday_match.end() :])
            rules.append(replace(body, possible_exception=True))
            continue
        month_match = _MONTH_PREFIX.match(clause)
        if month_match:
            first_month, last_month = int(month_match["first"]), int(month_match["last"])
            if not (1 <= first_month <= 12 and 1 <= last_month <= 12) or (
                month_match["next_year"] and first_month <= last_month
            ):
                rules.append(_unknown_rule())
                continue
            body = _parse_body(clause[month_match.end() :])
            months = frozenset(
                (first_month + offset - 1) % 12 + 1
                for offset in range((last_month - first_month) % 12 + 1)
            )
            rules.append(
                _Rule(None, None, None, body.status, body.intervals, body.conditional, months)
            )
            continue
        # A clearly separated night session does not close the daytime session
        # on its weekly rest day. Ambiguous/conditional night text stays unknown.
        night_match = re.fullmatch(
            rf"夜场\s*(?P<hours>{_TIME}\s*[-~至到—–]\s*{_TIME})"
            rf"\s*\((?P<closed>{_WEEKDAY})\s*(?:闭园|闭馆|不开放)\)",
            clause,
        )
        if night_match:
            body = _parse_body(night_match["hours"])
            night_weekdays = frozenset(set(range(1, 8)) - {_weekday(night_match["closed"])})
            rules.append(
                _Rule(night_weekdays, None, None, body.status, body.intervals, body.conditional)
            )
            continue
        date_match = _DATE_PREFIX.match(clause)
        if date_match:
            try:
                first = _date(date_match["first"])
                last = _date(date_match["last"]) if date_match["last"] else first
                if last < first:
                    raise ValueError("inverted date range")
            except ValueError:
                rules.append(_unknown_rule())
                continue
            body = _parse_body(clause[date_match.end() :])
            rules.append(_Rule(None, first, last, body.status, body.intervals, body.conditional))
            continue
        matches = list(_WEEKDAYS.finditer(clause))
        if matches and not clause[: matches[0].start()].strip(" :"):
            weekdays: set[int] = set()
            position = 0
            valid = True
            for match in matches:
                if clause[position : match.start()].strip(" 、,/:和及"):
                    valid = False
                    break
                first_weekday = _weekday(match["first"])
                last_weekday = _weekday(match["last"]) if match["last"] else first_weekday
                weekdays.update(
                    (first_weekday + offset - 1) % 7 + 1
                    for offset in range((last_weekday - first_weekday) % 7 + 1)
                )
                position = match.end()
            if valid:
                body = _parse_body(clause[position:])
                rules.append(
                    _Rule(
                        frozenset(weekdays),
                        None,
                        None,
                        body.status,
                        body.intervals,
                        body.conditional,
                    )
                )
                continue
        daily = re.match(r"^(每天|每日|全年|全天24小时|24小时)", clause)
        if daily:
            body = _parse_body(clause if "24小时" in daily[0] else clause[daily.end() :])
            rules.append(
                _Rule(
                    frozenset(range(1, 8)),
                    None,
                    None,
                    body.status,
                    body.intervals,
                    body.conditional,
                )
            )
        else:
            # Unqualified times / undefined seasons are not evidence of opening.
            rules.append(_unknown_rule())
    return rules


def _default_weekly_extension(clauses: list[str]) -> list[_Rule] | None:
    """AMap's base hours + explicit weekend/holiday extension, no holiday guess.

    Outside the stated weekdays only the base interval is guaranteed in BOTH
    normal and extended scenarios. A shortened/closed exception is not accepted.
    Bare times alone remain unsupported.
    """
    if len(clauses) != 2 or not _INTERVAL.fullmatch(clauses[0]):
        return None
    weekday = _WEEKDAYS.match(clauses[1])
    if weekday is None:
        return None
    remaining = clauses[1][weekday.end() :].strip()
    if not remaining.startswith("/节假日"):
        return None
    base = _parse_body(clauses[0])
    extended = _parse_body(remaining.removeprefix("/节假日"))
    if (
        base.status is not HoursDayStatus.OPEN
        or extended.status is not HoursDayStatus.OPEN
        or extended.conditional
        or len(base.intervals) != 1
        or len(extended.intervals) != 1
    ):
        return None
    normal, special = base.intervals[0], extended.intervals[0]
    if (
        special.opens_at > normal.opens_at
        or special.closes_at < normal.closes_at
        or special.last_entry_at is not None
    ):
        return None
    first = _weekday(weekday["first"])
    last = _weekday(weekday["last"]) if weekday["last"] else first
    weekdays = frozenset((first + i - 1) % 7 + 1 for i in range((last - first) % 7 + 1))
    return [
        replace(extended, weekdays=weekdays),
        replace(base, weekdays=frozenset(set(range(1, 8)) - weekdays)),
    ]


def _parse_body(text: str) -> _Rule:
    if text.count("(") != text.count(")"):
        return _unknown_rule()
    conditional = bool(_CONDITION.search(text))
    # Only this well-known conditional annotation can retain its baseline rule.
    text = re.sub(r"\(法定节假日除外\)", "", text).strip(" :")
    entry_matches = list(_LAST_ENTRY.finditer(text))
    last_entry = None
    if entry_matches:
        try:
            times = {_clock(match["before"] or match["after"]) for match in entry_matches}
            if len(times) != 1:
                return _unknown_rule()
            last_entry = times.pop()
        except ValueError:
            return _unknown_rule()
        text = _LAST_ENTRY.sub("", text)
    closed = _CLOSED.search(text)
    matches = list(_INTERVAL.finditer(text))
    all_day = bool(re.fullmatch(r"(?:全天)?24小时(?:营业|开放)?", text.strip()))
    remainder = _INTERVAL.sub("", text)
    remainder = _CLOSED.sub("", remainder)
    remainder = re.sub(r"开放|营业|全天24小时|24小时", "", remainder).strip(" :、,/()")
    if remainder or (closed and (matches or all_day)):
        return _unknown_rule()
    if closed:
        return _Rule(None, None, None, HoursDayStatus.CLOSED, (), conditional)
    try:
        intervals = [
            HoursInterval(opens_at=_clock(match[1]), closes_at=_clock(match[2], end=True))
            for match in matches
        ]
        if all_day:
            intervals = [HoursInterval(opens_at=time(0), closes_at=time(23, 59, 59))]
        if last_entry is not None:
            if not intervals or not any(
                item.opens_at <= last_entry <= item.closes_at for item in intervals
            ):
                return _unknown_rule()
            intervals = [
                HoursInterval(
                    opens_at=item.opens_at,
                    closes_at=item.closes_at,
                    last_entry_at=min(last_entry, item.closes_at),
                )
                for item in intervals
                if item.opens_at <= last_entry
            ]
    except ValueError:
        return _unknown_rule()
    if not intervals:
        return _unknown_rule()
    return _Rule(None, None, None, HoursDayStatus.OPEN, tuple(intervals), conditional)


def _combine(
    rules: list[_Rule], *, ignore_condition: bool = False
) -> tuple[HoursDayStatus, tuple[HoursInterval, ...], str]:
    if not rules:
        return HoursDayStatus.UNKNOWN, (), "没有覆盖该日期的可验证营业规则，不能视为全天开放。"
    exceptions = [rule for rule in rules if rule.possible_exception]
    if exceptions:
        baseline = [rule for rule in rules if not rule.possible_exception]
        normal_status, normal_intervals, normal_reason = _combine(
            baseline, ignore_condition=ignore_condition
        )
        if normal_status is HoursDayStatus.OPEN and all(
            rule.status is HoursDayStatus.OPEN
            and not rule.conditional
            and {_interval_key(item) for item in rule.intervals}
            == {_interval_key(item) for item in normal_intervals}
            for rule in exceptions
        ):
            return normal_status, normal_intervals, normal_reason
        return HoursDayStatus.UNKNOWN, (), "常规与节假日规则不同，尚未核实例外适用性。"
    statuses = {rule.status for rule in rules}
    if HoursDayStatus.OPEN in statuses and HoursDayStatus.CLOSED in statuses:
        return HoursDayStatus.CONFLICT, (), "同一日期的开放与关闭规则冲突。"
    if any(rule.conditional for rule in rules) and not ignore_condition:
        return HoursDayStatus.UNKNOWN, (), "该日期的规则包含节假日等例外，例外适用性尚未核实。"
    if HoursDayStatus.UNKNOWN in statuses:
        return (
            HoursDayStatus.UNKNOWN,
            (),
            "营业描述包含无法完整识别的日期、季节、临时或跨午夜规则，需核实。",
        )
    if statuses == {HoursDayStatus.CLOSED}:
        return HoursDayStatus.CLOSED, (), "适用的来源规则明确关闭/闭馆。"
    unique = {_interval_key(item): item for rule in rules for item in rule.intervals}
    intervals = tuple(sorted(unique.values(), key=lambda item: item.opens_at))
    if len(intervals) > 10 or any(
        a.closes_at > b.opens_at for a, b in zip(intervals, intervals[1:], strict=False)
    ):
        return HoursDayStatus.CONFLICT, (), "来源时段重叠或超出可验证数量，需要重新核实。"
    return HoursDayStatus.OPEN, intervals, "依据适用的来源规则；临时公告与预约要求仍需另行核验。"


def _unknown_rule() -> _Rule:
    return _Rule(None, None, None, HoursDayStatus.UNKNOWN, ())


def _weekday(text: str) -> int:
    return {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "日": 7,
        "天": 7,
        **{str(value): value for value in range(1, 8)},
    }[text[-1]]


def _date(value: str) -> date:
    parts = re.fullmatch(r"(\d{4})[-年](\d{1,2})[-月](\d{1,2})日?", value)
    if parts is None:
        raise ValueError("unsupported date")
    return date(*(int(item) for item in parts.groups()))


def _clock(value: str, *, end: bool = False) -> time:
    hours, minutes = (int(part) for part in value.split(":"))
    if end and (hours, minutes) == (24, 0):
        return time(23, 59, 59)
    return time(hours, minutes)


def _interval_key(item: HoursInterval) -> tuple[time, time, time | None]:
    return item.opens_at, item.closes_at, item.last_entry_at


def _signature(items: tuple[HoursInterval, ...]) -> set[tuple[time, time]]:
    # Today's field commonly omits last-entry metadata; omission is not contradiction.
    return {(item.opens_at, item.closes_at) for item in items}


def _cutoff_disagrees(left: tuple[HoursInterval, ...], right: tuple[HoursInterval, ...]) -> bool:
    return any(
        first.opens_at == second.opens_at
        and first.last_entry_at is not None
        and second.last_entry_at is not None
        and first.last_entry_at != second.last_entry_at
        for first in left
        for second in right
    )
