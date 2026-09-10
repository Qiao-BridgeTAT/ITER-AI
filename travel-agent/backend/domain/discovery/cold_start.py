"""Readable projection of user-submitted defaults, never inferred trip requirements."""

from backend.contracts.enums import DayReturn, DayStart, FiveLevel, MobilityTolerance, PriorityGoal
from backend.contracts.v4.state import ColdStartProfileSnapshot
from backend.contracts.v4.task_book import EvidenceBackedText

_START = {
    DayStart.BEFORE_07: "7 点前",
    DayStart.AROUND_08: "8 点左右",
    DayStart.AROUND_09: "9 点左右",
    DayStart.AROUND_10: "10 点左右",
    DayStart.AFTER_11: "11 点后",
    DayStart.FLEXIBLE: "灵活安排",
}
_RETURN = {
    DayReturn.BEFORE_20: "20 点前",
    DayReturn.AROUND_21: "21 点左右",
    DayReturn.AFTER_22: "22 点后",
    DayReturn.FLEXIBLE: "灵活安排",
}
_PACE = dict(
    zip(FiveLevel, ("很松弛", "轻松一点", "松紧平衡", "充实一些", "尽量多看"), strict=True)
)
_CLASSIC = dict(
    zip(FiveLevel, ("经典优先", "经典多一些", "两边平衡", "兴趣多一些", "跟着兴趣走"), strict=True)
)
_TRANSIT = dict(
    zip(
        FiveLevel,
        ("公共交通优先", "公共交通为主", "逐段综合比较", "打车为主", "打车优先"),
        strict=True,
    )
)
_MOBILITY = {
    MobilityTolerance.NEVER: "一般不考虑",
    MobilityTolerance.WITHIN_5: "5 分钟内",
    MobilityTolerance.AROUND_10: "10 分钟左右",
    MobilityTolerance.FIFTEEN_PLUS: "15 分钟以上也可",
}
_GOAL = {
    PriorityGoal.MUST_SEE_PLACES: "想去的地方不留遗憾",
    PriorityGoal.COMFORTABLE_STAY: "住得舒服",
    PriorityGoal.SATISFYING_FOOD: "吃得满意",
    PriorityGoal.SMOOTH_ROUTES: "路线顺、少折腾",
    PriorityGoal.GOOD_VALUE: "整体花费划算",
}


def cold_start_default_notes(profile: ColdStartProfileSnapshot | None) -> list[EvidenceBackedText]:
    if profile is None:
        return []
    value = profile.preferences
    notes = (
        f"作息：{_START[value.day_start]}出门，{_RETURN[value.day_return]}返回",
        f"步调：{_PACE[FiveLevel(value.pace_level)]}",
        f"经典与兴趣：{_CLASSIC[FiveLevel(value.classic_niche_level)]}",
        f"行走：步行{_MOBILITY[value.walking_tolerance]}，骑行{_MOBILITY[value.bike_tolerance]}",
        f"出行：{_TRANSIT[FiveLevel(value.transit_taxi_level)]}",
        "优先目标：" + "、".join(_GOAL[item] for item in value.priority_goals),
    )
    return [
        EvidenceBackedText(
            value=f"长期默认（本次明确要求优先）｜{text}",
            source_evidence_refs=profile.source_evidence_refs,
        )
        for text in notes
    ]
