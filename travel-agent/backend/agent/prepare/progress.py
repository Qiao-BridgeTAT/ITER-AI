"""Coarse presentation milestones; never business decisions or model input."""

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from enum import StrEnum


class AttractionProgress(StrEnum):
    PREFERENCE_DISCOVERY = "attraction_preference_discovery"
    PREFERENCE_READY = "attraction_preference_ready"
    SPECIFIC_SEARCH = "attraction_specific_search"
    SPECIFIC_SUPPLEMENT = "attraction_specific_supplement"
    SPECIFIC_READY = "attraction_specific_ready"


class DiningProgress(StrEnum):
    PREFERENCE_DISCOVERY = "dining_preference_discovery"
    PREFERENCE_READY = "dining_preference_ready"
    SPECIFIC_SEARCH = "dining_specific_search"
    SPECIFIC_SUPPLEMENT = "dining_specific_supplement"
    SPECIFIC_CLASSIFY = "dining_specific_classify"
    SPECIFIC_READY = "dining_specific_ready"


DiscoveryProgress = AttractionProgress | DiningProgress


PROGRESS_COPY = {
    AttractionProgress.PREFERENCE_DISCOVERY: "正在整理当地的特色玩法…",
    AttractionProgress.PREFERENCE_READY: "正在准备景点偏好卡…",
    AttractionProgress.SPECIFIC_SEARCH: "正在查找符合偏好的景点…",
    AttractionProgress.SPECIFIC_SUPPLEMENT: "正在补充更多景点方案…",
    AttractionProgress.SPECIFIC_READY: "正在为你准备输出合适的景点…",
    DiningProgress.PREFERENCE_DISCOVERY: "正在整理当地的特色风味…",
    DiningProgress.PREFERENCE_READY: "正在准备餐饮偏好卡呈现方式…",
    DiningProgress.SPECIFIC_SEARCH: "正在查找符合偏好的餐厅…",
    DiningProgress.SPECIFIC_SUPPLEMENT: "正在补充更多餐厅选择…",
    DiningProgress.SPECIFIC_CLASSIFY: "正在整理各家餐厅的风味与特色…",
    DiningProgress.SPECIFIC_READY: "正在为您梳理餐厅呈现信息…",
}

ProgressEmitter = Callable[[DiscoveryProgress], Awaitable[None]]
discovery_progress_emitter: ContextVar[ProgressEmitter | None] = ContextVar(
    "discovery_progress_emitter", default=None
)


async def report_attraction_progress(phase: AttractionProgress) -> None:
    emitter = discovery_progress_emitter.get()
    if emitter is not None:
        await emitter(phase)


async def report_dining_progress(phase: DiningProgress) -> None:
    emitter = discovery_progress_emitter.get()
    if emitter is not None:
        await emitter(phase)
