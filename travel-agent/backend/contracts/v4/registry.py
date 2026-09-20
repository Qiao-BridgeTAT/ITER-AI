"""Authoritative registry for generated V4 JSON Schema and TypeScript types."""

from __future__ import annotations

from pydantic import BaseModel

from backend.contracts.v4.cards import V4_CARD_CONTRACTS
from backend.contracts.v4.checkpoint import V4_CHECKPOINT_CONTRACTS
from backend.contracts.v4.commands import V4_COMMAND_CONTRACTS
from backend.contracts.v4.conversation import V4_CONVERSATION_CONTRACTS
from backend.contracts.v4.memory import CreateUserMemory, UserMemoryList, UserMemoryView
from backend.contracts.v4.place_introduction import PlaceIntroductionView
from backend.contracts.v4.plan_preview import PlannerPlacePreview, PlannerPlanPreview
from backend.contracts.v4.planner_workspace import V4_PLANNER_CONTRACTS
from backend.contracts.v4.prepare import V4_PREPARE_CONTRACTS
from backend.contracts.v4.semantic_operations import V4_SEMANTIC_OPERATION_CONTRACTS
from backend.contracts.v4.state import V4_STATE_CONTRACTS
from backend.contracts.v4.task_book import V4_TASK_BOOK_CONTRACTS

V4_PUBLIC_CONTRACTS: tuple[type[BaseModel], ...] = (
    CreateUserMemory,
    UserMemoryView,
    UserMemoryList,
    *V4_COMMAND_CONTRACTS,
    *V4_SEMANTIC_OPERATION_CONTRACTS,
    *V4_STATE_CONTRACTS,
    *V4_PREPARE_CONTRACTS,
    *V4_CARD_CONTRACTS,
    *V4_TASK_BOOK_CONTRACTS,
    *V4_CONVERSATION_CONTRACTS,
    *V4_CHECKPOINT_CONTRACTS,
    *V4_PLANNER_CONTRACTS,
    PlannerPlacePreview,
    PlannerPlanPreview,
    PlaceIntroductionView,
)

_names = [model.__name__ for model in V4_PUBLIC_CONTRACTS]
if len(set(_names)) != len(_names):
    raise RuntimeError("V4 public contract registry contains duplicate model names")
