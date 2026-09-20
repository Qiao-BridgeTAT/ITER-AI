"""User-owned explicit preferences and original travel feedback; no inferred writes."""

from datetime import UTC
from typing import Literal, cast
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.contracts.v4.memory import CreateUserMemory, UserMemoryList, UserMemoryView
from backend.persistence.models import Message, Trip, UserMemory


class UserMemorySourceError(ValueError):
    pass


class UserMemoryRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def list(self, user_id: UUID) -> UserMemoryList:
        async with self.sessions() as session:
            rows = (
                await session.scalars(
                    select(UserMemory)
                    .where(UserMemory.user_id == user_id)
                    .order_by(UserMemory.created_at.desc(), UserMemory.id)
                )
            ).all()
            return UserMemoryList(
                memories=tuple(
                    UserMemoryView(
                        memory_id=r.id,
                        kind=cast(Literal["preference", "feedback"], r.kind),
                        text=r.text,
                        trip_id=r.trip_id,
                        source_message_id=r.source_message_id,
                        created_at=r.created_at
                        if r.created_at.tzinfo
                        else r.created_at.replace(tzinfo=UTC),
                    )
                    for r in rows
                )
            )

    async def create(self, user_id: UUID, data: CreateUserMemory) -> UserMemoryList:
        async with self.sessions() as session, session.begin():
            if data.trip_id is not None:
                trip = await session.scalar(
                    select(Trip)
                    .where(Trip.id == data.trip_id, Trip.owner_user_id == user_id)
                    .with_for_update()
                )
                if trip is None:
                    raise UserMemorySourceError("memory source trip not found")
            if data.source_message_id is not None:
                message = await session.scalar(
                    select(Message).where(
                        Message.id == data.source_message_id,
                        Message.trip_id == data.trip_id,
                        Message.role == "user",
                        Message.status.in_(("accepted", "committed")),
                    )
                )
                if message is None or not message.text or data.text not in message.text:
                    raise UserMemorySourceError("memory must preserve an original user statement")
                existing = await session.scalar(
                    select(UserMemory).where(
                        UserMemory.user_id == user_id,
                        UserMemory.source_message_id == data.source_message_id,
                        UserMemory.kind == data.kind,
                    )
                )
            else:
                existing = await session.scalar(
                    select(UserMemory).where(
                        UserMemory.user_id == user_id,
                        UserMemory.kind == data.kind,
                        UserMemory.text == data.text,
                    )
                )
            if existing is None:
                session.add(
                    UserMemory(
                        user_id=user_id,
                        kind=data.kind,
                        text=data.text,
                        trip_id=data.trip_id,
                        source_message_id=data.source_message_id,
                    )
                )
        return await self.list(user_id)

    async def delete(self, user_id: UUID, memory_id: UUID) -> None:
        async with self.sessions() as session, session.begin():
            await session.execute(
                delete(UserMemory).where(UserMemory.id == memory_id, UserMemory.user_id == user_id)
            )

    async def capture_explicit_message(
        self, user_id: UUID, trip_id: UUID, message_id: UUID, text: str
    ) -> None:
        text = text.strip()
        if len(text) > 2000:
            return
        # Deliberately narrow: ordinary trip preferences and model reflections never become memory.
        for prefix, kind in (
            ("以后请记住", "preference"),
            ("请长期记住", "preference"),
            ("旅行反馈：", "feedback"),
            ("旅行反馈:", "feedback"),
        ):
            if text.startswith(prefix):
                await self.create(
                    user_id,
                    CreateUserMemory(
                        kind=cast(Literal["preference", "feedback"], kind),
                        text=text,
                        trip_id=trip_id,
                        source_message_id=message_id,
                        explicitly_confirmed=True,
                    ),
                )
                return
