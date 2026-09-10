"""Consistent public error responses with no internal exception leakage."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette import status

from backend.application.rest_service import (
    RestActorError,
    RestConflictError,
    RestResourceNotFoundError,
)


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    request_id: str
    details: list[dict[str, Any]] | None = None


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unavailable"))


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RestResourceNotFoundError)
    async def handle_not_found(request: Request, _exc: RestResourceNotFoundError) -> JSONResponse:
        body = ErrorBody(
            code="resource_not_found",
            message="The requested resource was not found.",
            request_id=_request_id(request),
        )
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=body.model_dump())

    @app.exception_handler(RestConflictError)
    async def handle_conflict(request: Request, exc: RestConflictError) -> JSONResponse:
        body = ErrorBody(code="request_conflict", message=str(exc), request_id=_request_id(request))
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content=body.model_dump())

    @app.exception_handler(RestActorError)
    async def handle_actor_error(request: Request, _exc: RestActorError) -> JSONResponse:
        body = ErrorBody(
            code="actor_not_allowed",
            message="The current session cannot perform this operation.",
            request_id=_request_id(request),
        )
        return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content=body.model_dump())

    @app.exception_handler(ServiceError)
    async def handle_service_error(request: Request, exc: ServiceError) -> JSONResponse:
        body = ErrorBody(code=exc.code, message=exc.message, request_id=_request_id(request))
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            {"location": list(error["loc"]), "type": error["type"]} for error in exc.errors()
        ]
        body = ErrorBody(
            code="request_validation_failed",
            message="The request did not match the public contract.",
            request_id=_request_id(request),
            details=details,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=body.model_dump()
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, _exc: Exception) -> JSONResponse:
        body = ErrorBody(
            code="internal_error",
            message="The service could not complete the request.",
            request_id=_request_id(request),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=body.model_dump(),
        )
