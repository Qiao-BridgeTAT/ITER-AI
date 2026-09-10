"""Request correlation shared by every HTTP route."""

from __future__ import annotations

from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"


class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER, "").strip() or str(uuid4())
        request.state.request_id = request_id[:128]
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        return response
