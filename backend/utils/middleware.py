from __future__ import annotations

import time
import uuid
import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.logger = logging.getLogger("spam_app")

    async def dispatch(self, request: Request, call_next: Callable):  # type: ignore[override]
        req_id = uuid.uuid4().hex
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
        # Attach request id
        response.headers.setdefault("X-Request-ID", req_id)

        self.logger.info(
            "request id=%s method=%s path=%s status=%s duration_ms=%.2f",
            req_id,
            request.method,
            request.url.path,
            getattr(response, "status_code", "-"),
            duration_ms,
        )
        return response

