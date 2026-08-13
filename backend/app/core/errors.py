"""
RadAssist AI — Error handling and request correlation (Phase 6, Step 5)

════════════════════════════════════════════════════════════════════
⚠️  EXCEPTION TEXT WAS BEING RETURNED TO CALLERS
════════════════════════════════════════════════════════════════════
Several endpoints did this:

    raise HTTPException(status_code=500, detail=f"Internal error: {e}")

Which hands whoever made the request whatever the exception happened to say —
file paths, driver messages, occasionally a connection string with credentials
in it. Nobody writes that on purpose; it accumulates because it is genuinely
useful while developing and then ships.

The information is not lost. It goes to the log, tied to a reference the
caller is given:

    response  500  {"detail": "Internal error.", "request_id": "b7f3a2e1"}
    log            b7f3a2e1  ValueError: ...  <full traceback>

The person debugging gets everything; the caller gets a string to quote.

════════════════════════════════════════════════════════════════════
WHY A REQUEST ID AND NOT JUST A TIMESTAMP
════════════════════════════════════════════════════════════════════
"It broke around 3pm" against a log with concurrent streaming requests is not
a search, it is an archaeology project. An id turns an incident report into
one grep — and it is the difference between a pilot user's complaint being
actionable and being noted.
"""

from __future__ import annotations

import logging
import secrets
import time

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Short enough to read down a phone line, long enough not to collide within a
# log retention window. 8 hex characters is ~4 billion values.
_ID_BYTES = 4

REQUEST_ID_HEADER = "X-Request-ID"


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def install(app: FastAPI) -> None:
    """Attach the middleware and handlers. Called once from main."""

    @app.middleware("http")
    async def _correlate(request: Request, call_next):
        """
        Give every request an id, log how it went, and return the id.

        ⚠️  AN INBOUND X-Request-ID IS NOT TRUSTED.
        Accepting a client-supplied value would let anyone forge or collide
        with an id, which is enough to make the logs untrustworthy exactly
        when they matter. Behind a gateway that already correlates requests,
        this is the place to adopt its id — deliberately, once, rather than
        by default.
        """
        request_id = secrets.token_hex(_ID_BYTES)
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Logged HERE with the traceback, because the handler below only
            # sees the exception and not the timing.
            elapsed = (time.perf_counter() - started) * 1000
            logger.exception(
                "%s %s %s failed after %.0fms",
                request_id, request.method, request.url.path, elapsed,
            )
            raise

        elapsed = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id

        # ⚠️  PATH ONLY — NEVER THE QUERY STRING OR THE BODY.
        # A chat request body contains dictated clinical findings. Logging
        # request content would undo Step 0, which stopped report text
        # reaching the logs via SQL echo.
        if elapsed > 3000 or response.status_code >= 400:
            logger.info(
                "%s %s %s → %s (%.0fms)",
                request_id, request.method, request.url.path,
                response.status_code, elapsed,
            )
        return response

    # ── Handlers ────────────────────────────────────────────

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        """
        Deliberate errors — 401, 404, 429 and the rest.

        `detail` here is written by us for the caller, so it is safe to return
        as-is. That is the distinction from the handler below: this text was
        composed; that text escaped.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "request_id": _request_id(request),
            },
            headers={
                **(exc.headers or {}),
                REQUEST_ID_HEADER: _request_id(request),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        """
        422 from Pydantic.

        ⚠️  THE INPUT IS STRIPPED OUT OF THE RESPONSE.
        Pydantic's default error payload echoes the offending value back. On
        a login that means the password appears in the response body and in
        any client-side error log. The field location and the reason are
        enough to fix a request; the value is not needed and is sometimes a
        credential.
        """
        safe = [
            {
                "field": ".".join(str(p) for p in err.get("loc", ())),
                "message": err.get("msg", "invalid"),
                "type": err.get("type", "value_error"),
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": "Request validation failed.",
                "errors": safe,
                "request_id": _request_id(request),
            },
            headers={REQUEST_ID_HEADER: _request_id(request)},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        """
        Anything that got away.

        ⚠️  THE RESPONSE NEVER CONTAINS THE EXCEPTION.
        Not its message, not its type, not a traceback. All of it is already
        in the log against this request id — which is what the caller is
        given instead.

        In DEBUG the type is included, because a developer staring at their
        own machine benefits and there is no third party to leak to. That is
        a deliberate exception and the only one.
        """
        request_id = _request_id(request)
        logger.exception("%s unhandled error on %s", request_id, request.url.path)

        body = {
            "detail": "Internal error.",
            "request_id": request_id,
        }
        if settings.DEBUG:
            body["debug_type"] = type(exc).__name__

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=body,
            headers={REQUEST_ID_HEADER: request_id},
        )
