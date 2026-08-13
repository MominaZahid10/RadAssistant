"""
Error handling and request correlation (Phase 6, Step 5).

⚠️  ENDPOINTS WERE RETURNING EXCEPTION TEXT TO CALLERS.

    raise HTTPException(status_code=500, detail=f"Internal error: {e}")

That hands whoever made the request whatever the exception happened to say —
file paths, driver messages, occasionally a connection string with credentials
in it. Nobody writes it on purpose; it accumulates because it is useful while
developing, and then it ships.

The information is not lost. It goes to the log against a request id that the
caller is given instead.
"""

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

from app.core import errors


def _code_only(source: str) -> str:
    """
    Strip comments and docstrings.

    Same reason as tests/test_ownership.py: the comments here quote the very
    pattern they exist to forbid, so a naive substring search fails on the
    explanation rather than on the code.
    """
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _unquoted(source: str) -> str:
    """
    Code with comments, docstrings AND quote characters removed.

    ⚠️  ast.unparse NORMALISES QUOTES.
    It emits 'request_id' where the source wrote "request_id", so a literal
    assertion fails on a difference that has no meaning. Dropping quote
    characters from both sides compares what the code says rather than how it
    was typed.
    """
    return _code_only(source).replace('"', "").replace("'", "")


# ══════════════════════════════════════════════════════════════
# NO EXCEPTION TEXT IN ANY RESPONSE
# ══════════════════════════════════════════════════════════════


ENDPOINT_FILES = sorted(Path("app/api/v1/endpoints").glob("*.py"))


@pytest.mark.parametrize(
    "path", ENDPOINT_FILES, ids=lambda p: p.name
)
def test_no_endpoint_interpolates_an_exception_into_a_response(path):
    """
    ⚠️  THE REGRESSION GUARD FOR THE WHOLE STEP.
    Every file is checked, so a new endpoint written next month is covered
    without anyone remembering to add it here.
    """
    code = _code_only(path.read_text(encoding="utf-8"))

    for forbidden in (
        'detail=f"Internal error: {e}"',
        'detail=f"Internal error: {exc}"',
        '"error": f"Internal error: {e}"',
    ):
        assert forbidden not in code, f"{path.name} returns exception text"


def test_chat_logs_the_traceback_instead_of_returning_it():
    from app.api.v1.endpoints import chat as chat_module

    code = _unquoted(inspect.getsource(chat_module._complete_response))
    assert "logger.exception" in code
    assert "detail=Internal error." in code


def test_the_sse_error_event_carries_no_exception_text():
    """
    An SSE error reaches the browser and lands in client-side logs, so it
    needs the same treatment as an HTTP body.
    """
    from app.api.v1.endpoints import chat as chat_module

    code = _unquoted(inspect.getsource(chat_module._sse_generator))
    assert "error: Internal error." in code


def test_image_processing_failures_are_not_stored_verbatim():
    """
    ⚠️  error_message IS RETURNED BY GET /images/{id}.
    A background task wrote the raw exception into it, so a driver message
    would be served back through the API. The two *handled* cases stay
    verbatim — those are sentences written for the user.
    """
    from app.api.v1.endpoints import images as images_module

    code = _code_only(inspect.getsource(images_module._process_upload))
    assert 'f"Internal error: {e}"' not in code
    assert "logger.exception" in code
    # The user-facing messages survive.
    assert "DicomError" in code and "ImageProcessingError" in code


# ══════════════════════════════════════════════════════════════
# WHAT THE CALLER GETS INSTEAD
# ══════════════════════════════════════════════════════════════


def test_the_unhandled_handler_returns_no_exception_detail():
    code = _unquoted(inspect.getsource(errors.install))
    assert "detail: Internal error." in code
    # Not the message, not the type, not a traceback.
    assert "str(exc)" not in code
    assert "traceback" not in code.lower()


def test_debug_mode_is_the_only_exception_and_only_leaks_the_type():
    """
    A developer on their own machine benefits and there is no third party to
    leak to. It is the class name, never the message — messages are where
    paths and credentials live.
    """
    code = _code_only(inspect.getsource(errors.install))
    assert "settings.DEBUG" in code
    assert "type(exc).__name__" in code


def test_every_response_carries_a_request_id():
    code = _unquoted(inspect.getsource(errors.install))
    assert code.count("REQUEST_ID_HEADER") >= 3      # middleware + handlers
    assert "request_id:" in code


def test_deliberate_errors_keep_their_message():
    """
    401, 404 and 429 have text we composed for the caller. The distinction is
    that this text was written; the other escaped.
    """
    code = _unquoted(inspect.getsource(errors.install))
    assert "detail: exc.detail" in code


# ══════════════════════════════════════════════════════════════
# VALIDATION ERRORS
# ══════════════════════════════════════════════════════════════


def test_validation_errors_do_not_echo_the_submitted_value():
    """
    ⚠️  PYDANTIC'S DEFAULT PAYLOAD INCLUDES THE INPUT.
    On a login that means the password appears in the response body and in
    any client-side error log. The field and the reason are enough to fix a
    request; the value is sometimes a credential.
    """
    code = _unquoted(inspect.getsource(errors.install))
    assert "field:" in code and "message:" in code
    # The key Pydantic uses for the offending value.
    assert "err.get(input" not in code


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════


def test_the_request_body_is_never_logged():
    """
    ⚠️  A CHAT BODY CONTAINS DICTATED CLINICAL FINDINGS.
    Logging request content would undo Step 0, which stopped report text
    reaching the logs through SQL echo. Path only — no query string, no body.
    """
    code = _code_only(inspect.getsource(errors.install))
    assert "request.url.path" in code
    assert "await request.body()" not in code
    assert "request.url.query" not in code


def test_an_inbound_request_id_is_not_trusted():
    """
    Accepting a client-supplied id lets anyone forge or collide with one,
    which makes the logs untrustworthy exactly when they matter.
    """
    code = _code_only(inspect.getsource(errors.install))
    assert "secrets.token_hex" in code
    assert 'headers.get("x-request-id")' not in code.lower()


def test_handlers_are_installed_before_the_routes():
    """A handler registered after the router still applies, but the ordering
    makes the intent legible — and the middleware must wrap everything."""
    source = Path("app/main.py").read_text(encoding="utf-8")
    assert source.index("errors.install(app)") < source.index(
        "app.include_router(api_v1_router)"
    )
