import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass, replace

from starlette.types import ASGIApp, Receive, Scope, Send


@dataclass(frozen=True)
class ReviewLogContext:
    review_request_id: int | None = None
    revision_id: int | None = None
    diff_id: int | None = None


_empty_context = ReviewLogContext()
_review_context = ContextVar("review_log_context", default=_empty_context)
_process_path = re.compile(r"/internal/process/(\d+)")
logger = logging.getLogger(__name__)


def bind_review_context(**fields: int | None) -> None:
    """Add identifiers for the review being handled by the current request."""
    _review_context.set(replace(_review_context.get(), **fields))


def install_review_logging() -> None:
    """Annotate every logger, including library and Uvicorn loggers.

    Prefix the message at record creation so existing handlers and formatters
    retain their formatting while all of them include the review identifiers.
    Child asyncio tasks and asyncio.to_thread inherit the request's context.
    """
    previous_factory = logging.getLogRecordFactory()
    if getattr(previous_factory, "_review_logging", False):
        return

    def record_factory(*args, **kwargs):
        record = previous_factory(*args, **kwargs)
        context = _review_context.get()
        fields = vars(context)
        for name, value in fields.items():
            setattr(record, name, value)
        prefix = " ".join(
            f"{name}={value}" for name, value in fields.items() if value is not None
        )
        if prefix:
            record.msg = f"[{prefix}] {record.msg}"
        return record

    record_factory._review_logging = True
    logging.setLogRecordFactory(record_factory)


class ReviewLoggingMiddleware:
    """Scope review identifiers to an entire ASGI request, including cleanup.

    Use pure ASGI middleware so endpoint context updates remain visible to
    response/access logging and dependency teardown in the same task.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        match = _process_path.fullmatch(scope["path"])
        context = ReviewLogContext(review_request_id=int(match[1]) if match else None)
        token = _review_context.set(context)
        try:
            await self.app(scope, receive, send)
        except Exception:
            # Uvicorn logs exceptions after our context has been reset. Record
            # the failure here too, while its review identifiers are available.
            logger.exception("Unhandled request error")
            raise
        finally:
            _review_context.reset(token)
