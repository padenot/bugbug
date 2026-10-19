import asyncio
import logging
from typing import Annotated

import httpx
import pytest
from app.logging_context import (
    ReviewLoggingMiddleware,
    bind_review_context,
    install_review_logging,
)
from fastapi import Depends, FastAPI


@pytest.fixture(autouse=True)
def review_logging():
    previous_factory = logging.getLogRecordFactory()
    install_review_logging()
    yield
    logging.setLogRecordFactory(previous_factory)


async def test_concurrent_reviews_include_library_child_task_and_access_logs(caplog):
    caplog.set_level(logging.INFO)
    both_started = asyncio.Event()
    started = 0

    async def app(scope, receive, send):
        nonlocal started
        review_id = int(scope["path"].rsplit("/", 1)[1])
        logging.getLogger("app.database").info("Loading review %s", review_id)
        bind_review_context(revision_id=334842 + review_id, diff_id=1410580 + review_id)
        started += 1
        if started == 2:
            both_started.set()
        await both_started.wait()

        async def child():
            logging.getLogger("httpx").info("Fetching parent for %s", review_id)

        await asyncio.create_task(child())
        await asyncio.to_thread(
            logging.getLogger("provider").info, "Model request for %s", review_id
        )
        try:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})
        finally:
            logging.getLogger("app.database").info("Cleanup for %s", review_id)

    middleware = ReviewLoggingMiddleware(app)

    async def request(review_id):
        async def send(message):
            if message["type"] == "http.response.start":
                logging.getLogger("uvicorn.access").info("Response for %s", review_id)

        await middleware(
            {"type": "http", "path": f"/internal/process/{review_id}"}, None, send
        )

    await asyncio.gather(request(1), request(2))
    logging.getLogger("app").info("Outside a review")

    for record in caplog.records[:-1]:
        review_id = record.args[0]
        assert record.review_request_id == review_id
        assert f"review_request_id={review_id}" in record.getMessage()
        if record.getMessage().endswith(f"Loading review {review_id}"):
            assert record.revision_id is None
        else:
            assert record.revision_id == 334842 + review_id
            assert record.diff_id == 1410580 + review_id
            assert f"revision_id={334842 + review_id}" in record.getMessage()
            assert f"diff_id={1410580 + review_id}" in record.getMessage()
    assert caplog.records[-1].review_request_id is None
    assert caplog.records[-1].getMessage() == "Outside a review"


async def test_endpoint_context_reaches_dependency_cleanup_and_response(caplog):
    # Exercise FastAPI itself: BaseHTTPMiddleware would lose endpoint updates
    # when logging the response from the outer middleware task.
    caplog.set_level(logging.INFO)
    app = FastAPI()
    app.add_middleware(ReviewLoggingMiddleware)

    async def dependency():
        try:
            yield
        finally:
            logging.getLogger("app.database").info("Dependency cleanup")

    @app.get("/internal/process/{review_request_id}")
    async def endpoint(
        review_request_id: int, resource: Annotated[None, Depends(dependency)]
    ):
        bind_review_context(revision_id=334842, diff_id=1410580)
        logging.getLogger("httpx").info("Fetched parent")
        return {"ok": True}

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        if message["type"] == "http.response.start":
            logging.getLogger("uvicorn.access").info("Access log")

    await app(
        {
            "type": "http",
            "method": "GET",
            "path": "/internal/process/3457",
            "query_string": b"",
            "headers": [],
        },
        receive,
        send,
    )
    assert len(caplog.records) == 3
    for record in caplog.records:
        assert record.review_request_id == 3457
        assert record.revision_id == 334842
        assert record.diff_id == 1410580


@pytest.mark.parametrize("error", [RuntimeError("failure"), asyncio.CancelledError()])
async def test_context_is_reset_after_failure_or_cancellation(caplog, error):
    caplog.set_level(logging.INFO)

    async def app(scope, receive, send):
        bind_review_context(revision_id=334842, diff_id=1410580)
        raise error

    with pytest.raises(type(error)):
        await ReviewLoggingMiddleware(app)(
            {"type": "http", "path": "/internal/process/3457"}, None, None
        )

    logging.getLogger("app").info("Outside a review")
    if isinstance(error, RuntimeError):
        assert caplog.records[0].review_request_id == 3457
        assert caplog.records[0].exc_info[1] is error
    assert caplog.records[-1].review_request_id is None


async def test_install_is_idempotent_and_unrelated_requests_have_no_review(caplog):
    caplog.set_level(logging.INFO)
    factory = logging.getLogRecordFactory()
    install_review_logging()
    assert logging.getLogRecordFactory() is factory

    app = FastAPI()
    app.add_middleware(ReviewLoggingMiddleware)

    @app.get("/health")
    async def health():
        logging.getLogger("app").info("Healthy")
        return {"ok": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        assert (await client.get("/health")).status_code == 200
    assert all(record.review_request_id is None for record in caplog.records)
