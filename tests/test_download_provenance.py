"""Completed downloads keep the provenance of their pending event."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from osworld_agent.browser_session import browser_thread
from test_browser_lifecycle import session


owner = browser_thread(lambda self, fn: fn())


def test_fast_async_download_keeps_source_page_after_pending_disappears(session):
    page = SimpleNamespace(url="https://portal.test/library")

    async def save(path):
        Path(path).write_bytes(b"downloaded content")

    download = SimpleNamespace(url="https://files.test/signed-transfer", suggested_filename="slides.pptx",
                               save_as=save, cancel=AsyncMock())

    def queue():
        session._await(asyncio.sleep(0))
        session._page_target_ids[id(page)] = "source-page"
        session._queue_download(page, download)
        pending = dict(next(iter(session._active_downloads.values())))
        # The initiating document may change before the transfer completes.
        page.url = "https://portal.test/other"
        session._page_target_ids[id(page)] = "other-page"
        session._await(asyncio.sleep(0))
        return pending

    pending = owner(session, queue)
    metadata = session.event_metadata()
    assert not metadata["pending_downloads"]
    completed = metadata["artifacts"][0]
    assert completed["completed"] is True
    assert completed["download_id"] == pending["download_id"]
    assert completed["page_id"] == pending["page_id"] == "source-page"
    assert completed["page_url"] == pending["page_url"] == "https://portal.test/library"
    assert completed["started_at"] == pending["started_at"]
    assert completed["completed_at"] >= completed["started_at"]


def test_sync_drain_keeps_pending_source_page_and_start_time(session):
    page = SimpleNamespace(url="https://portal.test/library")
    download = SimpleNamespace(url="https://files.test/attachment", suggested_filename="document.pdf",
                               save_as=lambda path: Path(path).write_bytes(b"%PDF-fixture"), cancel=Mock())

    def queue_and_drain():
        assert session._loop is None
        session._page_target_ids[id(page)] = "source-page"
        session._queue_download(page, download)
        pending = dict(next(iter(session._active_downloads.values())))
        page.url = "https://portal.test/other"
        session._page_target_ids[id(page)] = "other-page"
        session._drain_downloads()
        return pending

    pending = owner(session, queue_and_drain)
    metadata = session.event_metadata()
    assert not metadata["pending_downloads"]
    completed = metadata["artifacts"][0]
    assert completed["page_id"] == pending["page_id"] == "source-page"
    assert completed["page_url"] == pending["page_url"] == "https://portal.test/library"
    assert completed["started_at"] == pending["started_at"]


def test_record_artifact_remains_compatible_without_optional_provenance(session, tmp_path):
    path = tmp_path / "existing-caller.pdf"
    path.write_bytes(b"%PDF-fixture")
    completed = session._record_artifact(path=str(path), url="https://files.test/file.pdf",
                                         final_url="https://files.test/file.pdf", download_id="existing-caller")
    assert completed["completed"] is True
    assert "page_id" not in completed
    assert "started_at" not in completed
