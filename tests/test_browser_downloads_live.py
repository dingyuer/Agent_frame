"""Local Chromium checks for resource evidence and download lifecycle APIs."""
import time
from pathlib import Path

from osworld_agent.browser_session import browser_thread
from test_browser_live import live_browser, click, observe


def test_live_download_candidates_and_response_mime(live_browser):
    session, base = live_browser
    observation = observe(session)
    info = observation.info
    assert info["content_type"] == "text/html"
    assert info["document_kind"] == "html"
    candidate = next(c for c in info["download_candidates"] if c["url"] == base + "/paper.pdf")
    assert candidate["source"] == "anchor" and candidate["download_attribute"] is True
    assert candidate["page_id"] == info["page_id"]
    assert candidate["snapshot_id"] == info["snapshot_id"]
    assert observation.context[0]["download_candidates"] == info["download_candidates"]


def test_live_http_download_rename_and_alias_cache(live_browser):
    session, base = live_browser
    first = session.download(base + "/private.pdf", filename="original.pdf")
    identity = first["artifact_id"]
    result = session.rename_artifact(identity, "requested.pdf")
    assert result["status"] == "executed", result
    assert not Path(first["path"]).exists()
    assert Path(result["path"]).is_file()
    cached = session.download(base + "/private.pdf#page=1")
    assert cached["status"] == "already_downloaded"
    assert cached["artifact_id"] == identity and cached["path"] == result["path"]
    assert cached["artifacts"][0]["version"] == 2
    assert cached["artifacts"][0]["initial_sha256"] == first["artifacts"][0]["sha256"]


def test_live_pending_download_is_idempotent_and_cancelable(live_browser):
    session, base = live_browser
    result = click(session, "slow")
    assert result["status"] == "executed", result
    deadline = time.monotonic() + 1.5
    info = session.event_metadata()
    while not info["pending_downloads"] and time.monotonic() < deadline:
        time.sleep(.02)
        info = session.event_metadata()
    assert info["pending_downloads"], info
    pending = info["pending_downloads"][0]
    assert pending["page_id"] == session._task_page_id
    assert "bytes" not in pending
    duplicate = session.download(base + "/slow.pdf")
    assert duplicate["status"] == "in_progress" and duplicate["dispatched"] is False
    assert duplicate["download_id"] == pending["download_id"]
    canceled = session.cancel_download(pending["download_id"])
    assert canceled["status"] == "executed"
    assert not canceled["pending_downloads"] and not canceled["artifacts"]
    assert canceled["download_events"][-1]["status"] == "canceled"


def test_live_main_document_response_is_observed_without_body(live_browser):
    session, _ = live_browser
    owner = browser_thread(lambda self: self._main_responses.get(id(self._pick_page(self._connect()[1]))))
    record = owner(session)
    assert record and record["content_type"] == "text/html"
    assert set(record) == {"url", "content_type", "status"}
