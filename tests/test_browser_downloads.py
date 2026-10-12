"""Download resource evidence, stable identity, and no-clobber lifecycle tests."""
import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from osworld_agent.browser_session import BrowserSession, browser_thread
from test_browser_lifecycle import EventBrowser, session


def http_fixture(session, *, url="https://arxiv.org/pdf/2505.16301", headers=None, body=b"%PDF-1.7 fixture"):
    browser = EventBrowser()
    context = browser.new_context()
    page = context.new_page()
    page.url = "https://arxiv.org/abs/2505.16301"
    response = SimpleNamespace(status=200, url=url,
        headers=headers if headers is not None else {"content-type": "application/pdf"},
        body=lambda: body, dispose=Mock())
    context.request.get.return_value = response
    session._connect = Mock(return_value=(None, browser))
    return page, context, response


def test_pdf_mime_adds_extension_to_dotted_arxiv_identifier(session):
    _, context, response = http_fixture(session)
    session.options.download_timeout_s = 12.5
    result = session.download(response.url)
    assert result["status"] == "executed"
    assert Path(result["path"]).name == "2505.16301.pdf"
    assert context.request.get.call_args.kwargs["timeout"] == 12500
    artifact = result["artifacts"][0]
    assert artifact["artifact_id"] == artifact["download_id"] == result["artifact_id"]
    assert artifact["sha256"] == artifact["initial_sha256"]
    assert artifact["version"] == 1 and artifact["rename_history"] == []


@pytest.mark.parametrize("disposition,expected", [
    ('attachment; filename="fallback.pdf"; filename*=UTF-8\'\'%E8%AE%BA%E6%96%87.pdf', "论文.pdf"),
    ('attachment; filename*=UTF-8\'\'%E8%AE%BA%E6%96%87.pdf; filename="fallback.pdf"', "论文.pdf"),
    ('attachment; filename="report; final.pdf"', "report; final.pdf"),
    ('attachment; filename="../../escape.pdf"', "escape.pdf"),
])
def test_content_disposition_extended_name_precedence(disposition, expected):
    assert BrowserSession._response_filename(disposition, "https://host/irrelevant", "application/pdf") == expected


def test_explicit_filename_and_redirect_alias_do_not_redownload(session):
    _, context, response = http_fixture(session)
    original = "https://arxiv.org/dl/observed?signature=unchanged"
    first = session.download(original, filename="first.pdf")
    second = session.download(response.url + "#page=1", filename="requested.pdf")
    assert second["status"] == "already_downloaded"
    assert context.request.get.call_count == 1
    assert second["artifact_id"] == first["artifact_id"]
    assert second["url"] == response.url + "#page=1"
    assert Path(second["path"]).name == "requested.pdf"
    assert second["artifacts"][0]["version"] == 2


def test_rename_preserves_hash_inode_and_full_history(session):
    _, context, response = http_fixture(session)
    first = session.download(response.url)
    original = first["artifacts"][0]
    inode = Path(original["path"]).stat().st_ino
    renamed = session.rename_artifact(first["artifact_id"], "paper.pdf")
    renamed_again = session.rename_artifact(first["artifact_id"], "final.pdf")
    final = renamed_again["artifacts"][0]
    assert renamed["status"] == renamed_again["status"] == "executed"
    assert final["version"] == 3
    assert final["sha256"] == final["initial_sha256"] == original["sha256"]
    assert Path(final["path"]).stat().st_ino == inode
    assert not Path(original["path"]).exists()
    assert [e["to_version"] for e in final["rename_history"]] == [2, 3]
    assert final["rename_history"][0]["from_path"] == original["path"]
    assert session.event_metadata()["artifact_events"][-1]["event"] == "artifact_renamed"
    assert session.download(response.url)["path"] == final["path"]
    assert context.request.get.call_count == 1


def test_rename_refuses_collision_and_modified_contents(session):
    _, _, response = http_fixture(session)
    first = session.download(response.url)
    destination = Path(session.downloads_dir) / "occupied.pdf"
    destination.write_bytes(b"other artifact")
    assert session.rename_artifact(first["artifact_id"], destination.name)["status"] == "rejected"
    assert destination.read_bytes() == b"other artifact"
    assert Path(first["path"]).exists()
    Path(first["path"]).write_bytes(b"changed")
    assert session.rename_artifact(first["artifact_id"], "new.pdf")["status"] == "rejected"
    assert session.event_metadata()["artifacts"][0]["version"] == 1


@pytest.mark.parametrize("filename", ["../out.pdf", "/tmp/out.pdf", "CON.pdf", "", "a/b.pdf"])
def test_requested_filename_is_rejected_before_network(session, filename):
    _, context, response = http_fixture(session)
    assert session.download(response.url, filename=filename)["status"] == "rejected"
    context.request.get.assert_not_called()


def test_pending_duplicate_download_returns_same_identity_without_request(session):
    _, context, response = http_fixture(session)
    active = {"url": response.url, "download_id": "existing", "artifact_id": "existing", "status": "in_progress"}
    session._active_downloads["existing"] = active
    result = session.download(response.url + "#page=2")
    assert result["status"] == "in_progress" and result["dispatched"] is False
    assert result["artifact_id"] == "existing"
    assert "bytes" not in result["pending_downloads"][0]
    context.request.get.assert_not_called()


def test_parallel_same_name_downloads_are_reserved_and_duplicate_is_canceled(session):
    owner = browser_thread(lambda self, fn: fn())
    owner(session, lambda: session._await(asyncio.sleep(0)))
    release = asyncio.Event()
    async def save(path):
        await release.wait()
        Path(path).write_bytes(b"downloaded")
    one = SimpleNamespace(url="https://host/one", suggested_filename="paper.pdf", save_as=save, cancel=AsyncMock())
    two = SimpleNamespace(url="https://host/two", suggested_filename="paper.pdf", save_as=save, cancel=AsyncMock())
    duplicate = SimpleNamespace(url=one.url, suggested_filename="paper.pdf", save_as=AsyncMock(), cancel=AsyncMock())
    page = SimpleNamespace(url="https://host/")
    for item in (one, two, duplicate):
        owner(session, lambda item=item: session._queue_download(page, item))
    owner(session, lambda: session._await(asyncio.sleep(0)))
    assert len(session.event_metadata()["pending_downloads"]) == 2
    duplicate.cancel.assert_awaited_once()
    duplicate.save_as.assert_not_called()
    owner(session, release.set)
    owner(session, lambda: session._await(asyncio.sleep(.01)))
    artifacts = session.event_metadata()["artifacts"]
    assert len(artifacts) == 2 and len({a["path"] for a in artifacts}) == 2
    assert not session.event_metadata()["pending_downloads"]


def test_explicit_cancel_preserves_failure_event_and_no_completed_artifact(session):
    owner = browser_thread(lambda self, fn: fn())
    owner(session, lambda: session._await(asyncio.sleep(0)))
    async def save(path):
        await asyncio.Event().wait()
    item = SimpleNamespace(url="https://host/paper", suggested_filename="paper.pdf", save_as=save, cancel=AsyncMock())
    owner(session, lambda: session._queue_download(SimpleNamespace(url="https://host/"), item))
    owner(session, lambda: session._await(asyncio.sleep(0)))
    identity = session.event_metadata()["pending_downloads"][0]["download_id"]
    assert session.cancel_download(identity)["status"] == "executed"
    info = session.event_metadata()
    assert not info["pending_downloads"] and not info["artifacts"]
    assert info["download_events"][-1]["status"] == "canceled"
    item.cancel.assert_awaited_once()


def test_resource_evidence_preserves_full_urls_and_arxiv_metadata(session):
    page, _, _ = http_fixture(session)
    long_url = "https://arxiv.org/pdf/2505.16301?token=" + "a" * 650
    page.evaluate = Mock(return_value={"content_type": "text/html", "base_url": page.url,
        "links": [{"url": long_url, "label": "Download PDF", "download": True}], "embeds": [],
        "citation": {"citation_title": ["Observed title"], "citation_author": ["Alice", "Bob"],
                     "citation_pdf_url": [long_url]}})
    info = session._resource_metadata(page, {"page_id": "p", "document_id": 12})
    assert info["document_kind"] == "html"
    assert info["download_candidates"][0]["url"] == long_url
    assert info["paper_metadata"]["canonical_id"] == "2505.16301"
    assert info["paper_metadata"]["authors"] == ["Alice", "Bob"]
    assert info["paper_metadata"]["pdf_url"] == long_url
    page.url = "https://untrusted.example/abs/2505.16301"
    assert "paper_metadata" not in session._resource_metadata(page, {})


def test_pdf_viewer_evidence_is_not_desktop_or_invented_url(session):
    page, _, _ = http_fixture(session)
    page.url = "https://arxiv.org/pdf/2505.16301"
    page.evaluate = Mock(return_value={"content_type": "text/html", "embeds": [{
        "type": "application/pdf", "src": page.url, "original_url": ""}]})
    info = session._resource_metadata(page, {})
    assert info["document_kind"] == "pdf_viewer"
    assert all(c["url"] == page.url for c in info["download_candidates"])
    page.evaluate.return_value = None
    page.url = "https://example.test/page"
    unknown = session._resource_metadata(page, {})
    assert "document_kind" not in unknown and "content_type" not in unknown
    assert unknown["download_candidates"] == []


def test_html_embedding_or_viewer_page_is_not_a_pdf_download_candidate(session):
    page, _, _ = http_fixture(session)
    page.url = "https://host.test/page.html"
    page.evaluate = Mock(return_value={"content_type": "text/html", "embeds": [{
        "type": "application/pdf", "src": "/actual.pdf"}]})
    info = session._resource_metadata(page, {})
    assert [c["url"] for c in info["download_candidates"]] == ["https://host.test/actual.pdf"]
    page.evaluate.return_value = {"content_type": "text/html", "pdf_viewer": True, "embeds": []}
    assert session._resource_metadata(page, {})["download_candidates"] == []
