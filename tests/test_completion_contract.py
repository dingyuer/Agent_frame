"""Contracts require current observable evidence and distinct real artifacts."""
import hashlib
import os
from types import SimpleNamespace

import pytest

from osworld_agent.verifiers import CompletionVerifier


URL = "https://example.com/form"


def page(*nodes, url=URL, warnings=None):
    return SimpleNamespace(context=[{"kind": "browser_ax", "url": url,
                                     "nodes": list(nodes), "warnings": warnings or []}])


def contract(*checks):
    return CompletionVerifier("Complete the supplied task", {"checks": list(checks)})


def download_event(path, url="https://example.com/a.pdf", status="executed"):
    return {"action": "download", "status": status, "url": url, "path": str(path), "verified": True}


def test_all_checks_required_and_final_state_is_not_latched():
    verifier = contract({"kind": "url_equals", "url": URL},
                        {"kind": "element_state", "role": "textbox", "name": "Email", "value": "a@example.com"},
                        {"kind": "text_present", "text": "Submitted", "url": URL})
    field = {"role": "textbox", "name": "Email", "value": "a@example.com"}
    verifier.observe(page(field))
    assert verifier.evaluate()["status"] == "failed"
    verifier.observe(page(field, {"role": "status", "name": "Submitted"}))
    assert verifier.evaluate()["status"] == "passed"
    field["value"] = "wrong@example.com"
    verifier.observe(page(field, {"role": "status", "name": "Submitted"}))
    assert verifier.evaluate()["status"] == "failed"
    verifier.observe(None)
    assert verifier.evaluate()["status"] == "unknown"


def test_milestone_requires_real_evidence_and_survives_reobserve():
    verifier = contract({"kind": "url_equals", "url": URL, "scope": "milestone"},
                        {"kind": "text_present", "text": "Submitted", "url": "https://example.com/done"})
    verifier.observe(None, {"action": "goto", "status": "executed", "url": URL})
    assert verifier.evaluate()["checks"][0]["status"] == "unknown"
    verifier.observe(page())
    # No evaluate between this evidence and the later observation.
    verifier.observe(page({"role": "status", "name": "Submitted"}, url="https://example.com/done"))
    assert verifier.evaluate()["status"] == "passed"
    verifier.observe(None)
    result = verifier.evaluate()
    assert result["status"] == "unknown"
    assert result["checks"][0]["status"] == "passed"
    assert result["checks"][1]["status"] == "unknown"


def test_duplicate_names_are_ambiguous_and_element_id_can_disambiguate():
    checks = {"kind": "element_state", "role": "textbox", "name": "Email", "value": "correct"}
    evidence = page({"id": "one", "role": "textbox", "name": "Email", "value": "correct"},
                    {"id": "two", "role": "textbox", "name": "Email", "value": "wrong"})
    verifier = contract(checks)
    verifier.observe(evidence)
    assert verifier.evaluate()["status"] == "unknown"
    verifier = contract({**checks, "element_id": "one"})
    verifier.observe(evidence)
    assert verifier.evaluate()["status"] == "passed"


def test_incomplete_ax_and_missing_state_remain_unknown():
    verifier = contract({"kind": "element_state", "role": "checkbox", "name": "Agree", "states": {"checked": True}})
    verifier.observe(page({"role": "checkbox", "name": "Agree", "states": {}}))
    assert verifier.evaluate()["status"] == "unknown"
    verifier.observe(page(warnings=["node_limit: omitted nodes"]))
    assert verifier.evaluate()["status"] == "unknown"
    verifier.observe(page({"role": "checkbox", "name": "Agree", "states": {"checked": True}},
                          warnings=["node_limit: omitted nodes"]))
    assert verifier.evaluate()["status"] == "unknown"
    verifier.observe(page({"role": "checkbox", "name": "Agree", "states": {"checked": False}}))
    assert verifier.evaluate()["status"] == "failed"


def test_text_present_is_not_a_substring_or_model_claim():
    verifier = contract({"kind": "text_present", "text": "Submitted"})
    verifier.observe(page({"role": "status", "name": "Not Submitted"}))
    assert verifier.evaluate()["status"] == "failed"
    verifier.observe({"text": "Submitted", "memory": "Submitted", "answer": "Submitted"})
    assert verifier.evaluate()["status"] == "unknown"


def test_download_rechecks_deleted_or_modified_file_even_for_milestone(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"artifact bytes")
    verifier = contract({"kind": "download", "url": "https://example.com/a.pdf", "scope": "milestone",
                         "filename": "a.pdf", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    verifier.observe(None, download_event(path, status="already_downloaded"))
    verifier.observe(None)
    assert verifier.evaluate()["status"] == "passed"
    path.write_bytes(b"modified bytes")
    assert verifier.evaluate()["status"] == "failed"
    path.unlink()
    assert verifier.evaluate()["status"] == "failed"


def test_download_needs_matching_success_event_and_nonempty_file(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"real bytes")
    verifier = contract({"kind": "download", "url": "https://example.com/a.pdf"})
    for event in (download_event(path, url="https://example.com/b.pdf"),
                  download_event(path, status="rejected"),
                  {"action": "download", "verified": True, "path": str(path)}):
        verifier.observe(None, event)
        assert verifier.evaluate()["status"] != "passed"
    path.write_bytes(b"")
    verifier.observe(None, download_event(path))
    assert verifier.evaluate()["status"] == "failed"


def test_two_downloads_cannot_use_same_file_or_hard_link(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    a.write_bytes(b"artifact")
    os.link(a, b)
    verifier = contract({"kind": "download", "url": "https://example.com/a.pdf"},
                        {"kind": "download", "url": "https://example.com/b.pdf"})
    verifier.observe(None, download_event(a))
    verifier.observe(None, download_event(b, url="https://example.com/b.pdf"))
    assert verifier.evaluate()["status"] == "failed"
    b.unlink()
    b.write_bytes(b"another artifact")
    verifier.observe(None, download_event(b, url="https://example.com/b.pdf"))
    assert verifier.evaluate()["status"] == "passed"


def test_download_assignment_preserves_specific_filename_match(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    a.write_bytes(b"A")
    b.write_bytes(b"B")
    verifier = contract({"kind": "download", "url": "https://example.com/a.pdf"},
                        {"kind": "download", "url": "https://example.com/a.pdf", "filename": "a.pdf"})
    verifier.observe(None, download_event(a))
    verifier.observe(None, download_event(b))
    assert verifier.evaluate()["status"] == "passed"


def test_same_path_cannot_represent_two_download_urls(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"artifact")
    verifier = contract({"kind": "download", "url": "https://example.com/a.pdf"},
                        {"kind": "download", "url": "https://example.com/b.pdf"})
    verifier.observe(None, download_event(path))
    verifier.observe(None, download_event(path, url="https://example.com/b.pdf"))
    assert verifier.evaluate()["status"] != "passed"


def test_file_appearing_after_download_event_needs_new_event(tmp_path):
    path = tmp_path / "a.pdf"
    verifier = contract({"kind": "download", "url": "https://example.com/a.pdf"})
    verifier.observe(None, download_event(path))
    path.write_bytes(b"unattributed file")
    assert verifier.evaluate()["status"] == "failed"


def test_download_hash_and_size_requirements_are_checked(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(b"artifact")
    for extra in ({"sha256": "0" * 64}, {"min_bytes": 100}, {"filename": "b.pdf"}):
        verifier = contract({"kind": "download", "url": "https://example.com/a.pdf", **extra})
        verifier.observe(None, download_event(path))
        assert verifier.evaluate()["status"] == "failed"


def test_url_normalization_does_not_discard_path_query_or_fragment():
    verifier = contract({"kind": "url_equals", "url": "https://EXAMPLE.com:443"})
    verifier.observe(page(url="https://example.com/"))
    assert verifier.evaluate()["status"] == "passed"
    for url in ("https://example.com/other", "https://example.com/?wrong=yes", "https://example.com/#other"):
        verifier.observe(page(url=url))
        assert verifier.evaluate()["status"] == "failed"


def test_inference_is_anchored_and_empty_spec_remains_unknown():
    for task in ("请打开 https://example.com。", "please visit https://example.com", "下载 https://example.com/a.pdf"):
        verifier = CompletionVerifier(task)
        assert verifier.source == "task"
        assert verifier.spec["checks"][0]["id"] == "check_1"
    for task in ("first open https://example.com", "open https://example.com and submit", "打开这个页面",
                 "打开 https://example.com并提交", "下载 https://example.com/a.pdf 和 https://example.com/b.pdf"):
        assert CompletionVerifier(task).evaluate()["status"] == "unknown"
    assert CompletionVerifier("open https://example.com", {"checks": []}).evaluate()["status"] == "unknown"


@pytest.mark.parametrize("spec", [
    {}, {"checks": "anything"}, {"checks": [], "extra": True},
    {"checks": [{"kind": "llm_judge"}]},
    {"checks": [{"kind": "url_equals", "url": "not a url"}]},
    {"checks": [{"kind": "url_equals", "url": URL, "scope": "anything"}]},
    {"checks": [{"kind": "text_present", "text": "Submitted", "model_approved": True}]},
    {"checks": [{"kind": "element_state", "role": "textbox", "name": "Email"}]},
    {"checks": [{"kind": "element_state", "role": "textbox", "name": "Email", "value": 12}]},
    {"checks": [{"kind": "download", "url": URL, "min_bytes": True}]},
    {"checks": [{"kind": "download", "url": URL, "sha256": "bad"}]},
    {"checks": [{"kind": "download", "url": URL, "filename": "../a.pdf"}]},
    {"checks": [{"kind": "url_equals", "url": URL, "id": "duplicate"},
                {"kind": "url_equals", "url": URL, "id": "duplicate"}]},
])
def test_invalid_contract_is_rejected(spec):
    with pytest.raises(ValueError):
        CompletionVerifier("task", spec)


def test_spec_and_observation_are_copied():
    original = {"checks": [{"kind": "text_present", "text": "Submitted"}]}
    verifier = CompletionVerifier("task", original)
    original["checks"][0]["text"] = "Not Submitted"
    exposed = verifier.spec
    exposed["checks"][0]["text"] = "Not Submitted"
    observation = page({"name": "Submitted"})
    verifier.observe(observation)
    observation.context[0]["nodes"].clear()
    assert verifier.evaluate()["status"] == "passed"
