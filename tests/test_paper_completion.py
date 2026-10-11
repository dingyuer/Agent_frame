"""Paper completion comes from actual PDF content, not names or model claims."""
import json
import shutil
import hashlib
from types import SimpleNamespace

import pytest

from osworld_agent.task_specs import infer_paper_goal, task_profile
from osworld_agent.verifiers import CompletionVerifier


def pdf_file(path, *lines, metadata_title=None):
    """Make a real text PDF using pypdf; no remote model/browser or OCR needed."""
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({
        NameObject("/F1"): writer._add_object(font)})})
    text = "BT /F1 12 Tf 40 740 Td 18 TL\n"
    for index, line in enumerate(lines):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        text += ("T* " if index else "") + f"({escaped}) Tj\n"
    text += "ET"
    stream = DecodedStreamObject()
    stream.set_data(text.encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    if metadata_title:
        writer.add_metadata({"/Title": metadata_title})
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def event(path, ident="d1", **extra):
    return {"download_id": ident, "url": "https://example.org/paper.pdf",
            "final_url": "https://cdn.example.org/paper.pdf", "source_url": "https://example.org/paper",
            "status": "completed", "completed": True, "path": str(path), **extra}


def paper_verifier(**extra):
    return CompletionVerifier("download a paper", {"checks": [{"kind": "paper_download", "title": "AAA", **extra}]})


def test_natural_title_gets_public_goal_and_workflow():
    verifier = CompletionVerifier("请下载一篇叫做AAA的论文")
    check = verifier.spec["checks"][0]
    assert {k: check[k] for k in ("kind", "title", "count", "format")} == {
        "kind": "paper_download", "title": "AAA", "count": 1, "format": "pdf"}
    assert "url" not in check
    profile = task_profile("请下载一篇叫做AAA的论文", verifier.public_spec)
    assert [s["phase"] for s in profile["workflow"]] == ["search", "identify", "locate_fulltext", "download", "verify"]
    assert profile["goals"][0]["title"] == "AAA"
    assert verifier.evaluate()["status"] == "unknown"


@pytest.mark.parametrize("task", [
    "下载两篇叫做AAA的论文", "请下载一篇叫做AAA的论文并总结", "下载 AAA 和 BBB 两篇论文",
    "请下载一篇叫做AAA和BBB的论文", "下载这篇论文", "下载论文AAA并发邮件", "请下载一篇叫做AAA的论文，然后删除文件",
])
def test_ambiguous_or_compound_title_is_not_inferred(task):
    assert infer_paper_goal(task) is None


def test_explicitly_quoted_title_preserves_conjunction():
    assert infer_paper_goal('download a paper titled "Attention and Memory"')["title"] == "Attention and Memory"
    assert infer_paper_goal("下载论文《图像与语言》")["title"] == "图像与语言"


def test_real_pdf_title_and_completed_event_pass(tmp_path):
    path = pdf_file(tmp_path / "unrelated-name.bin", "AAA", "Alice Smith", "Abstract", "Research text.")
    verifier = CompletionVerifier("请下载一篇叫做AAA的论文")
    verifier.observe(None, {"artifacts": [event(path)]})
    result = verifier.evaluate()
    assert result["status"] == "passed"
    assert result["checks"][0]["matched_title"] == "AAA"
    assert result["checks"][0]["pages"] == 1


def test_pdf_magic_and_filename_are_not_enough(tmp_path):
    for content in (b"<html>AAA login</html>", b"%PDF-1.7\nnot a real document"):
        path = tmp_path / "AAA.pdf"
        path.write_bytes(content)
        verifier = paper_verifier()
        verifier.observe(None, {"artifacts": [event(path)]})
        assert verifier.evaluate()["status"] == "failed"


def test_metadata_only_and_body_reference_do_not_establish_title(tmp_path):
    for lines in (("Wrong paper", "Abstract", "AAA"), ("Wrong paper", "References", "AAA")):
        path = pdf_file(tmp_path / "AAA.pdf", *lines, metadata_title="AAA")
        verifier = paper_verifier()
        verifier.observe(None, {"artifacts": [event(path)]})
        assert verifier.evaluate()["status"] == "unknown"


def test_multiline_title_is_exact_normalized_heading(tmp_path):
    path = pdf_file(tmp_path / "paper.pdf", "Attention Is All", "You Need", "Alice", "Abstract")
    verifier = paper_verifier(title="Attention Is All You Need")
    verifier.observe(None, {"artifacts": [event(path)]})
    assert verifier.evaluate()["status"] == "passed"
    other = paper_verifier(title="Attention")
    other.observe(None, {"artifacts": [event(path)]})
    assert other.evaluate()["status"] == "unknown"


@pytest.mark.parametrize("actual,expected", [("C++", "C"), ("X > Y", "X < Y"), ("AB C", "A BC")])
def test_title_normalization_preserves_meaningful_punctuation_and_words(tmp_path, actual, expected):
    path = pdf_file(tmp_path / "paper.pdf", actual, "Abstract")
    verifier = paper_verifier(title=expected)
    verifier.observe(None, {"artifacts": [event(path)]})
    assert verifier.evaluate()["status"] == "unknown"


def test_paper_requires_real_success_event_not_model_identity_claim(tmp_path):
    path = pdf_file(tmp_path / "paper.pdf", "AAA", "Abstract")
    verifier = paper_verifier()
    for info in ({"artifacts": [event(path, status="failed")]},
                 {"artifacts": [event(path, completed=False)]},
                 {"artifacts": [event(path, url="bad", source_url="bad", final_url="bad")]},
                 {"paper_title": "AAA", "path": str(path), "verified": True}):
        verifier.observe(None, info)
        assert verifier.evaluate()["status"] == "unknown"


def test_deleted_modified_and_cumulative_artifacts_never_rebaseline(tmp_path):
    path = pdf_file(tmp_path / "paper.pdf", "AAA", "Abstract")
    verifier = paper_verifier(scope="milestone")
    artifact = event(path)
    verifier.observe(None, {"artifacts": [artifact]})
    assert verifier.evaluate()["status"] == "passed"
    pdf_file(path, "AAA", "Another revision", "Abstract")
    verifier.observe(None, {"artifacts": [artifact]})
    assert verifier.evaluate()["status"] == "failed"
    verifier.observe(None, {"artifacts": [event(path, ident="new-download")]})
    assert verifier.evaluate()["status"] == "passed"
    path.unlink()
    assert verifier.evaluate()["status"] == "failed"


def test_async_download_completion_is_consumed_from_later_observation(tmp_path):
    path = pdf_file(tmp_path / "paper.pdf", "AAA", "Abstract")
    verifier = paper_verifier()
    verifier.observe(SimpleNamespace(context=[], info={}), {"action": "click", "status": "executed"})
    assert verifier.evaluate()["status"] == "unknown"
    artifact = event(path)
    verifier.observe(SimpleNamespace(context=[], info={"artifacts": [artifact]}))
    assert verifier.evaluate()["status"] == "passed"
    pdf_file(path, "AAA", "Modified", "Abstract")
    # Duplicates across observation and execution must retain the first digest.
    verifier.observe({"context": [], "info": {"artifacts": [artifact]}}, {"artifacts": [artifact]})
    assert verifier.evaluate()["status"] == "failed"
    verifier.observe(None, {"artifacts": [event(path, ident="new-download")]})
    assert verifier.evaluate()["status"] == "passed"
    path.unlink()
    assert verifier.evaluate()["status"] == "failed"


def test_distinct_paper_count_rejects_copied_bytes(tmp_path):
    a = pdf_file(tmp_path / "a.pdf", "AAA", "Alice", "Abstract")
    b = tmp_path / "b.pdf"
    shutil.copyfile(a, b)
    verifier = paper_verifier(count=2)
    verifier.observe(None, {"artifacts": [event(a, "a"), event(b, "b")]})
    assert verifier.evaluate()["status"] == "failed"
    pdf_file(b, "AAA", "Bob", "Abstract")
    verifier.observe(None, {"artifacts": [event(b, "b2")]})
    assert verifier.evaluate()["status"] == "passed"


def test_optional_identity_fields_come_from_pdf_text(tmp_path):
    path = pdf_file(tmp_path / "paper.pdf", "AAA", "Alice Smith 2025", "arXiv:2501.01234v2", "doi:10.1234/paper.1", "Abstract")
    verifier = paper_verifier(author="Alice Smith", year=2025, arxiv_id="2501.01234", doi="https://doi.org/10.1234/paper.1")
    verifier.observe(None, {"artifacts": [event(path)]})
    assert verifier.evaluate()["status"] == "passed"
    for fields in ({"author": "Alice Jones"}, {"year": 2024}, {"arxiv_id": "2501.01234v3"}, {"doi": "10.1234/wrong"}):
        verifier = paper_verifier(**fields)
        verifier.observe(None, {"artifacts": [event(path)]})
        assert verifier.evaluate()["status"] == "unknown"


def test_identity_qualifiers_in_body_or_references_do_not_count(tmp_path):
    path = pdf_file(tmp_path / "paper.pdf", "AAA", "Wrong Author", "Abstract", "Alice Smith 2025", "arXiv:2501.01234", "10.1234/paper")
    for fields in ({"author": "Alice Smith"}, {"year": 2025}, {"arxiv_id": "2501.01234"}, {"doi": "10.1234/paper"}):
        verifier = paper_verifier(**fields)
        verifier.observe(None, {"artifacts": [event(path)]})
        assert verifier.evaluate()["status"] == "unknown"


def test_missing_pdf_dependency_stays_unknown(tmp_path, monkeypatch):
    import builtins
    path = pdf_file(tmp_path / "paper.pdf", "AAA", "Abstract")
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    verifier = paper_verifier()
    verifier.observe(None, {"artifacts": [event(path)]})
    assert verifier.evaluate()["status"] == "unknown"


def test_private_spec_and_failure_details_are_not_public():
    private = {"checks": [{"kind": "url_equals", "id": "SECRET_ID", "url": "https://secret.example/SECRET_TARGET"}]}
    verifier = CompletionVerifier("请下载一篇叫做AAA的论文", private_spec=private)
    assert len(verifier.spec["checks"]) == 2
    assert len(verifier.public_spec["checks"]) == 1
    verifier.observe({"kind": "browser_ax", "url": "https://secret.example/SECRET_ACTUAL", "nodes": []})
    full = verifier.evaluate()
    assert "SECRET_ACTUAL" in json.dumps(full)
    public = verifier.public_report(full)
    encoded = json.dumps({"spec": verifier.public_spec, "report": public})
    assert "SECRET" not in encoded and "secret.example" not in encoded
    assert public["status"] == "failed"
    assert public["checks"][-1]["id"] == "private_validation"


def test_private_hash_binding_verifies_same_public_paper(tmp_path):
    path = pdf_file(tmp_path / "paper.pdf", "AAA", "Abstract")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    verifier = CompletionVerifier("请下载一篇叫做AAA的论文", private_spec={"checks": [
        {"kind": "download", "bind_to": "check_1", "sha256": digest}]})
    verifier.observe(None, {"artifacts": [event(path)]})
    assert verifier.evaluate()["status"] == "passed"
    assert digest not in json.dumps(verifier.public_spec)
    # Private expected values are never exposed; observed public file hashes can
    # be evidence, so only the spec and private failure messages are secret.


def test_public_title_and_private_hash_cannot_be_spliced_from_different_files(tmp_path):
    a = pdf_file(tmp_path / "a.pdf", "AAA", "Abstract")
    b = pdf_file(tmp_path / "b.pdf", "BBB", "Abstract")
    verifier = CompletionVerifier("请下载一篇叫做AAA的论文", private_spec={"checks": [
        {"kind": "download", "bind_to": "check_1", "sha256": hashlib.sha256(b.read_bytes()).hexdigest()}]})
    verifier.observe(None, {"artifacts": [event(a, "a"), event(b, "b")]})
    assert verifier.evaluate()["status"] == "failed"


def test_binding_chooses_joint_candidate_instead_of_first_public_match(tmp_path):
    a = pdf_file(tmp_path / "a.pdf", "AAA", "Alice", "Abstract")
    b = pdf_file(tmp_path / "b.pdf", "AAA", "Bob", "Abstract")
    verifier = CompletionVerifier("请下载一篇叫做AAA的论文", private_spec={"checks": [
        {"kind": "download", "bind_to": "check_1", "sha256": hashlib.sha256(b.read_bytes()).hexdigest()}]})
    verifier.observe(None, {"artifacts": [event(a, "a"), event(b, "b")]})
    result = verifier.evaluate()
    assert result["status"] == "passed"
    assert all(c["path"] == str(b) for c in result["checks"])


def test_bound_constraint_applies_to_every_file_in_public_count(tmp_path):
    a = pdf_file(tmp_path / "a.pdf", "AAA", "Alice", "Abstract")
    b = pdf_file(tmp_path / "b.pdf", "AAA", "Bob", "Abstract")
    public = {"checks": [{"kind": "paper_download", "title": "AAA", "count": 2}]}
    private = {"checks": [{"kind": "paper_download", "title": "AAA", "author": "Alice", "bind_to": "check_1"}]}
    verifier = CompletionVerifier("download two papers", public, private)
    verifier.observe(None, {"artifacts": [event(a, "a"), event(b, "b")]})
    assert verifier.evaluate()["status"] == "failed"
    pdf_file(b, "AAA", "Alice", "Revised", "Abstract")
    verifier.observe(None, {"artifacts": [event(b, "b2")]})
    result = verifier.evaluate()
    assert result["status"] == "passed"
    assert all(c["count"] == 2 for c in result["checks"])


def test_binding_means_same_file_not_equal_bytes_from_other_source(tmp_path):
    a = pdf_file(tmp_path / "a.pdf", "AAA", "Abstract")
    b = tmp_path / "b.pdf"
    shutil.copyfile(a, b)
    public = {"checks": [{"kind": "paper_download", "title": "AAA", "url": "https://source.example/a"}]}
    private = {"checks": [{"kind": "download", "bind_to": "check_1", "url": "https://source.example/b"}]}
    verifier = CompletionVerifier("download paper", public, private)
    verifier.observe(None, {"artifacts": [event(a, "a", url="https://source.example/a"),
                                          event(b, "b", url="https://source.example/b")]})
    assert verifier.evaluate()["status"] == "failed"


@pytest.mark.parametrize("public,private", [
    ({"checks": [{"kind": "paper_download", "title": "AAA"}]},
     {"checks": [{"kind": "download", "bind_to": "missing", "sha256": "0" * 64}]}),
    ({"checks": [{"kind": "text_present", "text": "AAA"}]},
     {"checks": [{"kind": "download", "bind_to": "check_1", "sha256": "0" * 64}]}),
    ({"checks": [{"kind": "paper_download", "title": "AAA"}]},
     {"checks": [{"kind": "paper_download", "title": "AAA", "bind_to": "check_1", "count": 2}]}),
    ({"checks": [{"kind": "paper_download", "title": "AAA", "bind_to": "check_1"}]}, None),
    ({"checks": [{"kind": "paper_download", "title": "AAA"}]},
     {"checks": [{"kind": "text_present", "text": "AAA", "bind_to": "check_1"}]}),
])
def test_invalid_private_binding_fails_before_execution(public, private):
    with pytest.raises(ValueError):
        CompletionVerifier("task", public, private)


@pytest.mark.parametrize("fields", [{"count": 0}, {"count": True}, {"count": 21}, {"format": "html"},
                                        {"title": "!!!"}, {"year": "2025"}, {"doi": "wrong"}, {"arxiv_id": "wrong"}])
def test_invalid_paper_contract_is_rejected(fields):
    with pytest.raises(ValueError):
        paper_verifier(**fields)
