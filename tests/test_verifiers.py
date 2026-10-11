"""Legacy completion helper must not turn action success into task success."""
from osworld_agent.verifiers import check_completion


def test_navigation_requires_current_target_observation():
    info = {"action": "goto", "status": "executed", "url": "https://arxiv.org"}
    assert check_completion("打开 https://arxiv.org", info) == ""
    info["observation"] = {"context": [{"kind": "browser_ax", "url": "https://arxiv.org/", "nodes": []}]}
    assert check_completion("打开 https://arxiv.org", info) == "navigation"
    info["observation"]["context"][0]["url"] = "https://other.example/"
    assert check_completion("打开 https://arxiv.org", info) == ""


def test_compound_and_ambiguous_tasks_never_auto_complete():
    info = {"action": "goto", "status": "executed", "url": "https://example.com",
            "observation": {"context": [{"kind": "browser_ax", "url": "https://example.com", "nodes": []}]}}
    for task in ("打开 https://example.com 并填写表单后提交", "访问这个页面", "打开 example.com",
                 "open https://example.com and submit the form", "下载 A 和 B 两篇论文"):
        assert check_completion(task, info) == ""


def test_download_uses_real_bytes_not_verified_flag(tmp_path):
    path = tmp_path / "paper.pdf"
    info = {"action": "download", "status": "executed", "url": "https://example.com/paper.pdf",
            "verified": True, "path": str(path), "bytes": 100}
    task = "下载 https://example.com/paper.pdf"
    assert check_completion(task, info) == ""
    path.write_bytes(b"a real artifact")
    info["verified"] = False
    assert check_completion(task, info) == "download"
    assert check_completion("下载这篇论文", info) == ""
    assert check_completion("下载 https://example.com/other.pdf", info) == ""
    path.unlink()
    assert check_completion(task, info) == ""


def test_no_match():
    info = {"action": "click", "status": "executed"}
    assert check_completion("下载这篇论文", info) == ""
    assert check_completion("打开页面", info) == ""
