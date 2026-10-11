"""Real browser demonstration, policy-compatible supervision, and portable export."""
import json
import zipfile

from osworld_agent.viz.collection import CollectionController
from test_browser_live import live_browser


def test_live_human_download_review_and_portable_export(live_browser, tmp_path):
    session, base = live_browser
    capture = CollectionController(tmp_path / "capture", browser_factory=lambda *_: session)
    try:
        state = capture.start({"task": "请下载一篇叫做AAA的论文", "start_url": base})
        # Actual rendered node bounds, without hard-coding fixture positions.
        node = next(node for node in capture.observation.context[0]["nodes"] if node.get("id") == "pdf")
        x, y, width, height = node["bounds"]
        result = capture.action({"session_id": state["session_id"], "observation_id": state["observation"]["id"],
            "request_id": "download", "kind": "click", "x": x + width / 2, "y": y + height / 2})
        assert result["execution"]["status"] == "executed", result
        assert result["steps"][0]["action"]["args"]["target"]
        capture.review({"session_id": state["session_id"], "step": "h_1", "keep": True})
        result = capture.stop({"session_id": state["session_id"]})
        assert result["verification"]["status"] == "passed", result
        bundle = capture.export(state)
        with zipfile.ZipFile(bundle) as archive:
            rows = [json.loads(line) for line in archive.read("sft.jsonl").splitlines()]
            assert len(rows) == 1 and rows[0]["human_reviewed"] and rows[0]["independent_verified"]
            assert all(path in archive.namelist() for path in rows[0]["images"])
            completion = json.loads(rows[0]["completion"][0]["content"])
            assert completion["next_action"]["type"] == "click"
            assert "x" not in completion["next_action"]
            assert "AAA" in str(rows[0]["prompt"])
    finally:
        capture.close()
