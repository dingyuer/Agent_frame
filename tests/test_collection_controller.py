import copy
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from osworld_agent.viz.collection import CollectionController, CollectionError, collection_router


class Browser:
    def __init__(self):
        self.captures = 0
        self.executed = []
        self.closed = False
        self.sensitive = False
        self.url = "https://example.test/"
    def begin_episode(self, **kwargs):
        return {"isolated": True}
    def goto(self, url):
        self.url = url
        self.executed.append(("goto", url))
        return {"status": "executed"}
    def collect_observation(self):
        self.captures += 1
        return Image.new("RGB", (100, 100)), {"kind": "browser_ax", "nodes": [], "url": self.url}, {
            "url": self.url, "screenshot_size": [100, 100], "screenshot_space": "css_viewport",
            "snapshot_id": str(self.captures), "page_id": "p", "document_id": "d"}
    def inspect_point(self, *args):
        return {"status": "inspected", "target_label": "Search", "bounds": [1, 1, 40, 20],
                "editable": True, "sensitive": self.sensitive}
    inspect_focused = inspect_point
    def execute_viewport_action(self, action, metadata):
        self.executed.append((action.action, action.args))
        return {"status": "executed", "dispatched": True}
    def close(self):
        self.closed = True


class Recorder:
    def __init__(self, task, directory, **kwargs):
        self.steps = []
        self.preparations = []
        self.directory = directory
        self.verification = {"status": "unknown"}
    def prepare_observation(self, before):
        result = {"preparation_id": before.info["observation_id"], "policy_input": [{"history_steps": len(self.steps)}],
                  "prompt_metadata": {}}
        self.preparations.append(result)
        return result
    def record_transition(self, action, before, after, **kwargs):
        self.steps.append({"step": f"h_{len(self.steps) + 1}", "action": {"action": action.action, "args": action.args},
                           "metadata": kwargs["metadata"]})
    def summary(self):
        return {"steps": copy.deepcopy(self.steps), "verification": self.verification}
    def stop(self, **kwargs):
        self.stopped_with = kwargs
        return self.verification
    def review_step(self, *args, **kwargs):
        self.review = kwargs
    def export_sft(self):
        path = self.directory / "sft.jsonl"
        path.write_text(json.dumps({"test": True}) + "\n")
        return {"path": str(path), "count": 1}
    def close(self):
        pass


@pytest.fixture
def capture(tmp_path):
    browser = Browser()
    controller = CollectionController(tmp_path, browser_factory=lambda *_: browser, recorder_factory=Recorder)
    state = controller.start({"task": "Search a paper"})
    yield controller, browser, state
    controller.close()


def command(state, **kwargs):
    return {"session_id": state["session_id"], "observation_id": state["observation"]["id"],
            "request_id": "req1", "kind": "click", "x": 10, "y": 10, **kwargs}


def test_duplicate_requests_never_repeat_browser_actions(capture):
    controller, browser, state = capture
    payload = command(state)
    result = controller.action(payload)
    assert len(result["steps"]) == 1
    assert controller.action(payload)["duplicate"]
    assert len(browser.executed) == 1
    with pytest.raises(CollectionError, match="不同操作"):
        controller.action({**payload, "x": 20})


def test_post_observation_prompt_includes_completed_action(capture):
    controller, _, state = capture
    controller.action(command(state))
    assert controller.prepared["policy_input"] == [{"history_steps": 1}]
    assert controller.recorder.steps[0]["metadata"]["frozen_policy_input"] == [{"history_steps": 0}]


def test_stale_pause_and_resume_do_not_dispatch(capture):
    controller, browser, state = capture
    controller.pause(state)
    with pytest.raises(CollectionError):
        controller.action(command(state))
    resumed = controller.resume(state)
    assert resumed["observation"]["id"] != state["observation"]["id"]
    with pytest.raises(CollectionError, match="画面已更新"):
        controller.action(command(state))
    assert not browser.executed and not controller.recorder.steps


def test_state_polling_does_not_change_observation_or_steps(capture):
    controller, browser, state = capture
    assert controller.state() == state
    assert browser.captures == 1
    new = controller.frame(state)
    assert new["observation"]["id"] != state["observation"]["id"]
    assert not new["steps"]


def test_sensitive_input_is_not_dispatched_or_recorded(capture):
    controller, browser, state = capture
    browser.sensitive = True
    with pytest.raises(CollectionError, match="密码"):
        controller.action(command(state, kind="type", text="DO_NOT_RECORD"))
    assert not browser.executed and not controller.recorder.steps
    assert "DO_NOT_RECORD" not in str(controller.state())


def test_stop_does_not_turn_human_claim_into_verified_success(capture):
    controller, browser, state = capture
    result = controller.stop({**state, "claimed_complete": True})
    assert result["status"] == "stopped" and result["verification"]["status"] == "unknown"
    assert browser.closed


def test_failed_final_capture_invalidates_previous_page_evidence(capture):
    controller, browser, state = capture
    def unavailable():
        raise RuntimeError("page unavailable")
    browser.collect_observation = unavailable
    result = controller.stop(state)
    assert result["status"] == "stopped" and browser.closed
    final = controller.recorder.stopped_with["final_observation"]
    assert final.context == []
    assert final.info["observation_error"] == "final_capture_failed"


def test_final_capture_failure_cannot_reuse_a_previously_passed_url(tmp_path):
    browser = Browser()
    controller = CollectionController(tmp_path, browser_factory=lambda *_: browser)
    try:
        state = controller.start({"task": "open https://example.test/"})
        after_click = controller.action(command(state))
        assert after_click["verification"]["status"] == "passed"
        def unavailable():
            raise RuntimeError("page closed before final verification")
        browser.collect_observation = unavailable
        stopped = controller.stop(state)
        assert stopped["verification"]["status"] == "unknown"
        outcome = controller.recorder.summary()["outcome"]
        assert not outcome["independent_verified"] and not outcome["reward_valid"]
    finally:
        controller.close()


def test_api_origin_check_and_browser_free_status(capture):
    controller, browser, state = capture
    app = FastAPI()
    app.include_router(collection_router(controller))
    with TestClient(app) as client:
        assert client.get("/collect/status").json()["status"] == "recording"
        denied = client.post("/collect/action", json=command(state), headers={"Origin": "https://foreign.test"})
        assert denied.status_code == 403 and not browser.executed
        result = client.post("/collect/action", json=command(state), headers={"Origin": "http://testserver"})
        assert result.status_code == 200 and result.json()["ok"]


def test_bad_task_does_not_launch_browser(tmp_path):
    controller = CollectionController(tmp_path, browser_factory=lambda *_: pytest.fail("unexpected launch"))
    with pytest.raises(CollectionError):
        controller.start({"task": ""})
    with pytest.raises(CollectionError):
        controller.start({"task": "task", "start_url": "file:///secret"})


def test_nested_scroll_uses_the_inference_container_action(capture):
    controller, browser, state = capture
    browser.inspect_scroll_target = lambda *_: {
        "status": "inspected", "root_scroll": False, "target_ref": "region_1", "target_label": "Results"}
    browser.route = lambda action, before: {"channel": "browser_dom", "target_ref": action.args["target_ref"]}
    def execute(action, route):
        browser.executed.append(("dom_scroll", dict(action.args), route))
        return {"status": "executed", "dispatched": True}
    browser.execute = execute
    result = controller.action(command(state, kind="scroll", direction="down", amount=2))
    step = result["steps"][0]
    assert browser.executed[0][0] == "dom_scroll"
    assert step["action"]["args"] == {"target": "Results", "target_ref": "region_1", "direction": "down", "amount": 2}
    assert "sft_exclude_reason" not in step["metadata"]


@pytest.mark.parametrize("root_scroll", [True, False])
def test_unmapped_scroll_keeps_raw_input_without_fabricating_page_scroll(capture, root_scroll):
    controller, browser, state = capture
    browser.inspect_scroll_target = lambda *_: {"status": "inspected", "root_scroll": root_scroll}
    result = controller.action(command(state, kind="scroll", direction="up", amount=1))
    assert browser.executed == [("scroll", {"direction": "up", "amount": 1, "x": 10, "y": 10})]
    metadata = result["steps"][0]["metadata"]
    assert metadata.get("sft_exclude_reason") == (None if root_scroll else "nested_scroll_target_not_exposed")


def test_stale_container_does_not_dispatch_scroll(capture):
    controller, browser, state = capture
    browser.inspect_scroll_target = lambda *_: {"status": "inspected", "root_scroll": False, "target_ref": "region_1"}
    browser.route = lambda *_: {"channel": "reobserve"}
    result = controller.action(command(state, kind="scroll", direction="down", amount=1))
    assert result["execution"]["status"] == "rejected"
    assert not browser.executed


@pytest.mark.parametrize("properties", [{"role": "slider"}, {"tag": "canvas"}, {"input_type": "range"}])
def test_position_sensitive_clicks_are_retained_with_training_exclusion(capture, properties):
    controller, browser, state = capture
    browser.inspect_point = lambda *_: {"status": "inspected", "target_label": "Control", **properties}
    result = controller.action(command(state))
    assert len(browser.executed) == 1
    assert result["steps"][0]["metadata"]["sft_exclude_reason"] == "position_sensitive_target"


@pytest.mark.parametrize("has_text, overwrite, excluded", [(False, False, False), (True, False, True), (True, True, False), (None, False, True)])
def test_caret_sensitive_typing_does_not_become_an_ambiguous_label(capture, has_text, overwrite, excluded):
    controller, browser, state = capture
    browser.inspect_focused = lambda *_: {"status": "inspected", "target_label": "Search", "editable": True,
        "bounds": [1, 1, 40, 20], "focused_backend_id": 7, "editable_has_text": has_text}
    def insert(text, binding, **kwargs):
        browser.executed.append((text, kwargs))
        return {"status": "executed", "dispatched": True}
    browser.insert_focused_text = insert
    result = controller.action(command(state, kind="type", text="中文", overwrite=overwrite))
    assert browser.executed == [("中文", {"overwrite": overwrite, "expected_backend_id": 7})]
    assert bool(result["steps"][0]["metadata"].get("sft_exclude_reason")) == excluded
