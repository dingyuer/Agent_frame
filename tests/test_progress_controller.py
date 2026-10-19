"""Semantic progress accounting stays independent of queue advancement."""
import copy

import pytest

from osworld_agent.actions import Action
from osworld_agent.env import Observation
from osworld_agent.memory import StructuredMemory
from osworld_agent.pipeline_types import Decision, Execution, StepState
from osworld_agent.progress_controller import ProgressController, ProgressPorts


def observation(ref="old:1", *, document=10, name="Download PDF", value=""):
    url = "https://example.test/paper"
    return Observation(screenshot=b"same", info={"page_id": "page", "document_id": document, "url": url},
        context=[{"kind": "browser_ax", "url": url, "nodes": [
            {"ref": ref, "role": "button", "name": name, "value": value,
             "href": "https://example.test/paper.pdf"}]}])


def controller(*, difference=0.0, repeats=lambda: 3):
    def page(obs):
        return {} if obs is None else {key: obs.info.get(key) for key in ("page_id", "document_id", "url")}

    return ProgressController(ProgressPorts(
        observation_page=page,
        download_id=lambda item: str(item.get("download_id") or item.get("artifact_id") or item.get("id") or ""),
        screen_diff=lambda before, after: difference,
        early_stop_repeat=repeats,
    ))


def rejection(obs, code="stale"):
    return Execution(kind="browser_dom", obs=obs,
        info={"status": "rejected", "dispatched": False, "error_code": code})


def test_same_semantic_target_accumulates_failure_across_refs_and_spelling():
    progress = controller()
    before = observation()
    state = StepState(obs=before, memory=StructuredMemory())
    first = Decision(action=Action("click", {"target_ref": "old:1", "target": "Download PDF"}))
    assert progress._detect_stall(state, first, rejection(before), before.screenshot) is None
    fresh = observation("fresh:99")
    state.obs = fresh
    second = Decision(action=Action("click", {"target_ref": "fresh:99", "target": "DOWNLOAD 'PDF'"}))
    reason = progress._detect_stall(state, second, rejection(fresh), fresh.screenshot)
    assert "stale 已发生 2 次" in reason
    assert len(state.failure_counts) == 1
    assert state.failure_count == state.memory.stall_count == 2


def test_semantic_alias_survives_target_disappearing_from_fresh_snapshot():
    progress = controller()
    before = observation()
    state = StepState(obs=before, memory=StructuredMemory())
    action = Action("click", {"target_ref": "old:1", "target": "Download PDF"})
    key = progress._semantic_action_key(action, before, state)
    after = copy.deepcopy(before)
    after.context[0]["nodes"] = []
    assert progress._semantic_action_key(action, after, state) == key


@pytest.mark.parametrize("change", ["navigation", "input_value", "artifact", "verified_effect", "visual"])
def test_real_progress_clears_stall_without_resetting_lifetime_failure_count(change):
    progress = controller(difference=3.0 if change == "visual" else 0.0)
    before = observation()
    state = StepState(obs=before, memory=StructuredMemory())
    decision = Decision(action=Action("click", {"target_ref": "old:1", "target": "Download PDF"}))
    progress._detect_stall(state, decision, rejection(before), before.screenshot)
    state.memory.forbidden_repeats.append("old-failed-strategy")
    state.shell_outputs.append("old output")
    state.no_change_count = 3
    after = copy.deepcopy(before)
    info = {"status": "executed"}
    if change == "navigation":
        after.info["document_id"] = 11
    elif change == "input_value":
        after.context[0]["nodes"][0]["value"] = "new query"
    elif change == "artifact":
        info["artifacts"] = [{"download_id": "new-file", "completed": True}]
    elif change == "verified_effect":
        info["effect_verified"] = True
    assert progress._detect_stall(state, decision, Execution(kind="browser_dom", obs=after, info=info),
                                  before.screenshot, before) is None
    assert state.failure_counts == {}
    assert state.memory.stall_count == state.no_change_count == 0
    assert state.action_keys == state.shell_outputs == state.memory.forbidden_repeats == []
    assert state.semantic_targets == {}
    assert state.failure_class == state.failure_key == ""
    assert state.failure_count == 1


@pytest.mark.parametrize("info", [
    {"status": "executed", "pending_downloads": [{"download_id": "running"}]},
    {"status": "in_progress"},
    {"status": "pending"},
])
def test_unfinished_download_does_not_increment_failure_or_call_fact_ports(info):
    def unexpected(*args):
        raise AssertionError("pending operation must return before observation comparisons")

    progress = ProgressController(ProgressPorts(unexpected, unexpected, unexpected, unexpected))
    before = observation()
    state = StepState(obs=before, memory=StructuredMemory())
    state.failure_counts["existing"] = 1
    state.failure_count = state.memory.stall_count = 1
    decision = Decision(action=Action("click", {"target_ref": "old:1"}))
    assert progress._detect_stall(state, decision, Execution(kind="browser_dom", obs=before, info=info),
                                  before.screenshot) is None
    assert state.failure_counts == {"existing": 1}
    assert state.failure_count == state.memory.stall_count == 1
    assert state.action_keys == []


def test_download_timeout_is_counted_even_when_pending_metadata_remains():
    progress = controller()
    before = observation()
    state = StepState(obs=before, memory=StructuredMemory())
    decision = Decision(action=Action("click", {"target_ref": "old:1"}))
    execution = Execution(kind="browser_dom", obs=before, info={"status": "pending",
        "pending_downloads": [{"download_id": "running"}], "wait_error_code": "download_timeout"})
    assert progress._detect_stall(state, decision, execution, before.screenshot) is None
    assert "download_timeout 已发生 2 次" in progress._detect_stall(state, decision, execution, before.screenshot)


def test_no_progress_threshold_uses_live_setting_and_history_remains_bounded():
    setting = [4]
    progress = controller(repeats=lambda: setting[0])
    before = observation()
    state = StepState(obs=before, memory=StructuredMemory())
    decision = Decision(action=Action("click", {"target_ref": "old:1"}))
    execution = Execution(kind="browser_dom", obs=before, info={"status": "executed"})
    for _ in range(2):
        assert progress._detect_stall(state, decision, execution, before.screenshot) is None
    setting[0] = 3
    assert "no_progress 已发生 3 次" in progress._detect_stall(state, decision, execution, before.screenshot)
    for _ in range(5):
        progress._detect_stall(state, decision, execution, before.screenshot)
    assert len(state.action_keys) == 6


def test_ref_and_snapshot_changes_alone_do_not_count_as_real_progress():
    progress = controller(repeats=lambda: 1)
    before = observation()
    after = observation("fresh:2")
    state = StepState(obs=before, memory=StructuredMemory())
    decision = Decision(action=Action("click", {"target_ref": "old:1"}))
    reason = progress._detect_stall(state, decision,
        Execution(kind="browser_dom", obs=after, info={"status": "executed"}), before.screenshot, before)
    assert "no_progress 已发生 1 次" in reason
