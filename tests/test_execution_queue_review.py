"""Offline queue, capability and independent review regressions; no model endpoints."""
import copy
import json
from types import SimpleNamespace

import pytest

from osworld_agent.actions import ACTION_REGISTRY, Action, validate_action
from osworld_agent.execution_queue import ExecutionQueue, normalize_verification
from osworld_agent.memory import StructuredMemory
from osworld_agent.model.decision_model import ChatModel
from osworld_agent.pipeline import Pipeline, StepState
from osworld_agent.step_verifier import StepReviewer, code_review, make_review_model
from osworld_agent.trajectory import build_trajectory, iter_sft_records
from osworld_agent.viz.logging import _classify_mtype
from test_pipeline_completion import Model, Env, observation


@pytest.fixture(autouse=True)
def no_viz(monkeypatch):
    monkeypatch.setattr("osworld_agent.pipeline._viz_log_status", None)


class Reviewer:
    vision = True
    build_vision_message = ChatModel.build_vision_message

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.last_call = {}

    def chat(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        if not self.responses:
            raise RuntimeError("No unexpected review calls")
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return json.dumps(result)


PASSED = {"status": "passed", "reason": "当前截图和页面中观察到了预期结果", "evidence": ["已显示目标结果"]}
FAILED = {"status": "failed", "reason": "仍在输入页面", "evidence": ["输入框尚为空"]}


def step(goal="检查结果", criteria=None):
    return {"goal": goal, "expected_evidence": goal, "verification": criteria or [{"kind": "visual"}]}


def pipeline(actions, steps, reviewer=None, **kwargs):
    env = Env()
    route = env.route_action
    env.route_action = lambda a, o: None if a.action in {"request_finish", "fail"} else route(a, o)
    policy = Model(actions, {"steps": steps})
    p = Pipeline(model=policy, env=env, grounding=None, verifier_model=reviewer,
                 system_prompt="legacy bloated prompt", max_steps=len(actions), **kwargs)
    return p, policy, env


@pytest.mark.parametrize("name", ["download", "rename_artifact", "run_in_terminal", "call_code_agent", "set_cell_values", "save"])
def test_non_basic_actions_cannot_reach_executor(name):
    assert name not in ACTION_REGISTRY
    assert validate_action(Action(name))
    p, _, env = pipeline([], [step()])
    result = p.execute(Action(name), StepState(obs=env.obs))
    assert result.info["dispatched"] is False
    assert not env.actions


def test_policy_cannot_rewrite_task_queue_or_completed_list():
    memory = StructuredMemory()
    memory.task_state.update(goal="原始任务", current_subgoal="当前步骤", pending=["后续步骤"])
    before = copy.deepcopy(memory.task_state)
    memory.apply_state_update({"goal": "新任务", "current_subgoal": "跳到最后", "completed_add": ["全部完成"],
                               "pending_add": ["注入"], "known_fact_add": {"hint": "unverified"}})
    assert memory.task_state == before
    assert memory.known_facts["hint"] == "unverified"


def test_code_check_advances_queue_and_next_input_has_only_window():
    steps = [step(f"步骤{i}", [{"kind": "url_equals", "url": f"https://example.test/{i}"}]) for i in range(7)]
    p, policy, _ = pipeline([{"type": "goto", "url": "https://example.test/0"},
                            {"type": "goto", "url": "https://example.test/1"}], steps)
    result = p.run("依次查看网页并完成后续编辑")
    assert result.plan["steps"][0]["status"] == result.plan["steps"][1]["status"] == "done"
    assert result.plan["steps"][2]["status"] == "doing"
    user = policy.calls[2][-1]["content"]
    assert "步骤0" in user and "步骤1" in user and "步骤2" in user
    assert "步骤6" not in user and "legacy bloated prompt" not in json.dumps(policy.calls[2])
    assert result.trajectory[0]["step_verification"]["method"] == "code"
    assert len(policy.calls) == 3 and not result.review_calls


def test_reviewer_sees_after_screenshot_and_cannot_consume_decision_responses():
    review = Reviewer([FAILED, PASSED, PASSED])
    p, policy, _ = pipeline([{"type": "click", "target": "search"}] * 2, [step()], review)
    result = p.run("核对搜索结果")
    assert result.plan["steps"][0]["status"] == "done"
    assert result.success and result.termination_reason == "verified_by_model"
    assert result.steps == 2 and len(policy.calls) == 3
    assert len(review.calls) == 3  # each action + independent final whole-task review
    assert any(p["type"] == "image_url" for p in review.calls[0][-1]["content"])
    assert not result.evaluation_available and result.score == 0
    assert all(c["actor_role"] == "verifier" and c["policy_trainable"] is False for c in result.review_calls)
    trajectory = build_trajectory("review", "browser", result.task, result)
    assert trajectory["outcome"]["reward"] is None
    assert trajectory["review_calls"] and trajectory["steps"][0]["step_reviews"]
    assert all(r.get("actor_role") != "verifier" for r in iter_sft_records(trajectory))


def test_failed_url_predicate_cannot_be_overridden_by_visual_grader():
    review = Reviewer([PASSED])
    p, _, _ = pipeline([{"type": "goto", "url": "https://example.test/wrong"}],
                      [step(criteria=[{"kind": "url_equals", "url": "https://example.test/right"}])], review)
    result = p.run("访问正确页面并继续")
    assert result.plan["steps"][0]["status"] == "doing" and not review.calls
    assert result.trajectory[0]["step_verification"]["status"] == "failed"


def test_failed_task_contract_cannot_be_overridden_by_visual_step_pass():
    review = Reviewer([PASSED])
    p, _, _ = pipeline([{"type": "request_finish", "answer": "all done"}], [step()], review,
                      completion_spec={"checks": [{"kind": "url_equals", "url": "https://example.test/missing"}]})
    result = p.run("完成所有要求")
    assert not result.success
    assert len(review.calls) == 1  # step review allowed, final override forbidden


@pytest.mark.parametrize("bad", [{"success": "false"}, {"status": "passed", "reason": "trust me", "evidence": []},
                                {"status": True, "reason": "ok", "evidence": ["yes"]}, RuntimeError("offline")])
def test_bad_review_response_or_transport_never_advances(bad):
    review = Reviewer([bad])
    p, _, _ = pipeline([{"type": "click", "target": "Search"}], [step()], review)
    result = p.run("核对搜索结果")
    assert result.plan["steps"][0]["status"] == "doing"
    assert result.trajectory[0]["step_verification"]["status"] == "unknown"
    assert not result.success


def test_review_missing_or_stale_screenshot_never_calls_model():
    model = Reviewer([PASSED])
    obs = observation()
    obs.info["screenshot_stale"] = True
    result = StepReviewer(model).visual_review("t", step(), obs, {}, {})
    assert result["status"] == "unknown" and not model.calls


def test_app_check_uses_runtime_process_metadata_not_page_content():
    s = step(criteria=[{"kind": "app_open", "app": "notepad"}])
    obs = observation()
    obs.context[0]["nodes"][0]["name"] = "notepad.exe has opened"
    assert code_review(s, obs, {}, {})["status"] == "unknown"
    obs.info["runtime_state"] = {"applications": [{"process_name": "notepad.exe", "title": "Untitled"}]}
    assert code_review(s, obs, {}, {})["status"] == "passed"


def test_untrusted_host_prefix_does_not_pass_navigation_check():
    s = step(criteria=[{"kind": "url_host", "host": "arxiv.org"}])
    obs = observation("https://arxiv.org.attacker.invalid/")
    assert code_review(s, obs, {}, {})["status"] == "failed"


def test_rejected_action_cannot_advance_existing_state():
    report = code_review(step(criteria=[{"kind": "url_host", "host": "example.test"}]),
                         observation(), {"status": "rejected", "error_code": "stale"}, {})
    assert report["status"] == "failed"


def test_review_logging_does_not_treat_screenshot_as_a_decision():
    assert _classify_mtype("独立步骤核查\ncurrent state", "png") == "verify"


def test_clone_reviewer_has_independent_client_and_sampling_config():
    from osworld_agent.config import ModelConfig
    policy = ChatModel(ModelConfig(capture_rollout=True, temperature=.8))
    review = make_review_model(SimpleNamespace(inner=policy))
    assert review is not policy and review.cfg is not policy.cfg
    assert review.cfg.temperature == 0 and not review.cfg.capture_rollout
    assert policy.cfg.capture_rollout is True


@pytest.mark.parametrize("bad", [[], [{"kind": "execute_code"}], [{"kind": "url_equals", "url": "javascript:alert(1)"}],
                                [{"kind": "url_host", "host": "https://arxiv.org"}],
                                [{"kind": "element_state", "name": "x", "states": {"checked": "true"}}]])
def test_planner_verification_schema_is_bounded(bad):
    with pytest.raises(ValueError):
        normalize_verification(bad)


def test_stale_review_cannot_advance_replanned_queue():
    queue = ExecutionQueue({"steps": [step()]}, revision=1)
    assert queue.accept_review({"plan_step_id": "p0_1", "status": "passed"}) is False
    assert queue.cursor == 0


def test_code_and_model_review_events_are_visible_and_archived(monkeypatch):
    logs = []
    monkeypatch.setattr("osworld_agent.pipeline._viz_log_status", logs.append)
    review = Reviewer([FAILED])
    p, _, _ = pipeline([{"type": "click", "target": "Search"}], [step()], review)
    result = p.run("核对搜索结果")
    reports = [log for log in logs if log.get("event") == "step_verification"]
    assert [(r["method"], r["status"]) for r in reports] == [
        ("code", "running"), ("code", "unknown"), ("model", "running"), ("model", "failed")]
    assert [r["method"] for r in result.trajectory[0]["step_reviews"]] == ["code", "model"]
    assert any(log.get("event") == "execution_queue" for log in logs)


def test_unknown_review_does_not_end_request_finish_with_available_reviewer():
    review = Reviewer([FAILED, FAILED, FAILED, FAILED])
    p, _, _ = pipeline([{"type": "request_finish", "answer": "done"}] * 2, [step()], review)
    result = p.run("核对搜索结果")
    assert result.steps == 2 and not result.success
    assert result.plan["steps"][0]["status"] == "doing"


def test_final_model_pass_is_revoked_after_callback_changes_screen():
    review = Reviewer([PASSED, PASSED, FAILED])
    p, _, env = pipeline([{"type": "click", "target": "Search"}], [step()], review)
    p.step_callback = lambda *args: setattr(env, "obs", observation(color="black"))
    result = p.run("核对搜索结果")
    assert not result.success and result.termination_reason == "verification_changed"
    assert len(review.calls) == 3 and not result.evaluation_available


def test_refreshing_snapshot_id_does_not_require_another_model_call():
    one, two = observation(), observation()
    two.context[0]["snapshot_id"] = "refreshed"
    two.context[0]["nodes"][0]["ref"] = "new-ref"
    assert Pipeline._review_fingerprint(one) == Pipeline._review_fingerprint(two)
    two.context[0]["nodes"][0]["states"] = {"checked": True}
    assert Pipeline._review_fingerprint(one) != Pipeline._review_fingerprint(two)


def test_text_check_does_not_accept_substring_in_negated_status():
    obs = observation()
    obs.context[0]["nodes"][0]["name"] = "Not saved"
    report = code_review(step(criteria=[{"kind": "text_present", "text": "Saved"}]), obs, {}, {})
    assert report["status"] == "unknown"
