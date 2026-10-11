"""Exercise complete offline episodes; no model endpoint or browser is contacted."""
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from osworld_agent.actions import Action
from osworld_agent.env import Observation, StepResult
from osworld_agent.model.grounding import GroundingResult
from osworld_agent.pipeline import Pipeline, StepState
from osworld_agent.memory import StructuredMemory
from osworld_agent.trajectory import build_trajectory, iter_sft_records


def observation(url="https://example.test/", value="", color="white"):
    return Observation(screenshot=Image.new("RGB", (100, 80), color),
        context=[{"kind": "browser_ax", "snapshot_id": value or "initial", "url": url,
                  "nodes": [{"ref": "1", "role": "textbox", "name": "Email", "value": value}]}],
        info={"screenshot_space": "css_viewport", "screenshot_size": [100, 80], "page_id": "task-page"})


class Model:
    vision = False

    def __init__(self, actions, plan=None):
        self.responses = [json.dumps(plan or {"steps": [{"goal": "执行并检查任务", "expected_evidence": "观察目标状态"}]})]
        self.responses += [json.dumps({"next_action": a, "state_update": {},
                                       "expected_effect": "目标状态更新"}) for a in actions]
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        assert self.responses, "Unexpected model call (including model-based verification)"
        return self.responses.pop(0)

    def build_vision_message(self, text, images):
        return {"role": "user", "content": text}


class Env:
    evaluation_available = False

    def __init__(self, tmp_path=None):
        self.obs = observation()
        self.reset_count = 0
        self.observe_count = 0
        self.actions = []
        self.tmp_path = tmp_path

    def reset(self, task):
        self.reset_count += 1
        return self.obs

    def observe(self):
        self.observe_count += 1
        return self.obs

    def route_action(self, action, obs):
        return {"channel": "browser_dom"}

    def execute_routed(self, action, route):
        self.actions.append(action)
        info = {"status": "executed", "dispatched": True, "action": action.action}
        if action.action == "goto":
            self.obs = observation(action.args["url"])
            info["url"] = action.args["url"]
        elif action.action == "download":
            filename = action.args["url"].rsplit("/", 1)[-1]
            path = self.tmp_path / filename
            path.write_bytes(b"%PDF-test bytes")
            info.update(url=action.args["url"], path=str(path), verified=True, bytes=path.stat().st_size)
        return StepResult(self.obs, info=info)

    def step(self, action):
        self.actions.append(action)
        return StepResult(self.obs, info={"status": "executed", "dispatched": True})


def pipeline(model, env, spec=None, steps=6, grounding=None):
    # Terminal actions must remain in Pipeline, not be routed by this test double.
    route = env.route_action
    env.route_action = lambda a, o: None if a.action in {"request_finish", "fail"} else route(a, o)
    return Pipeline(model=model, env=env, grounding=grounding, system_prompt="test",
                    completion_spec=spec, max_steps=steps)


@pytest.fixture(autouse=True)
def no_viz_calls(monkeypatch):
    monkeypatch.setattr("osworld_agent.pipeline._viz_log_status", None)


def test_navigation_verifies_exact_actual_url_and_exports_evidence():
    model = Model([{"type": "goto", "url": "https://example.test/target"}])
    p = pipeline(model, Env())
    result = p.run("打开 https://example.test/target")
    assert result.success and result.steps == 1
    assert result.termination_reason == "verified"
    assert result.verification["status"] == "passed"
    assert result.evaluation_available and result.reward_source == "completion_contract"
    assert len(model.calls) == 2  # mandatory planner + one decision, no LLM verifier
    traj = build_trajectory("id", "browser", result.task, result)
    assert traj["outcome"]["verification"]["status"] == "passed"
    assert len(list(iter_sft_records(traj))) == 1


def test_compound_task_does_not_end_after_navigation_or_model_claim():
    model = Model([{"type": "goto", "url": "https://example.test/"},
                   {"type": "request_finish", "answer": "全部完成 CONFIRM"}])
    env = Env()
    result = pipeline(model, env).run("打开网站并填写表单后提交")
    assert result.steps == 2 and not result.success
    assert result.termination_reason == "completion_unverifiable"
    assert result.verification["status"] == "unknown"
    assert not result.evaluation_available
    assert result.trajectory[-1]["assessment"]["action_status"] == "uncertain"
    assert env.observe_count == 1
    assert len(model.calls) == 3


def test_two_downloads_require_two_real_artifacts(tmp_path):
    urls = ["https://example.test/a.pdf", "https://example.test/b.pdf"]
    spec = {"checks": [{"id": name, "kind": "download", "url": url}
                       for name, url in zip(["a", "b"], urls)]}
    model = Model([{"type": "download", "url": u} for u in urls])
    result = pipeline(model, Env(tmp_path), spec).run("下载两篇论文")
    assert result.success and result.steps == 2
    assert result.trajectory[0]["verification"]["status"] != "passed"
    assert all(c["status"] == "passed" for c in result.verification["checks"])


def test_finish_refreshes_actual_state_instead_of_using_previous_success():
    env = Env()
    env.obs = observation(value="correct")
    env.observe = lambda: observation(value="wrong")
    spec = {"checks": [{"kind": "element_state", "role": "textbox", "name": "Email", "value": "correct"}]}
    model = Model([{"type": "request_finish", "answer": "已填写"}])
    result = pipeline(model, env, spec, steps=1).run("填写邮箱")
    assert not result.success and result.verification["status"] == "failed"
    assert len(model.calls) == 2


def test_invalid_planner_retries_before_any_environment_operation():
    model = Model([])
    model.responses = ["not json", '{"steps": []}']
    env = Env()
    result = pipeline(model, env).run("某个任务")
    assert result.termination_reason == "planning_failed"
    assert not result.success and result.steps == 0
    assert env.reset_count == 0 and env.actions == []
    assert len(model.calls) == 2


def test_planner_repair_and_proposed_conditions_never_authorize_success():
    model = Model([])
    model.responses = ["bad", json.dumps({"steps": [{"goal": "做事", "expected_evidence": "出现目标状态"}],
        "completion_spec": {"checks": [{"kind": "url_equals", "url": "https://example.test/"}]}}),
        '{"next_action":{"type":"request_finish","answer":"完成"}}']
    result = pipeline(model, Env()).run("进行复杂任务")
    assert not result.success and result.termination_reason == "completion_unverifiable"
    assert result.plan["completion_spec"] == {"checks": []}


def test_reobserve_refreshes_without_resetting_task():
    env = Env()
    env.obs = observation(value="fresh")
    p = pipeline(Model([]), env)
    state = StepState(obs=observation(value="old"), memory=StructuredMemory(),
                      route={"channel": "reobserve", "reason": "stale"})
    execution = p.execute(Action("click", {"target": "Search"}), state)
    assert execution.obs is env.obs
    assert env.observe_count == 1 and env.reset_count == 0


@pytest.mark.parametrize("code,dispatched,expected", [
    ("not_found", False, 1), ("unsupported", False, 1),
    ("blocked", False, 0), ("stale", False, 0), ("ambiguous", False, 0),
    ("not_found", True, 0)])
def test_visual_recovery_is_bounded_and_uses_fresh_screenshot(code, dispatched, expected):
    env = Env()
    old, fresh = observation(color="red"), observation(color="blue")
    env.obs = fresh
    env.execute_routed = lambda a, r: StepResult(old, info={"status": "rejected",
        "dispatched": dispatched, "error_code": code})
    shots = []
    def locate(shot, target):
        shots.append(shot)
        return GroundingResult(10, 20)
    p = pipeline(Model([]), env, grounding=SimpleNamespace(locate=locate))
    state = StepState(obs=old, memory=StructuredMemory(), route={"channel": "browser_dom"})
    p.execute(Action("click", {"target": "Search"}), state)
    assert len(shots) == expected and len(env.actions) == expected
    if expected:
        assert shots[0] is fresh.screenshot
        assert env.actions[0].args["coordinate_space"] == "css_viewport"
    assert env.reset_count == 0


def test_environment_evaluator_is_authoritative_and_called_once():
    env = Env()
    env.evaluation_available = True
    evaluated = []
    def evaluate():
        evaluated.append(True)
        return 0.0
    env.evaluate = evaluate
    result = pipeline(Model([{"type": "request_finish", "answer": "任务完成"}]), env).run("编辑文档")
    assert not result.success and result.score == 0
    assert result.evaluation_available and result.reward_source == "environment_evaluator"
    assert len(evaluated) == 1


def test_state_update_cannot_rewrite_task_or_contract():
    memory = StructuredMemory()
    memory.task_state["goal"] = "original"
    memory.apply_state_update({"goal": "easier task", "completion_spec": {"checks": []}})
    assert memory.task_state["goal"] == "original"


def test_final_artifact_recheck_revokes_earlier_success(tmp_path):
    url = "https://example.test/a.pdf"
    p = pipeline(Model([{"type": "download", "url": url}]), Env(tmp_path))
    p.step_callback = lambda *args: (tmp_path / "a.pdf").unlink()
    result = p.run("下载 " + url)
    assert not result.success and result.score == 0
    assert result.verification["status"] == "failed"
    assert result.termination_reason == "verification_changed"
    assert result.trajectory[-1]["verification"]["status"] == "failed"
    assert result.trajectory[-1]["truncated"]


def test_evaluator_error_revokes_contract_success():
    env = Env()
    env.evaluation_available = True
    def broken():
        raise RuntimeError("evaluator offline")
    env.evaluate = broken
    result = pipeline(Model([{"type": "goto", "url": "https://example.test/"}]), env).run("打开 https://example.test/")
    assert not result.success and not result.evaluation_available
    assert result.verification["status"] == "unknown"
    assert result.termination_reason == "evaluation_error"
    assert result.trajectory[0]["truncated"]
    assert result.trajectory[0]["assessment"]["action_status"] == "uncertain"


def test_evaluator_failure_revokes_contract_success_and_answer():
    env = Env()
    env.evaluation_available = True
    env.evaluate = lambda: 0.0
    result = pipeline(Model([{"type": "goto", "url": "https://example.test/"}]), env).run("打开 https://example.test/")
    assert not result.success and result.evaluation_available
    assert result.verification["status"] == "failed"
    assert result.termination_reason == "evaluation_failed"
    assert "未通过" in result.final_answer
    assert result.trajectory[-1]["terminated"]
    assert not result.trajectory[-1]["truncated"]


@pytest.mark.parametrize("score,reason", [(1, "verified_by_environment"), (0, "evaluation_failed")])
def test_evaluator_finalizes_transition_flags(score, reason):
    env = Env()
    env.evaluation_available = True
    env.evaluate = lambda: score
    result = pipeline(Model([{"type": "request_finish", "answer": "完成"}]), env).run("编辑文档")
    assert result.success is bool(score)
    assert result.termination_reason == reason
    assert result.trajectory[-1]["terminated"]
    assert not result.trajectory[-1]["truncated"]


def test_full_plan_and_exact_criteria_survive_memory_pressure():
    checks = [{"id": f"check_{i}", "kind": "element_state", "role": "textbox",
               "name": f"字段{i}", "value": f"precise_value_{i}"} for i in range(12)]
    plan = {"steps": [{"goal": f"步骤{i}：找到表单中的指定字段并按照任务要求填写，然后检查显示的值",
                       "expected_evidence": f"字段{i}的值更新", "check_ids": [f"check_{i}"]}
                      for i in range(12)]}
    model = Model([{"type": "request_finish", "answer": "完成"}], plan)
    pipeline(model, Env(), {"checks": checks}, steps=1).run("填写全部字段")
    decision_prompt = model.calls[1][-1]["content"]
    for i in range(12):
        assert f"步骤{i}：" in decision_prompt
        assert f"precise_value_{i}" in decision_prompt


def test_browser_recovery_never_replays_target_on_desktop():
    env = Env()
    env.obs = observation()
    env.obs.info = {"screenshot_space": "screen_pixels"}
    env.execute_routed = lambda a, r: StepResult(env.obs, info={"status": "rejected",
        "dispatched": False, "error_code": "not_found"})
    p = pipeline(Model([]), env, grounding=SimpleNamespace(locate=lambda *a: pytest.fail("must replan")))
    state = StepState(obs=observation(), memory=StructuredMemory(), route={"channel": "browser_dom"})
    result = p.execute(Action("click", {"target": "Submit"}), state)
    assert result.obs is env.obs and not env.actions
