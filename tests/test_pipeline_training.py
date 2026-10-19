"""End-to-end planning, evidence, privacy and durable collection without endpoints."""
import json

import pytest

from osworld_agent.config import ContextConfig
from osworld_agent.context import PromptCompiler, ContextBudgetError
from osworld_agent.env import StepResult
from osworld_agent.pipeline import Pipeline
from osworld_agent.trajectory import build_trajectory, iter_sft_records
from osworld_agent.training_events import read_events
from test_pipeline_completion import Model, Env, observation
from test_paper_completion import pdf_file, event


@pytest.fixture(autouse=True)
def no_viz_calls(monkeypatch):
    monkeypatch.setattr("osworld_agent.pipeline._viz_log_status", None)


def paper_plan(reverse=False):
    phases = ["search", "identify", "locate_fulltext", "download", "verify"]
    if reverse:
        phases.reverse()
    return {"steps": [{"phase": phase, "goal": f"{phase} AAA", "expected_evidence": f"{phase}证据"}
                      for phase in phases], "keywords": ["AAA"]}


def test_title_only_paper_task_rejects_download_before_identity_binding(tmp_path):
    path = pdf_file(tmp_path / "download.bin", "AAA", "Alice Smith", "Abstract", "Paper content")
    env = Env()
    env.execute_routed = lambda a, r: StepResult(env.obs, info={"status": "executed", "action": "click",
                                                              "artifacts": [event(path)]})
    model = Model([{"type": "click", "target": "PDF button"}], paper_plan())
    result = Pipeline(model=model, env=env, grounding=None, system_prompt="test",
                      event_path=str(tmp_path / "events.jsonl")).run("请下载一篇叫做AAA的论文")
    assert not result.success
    assert result.trajectory[0]['routing']['error_code'] == 'paper_identity_mismatch'
    assert not result.verification['checks'][0].get('matched_title')
    calls = [e["call"] for e in read_events(tmp_path / "events.jsonl") if e["event_type"] == "model_call"]
    assert [c["actor_role"] for c in calls][:2] == ["planner", "decision"]
    assert all(c["policy_trainable"] for c in calls)
    trajectory = build_trajectory("paper-AAA", "browser", result.task, result)
    assert len(list(iter_sft_records(trajectory, actor_role="planner"))) == 0
    assert len(list(iter_sft_records(trajectory))) == 0
    assert read_events(tmp_path / "events.jsonl")[-1]["event_type"] == "episode_end"


def test_private_target_never_enters_any_policy_prompt(tmp_path):
    secret = "https://private-answer.invalid/secret-url"
    env = Env()
    env.route_action = lambda a, o: None
    model = Model([{"type": "request_finish", "answer": "完成"}] * 2)
    private = {"checks": [{"id": "private-secret-id", "kind": "url_equals", "url": secret}]}
    result = Pipeline(model=model, env=env, grounding=None, system_prompt="test", max_steps=2,
                      private_completion_spec=private).run("找到用户要求的论文")
    policy = json.dumps(model.calls, ensure_ascii=False)
    assert secret not in policy and "private-secret-id" not in policy
    assert not result.success
    assert result.private_evaluation["spec"] == private


def test_teacher_planner_is_separate_and_frozen():
    teacher = Model([])
    student = Model([{"type": "goto", "url": "https://example.test/target"}])
    student.responses.pop(0)
    result = Pipeline(model=student, planner_model=teacher, env=Env(), grounding=None,
                      system_prompt="test").run("打开 https://example.test/target")
    assert result.success and len(teacher.calls) == len(student.calls) == 1
    assert result.planning_calls[0]["policy_trainable"] is False
    assert result.trajectory[0]["decision_calls"][0]["policy_trainable"] is True


def test_reverse_paper_workflow_is_rejected_before_environment_operations():
    model = Model([], paper_plan(reverse=True))
    model.responses *= 2
    env = Env()
    result = Pipeline(model=model, env=env, grounding=None, system_prompt="test").run("请下载一篇叫做AAA的论文")
    assert result.termination_reason == "planning_failed"
    assert env.reset_count == 0 and not env.actions


def test_latest_image_is_preserved_even_when_image_limits_are_zero():
    compiler = PromptCompiler(ContextConfig(max_images=0, structured_max_images=0))
    _, images, metadata = compiler.compile(system="test", task="t", memory="", auxiliary="", required="",
        blocks=[{"kind": "browser_ax", "nodes": [{"ref": "1", "role": "button", "name": "PDF"}]}],
        images=[("current", observation().screenshot)], require_image=True)
    assert len(images) == metadata["image_count"] == 1


def test_required_image_is_not_dropped_to_fit_context():
    compiler = PromptCompiler(ContextConfig(max_input_tokens=5000))
    with pytest.raises(ContextBudgetError):
        compiler.compile(system="S" * 2600, task="t", memory="", auxiliary="", blocks=[],
                         images=[("current", observation().screenshot)], require_image=True)


def test_exception_preserves_prior_events(tmp_path):
    model = Model([])
    def broken(*args):
        raise RuntimeError("invalid image transport")
    model.build_vision_message = broken
    with pytest.raises(RuntimeError):
        Pipeline(model=model, env=Env(), grounding=None, system_prompt="test",
                 event_path=str(tmp_path / "events.jsonl")).run("任务")
    events = read_events(tmp_path / "events.jsonl")
    assert any(e["event_type"] == "model_call" for e in events)
    assert events[-1]["event_type"] == "episode_error"


def test_private_contract_cannot_be_overridden_by_environment_success():
    env = Env()
    env.route_action = lambda a, o: None
    env.evaluation_available = True
    env.evaluate = lambda: 1.0
    model = Model([{"type": "request_finish", "answer": "完成"}])
    result = Pipeline(model=model, env=env, grounding=None, system_prompt="test",
        private_completion_spec={"checks": [{"kind": "url_equals", "url": "https://example.test/other"}]}).run("任务")
    assert not result.success and result.score == 0.0
    assert result.verification["status"] == "failed"


def test_satisfied_goal_does_not_become_negative_reward_on_bad_last_decision():
    model = Model([])
    model.responses.append("invalid decision")
    result = Pipeline(model=model, env=Env(), grounding=None, system_prompt="test", max_retries=1,
                      max_steps=1).run("打开 https://example.test/")
    assert result.success and result.evaluation_available and result.score == 1.0
    assert result.trajectory[-1]["valid_action"] is False


def test_replanning_is_bounded_and_keeps_same_environment():
    class ReplanningModel(Model):
        def chat(self, messages, **kwargs):
            if "steps" in messages[0]["content"] and messages[0].get("role") == "user":
                self.calls.append(messages)
                repair = "当前计划受阻" in messages[0]["content"]
                return json.dumps({"steps": [{"goal": "先检查阻碍，再进入详情继续任务" if repair else "换路径继续任务", "expected_evidence": "目标状态改变"}],
                                   "strategy_change": "根据失败证据调整入口与观察方式，核查新的页面状态"})
            self.calls.append(messages)
            return json.dumps({"next_action": {"type": "click", "target": "Unresponsive button"},
                               "expected_effect": "状态更新", "state_update": {}})

    env = Env()
    model = ReplanningModel([])
    result = Pipeline(model=model, env=env, grounding=None, system_prompt="test",
                      max_steps=12, early_stop_repeat=2, max_replans=1).run("执行任务")
    assert result.termination_reason == "stalled"
    assert [c["plan_stage"] for c in result.planning_calls] == ["initial", "replan"]
    assert env.reset_count == 1
    assert result.steps < 12 and not result.evaluation_available
    assert result.run_manifest["runtime_config"]["max_replans"] == 1
