"""Framework recovery and asynchronous settlement without a model endpoint."""
import copy
import json
from types import SimpleNamespace

import pytest

from osworld_agent.actions import Action
from osworld_agent.env import StepResult
from osworld_agent.memory import StructuredMemory
from osworld_agent.pipeline import AgentResult, Decision, Execution, Pipeline, StepState, WaitConfig
from osworld_agent.verifiers import CompletionVerifier
from test_pipeline_completion import Env, Model, observation


@pytest.fixture(autouse=True)
def no_viz(monkeypatch):
    monkeypatch.setattr("osworld_agent.pipeline._viz_log_status", None)


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = 100.0
        sleeps = []

        def sleep(self, duration):
            assert 0 < duration <= 0.25
            self.sleeps.append(duration)
            self.now += duration

    value = Clock()
    monkeypatch.setattr("osworld_agent.pipeline.time.monotonic", lambda: value.now)
    monkeypatch.setattr("osworld_agent.pipeline.time.sleep", value.sleep)
    return value


def make_pipeline(env=None, **kwargs):
    return Pipeline(model=Model([]), env=env or Env(), grounding=None, system_prompt="test", **kwargs)


def state_and_decision(ref="snapshot:1", target="Download PDF", page="task-page", doc=10):
    obs = observation()
    obs.info.update(page_id=page, document_id=doc)
    obs.context[0]["nodes"] = [{"ref": ref, "role": "button", "name": "Download PDF"}]
    state = StepState(obs=obs, memory=StructuredMemory())
    return state, Decision(action=Action("click", {"target": target, "target_ref": ref}))


def rejection(obs, code="stale"):
    return Execution(kind="browser_dom", obs=obs, info={"status": "rejected", "dispatched": False,
                     "error_code": code, "reason": "AX observation expired"})


def test_two_real_failures_trigger_once_despite_changed_refs_and_target_spelling():
    pipe = make_pipeline(early_stop_repeat=3)
    state, first = state_and_decision()
    assert pipe._compute_control_state(state, first, rejection(state.obs), state.obs.screenshot) == "RECOVER"
    state.obs.context[0]["nodes"][0]["ref"] = "fresh:99"
    second = Decision(action=Action("click", {"target": "DOWNLOAD 'PDF'", "target_ref": "fresh:99"}))
    assert pipe._compute_control_state(state, second, rejection(state.obs), state.obs.screenshot) == "STALL"
    assert state.memory.stall_count == state.failure_count == 2
    assert len(state.failure_counts) == 1


def test_failure_class_and_navigation_identity_are_separate_scopes():
    pipe = make_pipeline()
    state, decision = state_and_decision()
    for code in ("stale", "blocked"):
        assert pipe._compute_control_state(state, decision, rejection(state.obs, code), state.obs.screenshot) == "RECOVER"
    state.obs.info["document_id"] = 11
    assert pipe._compute_control_state(state, decision, rejection(state.obs), state.obs.screenshot) == "RECOVER"
    assert state.memory.stall_count == 1
    assert len(state.failure_counts) == 3


def test_real_progress_clears_failure_counts_and_forbidden_strategies():
    pipe = make_pipeline()
    state, decision = state_and_decision()
    pipe._compute_control_state(state, decision, rejection(state.obs), state.obs.screenshot)
    pipe._compute_control_state(state, decision, rejection(state.obs), state.obs.screenshot)
    after = copy.deepcopy(state.obs)
    after.context[0]["nodes"][0]["name"] = "Download complete"
    execution = Execution(kind="browser_dom", obs=after, info={"status": "executed", "dispatched": True})
    pipe._compute_control_state(state, decision, execution, state.obs.screenshot)
    assert state.memory.stall_count == 0
    assert state.failure_counts == {} and state.memory.forbidden_repeats == []


def test_old_ref_cannot_escape_failure_scope_when_duplicate_labels_appear():
    pipe = make_pipeline()
    state, decision = state_and_decision()
    pipe._compute_control_state(state, decision, rejection(state.obs), state.obs.screenshot)
    state.obs.context[0]["nodes"] = [{"ref": f"new:{i}", "role": "button", "name": "Download PDF"} for i in (1, 2)]
    assert pipe._compute_control_state(state, decision, rejection(state.obs), state.obs.screenshot) == "STALL"


def test_reobserve_exposes_exact_rejection_reason_and_dispatch_state():
    pipe = make_pipeline()
    state, decision = state_and_decision()
    state.route = {"channel": "reobserve", "error_code": "stale", "reason": "ref belongs to previous snapshot"}
    execution = pipe.execute(decision.action, state)
    pipe._update_program_info(decision.action, execution, state)
    assert state.execution_status == "rejected"
    assert state.execution_feedback["dispatched"] is False
    assert state.execution_feedback["reason"] in pipe._brief_outcome(execution)
    assert "previous snapshot" in pipe._program_info_text(state)


def test_replan_occurs_on_second_rejection_and_never_on_last_slot():
    class ReplanModel(Model):
        def chat(self, messages, **kwargs):
            self.calls.append(messages)
            if messages[0]["role"] == "user":
                repair = "当前计划受阻" in messages[0]["content"]
                return json.dumps({"steps": [{"goal": "先重新观察当前可操作元素再走详情入口" if repair else "换一个路径", "expected_evidence": "页面改变"}],
                                   "strategy_change": "根据失败证据调整入口与观察方式，核查新的页面状态"})
            return json.dumps({"next_action": {"type": "click", "target": "Download PDF",
                                                 "target_ref": f"snapshot-{len(self.calls)}:1"}})

    def run(steps):
        env = Env()
        env.route_action = lambda action, obs: {"channel": "reobserve", "reason": "expired ref"}
        model = ReplanModel([])
        return Pipeline(model=model, env=env, grounding=None, system_prompt="test", max_steps=steps,
                        max_replans=1).run("任务")

    result = run(8)
    assert result.steps == 4 and result.termination_reason == "stalled"
    assert [c["plan_stage"] for c in result.planning_calls] == ["initial", "replan"]
    recovery = [e for e in result.events if e["event_type"] == "framework_recovery"]
    assert recovery[0]["parent_decision_id"] == "a_2"
    assert all(e["policy_trainable"] is False for e in recovery)
    last = run(2)
    assert last.termination_reason == "max_steps"
    assert [c["plan_stage"] for c in last.planning_calls] == ["initial"]


def test_fifteen_step_budget_recovers_after_failures_thirteen_and_fourteen():
    class BudgetModel(Model):
        decisions = 0

        def chat(self, messages, **kwargs):
            self.calls.append(messages)
            if messages[0]["role"] == "user":
                repair = "当前计划受阻" in messages[0]["content"]
                return json.dumps({"steps": [{"goal": "依据刷新后的页面导航到目标网址" if repair else "执行剩余目标", "expected_evidence": "目标网址出现"}],
                                   "strategy_change": "根据失败证据调整入口与观察方式，核查新的页面状态"})
            self.decisions += 1
            action = ({"type": "goto", "url": "https://example.test/done"} if self.decisions == 15 else
                      {"type": "click", "target": "Download PDF", "target_ref": f"snapshot-{self.decisions}:1"})
            return json.dumps({"next_action": action})

    env = Env()

    def execute(action, route):
        env.actions.append(action)
        number = len(env.actions)
        if number in {13, 14}:
            return StepResult(env.obs, info={"status": "rejected", "dispatched": False, "error_code": "stale"})
        env.obs = observation(url=action.args.get("url", "https://example.test/"), value=str(number))
        return StepResult(env.obs, info={"status": "executed", "dispatched": True})

    env.execute_routed = execute
    result = Pipeline(model=BudgetModel([]), env=env, grounding=None, system_prompt="test",
                      max_steps=15, early_stop_repeat=3).run("打开 https://example.test/done")
    assert result.success and result.steps == 15
    assert result.termination_reason == "verified"
    recovery = [e for e in result.events if e["event_type"] == "framework_recovery" and e["status"] == "requested"]
    assert len(recovery) == 1 and recovery[0]["parent_decision_id"] == "a_14"
    assert recovery[0]["remaining_decisions"] == 1


def test_parse_failures_refresh_and_have_bounded_replanning():
    class InvalidModel(Model):
        def chat(self, messages, **kwargs):
            self.calls.append(messages)
            if messages[0]["role"] == "user":
                repair = "当前计划受阻" in messages[0]["content"]
                return json.dumps({"steps": [{"goal": "重新观察控件并生成合法动作" if repair else "恢复输出格式", "expected_evidence": "合法动作"}],
                                   "strategy_change": "根据失败证据调整入口与观察方式，核查新的页面状态"})
            return "literal invalid policy output"

    env = Env()
    result = Pipeline(model=InvalidModel([]), env=env, grounding=None, system_prompt="test",
                      max_steps=10, max_retries=2, max_replans=1).run("任务")
    assert result.steps == 4 and result.termination_reason == "stalled"
    assert env.reset_count == 1 and env.observe_count == 4 and env.actions == []
    transitions = [t for t in result.trajectory if "policy_output" in t]
    assert len(transitions) == 4
    assert all(t["policy_output"] == "literal invalid policy output" for t in transitions)
    assert all(len(t["decision_calls"]) == 2 for t in transitions)


def test_verification_precedes_recovery_even_when_failure_threshold_is_reached():
    env = Env()
    calls = []

    def execute(action, route):
        calls.append(action)
        if len(calls) == 2:
            env.obs = observation("https://example.test/done")
        return StepResult(env.obs, info={"status": "rejected", "dispatched": False, "error_code": "stale"})

    env.execute_routed = execute
    model = Model([{"type": "click", "target": "PDF"}] * 2)
    result = Pipeline(model=model, env=env, grounding=None, system_prompt="test", max_steps=3).run(
        "打开 https://example.test/done")
    assert result.success and result.steps == 2
    assert [c["plan_stage"] for c in result.planning_calls] == ["initial"]


class EventEnv(Env):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.event_reads = 0
        self.canceled = []

    def browser_event_metadata(self):
        value = self.events[min(self.event_reads, len(self.events) - 1)]
        self.event_reads += 1
        return copy.deepcopy(value)

    def cancel_download(self, ident):
        self.canceled.append(ident)
        return {"status": "executed"}


def metadata(pending=False, artifacts=None, dialog=None):
    info = {"artifacts": artifacts or [], "pending_downloads": [], "download_events": []}
    if pending:
        info["pending_downloads"] = [{"download_id": "d1", "status": "in_progress", "page_id": "task-page"}]
    if dialog:
        info["dialogs"] = [dialog]
    return info


def test_last_decision_waits_for_download_and_verifies_real_file(tmp_path, clock):
    path = tmp_path / "file.pdf"
    path.write_bytes(b"%PDF-test")
    url = "https://example.test/file.pdf"
    artifact = {"download_id": "d1", "status": "completed", "completed": True,
                "url": url, "path": str(path)}
    env = EventEnv([metadata(), metadata(True), metadata(True), metadata(artifacts=[artifact])])
    model = Model([{"type": "click", "target": "Download PDF"}])
    raw = model.responses[-1]
    result = Pipeline(model=model, env=env, grounding=None, system_prompt="test", max_steps=1,
                      completion_spec={"checks": [{"kind": "download", "url": url}]}).run("取得文件")
    assert result.success and result.steps == 1 and result.termination_reason == "verified"
    assert len(model.calls) == 2 and len(env.actions) == 1
    assert env.observe_count == 1 and env.event_reads == 4
    assert result.trajectory[0]["policy_output"] == raw
    waits = [e for e in result.events if e["event_type"] == "framework_wait"]
    assert [e["status"] for e in waits] == ["waiting", "completed"]
    assert all(e["policy_trainable"] is False for e in waits)
    assert result.trajectory[0]["execution_status"] == "executed"
    assert "artifacts" not in result.trajectory[0]["observation"]["info"]


def settle(env, **wait_config):
    pipe = make_pipeline(env, wait_config=wait_config)
    pipe.completion = CompletionVerifier("task")
    state, decision = state_and_decision()
    execution = Execution(kind="browser_dom", obs=state.obs, info={"status": "executed", "dispatched": True})
    return pipe, state, pipe._settle_execution(decision.action, execution, state)


def test_no_download_signal_has_only_bounded_start_grace(clock):
    env = EventEnv([metadata()])
    pipe, state, execution = settle(env, download_start_timeout_s=0.5)
    assert sum(clock.sleeps) == 0.5
    assert env.observe_count == 1 and env.event_reads == 3
    assert execution.info["framework_wait"]["status"] == "no_signal"
    assert state.memory.stall_count == 0 and not env.canceled


def test_download_jev_checks_each_second_but_cannot_claim_file_without_cdp(clock):
    class Judge:
        last_call = {}
        def __init__(self):
            self.calls = []
        def download_wait(self, before, after, action, events, *, remaining_s=None):
            self.calls.append((clock.now, dict(events)))
            return {'status': 'answered', 'latency_ms': 250,
                    'answers': {'effect': {'label': 'not_started'}, 'next': {'label': 'inspect'}}}

    env = EventEnv([metadata()])
    pipe = make_pipeline(env, wait_config={'download_start_timeout_s': 2.0, 'download_visual_assist': True})
    pipe.decision_judge = Judge()
    state, decision = state_and_decision()
    result = pipe._settle_execution(decision.action,
        Execution(kind='browser_dom', obs=state.obs, info={'status': 'executed', 'dispatched': True}), state)
    assert len(pipe.decision_judge.calls) == 2
    assert [round(t - pipe.decision_judge.calls[0][0]) for t, _ in pipe.decision_judge.calls] == [0, 1]
    assert result.info['framework_wait']['status'] == 'no_signal'
    assert result.info['status'] == 'uncertain'
    assert result.info['wait_error_code'] == 'download_not_confirmed'
    assert not env.canceled


def test_download_jev_wait_hint_extends_start_grace_but_is_bounded(clock):
    class Judge:
        last_call = {}
        def __init__(self): self.calls = 0
        def download_wait(self, *args, **kwargs):
            self.calls += 1
            return {'status': 'answered', 'latency_ms': 250,
                    'answers': {'effect': {'label': 'unknown'}, 'next': {'label': 'wait'}}}

    env = EventEnv([metadata()])
    pipe = make_pipeline(env, wait_config={'download_start_timeout_s': 2.0, 'download_visual_assist': True})
    pipe.decision_judge = Judge()
    state, decision = state_and_decision()
    result = pipe._settle_execution(decision.action,
        Execution(kind='browser_dom', obs=state.obs, info={'status': 'executed', 'dispatched': True}), state)
    assert 3 <= pipe.decision_judge.calls <= 9
    assert sum(clock.sleeps) == 8.0
    assert result.info['status'] == 'uncertain'


def test_pdf_viewer_visual_click_without_download_signal_is_uncertain(clock, monkeypatch):
    monkeypatch.setattr("osworld_agent.pipeline.is_pdf_observation", lambda obs: True)
    env = EventEnv([metadata()])
    pipe = make_pipeline(env, wait_config={"download_start_timeout_s": 0.5})
    pipe.completion = CompletionVerifier("task")
    state, _ = state_and_decision()
    action = Action("click", {"target": "PDF viewer download icon", "x": 1335, "y": 30})
    execution = Execution(kind="browser_dom", obs=state.obs,
                          info={"status": "executed", "dispatched": True, "resolved": "visual"})
    result = pipe._settle_execution(action, execution, state)
    assert result.info["framework_wait"]["status"] == "no_signal"
    assert result.info["status"] == "uncertain" and result.info["dispatched"] is True
    assert "observe_desktop" in result.info["wait_reason"]


def test_download_timeout_cancels_once_without_replay_or_policy_step(clock):
    env = EventEnv([metadata(True)])
    pipe, state, execution = settle(env, operation_timeout_s=0.5)
    assert env.canceled == ["d1"] and not env.actions
    assert execution.info["status"] == "uncertain" and execution.info["dispatched"] is True
    assert execution.info["wait_error_code"] == "download_timeout"
    assert execution.info["framework_wait"]["status"] == "timeout"
    assert sum(clock.sleeps) == 0.5 and env.observe_count == 1
    assert state.memory.stall_count == 0


def test_dialog_interrupts_pending_download_immediately(clock):
    env = EventEnv([metadata(True, dialog={"type": "confirm", "message": "Allow download?"})])
    _, state, execution = settle(env)
    assert not clock.sleeps and not env.canceled
    assert execution.info["framework_wait"]["status"] == "dialog"
    assert execution.obs.info["dialog"]["type"] == "confirm"
    assert state.recovery["status"] == "resolve_dialog"


def test_handling_dialog_on_final_decision_settles_existing_download(tmp_path, clock):
    path = tmp_path / "download.pdf"
    path.write_bytes(b"%PDF-real-file-evidence")
    url = "https://example.test/download.pdf"
    artifact = {"download_id": "d1", "completed": True, "status": "completed", "url": url, "path": str(path)}
    env = EventEnv([metadata(True, dialog={"type": "confirm", "message": "Continue?"}),
                    metadata(True), metadata(True), metadata(artifacts=[artifact])])
    pipe = make_pipeline(env, max_steps=2)
    pipe.completion = CompletionVerifier("task", {"checks": [{"kind": "download", "url": url}]})
    state, click = state_and_decision()
    first = pipe._settle_execution(click.action, Execution(kind="browser_dom", obs=state.obs,
                                   info={"status": "executed", "dispatched": True}), state)
    state.obs = first.obs
    assert set(state.waiting_download_ids) == {"d1"}
    state.step_id = 2
    action = Action("handle_dialog", {"accept": True})
    execution = pipe._settle_execution(action, Execution(kind="browser", obs=observation(),
                                     info={"status": "executed", "dispatched": True}), state)
    result = AgentResult(task="task")
    _, done = pipe.postprocess(2, Decision(action=action), execution, state, result)
    assert done and result.success and result.termination_reason == "verified"
    assert execution.info["framework_wait"]["status"] == "completed"
    assert set(state.waiting_download_ids) == set()
    assert sum(clock.sleeps) == 0.5 and not pipe.model.calls


def test_switching_to_desktop_keeps_pending_download_for_final_finish(tmp_path, clock):
    path = tmp_path / "download.pdf"
    path.write_bytes(b"%PDF-real-file-evidence")
    url = "https://example.test/download.pdf"
    artifact = {"download_id": "d1", "completed": True, "status": "completed", "url": url, "path": str(path)}
    dialog_pending = metadata(True, dialog={"type": "confirm", "message": "Continue?"})
    env = EventEnv([dialog_pending, dialog_pending, metadata(True), metadata(artifacts=[artifact])])
    pipe = make_pipeline(env, max_steps=3)
    pipe.completion = CompletionVerifier("task", {"checks": [{"kind": "download", "url": url}]})
    state, click = state_and_decision()
    first = pipe._settle_execution(click.action, Execution(kind="browser_dom", obs=state.obs,
                                   info={"status": "executed", "dispatched": True}), state)
    state.obs = first.obs
    desktop = observation()
    desktop.context = []
    desktop.info = {"screenshot_space": "desktop"}
    env.obs = desktop
    switched = pipe._settle_execution(Action("observe_desktop", {}), Execution(kind="browser", obs=desktop,
                                      info={"status": "executed", "dispatched": False}), state)
    state.obs = switched.obs
    assert set(state.waiting_download_ids) == {"d1"}
    assert not state.obs.info.get("page_id") and not state.obs.context
    state.step_id = 3
    finish = Action("request_finish", {"answer": "下载结束"})
    execution = pipe._settle_execution(finish, Execution(kind="finish_request", obs=desktop, info={}), state)
    result = AgentResult(task="task")
    _, done = pipe.postprocess(3, Decision(action=finish), execution, state, result)
    assert done and result.success and result.termination_reason == "verified"
    assert execution.info["framework_wait"]["status"] == "completed"
    assert set(state.waiting_download_ids) == set()
    assert sum(clock.sleeps) == 0.25 and not pipe.model.calls


@pytest.mark.parametrize("name,kind", [("handle_dialog", "browser"), ("observe_desktop", "browser"),
                                      ("request_finish", "finish_request")])
def test_nontrigger_actions_without_pending_transfer_have_no_start_grace(clock, name, kind):
    env = EventEnv([metadata()])
    pipe = make_pipeline(env)
    state, _ = state_and_decision()
    execution = pipe._settle_execution(Action(name, {}), Execution(kind=kind, obs=state.obs,
                                      info={"status": "executed", "dispatched": True}), state)
    assert not clock.sleeps and env.observe_count == 0
    assert "framework_wait" not in execution.info


def test_task_deadline_limits_pending_download_on_final_step(clock):
    env = EventEnv([metadata(True)])
    env._browser_session = SimpleNamespace(options=SimpleNamespace(download_timeout_s=999))
    model = Model([{"type": "click", "target": "Download PDF"}])
    result = Pipeline(model=model, env=env, grounding=None, system_prompt="test", max_steps=1,
                      wait_config={"task_timeout_s": 0.5}).run("任务")
    assert result.termination_reason == "task_timeout" and result.steps == 1
    assert env._browser_session.options.download_timeout_s == 0.5
    assert env.canceled == ["d1"] and sum(clock.sleeps) == 0.5


def test_unrelated_page_download_does_not_block_action(clock):
    data = metadata(True)
    data["pending_downloads"][0]["page_id"] = "unrelated-tab"
    env = EventEnv([data])
    _, _, execution = settle(env, download_start_timeout_s=0)
    assert not clock.sleeps and not env.canceled
    assert execution.info["status"] == "executed"


@pytest.mark.parametrize("terminal", ["failed", "completed"])
def test_unrelated_terminal_download_event_cannot_end_relevant_wait(clock, terminal):
    unrelated = {"download_id": "other", "page_id": "other-page", "status": terminal,
                 "completed": terminal == "completed", "url": "https://other.test/file"}
    data = metadata()
    data["artifacts" if terminal == "completed" else "download_events"] = [unrelated]
    env = EventEnv([data])
    _, _, execution = settle(env, download_start_timeout_s=0.5)
    assert sum(clock.sleeps) == 0.5
    assert execution.info["framework_wait"]["status"] == "no_signal"
    assert execution.info["status"] == "executed"


def test_paper_task_does_not_wait_after_ordinary_navigation_or_search(clock):
    env = EventEnv([metadata()])
    pipe = make_pipeline(env)
    pipe.completion = CompletionVerifier("task")
    state, _ = state_and_decision()
    state.task_profile = {"kind": "paper_download"}
    for action in (Action("goto", {"url": "https://arxiv.org/search"}), Action("click", {"target": "Search"})):
        execution = Execution(kind="browser_dom", obs=state.obs, info={"status": "executed", "dispatched": True})
        pipe._settle_execution(action, execution, state)
    assert not clock.sleeps and env.observe_count == 0


def test_declared_download_candidate_waits_even_without_pdf_button_label(clock):
    env = EventEnv([metadata()])
    pipe = make_pipeline(env, wait_config={"download_start_timeout_s": 0.5})
    pipe.completion = CompletionVerifier("task")
    state, decision = state_and_decision(target="Get document")
    state.obs.context[0]["nodes"][0].update(name="Get document", href="/files/123")
    state.obs.info["download_candidates"] = [{"url": "https://example.test/files/123"}]
    pipe._settle_execution(decision.action, Execution(kind="browser_dom", obs=state.obs,
                          info={"status": "executed", "dispatched": True}), state)
    assert sum(clock.sleeps) == 0.5


def test_model_call_crossing_deadline_cannot_dispatch_an_action(clock):
    class SlowDecisionModel(Model):
        def chat(self, messages, **kwargs):
            output = super().chat(messages, **kwargs)
            if messages[0]["role"] == "system":
                clock.now += 2
            return output

    env = Env()
    result = Pipeline(model=SlowDecisionModel([{"type": "goto", "url": "https://example.test/done"}]),
                      env=env, grounding=None, system_prompt="test", max_steps=2,
                      wait_config={"task_timeout_s": 1}).run("任务")
    assert result.termination_reason == "task_timeout" and not env.actions
    assert result.trajectory[0]["execution_info"]["dispatched"] is False


@pytest.mark.parametrize("config", [{"poll_interval_s": 0}, {"operation_timeout_s": -1},
                                  {"task_timeout_s": float("inf")}, {"download_start_timeout_s": float("nan")},
                                  {"poll_interval_s": True}, {"enabled": "false"}])
def test_wait_config_rejects_unbounded_or_invalid_values(config):
    with pytest.raises(ValueError):
        WaitConfig(**config)
