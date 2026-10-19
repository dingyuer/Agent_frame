"""Independent work overlaps; joins preserve evidence and failure precedence."""
import threading
from types import SimpleNamespace

import pytest

from osworld_agent.actions import Action
from osworld_agent.browser_session import BrowserSession
from osworld_agent.pipeline import AgentResult, Execution, StepState
from osworld_agent.memory import StructuredMemory
from osworld_agent.verifiers import CompletionVerifier
from test_latency_paths import named_setup
from test_pipeline_completion import Env, Model, pipeline, observation


def test_startup_prepares_all_inputs_concurrently_before_planning(monkeypatch):
    pipe = pipeline(Model([]), Env())
    pipe.completion = CompletionVerifier('test')
    barrier = threading.Barrier(3, timeout=3)
    main_thread = threading.get_ident()
    prepared = []
    def client():
        barrier.wait()
        prepared.append('client')
    def classify(*args, **kwargs):
        barrier.wait()
        prepared.append('classification')
        return {'status': 'answered', 'answers': {'task_type': {'label': 'browser'}}}
    def environment(task, state, result):
        assert threading.get_ident() == main_thread
        barrier.wait()
        state.task_profile = {'kind': 'generic'}
        state.obs = observation()
        prepared.append('environment')
    pipe.planner_model = SimpleNamespace(prepare_client=client)
    pipe.decision_judge = SimpleNamespace(classify_task=classify, last_call={})
    monkeypatch.setattr(pipe, '_prepare_initial_environment', environment)
    state, result = StepState(task='test'), AgentResult(task='test')
    pipe._prepare_initial_state('test', state, result)
    assert set(prepared) == {'client', 'classification', 'environment'}
    assert state.task_profile['execution_hint']['answers']['task_type']['label'] == 'browser'
    assert result.run_manifest['task_routing_hint'] == state.routing_hint['task']
    assert any(e['event_type'] == 'parallel_preparation' for e in pipe.events)


def test_four_questions_overlap_code_review_but_cannot_override_failure(monkeypatch):
    pipe = pipeline(Model([]), Env())
    barrier = threading.Barrier(2, timeout=3)
    calls = []
    def assess(*args, **kwargs):
        barrier.wait()
        calls.append('four_questions')
        return {'answers': {k: {'label': v} for k, v in
            [('settled', 'complete'), ('correct', 'yes'), ('next_mode', 'dom'), ('replan', 'no')]}}
    def code(*args, **kwargs):
        barrier.wait()
        calls.append('code')
        return {'method': 'code', 'status': 'failed', 'reason': 'wrong URL'}, {}
    pipe.decision_judge = SimpleNamespace(config=SimpleNamespace(transition_diagnostics=True),
        assess_transition=assess, last_call={})
    monkeypatch.setattr(pipe, '_code_step_review', code)
    monkeypatch.setattr(pipe, '_review_visual', lambda *a, **k: pytest.fail('must not override code failure'))
    state = StepState(task='test', obs=observation(), step_id=1, memory=StructuredMemory())
    pipe._update_plan_progress(state, Execution(kind='browser_dom', obs=state.obs,
        info={'status': 'executed'}), action=Action('click', {'target': 'test'}))
    assert set(calls) == {'four_questions', 'code'}
    assert state.step_review['status'] == 'failed'
    assert set(state.step_review['action_transition']['answers']) == {'settled', 'correct', 'next_mode', 'replan'}


def test_failed_code_work_joins_judge_before_propagating(monkeypatch):
    pipe = pipeline(Model([]), Env())
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    def assess(*a, **k):
        started.set()
        assert release.wait(3)
        finished.set()
        return {}
    def code(*a, **k):
        assert started.wait(3)
        release.set()
        raise ValueError('code failed')
    pipe.decision_judge = SimpleNamespace(config=SimpleNamespace(transition_diagnostics=True), assess_transition=assess)
    monkeypatch.setattr(pipe, '_code_step_review', code)
    with pytest.raises(ValueError, match='code failed'):
        pipe._parallel_action_review(StepState(task='test'), {'id': 's'}, Execution(kind='env'),
                                    Action('click', {}), None, {}, {})
    assert finished.is_set()


def test_final_capture_overlaps_file_precheck_and_detects_deletion(tmp_path, monkeypatch):
    pipe, state, path, calls = named_setup(tmp_path, monkeypatch)
    assert pipe._verify_finish(state, 'done')[0]
    checked, captured = threading.Event(), threading.Event()
    original = pipe._final_code_snapshot
    def code():
        result = original()
        checked.set()
        assert captured.wait(3)
        return result
    def observe():
        assert checked.wait(3)
        path.unlink()
        captured.set()
        return observation()
    monkeypatch.setattr(pipe, '_final_code_snapshot', code)
    monkeypatch.setattr(pipe, '_observe', observe)
    result = AgentResult(task=state.task, success=True)
    pipe._evaluate(result)
    assert not result.success
    assert result.verification['download_facts'][0]['status'] == 'failed'
    assert len(calls) == 1  # Code failure cannot be waived by another model call.


def test_screen_stability_wait_keeps_browser_events_running(monkeypatch):
    events = []
    page = SimpleNamespace(url='https://example.test',
        screenshot=lambda **kw: b'stable frame',
        wait_for_timeout=lambda duration: events.append(('pumped', duration)))
    monkeypatch.setattr('osworld_agent.browser_session.time.sleep',
                        lambda *a: pytest.fail('blocking the browser loop'))
    BrowserSession._wait_screen_stable(page)
    assert events == [('pumped', 350)]
