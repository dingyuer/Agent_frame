"""Closing a popup cannot erase accepted download evidence at task completion."""
import copy
from types import SimpleNamespace

from osworld_agent.actions import Action
from osworld_agent.env import Observation, StepResult
from osworld_agent.execution_queue import ExecutionQueue
from osworld_agent.memory import StructuredMemory
from osworld_agent.pipeline import AgentResult, Decision, Execution, Pipeline, StepState
from osworld_agent.step_verifier import code_review
from osworld_agent.verifiers import CompletionVerifier
from test_download_facts import completed
from test_pipeline_completion import Env, Model, observation


def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    verifier, path, event = completed(tmp_path)
    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt='test')
    pipe.completion = verifier
    state = StepState(task='去指定门户下载一个 PPT', memory=StructuredMemory(), obs=observation())
    state.consumed_downloads.add(event['artifact_id'])
    return pipe, state, path, event


def test_finish_gets_durable_file_facts_after_popup_closed(tmp_path, monkeypatch):
    pipe, state, path, event = prepared(tmp_path, monkeypatch)
    calls = []
    def review(state, step, execution, code, identity, before=None):
        calls.append(copy.deepcopy(code))
        assert not state.obs.info.get('artifacts')
        assert code['checks'] == []  # Generic task still has no invented contract.
        assert code['download_facts'][0]['status'] == 'passed'
        assert code['download_facts'][0]['format'] == 'pptx'
        return {**identity, 'method': 'decision_model', 'status': 'passed',
                'reason': '已核对其余任务要求', 'evidence': [str(path)]}
    monkeypatch.setattr(pipe, '_review_visual', review)
    assert pipe._verify_finish(state, 'done')[0]
    result = AgentResult(task=state.task, success=True, termination_reason='verified')
    pipe._evaluate(result)
    assert result.success and result.termination_reason == 'verified_by_model'
    assert result.verification['completion']['download_facts'][0]['status'] == 'passed'
    assert len(calls) == 1


def test_recheck_of_same_file_can_use_durable_receipt_without_current_artifact_array(tmp_path, monkeypatch):
    pipe, state, _, _ = prepared(tmp_path, monkeypatch)
    step = {'goal': '检查 PPT 下载成功', 'verification': [
        {'kind': 'artifact_download', 'format': 'ppt', 'purpose': 'recheck'}]}
    report, public = pipe._code_step_review(state, step, {})
    assert report['status'] == 'passed'
    step['verification'][0].pop('purpose')
    assert code_review(step, state.obs, {}, public)['status'] == 'unknown'


def test_changed_file_forces_new_final_review_even_when_page_is_identical(tmp_path, monkeypatch):
    pipe, state, path, _ = prepared(tmp_path, monkeypatch)
    calls = []
    def review(state, step, execution, code, identity, before=None):
        calls.append(code['download_facts'][0]['status'])
        ok = calls[-1] == 'passed'
        return {**identity, 'method': 'decision_model', 'status': 'passed' if ok else 'failed',
                'reason': '文件证据已核对', 'evidence': []}
    monkeypatch.setattr(pipe, '_review_visual', review)
    assert pipe._verify_finish(state, 'done')[0]
    path.unlink()
    result = AgentResult(task=state.task, success=True, termination_reason='verified')
    pipe._evaluate(result)
    assert calls == ['passed']  # Code invalidates the proof without asking a model to waive it.
    assert not result.success and result.termination_reason == 'verification_changed'
    assert result.verification['download_facts'][0]['status'] == 'failed'


def test_model_cannot_pass_file_deleted_during_its_review(tmp_path, monkeypatch):
    pipe, state, path, _ = prepared(tmp_path, monkeypatch)
    def review(state, step, execution, code, identity, before=None):
        assert code['download_facts'][0]['status'] == 'passed'
        path.unlink()
        return {**identity, 'method': 'decision_model', 'status': 'passed',
                'reason': '模型误报成功', 'evidence': []}
    monkeypatch.setattr(pipe, '_review_visual', review)
    ok, reason = pipe._verify_finish(state, 'done')
    assert not ok and '丢失或改变' in reason
    assert pipe._model_final_report is None


def test_model_pass_cannot_override_deleted_file_at_final_evaluation(tmp_path, monkeypatch):
    pipe, state, path, _ = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(pipe, '_review_visual', lambda state, step, execution, code, identity, before=None:
        {**identity, 'method': 'decision_model', 'status': 'passed', 'reason': '模型始终声称成功', 'evidence': []})
    assert pipe._verify_finish(state, 'done')[0]
    path.unlink()
    result = AgentResult(task=state.task, success=True, termination_reason='verified')
    pipe._evaluate(result)
    assert not result.success and result.termination_reason == 'verification_changed'


def test_old_failed_file_does_not_invalidate_unrelated_valid_final_receipt():
    old = {'artifact_id': 'wrong', 'status': 'failed'}
    valid = {'artifact_id': 'good', 'status': 'passed', 'sha256': 'x', 'bytes': 10, 'file_identity': [1, 2]}
    assert not Pipeline._download_proof_invalidated([old, valid], [old, dict(valid)])


def test_file_disappearing_in_final_observation_cannot_be_waived_by_model(tmp_path, monkeypatch):
    pipe, state, path, _ = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(pipe, '_review_visual', lambda state, step, execution, code, identity, before=None:
        {**identity, 'method': 'decision_model', 'status': 'passed', 'reason': '模型始终声称成功', 'evidence': []})
    assert pipe._verify_finish(state, 'done')[0]
    def observe_after_removal():
        path.unlink()
        return observation()
    pipe.env.observe = observe_after_removal
    result = AgentResult(task=state.task, success=True, termination_reason='verified')
    pipe._evaluate(result)
    assert not result.success and result.termination_reason == 'verification_changed'


def test_final_checks_file_format_even_when_navigation_goal_is_still_current(tmp_path, monkeypatch):
    pipe, state, path, _ = prepared(tmp_path, monkeypatch)
    state.memory.plan = {'steps': [
        {'goal': '进入方案库', 'verification': [{'kind': 'visual'}]},
        {'goal': '下载一个 PPT', 'verification': [{'kind': 'artifact_download', 'format': 'ppt'}]}]}
    calls = []
    monkeypatch.setattr(pipe, '_review_visual', lambda state, step, execution, code, identity, before=None:
        calls.append(code) or {**identity, 'method': 'decision_model', 'status': 'passed', 'reason': '通过'})
    assert pipe._verify_finish(state, 'done')[0]
    assert calls[0]['download_checks'][0]['status'] == 'passed'


def test_final_model_cannot_waive_missing_required_format(tmp_path, monkeypatch):
    pipe, state, _, _ = prepared(tmp_path, monkeypatch)
    state.memory.plan = {'steps': [
        {'goal': '下载 PDF', 'verification': [{'kind': 'artifact_download', 'format': 'pdf'}]}]}
    monkeypatch.setattr(pipe, '_review_visual', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('A model cannot waive a missing file format')))
    ok, reason = pipe._verify_finish(state, 'done')
    assert not ok and '所需格式' in reason


def test_identical_inconclusive_finish_is_not_repeated_or_replanned(tmp_path, monkeypatch):
    pipe, state, path, _ = prepared(tmp_path, monkeypatch)
    calls = []
    def review(state, step, execution, code, identity, before=None):
        calls.append(code)
        return {**identity, 'method': 'decision_model', 'status': 'unknown',
                'reason': '尚有对象身份要求未确认', 'evidence': []}
    monkeypatch.setattr(pipe, '_review_visual', review)
    assert not pipe._verify_finish(state, 'done')[0]
    state.queue = ExecutionQueue({'steps': [{'idx': 1, 'goal': '核对任务', 'expected_evidence': '原任务要求满足',
        'verification': [{'kind': 'visual'}]}]})
    monkeypatch.setattr(pipe, '_update_plan_progress', lambda *a, **k: None)
    def stalled(*args, **kwargs):
        state.memory.control_state = 'STALL'
    monkeypatch.setattr(pipe, '_compute_control_state', stalled)
    monkeypatch.setattr(pipe, '_recover_plan', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('A completed download must not be replanned for an unchanged finish review')))
    result = AgentResult(task=state.task)
    finish = Action('request_finish', {'answer': 'done'})
    _, stop = pipe.postprocess(2, Decision(action=finish),
        Execution(kind='finish_request', obs=state.obs), state, result)
    pipe._evaluate(result)
    assert stop and not result.success
    assert result.termination_reason == 'completion_inconclusive'
    assert result.verification['download_facts'][0]['status'] == 'passed'
    assert str(path) in result.final_answer and len(calls) == 1


def test_new_page_evidence_can_be_reviewed_after_unknown_finish(tmp_path, monkeypatch):
    pipe, state, _, _ = prepared(tmp_path, monkeypatch)
    calls = []
    def review(state, step, execution, code, identity, before=None):
        calls.append(code)
        return {**identity, 'method': 'decision_model', 'status': 'unknown' if len(calls) == 1 else 'passed',
                'reason': '复核', 'evidence': []}
    monkeypatch.setattr(pipe, '_review_visual', review)
    assert not pipe._verify_finish(state, 'done')[0]
    state.obs = observation(value='new matching attachment details')
    assert pipe._verify_finish(state, 'done')[0]
    assert len(calls) == 2


def test_whole_task_jev_supplemental_review_keeps_download_receipt(tmp_path, monkeypatch):
    pipe, state, _, _ = prepared(tmp_path, monkeypatch)
    pipe.env.obs = observation(value='new metadata')
    received = []
    def review(task, step, before, fresh, execution, code, **kwargs):
        received.append(code)
        return {'method': 'decision_model', 'status': 'passed', 'reason': '核查通过', 'latency_ms': 1}
    pipe.decision_judge = SimpleNamespace(config=SimpleNamespace(recheck_unknown=True), review=review, last_call=None)
    monkeypatch.setattr(pipe, '_record_typed_judge', lambda: None)
    typed = {'method': 'decision_model', 'status': 'unknown', 'reason': 'unknown', 'latency_ms': 1}
    identity = {'review_id': 'r', 'decision_step': 1}
    report = pipe._recheck_jev(state, {'id': 'whole_task', 'goal': state.task}, {}, {}, typed, None, identity)
    assert report['status'] == 'passed'
    assert received[0]['download_facts'][0]['status'] == 'passed'
