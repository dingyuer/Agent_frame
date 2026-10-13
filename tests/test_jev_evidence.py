"""Evidence selection, partial progress, bounded re-observation and historical review."""
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from osworld_agent.actions import Action
from osworld_agent.execution_queue import ExecutionQueue
from osworld_agent.judge_evidence import checkpoint, effect_review, select_frame
from osworld_agent.memory import StructuredMemory
from osworld_agent.model.decision_judge import DecisionJudge
from osworld_agent.pipeline import Pipeline, StepState, Execution
from osworld_agent.step_progress import StepProgressMonitor
from osworld_agent.verifiers import CompletionVerifier
from test_pipeline_completion import Model, Env, observation
from test_decision_judge import answer

CONFIG = {'url': 'http://judge.test/v1', 'model': 'vllm-sr/d3-flash', 'review_mode': 'exclusive'}
STEP = {'id': 'p0_1', 'goal': '输入关键词 roboclaw 并搜索', 'expected_evidence': '显示搜索结果',
        'verification': [{'kind': 'visual'}], 'check_ids': []}
IDENTITY = {'review_id': 'run:a1:p0_1', 'plan_step_id': 'p0_1', 'decision_step': 1, 'goal': STEP['goal']}


def judge_with(labels, requests, **config):
    pending = list(labels)
    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        label = pending.pop(0)
        return httpx.Response(200, json={'model': 'd3-flash', 'answers': {
            name: answer(question['criteria'], label) for name, question in payload['questions'].items()}})
    return DecisionJudge({**CONFIG, **config}, httpx.MockTransport(handle))


def test_relevant_late_nodes_modals_and_new_artifacts_survive_bounded_selection():
    before, after = observation(), observation()
    many = [{'ref': str(i), 'role': 'link', 'name': 'Unrelated link ' + str(i)} for i in range(150)]
    many += [{'ref': '150', 'role': 'textbox', 'name': 'Paper search', 'value': 'roboclaw'},
             {'ref': '151', 'role': 'alert', 'name': 'Permission dialog'}]
    after.context[0]['nodes'] = many
    before.info['artifacts'] = [{'artifact_id': str(i), 'path': 'unrelated.pdf'} for i in range(30)]
    after.info['artifacts'] = copy.deepcopy(before.info['artifacts']) + [{'artifact_id': 'latest', 'path': 'roboclaw.pdf', 'completed': True}]
    frame = select_frame(after, STEP, {'args': {'target': 'Paper search'}}, before)
    assert len(frame['nodes']) <= 40
    assert {'150', '151'} <= {n['ref'] for n in frame['nodes']}
    assert frame['artifacts'][0]['artifact_id'] == 'latest'
    assert frame['selection']['truncated'] and frame['selection']['nodes_total'] == 152


def test_exact_input_effect_is_not_subtask_completion_and_ambiguous_inputs_abstain():
    before, after = observation(), observation(value='roboclaw')
    execution = {'action': 'type', 'status': 'executed', 'args': {'target_ref': '1', 'text': 'roboclaw'}}
    assert effect_review(before, after, execution)['status'] == 'passed'
    assert effect_review(after, after, execution)['status'] == 'unknown'  # no new effect
    after.context[0]['nodes'].append(dict(after.context[0]['nodes'][0], ref='2'))
    assert effect_review(before, after, execution)['status'] == 'unknown'
    assert effect_review(before, after, {**execution, 'dispatched': False})['status'] == 'failed'


def test_progress_uses_one_forward_and_never_advances_queue():
    requests = []
    judge = judge_with(['progress'], requests)
    report = judge.review('搜索论文', STEP, observation(), observation(color='blue'),
                          {'action': 'click', 'status': 'executed', 'expected_effect': '搜索框获得焦点'}, {})
    assert report['step_outcome'] == 'progress' and report['status'] == 'unknown'
    assert report['action_effect']['method'] == 'decision_model' and report['action_effect']['status'] == 'passed'
    assert len(requests) == 1 and list(requests[0]['questions']) == ['step']
    queue = ExecutionQueue({'steps': [copy.deepcopy(STEP)]})
    assert not queue.accept_review({**report, 'plan_step_id': 'p0_1'}) and queue.cursor == 0


def test_code_input_effect_does_not_become_false_failure_when_goal_is_incomplete():
    judge = judge_with(['failed'], [])
    report = judge.review('搜索论文', STEP, observation(), observation(value='roboclaw'),
        {'action': 'type', 'args': {'target_ref': '1', 'text': 'roboclaw'}, 'status': 'executed'}, {})
    assert report['status'] == 'failed'  # goal not yet achieved
    assert report['action_effect']['status'] == 'passed' and report['action_effect']['method'] == 'code'


def setup_review(monkeypatch, labels, fresh=None, spec=None, recheck=True):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    env = Env()
    env.obs = fresh or observation()
    pipe = Pipeline(model=Model([]), env=env, grounding=None, system_prompt='test', decision_judge=CONFIG)
    pipe.completion = CompletionVerifier('核对搜索结果', spec)
    state = StepState(task='核对搜索结果', obs=observation(), step_id=1, memory=StructuredMemory())
    pipe.completion.observe(state.obs)
    state.verification = pipe.completion.evaluate()
    requests = []
    pipe.decision_judge = judge_with(labels, requests, recheck_unknown=recheck)
    return pipe, state, requests


def review(pipe, state, step=None):
    return pipe._review_visual(state, step or STEP, {'status': 'executed', 'action': 'click'},
                              {'method': 'code', 'status': 'unknown'}, IDENTITY, before=observation())


def test_unchanged_evidence_does_not_call_model_twice_or_consume_steps(monkeypatch):
    pipe, state, requests = setup_review(monkeypatch, ['unknown', 'unknown'])
    report = review(pipe, state)
    assert report['status'] == 'unknown' and report['recheck']['status'] == 'unchanged'
    assert len(requests) == 1 and pipe.env.observe_count == 1 and not pipe.env.actions
    assert state.step_id == 1
    report = review(pipe, state)
    assert report['recheck']['status'] == 'budget_exhausted' and pipe.env.observe_count == 1
    assert len(requests) == 2  # one fresh primary review, no second retry


def test_new_evidence_gets_one_recheck_and_all_calls_are_verifier_only(monkeypatch):
    pipe, state, requests = setup_review(monkeypatch, ['unknown', 'passed'], observation(value='results', color='blue'))
    report = review(pipe, state)
    assert report['status'] == 'passed' and report['recheck']['model_calls'] == 1
    assert len(requests) == 2 and pipe.env.observe_count == 1 and not pipe.env.actions
    assert all(c['actor_role'] == 'verifier' and not c['policy_trainable'] for c in pipe.reviewer.calls)
    assert json.loads(requests[-1]['state'])['after']['nodes'][0]['value'] == 'results'
    assert report['latency_ms'] >= report['model_latency_ms']


def test_new_code_failure_blocks_even_a_passing_second_model_response(monkeypatch):
    pipe, state, requests = setup_review(monkeypatch, ['unknown', 'passed'], observation(url='https://wrong.test/'))
    step = {**STEP, 'verification': [{'kind': 'url_equals', 'url': 'https://required.test/'}]}
    report = review(pipe, state, step)
    assert report['method'] == 'code' and report['status'] == 'failed'
    assert len(requests) == 1 and report['recheck']['status'] == 'code_resolved'


def test_code_can_resolve_new_evidence_without_second_model_call(monkeypatch):
    pipe, state, requests = setup_review(monkeypatch, ['unknown'], observation(url='https://required.test/'))
    step = {**STEP, 'verification': [{'kind': 'url_equals', 'url': 'https://required.test/'}]}
    report = review(pipe, state, step)
    assert report['method'] == 'code' and report['status'] == 'passed' and len(requests) == 1


def test_api_error_and_progress_do_not_trigger_reobservation(monkeypatch):
    pipe, state, requests = setup_review(monkeypatch, ['progress'])
    assert review(pipe, state)['step_outcome'] == 'progress'
    assert pipe.env.observe_count == 0
    pipe.decision_judge = DecisionJudge(CONFIG, httpx.MockTransport(lambda r: httpx.Response(503)))
    assert review(pipe, state)['status'] == 'unknown'
    assert pipe.env.observe_count == 0


def test_disable_recheck_preserves_unknown(monkeypatch):
    pipe, state, requests = setup_review(monkeypatch, ['unknown'], observation(color='blue'), recheck=False)
    assert review(pipe, state)['status'] == 'unknown'
    assert len(requests) == 1 and pipe.env.observe_count == 0


def test_historical_checkpoints_are_bounded_labeled_and_do_not_mutate(monkeypatch):
    pipe, state, requests = setup_review(monkeypatch, ['passed'])
    for i, color in enumerate(['red', 'green', 'blue', 'yellow', 'pink', 'orange']):
        state.obs = observation(url=f'https://example.test/{i}', color=color)
        state.step_id = i + 1
        pipe._remember_judge_checkpoint(state, {**STEP, 'id': f'p0_{i+1}'}, {'method': 'code'})
    assert len(state.judge_checkpoints) == 4
    saved = copy.deepcopy(state.judge_checkpoints)
    state.obs = observation(color='white')
    report = pipe.decision_judge.review(state.task, {**STEP, 'id': 'whole_task'}, None, state.obs, {}, {}, checkpoints=state.judge_checkpoints)
    payload = requests[0]
    data = json.loads(payload['state'])
    assert len(payload['images']) == 3 and len(report['checkpoint_ids']) == 2
    assert data['image_order'][-1] == 'after_only'
    assert all(x.startswith('historical_checkpoint:') for x in data['image_order'][:-1])
    assert '过去的观察' in payload['questions']['step']['instructions']
    assert state.judge_checkpoints == saved and 'progress' not in payload['questions']['step']['criteria']


def test_visual_progress_allowance_cannot_mask_endless_stalling():
    monitor = StepProgressMonitor()
    for i in range(6):
        report = {'plan_step_id': 'p0_1', 'status': 'unknown', 'action_effect': {
            'method': 'decision_model', 'status': 'passed', 'evidence_fingerprint': str(i)}}
        progress = monitor.observe(STEP, report, observation(), observation(), Action('click'), {'status': 'executed'})
    assert progress['status'] == 'stalled' and progress['idle_decisions'] == 4


def test_ref_only_target_survives_renumbering_and_evidence_fingerprint_is_stable():
    from osworld_agent.judge_evidence import target_identity, evidence_fingerprint
    before, after = observation(), observation()
    before.context[0]['nodes'] = [{'ref': str(i), 'id': f'n{i}', 'role': 'button', 'name': f'Item {i}'} for i in range(120)]
    after.context[0]['nodes'] = [dict(n, ref=str(int(n['ref'])+200)) for n in before.context[0]['nodes']]
    execution = {'args': {'target_ref': '119'}, 'target_node': target_identity(before, {'target_ref': '119'})}
    frame = select_frame(after, {}, execution, before)
    assert frame['nodes'][0]['id'] == 'n119'
    assert evidence_fingerprint(before, {}, execution) == evidence_fingerprint(after, {}, execution)


def test_input_effect_cannot_cross_browser_document_identity():
    before, after = observation(), observation(value='value')
    after.info['page_id'] = 'another-tab'
    report = effect_review(before, after, {'action': 'type', 'status': 'executed', 'args': {'target_ref': '1', 'text': 'value'}})
    assert report['status'] == 'unknown'


def test_final_recheck_cannot_override_new_failure_of_hard_contract(monkeypatch):
    spec = {'checks': [{'kind': 'url_equals', 'url': 'https://example.test/'},
                       {'kind': 'unresolved_requirement', 'text': '搜索结果相关'}]}
    pipe, state, requests = setup_review(monkeypatch, ['unknown', 'passed'], observation(url='https://wrong.test/'), spec)
    assert pipe._model_review_allowed(state.verification)
    report = review(pipe, state, {**STEP, 'id': 'whole_task'})
    assert report['status'] != 'passed' and report['method'] == 'code'
    assert len(requests) == 1 and not pipe._model_review_allowed(state.verification)


def test_new_pending_download_is_not_misread_as_completion(monkeypatch):
    fresh = observation(color='blue')
    fresh.info['pending_downloads'] = [{'download_id': 'd1'}]
    pipe, state, requests = setup_review(monkeypatch, ['unknown', 'passed'], fresh)
    report = review(pipe, state)
    assert report['status'] == 'unknown' and report['recheck']['status'] == 'pending'
    assert len(requests) == 1 and state.obs.info['pending_downloads']


def test_large_evidence_is_reduced_without_losing_original_requirements():
    requests, obs = [], observation()
    obs.context[0]['nodes'] = [{'ref': str(i), 'role': 'textbox', 'name': f'Target {i}' + 'x'*300,
        'value': 'x'*300, 'states': {f'k{j}': 'x'*300 for j in range(10)}} for i in range(200)]
    original = copy.deepcopy(obs.context)
    judge = judge_with(['unknown'], requests)
    goal = {**STEP, 'unresolved_requirements': ['不得覆盖旧文件', '保留所有字段']}
    result = judge.review('必须完成原始任务的全部要求', goal, obs, obs, {}, {})
    assert result['judgment']['status'] == 'answered' and len(requests[0]['state']) <= 24000
    state = json.loads(requests[0]['state'])
    assert state['original_task'] == '必须完成原始任务的全部要求'
    assert state['step']['unresolved_requirements'] == goal['unresolved_requirements']
    assert state['after']['selection']['truncated'] and obs.context == original


def test_recheck_option_roundtrips_to_cli():
    import argparse
    from osworld_agent.model.decision_judge import add_judge_arguments, judge_config_from_args
    from osworld_agent.viz.server import _judge_cli_args
    parser = argparse.ArgumentParser()
    add_judge_arguments(parser)
    flags = _judge_cli_args({'decision_judge': {**CONFIG, 'recheck_unknown': False}})
    parsed = judge_config_from_args(parser.parse_args(flags))
    assert parsed['model'] == 'vllm-sr/d3-flash' and parsed['recheck_unknown'] is False


def test_model_progress_cannot_override_code_rejected_action():
    judge = judge_with(['progress'], [])
    report = judge.review('搜索论文', STEP, observation(), observation(),
                          {'action': 'click', 'status': 'rejected', 'dispatched': False}, {})
    assert report['status'] == 'unknown' and report['step_outcome'] == 'unknown'
    assert report['action_effect']['status'] == 'failed'
    assert '不能采信' in report['reason']
    assert report['judgment']['answers']['step']['label'] == 'progress'  # Preserve actual sampled output.
