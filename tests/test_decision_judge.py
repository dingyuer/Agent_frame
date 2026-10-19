"""Official typed API shape, bounded failures, hard-check precedence and provenance."""
import json

import httpx
import pytest

from osworld_agent.model.decision_judge import DecisionJudge, DecisionJudgeConfig, TASK_CHOICES
from osworld_agent.pipeline import Pipeline
from osworld_agent.trajectory import build_trajectory
from osworld_agent.viz.server import _judge_cli_args
from test_pipeline_completion import Env, Model, observation
from test_execution_queue_review import Reviewer, FAILED, PASSED


CONFIG = {'url': 'http://judge.test:8899/v1', 'model': 'vllm-sr/d3', 'revision': 'pinned-test'}


def answer(labels, selected, probability=.97):
    return {'type': 'choice', 'choice': selected,
            'probabilities': {k: probability if k == selected else (1-probability)/(len(labels)-1) for k in labels}}


def transport(requests, *, step='passed', confidence=.97):
    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == '/v1/systemone'
        answers = {}
        for key, question in payload['questions'].items():
            selected = {'task_type': 'mixed', 'surface': 'browser_chrome', 'effect': 'passed', 'step': step,
                        'settled': 'complete', 'correct': 'yes', 'next_mode': 'dom', 'replan': 'no'}[key]
            answers[key] = answer(question['criteria'], selected, confidence)
        return httpx.Response(200, json={'model': 'vllm-sr/d3', 'model_version': 'server-version', 'answers': answers})
    return httpx.MockTransport(handle)


def test_classification_and_two_image_review_follow_systemone_protocol():
    requests = []
    judge = DecisionJudge(CONFIG, transport(requests))
    classification = judge.classify_task('搜索论文并下载后在文件管理器改名')
    assert classification['answers']['task_type']['label'] == 'mixed'
    assert judge.last_call['purpose'] == 'task_routing'
    before, after = observation(value=''), observation(value='query')
    result = judge.review('task', {'goal': 'search', 'expected_evidence': 'results'}, before, after,
                          {'action': 'type', 'expected_effect': 'text appears'}, {})
    assert result['status'] == 'passed' and result['reward_valid'] is False
    assert result['judgment']['answers']['effect']['label'] == 'passed'
    assert len(requests[-1]['images']) == 2 and all(s.startswith('data:image/') for s in requests[-1]['images'])
    assert json.loads(requests[-1]['state'])['image_order'] == ['before', 'after']
    assert 'messages' not in requests[-1]
    assert result['judgment']['model_version'] == 'server-version'
    assert judge.last_call['actor_role'] == 'verifier' and not judge.last_call['policy_trainable']


def test_action_assessment_sends_before_after_ax_and_four_typed_questions():
    requests = []
    judge = DecisionJudge(CONFIG, transport(requests))
    before, after = observation(value=''), observation(value='query')
    result = judge.assess_transition('search paper', {'goal': 'type query', 'expected_evidence': 'query filled'},
        before, after, {'action': 'type', 'args': {'target_ref': '1', 'text': 'query'},
                        'status': 'executed', 'dispatched': True},
        {'method': 'code', 'status': 'passed', 'reason': 'input value changed'})
    assert result['status'] == 'answered'
    assert set(result['answers']) == {'settled', 'correct', 'next_mode', 'replan'}
    assert judge.last_call['purpose'] == 'action_transition'
    sent = json.loads(requests[-1]['state'])
    assert sent['before']['nodes'] and sent['after']['nodes']
    assert sent['execution']['args']['text'] == 'query'
    assert len(requests[-1]['images']) == 2


@pytest.mark.parametrize('bad', [
    {}, {'type': 'choice', 'error': 'max_length_exceeded'},
    {'type': 'choice', 'probabilities': {'browser': 1}},
    {'type': 'choice', 'probabilities': dict.fromkeys(TASK_CHOICES, .9)},
    {'type': 'choice', 'probabilities': dict.fromkeys(TASK_CHOICES, True)},
    {'type': 'choice', 'probabilities': dict.fromkeys(TASK_CHOICES, float('nan'))},
])
def test_invalid_model_scores_never_produce_confident_judgments(bad):
    judge = DecisionJudge(CONFIG)
    with pytest.raises(ValueError):
        judge._choose(bad, TASK_CHOICES)


def test_low_confidence_and_score_margin_require_abstention():
    judge = DecisionJudge({**CONFIG, 'threshold': .5, 'margin': .2})
    probabilities = {'browser': .51, 'desktop': .49, 'mixed': 0, 'cli': 0, 'unknown': 0}
    result = judge._choose({'type': 'choice', 'probabilities': probabilities}, TASK_CHOICES)
    assert result['label'] == 'unknown' and result['top_label'] == 'browser'


def test_http_failure_deadline_and_missing_image_are_bounded():
    calls = []
    def offline(request):
        calls.append(request)
        raise httpx.ConnectError('fixture failure')
    judge = DecisionJudge(CONFIG, httpx.MockTransport(offline))
    result = judge.classify_task('task', remaining_s=0)
    assert result['status'] == 'unknown' and not calls
    result = judge.classify_task('task')
    assert result['status'] == 'unknown' and len(calls) == 1
    after = observation()
    after.info['screenshot_stale'] = True
    result = judge.review('task', {}, None, after, {}, {})
    assert result['status'] == 'unknown' and len(calls) == 1 and judge.last_call == {}


@pytest.mark.parametrize('kwargs', [{'url': 'file:///tmp/model'}, {'url': 'http://user:key@host'},
                                   {'threshold': True}, {'timeout': 0}, {'margin': float('inf')}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DecisionJudgeConfig(**{**CONFIG, **kwargs})


def make_pipeline(monkeypatch, reviewer, requests, step_status='passed', confidence=.97, completion=None):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    plan = {'steps': [{'goal': '核对结果', 'expected_evidence': '结果符合要求'}]}
    model = Model([{'type': 'click', 'target': 'Search'}], plan)
    pipe = Pipeline(model=model, env=Env(), grounding=None, system_prompt='test', max_steps=1,
                    verifier_model=reviewer, completion_spec=completion, decision_judge=CONFIG)
    pipe.decision_judge = DecisionJudge(CONFIG, transport(requests, step=step_status, confidence=confidence))
    return pipe


def test_typed_step_pass_still_requires_whole_task_review_and_never_becomes_reward(monkeypatch):
    requests, reviewer = [], Reviewer([PASSED])
    result = make_pipeline(monkeypatch, reviewer, requests).run('核对搜索结果')
    assert result.success and result.termination_reason == 'verified_by_model'
    assert len(requests) == 3 and len(reviewer.calls) == 1  # task route, action assessment, local check
    assert result.trajectory[0]['step_verification']['method'] == 'decision_model'
    assert 'action_transition' in result.trajectory[0]['step_verification']
    assert not result.evaluation_available and result.reward_source == 'independent_model_review'
    assert all(c['policy_trainable'] is False for c in result.review_calls)
    data = build_trajectory('typed', 'browser', result.task, result)
    assert data['outcome']['reward_valid'] is False
    assert data['run_manifest']['runtime_config']['decision_judge']['model'] == CONFIG['model']


@pytest.mark.parametrize('status,confidence', [('passed', .6), ('pending', .97), ('unknown', .97)])
def test_uncertain_typed_result_falls_back_to_independent_reviewer(monkeypatch, status, confidence):
    requests, reviewer = [], Reviewer([FAILED])
    result = make_pipeline(monkeypatch, reviewer, requests, status, confidence).run('核对结果')
    assert len(reviewer.calls) == 1 and not result.success
    assert result.trajectory[0]['step_verification']['method'] == 'model'
    assert result.trajectory[0]['step_verification']['decision_judgment']['status'] == 'unknown'


def test_failed_code_predicate_preserves_action_assessment_without_semantic_override(monkeypatch):
    requests, reviewer = [], Reviewer([])
    pipe = make_pipeline(monkeypatch, reviewer, requests)
    plan = {'steps': [{'goal': '打开目标', 'expected_evidence': '目标URL',
                      'verification': [{'kind': 'url_equals', 'url': 'https://example.test/target'}]}]}
    pipe.model.responses[0] = json.dumps(plan)
    result = pipe.run('打开目标页面并完成任务')
    assert len(requests) == 2 and not reviewer.calls  # assessment remains; code failure is authoritative
    assert set(requests[-1]['questions']) == {'settled', 'correct', 'next_mode', 'replan'}
    assert result.trajectory[0]['step_verification']['status'] == 'failed' and not result.success


def test_frontend_and_cli_preserve_configuration_and_do_not_emit_credentials():
    assert _judge_cli_args({}) == []
    args = _judge_cli_args({'decision_judge': {**CONFIG, 'threshold': .95}})
    assert args[args.index('--judge-model') + 1] == CONFIG['model']
    assert float(args[args.index('--judge-threshold') + 1]) == .95
    assert 'api_key' not in DecisionJudgeConfig(**CONFIG, api_key='secret').public()
    with pytest.raises(ValueError):
        _judge_cli_args({'decision_judge': {**CONFIG, 'api_key': 'secret'}})


@pytest.mark.parametrize('step_id,purpose,scope', [
    ('p0_1', 'step_review', 'subtask'), ('whole_task', 'final_review', 'whole_task'),
    ('final', 'final_review', 'whole_task')])
def test_jev_archive_names_review_scope_without_guessing_from_question_id(step_id, purpose, scope):
    judge = DecisionJudge(CONFIG, transport([]))
    try:
        judge.review('task', {'id': step_id, 'goal': 'goal', 'expected_evidence': 'evidence'},
                     observation(), observation(), {'action': 'click'}, {})
        assert judge.last_call['purpose'] == purpose
        assert judge.last_call['review_scope'] == scope
        assert judge.last_call['plan_step_id'] == step_id
        # UI provenance is local metadata; the strict remote protocol is unchanged.
        assert 'purpose' not in judge.last_call['request']
    finally:
        judge.close()
