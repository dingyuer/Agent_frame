"""System One typed protocol, explicit verifier selection, and debug API boundary."""
import base64
import io
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from osworld_agent.model.decision_judge import DecisionJudge, DecisionJudgeConfig, validate_questions
from osworld_agent.viz.server import app, _judge_cli_args
from test_decision_judge import CONFIG, make_pipeline, transport
from test_execution_queue_review import Reviewer


def picture():
    out = io.BytesIO()
    Image.new('RGB', (64, 64), 'white').save(out, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode()


QUESTIONS = {
    'status': {'type': 'choice', 'instructions': 'Choose', 'criteria': {'passed': 'Yes', 'unknown': None}},
    'completed': {'type': 'noul', 'instructions': 'Complete?'},
    'rating': {'type': 'score', 'instructions': 'Rate', 'criteria': ['No', 'Partly', 'Yes']},
}
ANSWERS = {
    'status': {'type': 'choice', 'choice': 'passed', 'probabilities': {'passed': .97, 'unknown': .03}, 'confidence': .8},
    'completed': {'type': 'noul', 'noul': .966},
    'rating': {'type': 'score', 'score': 1.5, 'probabilities': {'0': .1, '1': .3, '2': .6}, 'confidence': .2,
               'legend': {'0': 'No', '1': 'Partly', '2': 'Yes'}},
}


def test_all_protocol_types_preserve_raw_usage_actual_model_and_auth():
    calls = []
    body = {'model': 'deployed-flash', 'answers': ANSWERS, 'usage': {'input_tokens': 245, 'output_tokens': 0}}
    def handle(request):
        calls.append(json.loads(request.content))
        assert request.headers['Authorization'] == 'Bearer secret'
        return httpx.Response(200, json=body)
    judge = DecisionJudge({**CONFIG, 'api_key': 'secret'}, httpx.MockTransport(handle))
    result = judge.decide('Task context', QUESTIONS, [picture()])
    assert result['status'] == 'answered' and result['raw_response'] == body
    assert result['model'] == 'deployed-flash' and result['usage']['output_tokens'] == 0
    assert result['answers']['completed']['noul'] == .966
    assert result['answers']['rating']['score'] == 1.5
    assert result['answers']['status']['top_probability'] == .97
    assert result['answers']['status']['entropy_confidence'] == .8
    assert result['latency_ms'] >= 0 and calls[0]['state'] == 'Task context'
    assert 'secret' not in json.dumps(judge.last_call)


def test_single_choice_and_malformed_score_are_handled():
    judge = DecisionJudge(CONFIG)
    assert judge._choose({'type': 'choice', 'choice': 'one', 'probabilities': {'one': 1}}, {'one': None})['label'] == 'one'
    for bad in [dict(ANSWERS['rating'], score=0), dict(ANSWERS['rating'], legend={}), dict(ANSWERS['rating'], confidence=True)]:
        with pytest.raises(ValueError):
            judge._answer(bad, QUESTIONS['rating'])
    with pytest.raises(ValueError):
        judge._answer({'type': 'noul', 'noul': True}, QUESTIONS['completed'])
    with pytest.raises(ValueError):
        judge._choose(dict(ANSWERS['status'], choice='unknown'), QUESTIONS['status']['criteria'])


@pytest.mark.parametrize('questions', [{}, {'x': {'type': 'noul', 'instructions': 'Yes?', 'extra': 1}},
    {'x': {'type': 'score', 'instructions': 'Rate', 'criteria': ['one']}},
    {'x': {'type': 'noul', 'instructions': 'Yes?', 'criteria': {'true': 'yes'}}}])
def test_invalid_questions_rejected(questions):
    with pytest.raises(ValueError):
        validate_questions(questions)


def test_remote_422_is_visible_and_not_retried():
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(422, json={'detail': [{'loc': ['images', 0], 'msg': 'invalid media'}]})
    result = DecisionJudge(CONFIG, httpx.MockTransport(handle)).decide('test', QUESTIONS)
    assert result['status'] == 'unknown' and result['http_status'] == 422
    assert result['detail'][0]['msg'] == 'invalid media' and len(calls) == 1


def exclusive_pipeline(monkeypatch, *, status='passed', confidence=.97):
    requests, reviewer = [], Reviewer([])
    pipeline = make_pipeline(monkeypatch, reviewer, requests)
    pipeline.decision_judge = DecisionJudge({**CONFIG, 'review_mode': 'exclusive'}, transport(requests, step=status, confidence=confidence))
    return pipeline, requests, reviewer


def test_exclusive_jev_reviews_steps_and_whole_task_without_generative_fallback(monkeypatch):
    pipeline, calls, reviewer = exclusive_pipeline(monkeypatch)
    result = pipeline.run('核对结果')
    assert result.success and not reviewer.calls
    assert len(calls) == 4  # classification, action assessment, step review, whole task
    assert set(calls[1]['questions']) == {'settled', 'correct', 'next_mode', 'replan'}
    assert list(calls[-1]['questions']) == ['step']  # one forward per semantic review
    assert '原始任务的所有要求' in json.loads(calls[-1]['state'])['step']['expected_evidence']
    assert result.trajectory[0]['step_verification']['latency_ms'] >= 0
    assert not result.evaluation_available and all(not c['policy_trainable'] for c in result.review_calls)


@pytest.mark.parametrize('status,confidence', [('passed', .6), ('pending', .97), ('unknown', .97)])
def test_exclusive_uncertainty_never_calls_generative_reviewer_or_passes(monkeypatch, status, confidence):
    pipeline, calls, reviewer = exclusive_pipeline(monkeypatch, status=status, confidence=confidence)
    result = pipeline.run('核对结果')
    assert not result.success and not reviewer.calls
    report = result.trajectory[0]['step_verification']
    assert report['method'] == 'decision_model' and report['status'] == 'unknown'


def test_exclusive_jev_cannot_override_failed_code(monkeypatch):
    pipeline, calls, reviewer = exclusive_pipeline(monkeypatch)
    pipeline.model.responses[0] = json.dumps({'steps': [{'goal': '打开网页', 'expected_evidence': '目标 URL',
        'verification': [{'kind': 'url_equals', 'url': 'https://target.invalid/'}]}]})
    result = pipeline.run('打开网页')
    assert not result.success and len(calls) == 2 and not reviewer.calls
    assert result.trajectory[0]['step_verification']['method'] == 'code'
    assert result.trajectory[0]['step_verification']['latency_ms'] >= 0


def test_explicit_mode_survives_cli_and_requires_url():
    args = _judge_cli_args({'decision_judge': {**CONFIG, 'review_mode': 'exclusive', 'timeout': 120}})
    assert args[args.index('--judge-review-mode') + 1] == 'exclusive'
    with pytest.raises(ValueError):
        DecisionJudgeConfig(review_mode='exclusive')


def test_manual_api_validates_media_and_does_not_read_local_paths(monkeypatch):
    calls = []
    def decide(self, state, questions, images):
        calls.append((state, questions, images))
        return {'status': 'answered', 'latency_ms': 5, 'raw_response': {'answers': ANSWERS}}
    monkeypatch.setattr(DecisionJudge, 'decide', decide)
    with TestClient(app) as client:
        body = {'config': CONFIG, 'state': {'task': 'check'}, 'questions': QUESTIONS, 'images': [picture()]}
        result = client.post('/jev/decide', json=body)
        assert result.status_code == 200 and result.json()['ok']
        for images in [['/etc/passwd'], ['https://example.test/image.png'], ['data:image/png;base64,bad!'], [picture()]*5]:
            assert client.post('/jev/decide', json={**body, 'images': images}).status_code == 422
        assert client.post('/jev/decide', json={**body, 'unknown': True}).status_code == 422
        assert client.post('/jev/decide', json={**body, 'state': 'x'*32001}).status_code == 422
    assert len(calls) == 1


def test_capability_probe_uses_models_and_keeps_health_unauthenticated(monkeypatch):
    from osworld_agent.viz.jev_debug import _probe
    seen = []
    original = httpx.Client
    def handle(request):
        seen.append(request)
        if request.url.path.endswith('/health'):
            assert 'Authorization' not in request.headers
            return httpx.Response(200, json={'max_input_tokens': None})
        assert request.headers['Authorization'] == 'Bearer env-secret'
        assert request.url.path.endswith('/v1/models')
        return httpx.Response(200, json={'data': [{'id': 'real-model', 'max_input_tokens': 8192}]})
    monkeypatch.setenv('DECISION_JUDGE_API_KEY', 'env-secret')
    monkeypatch.setattr('osworld_agent.viz.jev_debug.httpx.Client',
                        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handle)))
    result = _probe(DecisionJudgeConfig(url='http://example.test/prefix/v1/systemone'))
    assert result['ok'] and len(seen) == 2
    assert result['models']['data']['data'][0]['max_input_tokens'] == 8192
    assert 'env-secret' not in json.dumps(result) and result['latency_ms'] >= 0
