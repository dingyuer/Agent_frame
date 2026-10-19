"""Runtime maximum-probability labels retain strict typed-response validation."""
import json

import httpx
import pytest

from osworld_agent.model.decision_judge import DecisionJudge
from osworld_agent.model.judge_selection import RuntimeDecisionJudge


CONFIG = {'url': 'http://judge.test:8007', 'threshold': .9, 'margin': .15}
OPTIONS = {'passed': 'Complete', 'failed': 'Incorrect', 'pending': 'Waiting', 'unknown': 'Unknown'}
PROBABILITIES = {'passed': .51, 'failed': .49, 'pending': 0, 'unknown': 0}


def answer(probabilities=None, **extra):
    return {'type': 'choice', 'probabilities': dict(PROBABILITIES if probabilities is None else probabilities), **extra}


@pytest.mark.parametrize('policy,expected', [('threshold', 'unknown'), ('maximum_probability', 'passed')])
def test_selection_policy_changes_label_only_and_preserves_scores(policy, expected):
    judge = RuntimeDecisionJudge(CONFIG, selection_policy=policy)
    base = DecisionJudge(CONFIG)
    try:
        original = base._choose(answer(), OPTIONS)
        result = judge._choose(answer(), OPTIONS)
        assert result['label'] == expected
        assert result['selection_policy'] == policy
        assert result['confident'] is False
        assert result['confidence'] == .51
        assert {k: v for k, v in result.items() if k not in {'label', 'selection_policy'}} == {
            k: v for k, v in original.items() if k != 'label'}
    finally:
        judge.close()
        base.close()


def test_default_threshold_matches_installed_judge():
    judge = RuntimeDecisionJudge(CONFIG)
    base = DecisionJudge(CONFIG)
    try:
        result = judge._choose(answer(), OPTIONS)
        assert result.pop('selection_policy') == 'threshold'
        assert result == base._choose(answer(), OPTIONS)
    finally:
        judge.close()
        base.close()


@pytest.mark.parametrize('probabilities,expected', [
    ({'passed': .25, 'failed': .1, 'pending': .1, 'unknown': .55}, 'unknown'),
    ({'passed': .1, 'failed': .2, 'pending': .7, 'unknown': 0}, 'pending'),
    ({'passed': .2, 'failed': .6, 'pending': .1, 'unknown': .1}, 'failed'),
    ({'passed': .5, 'failed': .5, 'pending': 0, 'unknown': 0}, 'passed'),
])
def test_maximum_probability_keeps_real_unknown_and_uses_distribution(probabilities, expected):
    judge = RuntimeDecisionJudge(CONFIG, selection_policy='maximum_probability')
    try:
        # An unrelated model-written label is not the selected label.
        result = judge._choose(answer(probabilities, label='passed'), OPTIONS)
        assert result['label'] == expected
        assert result['label'] == result['top_label']
    finally:
        judge.close()


@pytest.mark.parametrize('bad', [
    None, [], {}, {'type': 'choice'}, {'probabilities': PROBABILITIES},
    answer(error='failed'), answer(type='score'),
    answer({'passed': 1}), answer({**PROBABILITIES, 'extra': 0}),
    answer({**PROBABILITIES, 'passed': '0.51'}),
    answer({**PROBABILITIES, 'passed': True}),
    answer({**PROBABILITIES, 'passed': float('nan')}),
    answer({**PROBABILITIES, 'passed': float('inf')}),
    answer({**PROBABILITIES, 'passed': -1}),
    answer({**PROBABILITIES, 'passed': 1.1}),
    answer({**PROBABILITIES, 'failed': .1}),
    answer(choice='failed'), answer(confidence='high'), answer(confidence=1.1),
])
@pytest.mark.parametrize('policy', ['threshold', 'maximum_probability'])
def test_invalid_schema_or_distribution_cannot_be_promoted(bad, policy):
    judge = RuntimeDecisionJudge(CONFIG, selection_policy=policy)
    try:
        with pytest.raises(ValueError):
            judge._choose(bad, OPTIONS)
    finally:
        judge.close()


@pytest.mark.parametrize('bad', [None, '', 'winner', 1, [], {}])
def test_unknown_selection_policy_is_rejected(bad):
    with pytest.raises(ValueError, match='selection_policy'):
        RuntimeDecisionJudge(CONFIG, selection_policy=bad)


@pytest.mark.parametrize('payload,expected', [
    (answer(), 'passed'),
    (answer({'passed': .15, 'failed': .1, 'pending': .1, 'unknown': .65}), 'unknown'),
])
def test_actual_typed_endpoint_path_uses_runtime_selection(payload, expected):
    def transport(request):
        body = json.loads(request.content)
        assert body['questions']['result']['criteria'] == OPTIONS
        return httpx.Response(200, json={'answers': {'result': payload}})
    judge = RuntimeDecisionJudge(CONFIG, httpx.MockTransport(transport), selection_policy='maximum_probability')
    try:
        result = judge.decide({'task': 'check'}, {'result': {
            'type': 'choice', 'criteria': OPTIONS, 'instructions': 'Check the evidence.'}})
        assert result['status'] == 'answered'
        assert result['answers']['result']['label'] == expected
        assert result['answers']['result']['selection_policy'] == 'maximum_probability'
        assert result['policy_trainable'] is False
    finally:
        judge.close()


def test_invalid_endpoint_response_remains_unknown_under_maximum_probability():
    judge = RuntimeDecisionJudge(CONFIG, httpx.MockTransport(lambda request: httpx.Response(
        200, json={'answers': {'result': answer({'passed': 1})}})), selection_policy='maximum_probability')
    try:
        result = judge.decide({}, {'result': {
            'type': 'choice', 'criteria': OPTIONS, 'instructions': 'Check the evidence.'}})
        assert result['status'] == 'unknown' and result['answers'] == {}
        assert result['error_type'] == 'ValueError'
    finally:
        judge.close()
