"""Planner spelling compatibility must preserve predicate meaning and raw data."""
import copy
import json

import pytest

from osworld_agent.execution_queue import normalize_verification
from osworld_agent.step_verifier import code_review
from test_pipeline_completion import Model, Env, observation, pipeline
from test_pipeline_training import paper_plan


@pytest.fixture(autouse=True)
def no_viz(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)


@pytest.mark.parametrize('kind,key,value', [
    ('url_host', 'host', 'scholar.google.com'),
    ('url_equals', 'url', 'https://arxiv.org/'),
    ('text_present', 'text', 'roboclaw'),
    ('app_open', 'app', 'notepad'),
])
def test_scalar_alias_is_equivalent_to_canonical_field_without_mutation(kind, key, value):
    raw = [{'kind': kind, 'value': ' ' + value + ' '}]
    before = copy.deepcopy(raw)
    canonical = [{'kind': kind, key: value}]
    assert normalize_verification(raw) == normalize_verification(canonical) == canonical
    assert raw == before
    assert normalize_verification([{'kind': kind, key: value, 'value': value}]) == canonical


@pytest.mark.parametrize('item', [
    {'kind': 'url_host', 'host': 'arxiv.org', 'value': 'scholar.google.com'},
    {'kind': 'url_host', 'host': '', 'value': 'arxiv.org'},
    {'kind': 'url_host', 'value': 'https://arxiv.org/'},
    {'kind': 'url_equals', 'value': 'javascript:alert(1)'},
    {'kind': 'text_present', 'value': ''},
    {'kind': 'text_present', 'value': ' '},
    {'kind': 'text_present', 'value': 'a' * 1001},
    {'kind': 'text_present', 'value': True},
    {'kind': 'text_present', 'value': 123},
    {'kind': 'text_present', 'value': {'text': 'roboclaw'}},
    {'kind': 'text_present', 'text': 'roboclaw', 'value': None},
    {'kind': 'element_state', 'value': 'Terms', 'states': {'checked': True}},
    {'kind': 'element_state', 'name': 'Terms', 'value': 'confirmed', 'states': {'checked': True}},
])
def test_ambiguous_conflicting_and_invalid_aliases_are_rejected(item):
    with pytest.raises(ValueError):
        normalize_verification([item])


def test_alias_does_not_turn_a_different_host_or_partial_title_into_success():
    host = {'verification': normalize_verification([{'kind': 'url_host', 'value': 'arxiv.org'}])}
    assert code_review(host, observation('https://arxiv.org.attacker.invalid'), {}, {})['status'] == 'failed'
    title = {'verification': normalize_verification([{'kind': 'text_present', 'value': 'roboclaw'}])}
    obs = observation()
    obs.context[0]['nodes'][0]['name'] = 'RoboClaw: An Agentic Framework'
    assert code_review(title, obs, {}, {})['status'] == 'unknown'


def test_reported_paper_task_plan_with_value_reaches_execution_on_first_attempt():
    plan = paper_plan()
    for step in plan['steps']:
        step['goal'] = step['goal'].replace('AAA', 'roboclaw')
    plan['steps'][0].update(goal='打开Google Scholar首页', expected_evidence='当前页面域名为scholar.google.com')
    plan['steps'][0]['verification'] = [{'kind': 'url_host', 'value': 'scholar.google.com'}]
    plan['steps'][1]['verification'] = [{'kind': 'text_present', 'value': 'roboclaw'}]
    plan['steps'].insert(-1, {'phase': 'download', 'goal': '改名为data.pdf', 'expected_evidence': '文件名称为data.pdf'})
    plan['keywords'] = ['roboclaw']
    model = Model([{'type': 'goto', 'url': 'https://scholar.google.com/'}], plan)
    sampled_plan = model.responses[0]
    env = Env()
    result = pipeline(model, env, steps=1).run('搜索论文roboclaw并下载，然后改名为data')
    assert result.termination_reason != 'planning_failed'
    assert len(result.planning_calls) == 1 and result.planning_calls[0]['valid_plan']
    assert result.planning_calls[0]['response'] == sampled_plan
    assert json.loads(sampled_plan)['steps'][0]['verification'][0]['value'] == 'scholar.google.com'
    assert result.plan['steps'][0]['verification'] == [{'kind': 'url_host', 'host': 'scholar.google.com'}]
    assert result.plan['steps'][1]['verification'] == [{'kind': 'text_present', 'text': 'roboclaw'}]
    assert env.reset_count == 1 and len(env.actions) == 1
    assert result.trajectory[0]['step_verification']['status'] == 'passed'
    assert not result.success  # Passing a navigation milestone is not a downloaded, renamed paper.


def test_conflict_repair_names_the_step_and_never_executes_bad_plan():
    plan = {'steps': [{'goal': '打开目标', 'expected_evidence': '目标域名', 'verification': [
        {'kind': 'url_host', 'host': 'arxiv.org', 'value': 'scholar.google.com'}]}]}
    model = Model([], plan)
    model.responses *= 2
    env = Env()
    result = pipeline(model, env, steps=1).run('打开目标并继续任务')
    assert result.termination_reason == 'planning_failed' and env.reset_count == 0 and not env.actions
    assert '第 1 个子目标 verification' in result.planning_calls[0]['validation_error']
    assert '冲突' in model.calls[1][-1]['content']
