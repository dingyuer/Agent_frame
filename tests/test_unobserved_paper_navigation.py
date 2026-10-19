"""Unbound paper identification cannot bypass observed search provenance."""
import json

import pytest

from osworld_agent.actions import Action
from osworld_agent.browser_feedback import search_navigation_error
from osworld_agent.env import Observation
from osworld_agent.execution_queue import ExecutionQueue
from osworld_agent.pipeline import Pipeline, StepState
from osworld_agent.verifiers import CompletionVerifier
from test_pipeline_completion import Env, Model
from test_paper_resolver import IDENTIFIER, resolve


TASK = '搜索论文Attention Is All You Need并下载'
GUESSED_URL = 'https://arxiv.org/abs/1706.03762'  # The regression action, never a runtime hint.


def observed_search(link=None):
    nodes = [{'ref': '1', 'role': 'textbox', 'id': 'query', 'name': 'Search term or terms',
              'value': 'Attention Is All You Need', 'direct': True},
             {'ref': '2', 'role': 'link', 'name': 'arXiv:9912.99999',
              'href': 'https://arxiv.org/abs/9912.99999', 'direct': True}]
    if link:
        nodes.append({'ref': '3', 'role': 'link', 'name': 'Observed candidate', 'href': link, 'direct': True})
    return Observation(context=[{'kind': 'browser_ax', 'snapshot_id': 'results',
        'url': 'https://arxiv.org/search/?query=Attention+Is+All+You+Need', 'nodes': nodes}],
        info={'active_surface': 'browser', 'screenshot_space': 'css_viewport'})


@pytest.mark.parametrize('phase', ['search', 'identify', 'locate_fulltext', 'download', 'verify', ''])
def test_unseen_familiar_paper_url_is_rejected_in_all_unbound_phases(phase):
    verifier = CompletionVerifier(TASK)
    assert search_navigation_error(Action('goto', {'url': GUESSED_URL}), TASK,
                                   {'phase': phase}, observed_search(), verifier)


def test_actual_observed_link_and_user_supplied_id_remain_usable():
    action = Action('goto', {'url': GUESSED_URL})
    verifier = CompletionVerifier(TASK)
    assert not search_navigation_error(action, TASK, {'phase': 'identify'}, observed_search(GUESSED_URL), verifier)
    assert not search_navigation_error(action, TASK + '，arXiv ID为1706.03762',
                                       {'phase': 'identify'}, observed_search(), verifier)
    assert not search_navigation_error(Action('goto', {'url': 'https://arxiv.org/search/?query=Attention'}),
                                       TASK, {'phase': 'identify'}, observed_search(), verifier)


def test_trusted_binding_allows_its_official_detail_and_pdf_when_links_are_not_visible():
    task = '搜索论文Roboclaw并下载'
    verifier = CompletionVerifier(task)
    verifier.bind_paper('check_1', resolve().binding)
    for kind in ('abs', 'pdf'):
        assert not search_navigation_error(Action('goto', {'url': f'https://arxiv.org/{kind}/{IDENTIFIER}'}),
                                           task, {'phase': 'identify'}, observed_search(), verifier)


def test_generic_task_is_not_restricted_by_paper_navigation_guard():
    task = '打开一个页面并查看内容'
    assert not search_navigation_error(Action('goto', {'url': GUESSED_URL}), task,
                                       {'phase': 'identify'}, observed_search(), CompletionVerifier(task))


def test_identification_phase_repairs_guess_before_any_dispatch(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    model = Model([])
    valid = 'https://arxiv.org/search/?query=Attention+Is+All+You+Need&order=-announced_date'
    model.responses = [json.dumps({'next_action': {'type': 'goto', 'url': GUESSED_URL}}),
                       json.dumps({'next_action': {'type': 'goto', 'url': valid}})]
    env = Env()
    pipe = Pipeline(model=model, env=env, grounding=None, system_prompt='test')
    pipe.completion = CompletionVerifier(TASK)
    state = StepState(task=TASK, obs=observed_search(), step_id=3)
    state.queue = ExecutionQueue({'steps': [{'phase': 'identify', 'goal': '从搜索结果核实目标论文身份',
        'expected_evidence': '官方论文元数据与原标题相符', 'check_ids': []}]})
    decision = pipe.decide([{'role': 'user', 'content': '根据当前搜索结果选择下一动作'}], state)
    assert decision.action.action == 'goto' and decision.action.args['url'] == valid
    assert len(decision.calls) == 2
    assert decision.calls[0]['validation_error_type'] == 'unobserved_search_result'
    assert decision.calls[0]['valid_action'] is False
    assert decision.calls[1]['valid_action'] is True
    assert state.step_id == 3 and not env.actions


def test_ground_rejects_the_same_guess_when_decide_is_bypassed():
    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt='test')
    pipe.completion = CompletionVerifier(TASK)
    state = StepState(task=TASK, obs=observed_search())
    state.queue = ExecutionQueue({'steps': [{'phase': 'identify', 'goal': '核实论文', 'expected_evidence': '身份匹配'}]})
    pipe.ground(Action('goto', {'url': GUESSED_URL}), state)
    assert state.route['channel'] == 'reobserve'
    assert state.route['error_code'] == 'unobserved_search_result'
    assert not pipe.env.actions
