"""One decision may explicitly type then Enter; neither primitive proves success."""
import pytest

from osworld_agent.actions import Action, validate_action
from osworld_agent.judge_evidence import effect_review
from osworld_agent.pipeline import Pipeline, StepState
from test_pipeline_completion import Env, Model, observation


def search_action(**overrides):
    return Action('type', {'target': 'Search', 'element_id': 'query', 'text': 'roboclaw',
                           'overwrite': True, 'submit_key': 'Enter', **overrides})


@pytest.mark.parametrize('args', [{'submit_key': 'Tab'}, {'overwrite': False}, {'element_id': ''}])
def test_compound_input_requires_exact_key_overwrite_and_dom_reference(args):
    assert validate_action(search_action(**args))


def test_compound_action_never_falls_back_to_gui(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt='test')
    monkeypatch.setattr(pipe.env, 'route_action', lambda *a: None)
    monkeypatch.setattr(pipe, '_resolve', lambda *a: pytest.fail('no visual compound fallback'))
    state = StepState(task='search', obs=observation())
    action = pipe.ground(search_action(), state)
    result = pipe.execute(action, state)
    assert result.info['dispatched'] is False
    assert not pipe.env.actions


def test_text_value_alone_does_not_verify_combined_search():
    before, after = observation(value=''), observation(value='roboclaw')
    execution = {'status': 'executed', 'dispatched': True, 'action': 'type',
                 'args': {'target_ref': '1', 'text': 'roboclaw', 'submit_key': 'Enter'}}
    assert effect_review(before, after, execution)['status'] == 'unknown'
    execution['args'].pop('submit_key')
    assert effect_review(before, after, execution)['status'] == 'passed'


def test_invalid_reference_feedback_is_preserved_for_compound_input(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt='test')
    state = StepState(task='search', obs=observation(), route={'channel': 'reobserve',
                      'reason': "target_ref '#query' 无效，请使用第一列 ref 5", 'error_code': 'stale'})
    result = pipe.execute(search_action(), state)
    assert result.info['dispatched'] is False
    assert '第一列 ref 5' in result.info['reason']


def test_compound_input_cannot_resubmit_an_exhausted_paper_query():
    from osworld_agent.paper_search import PaperSearchRecovery
    from test_paper_search_recovery import obs, type_query
    recovery = PaperSearchRecovery()
    recovery.facts(obs(), 'WorldGUI')
    action = type_query()
    assert not recovery.action_error(action, obs(value='WorldGUIWorldGUI'), 'WorldGUI')
    action.args['submit_key'] = 'Enter'
    assert recovery.action_error(action, obs(value='WorldGUIWorldGUI'), 'WorldGUI')
