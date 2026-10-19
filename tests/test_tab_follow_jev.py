"""Jev chooses only among browser-observed tabs after an ambiguous click."""
from unittest.mock import Mock

from osworld_agent.actions import Action
from osworld_agent.env import Observation, StepResult
from osworld_agent.pipeline import Execution, Pipeline, StepState


def test_ambiguous_popup_selection_is_scoped_to_real_tab_ids():
    candidates = [{'tab_id': 'ad', 'title': 'Advertisement', 'url': 'https://ads.test/'},
                  {'tab_id': 'detail', 'title': 'Solution detail', 'url': 'https://portal.test/detail'}]
    before = Observation(info={'url': 'https://portal.test/library'})
    after = Observation(info={'tab_follow_candidates': {'from_tab_id': 'parent', 'candidates': candidates}})
    selected = Observation(info={'url': 'https://portal.test/detail', 'page_id': 'detail'})
    judge = Mock()
    judge.last_call = {}
    judge.select_opened_tab.return_value = {'status': 'answered', 'answers': {'tab': {'label': 'tab_1'}}}
    pipe = Pipeline.__new__(Pipeline)
    pipe.decision_judge = judge
    pipe.env = Mock()
    pipe._task_deadline = None
    pipe._record_typed_judge = Mock()
    pipe._publish_diagnostic = Mock()
    pipe._step_environment = Mock(return_value=StepResult(observation=selected, info={'status': 'executed'}))
    result = pipe._follow_ambiguous_tabs(Action('click', {'target': 'Solution card'}),
        Execution(kind='browser_dom', obs=after, info={}), StepState(task='打开方案详情', obs=before))
    assert result.obs is selected
    assert result.info['tab_follow']['tab_id'] == 'detail'
    pipe._step_environment.assert_called_once()
    assert pipe._step_environment.call_args.args[0].args['tab_id'] == 'detail'


def test_ambiguous_popup_unknown_keeps_current_page():
    candidates = [{'tab_id': 'a'}, {'tab_id': 'b'}]
    pipe = Pipeline.__new__(Pipeline)
    pipe.decision_judge = Mock(last_call={})
    pipe.decision_judge.select_opened_tab.return_value = {
        'status': 'answered', 'answers': {'tab': {'label': 'unknown'}}}
    pipe._task_deadline = None
    pipe._record_typed_judge = Mock()
    pipe._step_environment = Mock()
    after = Observation(info={'tab_follow_candidates': {'candidates': candidates}})
    result = pipe._follow_ambiguous_tabs(Action('click', {'target': 'card'}),
        Execution(kind='browser_dom', obs=after, info={}), StepState(obs=Observation()))
    assert result.obs is after
    assert result.info['tab_follow']['status'] == 'ambiguous'
    pipe._step_environment.assert_not_called()
