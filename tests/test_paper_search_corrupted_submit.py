"""Damaged search inputs are repaired before dispatching a form submission."""
import pytest

from osworld_agent.actions import Action
from osworld_agent.paper_search import PaperSearchRecovery
from test_paper_search_recovery import obs, type_query


SUBMISSIONS = [Action('press', {'key': 'Enter'}),
               Action('click', {'target': 'Search', 'target_ref': 'submit'})]


@pytest.mark.parametrize('engine', ['scholar', 'arxiv'])
@pytest.mark.parametrize('repeated', ['WorldGUIWorldGUI', 'WorldGUIWorldGUIWorldGUI', '"WorldGUI""WorldGUI"'])
@pytest.mark.parametrize('submission', SUBMISSIONS)
def test_corrupted_query_cannot_be_submitted_by_search_or_focused_enter(engine, repeated, submission):
    recovery = PaperSearchRecovery()
    page = obs(value=repeated, engine=engine)
    facts = ' '.join(recovery.facts(page, 'WorldGUI'))
    assert '重复追加' in facts and 'overwrite:true' in facts
    error = recovery.action_error(submission, page, 'WorldGUI')
    assert '重复追加' in error and '不能提交' in error and 'overwrite:true' in error
    # The prescribed input repair remains available even if the original query
    # already has an observed no-match result in this recovery object's history.
    assert not recovery.action_error(type_query('WorldGUI', overwrite=True), page, 'WorldGUI')


@pytest.mark.parametrize('engine', ['scholar', 'arxiv'])
@pytest.mark.parametrize('submission', SUBMISSIONS)
def test_legitimate_expanded_query_can_be_submitted(engine, submission):
    recovery = PaperSearchRecovery()
    page = obs(value='WorldGUI paper', engine=engine)
    recovery.facts(page, 'WorldGUI')
    assert not recovery.action_error(submission, page, 'WorldGUI')


def test_enter_on_a_result_link_is_not_treated_as_corrupted_form_submission():
    page = obs(value='WorldGUIWorldGUI')
    page.context[0]['nodes'][0]['states']['focused'] = False
    assert not PaperSearchRecovery().action_error(Action('press', {'key': 'Enter'}), page, 'WorldGUI')


def test_result_inspection_remains_available_with_a_corrupted_search_field():
    page = obs(value='WorldGUIWorldGUI', match=True)
    action = Action('click', {'target': 'WorldGUI: A Desktop Benchmark', 'target_ref': 'result'})
    assert not PaperSearchRecovery().action_error(action, page, 'WorldGUI')


def test_clearing_corruption_does_not_reenable_an_exhausted_submission():
    recovery = PaperSearchRecovery()
    recovery.facts(obs(value='WorldGUIWorldGUI'), 'WorldGUI')
    assert not recovery.action_error(type_query(), obs(value='WorldGUIWorldGUI'), 'WorldGUI')
    assert recovery.action_error(SUBMISSIONS[0], obs(value='WorldGUI'), 'WorldGUI')
