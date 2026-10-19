"""Repeated browser behavior and late UI changes around input/Enter."""
import pytest

from test_browser_live import live_browser
from test_browser_interaction_live import load_fixture, page_eval
from test_type_submit_live import PAGE, execute
from test_type_submit import search_action


@pytest.mark.parametrize('query', ['roboclaw', '机器人 长期规划', 'Attention Is All You Need', 'a "quoted" & query'])
@pytest.mark.parametrize('input_type', ['text', 'search'])
def test_repeated_search_overwrites_and_submits_once_per_decision(live_browser, query, input_type):
    session, _ = live_browser
    load_fixture(session, PAGE.replace('type=search', 'type=' + input_type))
    for repeat in range(3):
        page_eval(session, "query.value='old query'")
        result = execute(session, search_action(text=query))
        assert result['status'] == 'executed', result
        assert page_eval(session, '[submits,enters,sent]') == [repeat + 1, repeat + 1, query]


@pytest.mark.parametrize('mutation', ["other.focus()", "query.value='replaced'", "query.disabled=true"])
def test_late_change_during_tab_bookkeeping_cannot_receive_enter(live_browser, monkeypatch, mutation):
    session, _ = live_browser
    load_fixture(session, PAGE)
    arm = session._arm_tab_follow
    def delayed_arm(browser, page, action):
        arm(browser, page, action)
        page.evaluate(mutation)  # Emulates an async handler during CDP bookkeeping.
    monkeypatch.setattr(session, '_arm_tab_follow', delayed_arm)
    result = execute(session, search_action())
    assert result['status'] != 'executed', result
    assert page_eval(session, '[submits,enters]') == [0, 0]
    assert session._pending_tab_follow is None
    assert result['compound_action']['primitives'][-1]['dispatched'] is False


def test_ref_description_mismatch_gives_actionable_feedback(live_browser):
    from test_browser_live import observe
    session, _ = live_browser
    load_fixture(session, PAGE)
    obs = observe(session)
    ref = session.resolve_id('query')
    action = search_action(element_id='', target_ref=ref, target='当前搜索框')
    route = session.route(action, obs)
    assert route['channel'] == 'reobserve'
    assert route['error_code'] == 'target_mismatch'
    assert 'Search' in route['reason'] and 'query' in route['reason']
    assert page_eval(session, '[submits,enters,query.value]') == [0, 0, '']
    action.args['target'] = 'Search'
    result = session.execute(action, session.route(action, obs))
    assert result['status'] == 'executed', result
    assert page_eval(session, '[submits,enters]') == [1, 1]


def test_focus_change_while_clearing_does_not_type_into_other_field(live_browser):
    session, _ = live_browser
    load_fixture(session, PAGE)
    page_eval(session, "query.value='old';query.addEventListener('input',()=>{if(!query.value)other.focus()})")
    result = execute(session, search_action())
    assert result['status'] != 'executed'
    assert page_eval(session, '[submits,enters,other.value]') == [0, 0, '']


def test_prevented_submit_is_not_automatically_replayed(live_browser):
    session, _ = live_browser
    load_fixture(session, PAGE)
    page_eval(session, "query.addEventListener('keydown',e=>{if(e.key==='Enter')e.preventDefault()})")
    result = execute(session, search_action())
    assert result['status'] == 'executed'  # dispatched input is not task success
    assert page_eval(session, '[submits,enters]') == [0, 1]


def test_new_tab_submission_happens_once(live_browser):
    session, base = live_browser
    load_fixture(session, '''<form action="''' + base + '''/child" target=_blank>
      <input id=query type=search aria-label=Search name=q><button>Search</button></form>''')
    result = execute(session, search_action())
    assert result['status'] == 'executed', result
    from test_browser_live import observe
    obs = observe(session)
    assert len(obs.info['tabs']) == 2
    assert '/child?q=roboclaw' in obs.info['url']
