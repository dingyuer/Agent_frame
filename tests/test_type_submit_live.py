"""Real Chromium events for bounded input-and-submit; no remote models."""
import pytest

from test_browser_live import live_browser, observe
from test_browser_interaction_live import load_fixture, page_eval
from test_type_submit import search_action


PAGE = '''<script>window.submits=0;window.enters=0;window.sent='';
 document.addEventListener('keydown',e=>{if(e.key==='Enter')enters++})</script>
 <form onsubmit="event.preventDefault();submits++;sent=query.value;document.getElementById('result').textContent='Results: '+sent">
 <input id=query aria-label=Search type=search><input id=other aria-label=Other>
 <button>Search</button></form><p id=result></p>'''


def execute(session, action):
    obs = observe(session)
    route = session.route(action, obs)
    assert route and route['channel'] == 'browser_dom'
    return session.execute(action, route)


def test_search_types_and_submits_exactly_once(live_browser):
    session, _ = live_browser
    load_fixture(session, PAGE)
    result = execute(session, search_action())
    assert result['status'] == 'executed', result
    assert page_eval(session, '[submits,enters,sent]') == [1, 1, 'roboclaw']
    primitives = result['compound_action']['primitives']
    assert [p['type'] for p in primitives] == ['type', 'press']
    assert all(p['status'] == 'executed' for p in primitives)


def test_plain_type_never_implicitly_submits(live_browser):
    session, _ = live_browser
    load_fixture(session, PAGE)
    action = search_action()
    action.args.pop('submit_key')
    result = execute(session, action)
    assert result['status'] == 'executed'
    assert page_eval(session, '[submits,enters,query.value]') == [0, 0, 'roboclaw']


@pytest.mark.parametrize('handler', ["other.focus()", "query.value='changed'", "query.remove()",
                                   "query.setAttribute('aria-expanded','true')"])
def test_input_handler_changes_block_enter(live_browser, handler):
    session, _ = live_browser
    load_fixture(session, PAGE)
    page_eval(session, "query.addEventListener('input',()=>{if(query.value==='roboclaw'){" + handler + "}})")
    result = execute(session, search_action())
    assert result['dispatched'] is True and result['status'] != 'executed', result
    assert page_eval(session, '[submits,enters]') == [0, 0]
    assert result['compound_action']['primitives'][0]['status'] == 'executed'
    assert result['compound_action']['primitives'][1]['dispatched'] is False


@pytest.mark.parametrize('element', ['<textarea id=query aria-label=Search></textarea>',
                                    '<input id=query aria-label=Search type=password>'])
def test_non_search_input_rejects_before_typing(live_browser, element):
    session, _ = live_browser
    load_fixture(session, element)
    result = execute(session, search_action())
    assert result['dispatched'] is False and result['status'] == 'rejected', result
    assert page_eval(session, 'query.value') == ''
