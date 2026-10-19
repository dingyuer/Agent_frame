"""Opt-in real Chromium fixtures for one-click hit tests and scroll effects."""
from osworld_agent.actions import Action
from osworld_agent.browser_session import browser_thread
from test_browser_live import live_browser, observe, click
import pytest


def page_eval(session, script):
    return browser_thread(lambda owner: owner._pick_page(owner._connect()[1]).evaluate(script))(session)


def load_fixture(session, html):
    browser_thread(lambda owner: owner._pick_page(owner._connect()[1]).set_content(html))(session)


def dom_scroll(session, element_id, amount=1, direction='down'):
    observation = observe(session)
    action = Action('scroll', {'element_id': element_id, 'amount': amount, 'direction': direction})
    route = session.route(action, observation)
    assert route and route['channel'] == 'browser_dom'
    return session.execute(action, route)


def test_live_card_click_skips_covered_title_and_independent_center_button(live_browser):
    session, _ = live_browser
    load_fixture(session, """<style>
      body{margin:20px} #card{position:relative;display:block;width:400px;height:280px;background:#ddd}
      h2{margin:0;padding:15px} #save{position:absolute;left:160px;top:110px;width:80px;height:60px}
      #cover{position:fixed;left:20px;top:20px;width:400px;height:85px;background:white;z-index:5}
      </style><script>window.counts={card:0,save:0,all:0};document.addEventListener('click',()=>counts.all++)</script>
      <div id=card role=link tabindex=0 aria-label="Open solution" onclick="counts.card++">
      <h2>Solution title</h2><button id=save onclick="event.stopPropagation();counts.save++">Save</button></div>
      <div id=cover>Partial overlay</div>""")
    result = click(session, 'card')
    assert result['status'] == 'executed', result
    assert result['actionability']['candidates_checked'] > 1
    assert page_eval(session, 'window.counts') == {'card': 1, 'save': 0, 'all': 1}


def test_live_full_cover_refuses_click_without_dismissing_or_repeating(live_browser):
    session, _ = live_browser
    load_fixture(session, """<script>window.counts={click:0,key:0};
      document.addEventListener('click',()=>counts.click++);document.addEventListener('keydown',()=>counts.key++)</script>
      <button id=target style="width:300px;height:200px">Target</button>
      <div style="position:fixed;inset:0;background:white;z-index:9">Overlay</div>""")
    result = click(session, 'target')
    assert result['status'] == 'rejected' and result['dispatched'] is False
    assert page_eval(session, 'window.counts') == {'click': 0, 'key': 0}


def test_live_card_cannot_click_a_nested_button_covering_all_of_it(live_browser):
    session, _ = live_browser
    load_fixture(session, """<script>window.counts=0</script>
      <div id=card role=link tabindex=0 aria-label="Card" style="position:relative;width:300px;height:200px">
      <button style="position:absolute;inset:0;width:100%;height:100%" onclick="counts++">Nested action</button></div>""")
    result = click(session, 'card')
    assert result['status'] == 'rejected' and result['dispatched'] is False
    assert page_eval(session, 'window.counts') == 0


@pytest.mark.parametrize('full_cover', [False, True])
def test_live_plain_div_registered_click_handler_is_an_independent_target(live_browser, full_cover):
    session, _ = live_browser
    size = 'inset:0;width:100%;height:100%' if full_cover else 'top:0;left:0;width:150px;height:60px'
    load_fixture(session, """<script>window.counts={card:0,child:0,all:0};document.addEventListener('click',()=>counts.all++)</script>
      <div id=card role=link tabindex=0 aria-label="Card" onclick="counts.card++"
        style="position:relative;width:400px;height:240px;background:#ddd">
      <div id=child style="position:absolute;""" + size + """">Child action text</div></div>
      <script>document.getElementById('child').addEventListener('click',e=>{e.stopPropagation();counts.child++})</script>""")
    result = click(session, 'card')
    counts = page_eval(session, 'window.counts')
    assert counts['child'] == 0, result
    if full_cover:
        assert result['dispatched'] is False and counts == {'card': 0, 'child': 0, 'all': 0}
    else:
        assert result['status'] == 'executed' and counts == {'card': 1, 'child': 0, 'all': 1}


def test_live_compound_target_is_rejected_when_listener_scan_is_unavailable(live_browser, monkeypatch):
    session, _ = live_browser
    load_fixture(session, """<script>window.counts=0</script><div id=card role=link tabindex=0
      aria-label="Card" onclick="counts++" style="width:300px;height:200px"><div>Plain child</div></div>""")
    monkeypatch.setattr(session, '_nested_listener_objects', lambda *args: (False, []))
    result = click(session, 'card')
    assert result['dispatched'] is False and 'listener_inspection_unavailable' in result['error']
    assert page_eval(session, 'window.counts') == 0


SCROLL_PAGE = """<style>body{margin:20px;height:2200px}#inner{width:350px;height:200px;overflow-y:scroll;overscroll-behavior:contain}
  #content{height:900px;background:linear-gradient(white,#aaa)}</style>
  <div id=inner role=region aria-label="Results"><div id=content>Scrollable results</div></div>"""


def test_live_dom_inner_scroll_reports_movement_and_bottom_noop(live_browser):
    session, _ = live_browser
    load_fixture(session, SCROLL_PAGE)
    result = dom_scroll(session, 'inner')
    feedback = result['scroll_feedback']
    assert result['status'] == 'executed' and feedback['available'] is True, result
    assert feedback['target_container']['id'] == 'inner'
    assert feedback['requested_delta'] == 120 and feedback['actual_delta'] > 0
    assert feedback['moved'] is True and feedback['noop'] is False
    page_eval(session, "document.getElementById('inner').scrollTop=700")
    bottom = dom_scroll(session, 'inner')
    assert bottom['scroll_feedback']['noop'] is True, bottom
    assert bottom['scroll_feedback']['at_bottom'] is True
    assert page_eval(session, 'window.scrollY') == 0


def test_live_visual_inner_scroll_keeps_pointer_wheel_semantics(live_browser):
    session, _ = live_browser
    load_fixture(session, SCROLL_PAGE)
    observation = observe(session)
    result = session.execute_viewport_action(Action('scroll', {'x': 100, 'y': 100, 'amount': 1, 'direction': 'down'}), observation.info)
    feedback = result['scroll_feedback']
    assert feedback['requested_delta'] == 300 and feedback['actual_delta'] > 0, result
    assert feedback['target_container']['id'] == 'inner'
    assert page_eval(session, 'window.scrollY') == 0


def test_live_page_scroll_reports_actual_delta_and_top_boundary(live_browser):
    session, _ = live_browser
    load_fixture(session, SCROLL_PAGE)
    observation = observe(session)
    result = session.execute_viewport_action(Action('scroll', {'amount': 1, 'direction': 'down'}), observation.info)
    feedback = result['scroll_feedback']
    assert feedback['target_container']['kind'] == 'page'
    assert feedback['actual_delta'] == 300 and feedback['requested_delta'] == 300
    page_eval(session, 'window.scrollTo(0,0)')
    top = session._scroll(Action('scroll', {'amount': 1, 'direction': 'up'}))
    assert top['scroll_feedback']['at_top'] is True and top['scroll_feedback']['noop'] is True


def test_live_smooth_page_scroll_is_observed_without_replaying_scrollby(live_browser):
    session, _ = live_browser
    load_fixture(session, """<style>html{scroll-behavior:smooth}body{height:3000px}</style>
      <p>Smooth scroll fixture</p><script>window.scrollCalls=0;const original=window.scrollBy.bind(window);
      window.scrollBy=(...args)=>{scrollCalls++;return original(...args)}</script>""")
    observation = observe(session)
    result = session.execute_viewport_action(Action('scroll', {'amount': 1, 'direction': 'down'}), observation.info)
    feedback = result['scroll_feedback']
    assert feedback['noop'] is not True, result
    assert feedback['moved'] is True or feedback['available'] is False
    assert page_eval(session, 'window.scrollCalls') == 1
