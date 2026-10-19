"""Real Chrome separates prompt selection from the actual dispatch route."""
import json
import socket
import threading
import time

import pytest


def test_history_serializes_resolved_gui_coordinate():
    from osworld_agent.viz.server import _local_traj_to_frontend
    rows = _local_traj_to_frontend({'steps': [{
        'step': 1, 'action': {'action': 'click', 'args': {'target': '方案卡片'}},
        'operation_mode': 'gui',
        'resolved_action': '{"action":"click","args":{"x":310,"y":420,"coordinate_space":"css_viewport"}}',
        'grounding': {'target': '方案卡片', 'coord': [310, 420]},
    }]})
    assert rows[0]['grounding'][0]['coord'] == [310, 420]
    assert rows[0]['resolved_action']['args']['x'] == 310


def test_history_serializes_full_prompt_profile_and_accepts_legacy_rows():
    from osworld_agent.viz.server import _local_traj_to_frontend
    profile = {
        'mode': 'dom', 'surface': 'browser', 'source': 'jev',
        'reason': 'Jev selected DOM for the current browser view.',
        'jev': {'label': 'dom', 'probabilities': {'dom': .91, 'gui': .09}},
    }
    metadata = {'prompt_profile': profile, 'template': 'dom'}
    rows = _local_traj_to_frontend({'steps': [
        {'step': 1, 'prompt_metadata': metadata, 'operation_mode': 'gui'},
        {'step': 2},
    ]})
    assert rows[0]['prompt_metadata'] == metadata
    assert rows[0]['operation_mode'] == 'gui'
    assert rows[1]['prompt_metadata'] == {}


@pytest.fixture(scope='module')
def operation_page(request):
    if not request.config.getoption('--run-browser-integration'):
        pytest.skip('pass --run-browser-integration to exercise the Studio UI')
    import httpx
    import uvicorn
    from playwright.sync_api import sync_playwright, expect
    from osworld_agent.viz.server import app

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    runtime = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='error'))
    thread = threading.Thread(target=runtime.run, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{port}'
    try:
        with httpx.Client(trust_env=False) as client:
            for _ in range(100):
                try:
                    if client.get(base).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(.05)
            else:
                raise AssertionError('Studio did not start')
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(channel='chrome', headless=True)
            except Exception as exc:
                pytest.skip(f'system Chrome unavailable for UI integration: {exc}')
            page = browser.new_page(viewport={'width': 1280, 'height': 800})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(base)
            yield page
            assert not errors
            browser.close()
    finally:
        runtime.should_exit = True
        thread.join(timeout=10)


def test_operation_mode_badge_and_jev_detail(operation_page):
    from playwright.sync_api import expect
    page = operation_page
    page.evaluate("""() => {
              resetLiveView();
              switchView('live');
              addCard({kind:'status',event:'task_start',instruction:'搜索论文',max_steps:5,ts:1});
              addCard({kind:'status',event:'step',step:1,ts:1});
              addCard({kind:'model',type:'decision',step:1,response:JSON.stringify({next_action:{type:'click',target:'15'}}),ts:1});
              addCard({kind:'status',event:'route',step:1,operation_mode:'dom',ts:1});
              addCard({kind:'status',event:'step',step:2,ts:1});
              addCard({kind:'model',type:'decision',step:2,response:JSON.stringify({next_action:{type:'click',target:'PDF toolbar'}}),ts:1});
              addCard({kind:'status',event:'route',step:2,operation_mode:'gui',ts:1});
              addCard({kind:'model',type:'grounding',step:2,response:'(310, 420)',screenshot:'',ts:1});
              addCard({kind:'status',event:'jev_transition',step:2,transition:{answers:{settled:{label:'complete'},correct:{label:'yes'},next_mode:{label:'dom'},replan:{label:'no'}}},ts:1});
              addCard({kind:'status',event:'download_jev_poll',step:2,effect:'started',next:'wait',elapsed_s:1.0,ts:1});
            }""")
    expect(page.locator('.compact-step[data-step="1"] .operation-mode')).to_have_text('DOM')
    expect(page.locator('.compact-step[data-step="2"] .operation-mode')).to_have_text('GUI')
    assert '(310, 420)' in page.locator('.compact-step[data-step="2"] .compact-action').inner_text()
    page.locator('.compact-step[data-step="2"] .compact-step-more > summary').click()
    assert 'Jev 动作后核查' in page.locator('.compact-step[data-step="2"] .compact-step-details').inner_text()
    assert 'Jev 下载复查' in page.locator('.compact-step[data-step="2"] .compact-step-details').inner_text()
    assert page.locator('[data-event="jev_transition"]').count() == 0
    history = page.evaluate("localStepCard('fixture',{step_num:1,action:JSON.stringify({action:'click',args:{target:'15'}}),routing:{channel:'browser_dom'},prompt:'long AX prompt'})")
    assert 'DOM' in history and history.index('click(') < history.index('展开模型输入')
    assert '<details class="model-details">' in history
    assert 'GUI' in page.evaluate("localStepCard('fixture',{step_num:2,action:JSON.stringify({action:'click',args:{target:'toolbar'}}),routing:{}})")
    assert '(310, 420)' in page.evaluate("localStepCard('fixture',{step_num:2,action:JSON.stringify({action:'click',args:{target:'toolbar'}}),operation_mode:'gui',grounding:[{coord:[310,420]}]})")


def test_live_prompt_source_is_compact_and_independent_of_actual_route(operation_page):
    from playwright.sync_api import expect
    page = operation_page
    profiles = [
        {'mode': 'dom', 'surface': 'browser', 'source': 'jev',
         'reason': 'Jev selected DOM.', 'jev': {'label': 'dom', 'confidence': .94}},
        {'mode': 'gui', 'surface': 'desktop', 'source': 'fallback',
         'reason': 'Use GUI for the desktop.', 'fallback_reason': 'Jev timed out.'},
        {'mode': 'observe', 'surface': 'browser', 'source': 'capability',
         'reason': 'The current surface only supports observation.'},
        {'mode': 'dialog', 'surface': 'desktop', 'source': 'capability',
         'reason': 'A native dialog owns focus.'},
    ]
    page.evaluate("""profiles => {
      resetLiveView();
      switchView('live');
      addCard({kind:'status',event:'task_start',instruction:'选模来源测试',max_steps:5,ts:1});
      profiles.forEach((profile,index) => {
        const step = index + 1;
        addCard({kind:'status',event:'decision_mode',step,profile,ts:1});
        addCard({kind:'status',event:'decision_mode',step,profile,ts:1});
        addCard({kind:'model',type:'decision',step,response:JSON.stringify({next_action:{type:'click',target:'toolbar'}}),ts:1});
        addCard({kind:'status',event:'route',step,operation_mode:'gui',ts:1});
      });
    }""", profiles)
    labels = ['Prompt DOM · Jev 选模', 'Prompt GUI · 规则兜底',
              'Prompt 观察 · 能力约束', 'Prompt 对话 · 能力约束']
    for step, (profile, label) in enumerate(zip(profiles, labels), 1):
        card = page.locator(f'.compact-step[data-step="{step}"]')
        expect(card.locator('.operation-mode')).to_have_text('GUI')
        expect(card.locator('.prompt-profile')).to_have_text(label)
        title = card.locator('.prompt-profile').get_attribute('title')
        assert profile['reason'] in title
        if profile.get('fallback_reason'):
            assert profile['fallback_reason'] in title
        expect(card.locator('.prompt-profile-detail')).to_have_count(1)
        detail = card.locator('.prompt-profile-detail').text_content().split('\n', 1)[1]
        assert json.loads(detail) == profile
    assert page.locator('[data-event="decision_mode"]').count() == 0
    expect(page.locator('.compact-step')).to_have_count(4)


def test_history_prompt_profile_reason_and_legacy_compatibility(operation_page):
    from playwright.sync_api import expect
    page = operation_page
    profile = {'mode': 'dom', 'surface': 'browser', 'source': 'jev',
               'reason': 'Use "DOM" for <browser> & semantic targets.',
               'jev': {'label': 'dom', 'probabilities': {'dom': .92}}}
    rows = [
        {'step_num': 1, 'action': json.dumps({'action': 'click', 'args': {'target': 'toolbar'}}),
         'operation_mode': 'gui', 'prompt_metadata': {'prompt_profile': profile}},
        {'step_num': 2, 'action': json.dumps({'action': 'click', 'args': {'target': '15'}}),
         'routing': {'channel': 'browser_dom'}},
    ]
    page.evaluate("""rows => {
      document.querySelectorAll('[id^="view-"]').forEach(view => view.classList.remove('active'));
      document.getElementById('view-local').classList.add('active');
      renderLocalDetail('prompt-mode-fixture', rows, '历史选模来源', null);
    }""", rows)
    cards = page.locator('#local-body > .card').filter(has=page.locator('.operation-mode'))
    expect(cards).to_have_count(2)
    first = cards.nth(0)
    expect(first.locator('.operation-mode')).to_have_text('GUI')
    expect(first.locator('.prompt-profile')).to_have_text('Prompt DOM · Jev 选模')
    assert profile['reason'] in first.locator('.prompt-profile').get_attribute('title')
    assert first.locator('[onmouseover], [onfocus], script').count() == 0
    first.locator('details.model-details > summary').click()
    assert json.loads(first.locator('.prompt-profile-detail').inner_text()) == profile
    expect(cards.nth(1).locator('.operation-mode')).to_have_text('DOM')
    expect(cards.nth(1).locator('.prompt-profile')).to_have_count(0)

    page.evaluate("""rows => {
      document.querySelectorAll('[id^="view-"]').forEach(view => view.classList.remove('active'));
      document.getElementById('view-history').classList.add('active');
      renderRunDetail('fixture', 'browser', 'prompt-mode', rows, '', {}, '');
    }""", rows)
    badge = page.locator('#history-body .prompt-profile')
    expect(badge).to_have_count(1)
    expect(badge).to_have_text('Prompt DOM · Jev 选模')
    expect(badge.locator('xpath=..').locator('.operation-mode')).to_have_text('GUI')
    assert profile['reason'] in badge.get_attribute('title')
    assert json.loads(page.locator('#history-body .prompt-profile-detail').text_content()) == profile
