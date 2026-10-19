"""Real Studio rendering separates local effects from subtask completion."""
import socket
import threading
import time
import json
import re

import pytest


@pytest.fixture(scope="module")
def review_page(request):
    if not request.config.getoption("--run-browser-integration"):
        pytest.skip("pass --run-browser-integration to exercise Studio rendering")
    import httpx
    import uvicorn
    from playwright.sync_api import sync_playwright
    from osworld_agent.viz.server import app

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(trust_env=False) as client:
            for _ in range(100):
                try:
                    if client.get(base).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(.05)
            else:
                raise AssertionError("Studio did not start")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(base)
            yield page
            assert not errors
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def start_step(page, instruction="验证显示测试"):
    page.evaluate("""instruction => {
      addCard({kind:'status',event:'task_start',instruction,max_steps:5,ts:1});
      addCard({kind:'status',event:'step',step:1,ts:1});
    }""", instruction)


def test_live_local_effect_and_goal_completion_use_separate_labels(review_page):
    from playwright.sync_api import expect
    page = review_page
    start_step(page)
    page.evaluate("""() => addCard({kind:'status',event:'step_verification',step:1,
      method:'decision_model',status:'unknown',step_outcome:'progress',
      action_effect:{method:'code',status:'passed',reason:'输入框值已改变'},ts:1})""")
    card = page.locator('.compact-step[data-step="1"]').last
    expect(card.locator('.compact-step-effect')).to_have_text('动作已验证')
    expect(card.locator('.compact-step-status')).to_have_text('子任务进行中')
    assert '子任务已完成' not in card.locator('.compact-step-head').inner_text()

    page.evaluate("""() => addCard({kind:'status',event:'step_verification',step:1,
      method:'decision_model',status:'unknown',step_outcome:'progress',progress_hint:true,
      action_effect:{method:'code',status:'unknown'},ts:1})""")
    expect(card.locator('.compact-step-effect')).to_have_text('动作有进展倾向')
    expect(card.locator('.compact-step-status')).to_have_text('子任务进行中')
    assert 'passed' not in card.locator('.compact-step-effect').get_attribute('class')

    page.evaluate("""() => addCard({kind:'status',event:'step_verification',step:1,
      method:'code',status:'passed',action_effect:{method:'code',status:'unknown'},ts:1})""")
    expect(card.locator('.compact-step-status')).to_have_text('子任务已完成')
    expect(card.locator('.compact-step-effect')).to_have_text('动作待核查')


def test_transition_displays_effective_correct_and_retains_original_probability_labels(review_page):
    from playwright.sync_api import expect
    page = review_page
    start_step(page, "Jev有效结论与原始概率")
    page.evaluate("""() => addCard({kind:'status',event:'jev_transition',step:1,
      transition:{effective_correct:'yes',answers:{settled:{label:'complete'},
        correct:{label:'unknown',top_label:'yes',probabilities:{yes:.71,no:.19,unknown:.10}},
        next_mode:{label:'dom'},replan:{label:'no'}}},ts:1})""")
    card = page.locator('.compact-step[data-step="1"]').last
    expect(card.locator('.compact-step-effect')).to_have_text('动作已验证')
    assert '子任务已完成' not in card.locator('.compact-step-head').inner_text()
    card.locator('.compact-step-more').evaluate('element => element.open = true')
    detail = card.locator('.compact-step-details').inner_text()
    assert '操作正确：yes' in detail
    assert 'Jev 原始标签与概率' in detail and '"label": "unknown"' in detail and '0.71' in detail
    page.evaluate("""() => addCard({kind:'status',event:'jev_transition',step:1,
      transition:{answers:{correct:{label:'no'}}},ts:1})""")
    expect(card.locator('.compact-step-effect')).to_have_text('动作失败')
    assert '操作正确：no' in card.locator('.compact-step-details').inner_text()


def test_gui_dispatch_is_distinct_from_grounding_and_verified_effect(review_page):
    from playwright.sync_api import expect
    page = review_page
    start_step(page, 'GUI下载动作派发尚未产生文件')
    page.evaluate("""() => {
      addCard({kind:'model',type:'decision',step:1,
        response:JSON.stringify({next_action:{type:'click',target:'下载图标'}}),ts:1});
      addCard({kind:'status',event:'route',step:1,operation_mode:'gui',ts:1});
      addCard({kind:'model',type:'grounding',step:1,response:'(1330, 30)',ts:1});
    }""")
    card = page.locator('.compact-step[data-step="1"]').last
    assert card.locator('.dispatch-state').is_hidden()
    assert '(1330, 30)' in card.locator('.compact-action').inner_text()
    page.evaluate("""() => {
      addCard({kind:'status',event:'executed',step:1,status:'uncertain',dispatched:true,ts:1});
      addCard({kind:'status',event:'step_verification',step:1,status:'failed',
        action_effect:{status:'unknown'},ts:1});
    }""")
    expect(card.locator('.dispatch-state')).to_have_text('已派发')
    expect(card.locator('.compact-step-effect')).to_have_text('动作待核查')
    expect(card.locator('.compact-step-status')).to_have_text('子任务尚未完成')
    assert '动作已验证' not in card.locator('.compact-step-head').inner_text()
    page.evaluate("""() => addCard({kind:'status',event:'executed',step:1,
      status:'rejected',dispatched:false,ts:1})""")
    expect(card.locator('.dispatch-state')).to_have_text('未派发')


def test_history_maps_rejection_and_gui_dispatch_without_claiming_download_success(review_page):
    from osworld_agent.viz.server import _local_traj_to_frontend
    rows = _local_traj_to_frontend({'steps': [
        {'step': 3, 'action': {'action': 'click', 'args': {'target': 'View PDF'}},
         'operation_mode': 'blocked', 'routing': {'channel': 'reobserve'},
         'execution_info': {'status': 'rejected', 'dispatched': False,
                            'error_code': 'paper_identity_mismatch'},
         'step_reviews': [{'review_id': 'r3', 'decision_step': 3, 'status': 'failed',
                           'action_effect': {'status': 'failed'}}]},
        {'step': 10, 'action': {'action': 'click', 'args': {'target': '下载图标'}},
         'operation_mode': 'gui', 'grounding': {'target': '下载图标', 'coord': [1330, 30]},
         'execution_info': {'status': 'uncertain', 'dispatched': True, 'resolved': 'visual'},
         'step_reviews': [{'review_id': 'r10', 'decision_step': 10, 'status': 'failed',
                           'action_effect': {'status': 'unknown'}}]}
    ]})
    assert rows[0]['execution_info']['dispatched'] is False
    assert rows[1]['grounding'][0]['coord'] == [1330, 30]
    html = review_page.evaluate("rows => rows.map(row => localStepCard('fixture',row))", rows)
    assert '未执行' in html[0] and '未派发' in html[0]
    assert 'GUI' in html[1] and '(1330, 30)' in html[1] and '已派发' in html[1]
    assert '动作待核查' in html[1] and '子任务尚未完成' in html[1]
    assert '动作已验证' not in html[1]


def test_same_action_keeps_deterministic_effect_when_following_conditions_are_observation_only(review_page):
    from playwright.sync_api import expect
    page = review_page
    actual = {'review_id': 'a4:download', 'decision_step': 1, 'plan_step_id': 'download',
              'method': 'code', 'status': 'passed',
              'action_effect': {'method': 'code', 'status': 'passed',
                                'evidence': ['completed-browser-download-id']},
              'action_transition': {'effective_correct': 'yes',
                                    'answers': {'correct': {'label': 'no', 'probabilities': {'yes': .07, 'no': .93}}}}}
    old_observation = {'review_id': 'a4:verify', 'decision_step': 1, 'plan_step_id': 'verify',
                       'method': 'code', 'status': 'passed', 'reason': '全部文件验收条件已满足',
                       'action_effect': {'method': 'code', 'status': 'failed',
                                         'reason': '动作未成功派发或执行器报告失败', 'evidence': []}}
    start_step(page, '真实下载完成后的后续条件核查')
    for report in (actual, old_observation):
        page.evaluate("report => addCard({kind:'status',event:'step_verification',step:1,...report,ts:1})", report)
    card = page.locator('.compact-step[data-step="1"]').last
    expect(card.locator('.compact-step-effect')).to_have_text('动作已验证')
    expect(card.locator('.compact-step-status')).to_have_text('子任务已完成')
    result = page.evaluate("""reviews => {
      const step={step_num:1,step_reviews:reviews,step_verification:reviews[1]},
        original=JSON.stringify(step),holder=document.createElement('div');
      holder.innerHTML=localStepCard('archive',step);
      return {head:holder.querySelector('.card-head').textContent,html:holder.innerHTML,
        unchanged:JSON.stringify(step)===original};
    }""", [actual, old_observation])
    assert '动作已验证' in result['head'] and '动作失败' not in result['head']
    assert '子任务已完成' in result['head'] and '状态核查' in result['html']
    assert '"label": "no"' in result['html'] and '0.93' in result['html']
    assert result['unchanged'] is True


def test_explicit_framework_observation_has_no_action_failure_badge_without_model_call(review_page):
    page = review_page
    result = page.evaluate("""() => {
      const actual={review_id:'a3:identify',decision_step:3,plan_step_id:'identify',status:'passed',
        action_effect:{method:'code',status:'passed',evidence:['https://arxiv.org/abs/observed-id']}},
        observed={review_id:'a3:pdf',decision_step:3,plan_step_id:'pdf',status:'passed',
          action_scope:'framework_observation',action_effect:{method:'code',status:'not_applicable',scope:'observation'}},
        card=document.createElement('div'),only=document.createElement('div');
      card.innerHTML=localStepCard('new',{step_num:3,step_reviews:[actual,observed]});
      only.innerHTML=localStepCard('observation',{step_num:3,step_reviews:[observed]});
      return {head:card.querySelector('.card-head').textContent,html:card.innerHTML,
        observedEffects:only.querySelectorAll('.action-effect-state').length};
    }""")
    assert '动作已验证' in result['head'] and '子任务已完成' in result['head']
    assert '动作失败' not in result['html']
    assert result['observedEffects'] == 0


def test_history_uses_latest_relevant_review_and_removes_superseded_unknown_stages(review_page):
    page = review_page
    result = page.evaluate("""() => {
      const step={step_num:1,action:JSON.stringify({action:'click',args:{target:'链接'}}),
        assessment:{action_status:'failure'},step_reviews:[
          {review_id:'r1',decision_step:1,method:'code',status:'unknown',reason:'obsolete-code'},
          {review_id:'r1',decision_step:1,method:'decision_model',status:'unknown',reason:'obsolete-jev'},
          {review_id:'r1',decision_step:1,method:'code',status:'passed',reason:'final-current',action_effect:{status:'passed'}},
          {review_id:'r1',decision_step:1,method:'code',status:'running',reason:'unfinished-stage'},
          {review_id:'other-step',decision_step:2,method:'code',status:'unknown',reason:'unrelated-step'}],
        step_verification:{review_id:'r1',decision_step:1,status:'unknown',reason:'older-snapshot'}};
      const original=JSON.stringify(step), holder=document.createElement('div');
      holder.innerHTML=localStepCard('fixture',step);
      return {html:holder.innerHTML,head:holder.querySelector('.card-head').textContent,
        reviews:holder.querySelectorAll('.review-card').length,unchanged:JSON.stringify(step)===original};
    }""")
    assert result['reviews'] == 1 and result['unchanged'] is True
    assert '动作已验证' in result['head'] and '子任务已完成' in result['head']
    assert 'failure' not in result['head']
    assert 'final-current' in result['html']
    for old in ('obsolete-code', 'obsolete-jev', 'unfinished-stage', 'unrelated-step', 'older-snapshot'):
        assert old not in result['html']


def test_history_retains_distinct_review_scopes_and_supports_legacy_assessments(review_page):
    page = review_page
    result = page.evaluate("""() => {
      const holder=document.createElement('div');
      holder.innerHTML=localStepCard('fixture',{step_num:1,step_reviews:[
        {review_id:'action-subtask',decision_step:1,plan_step_id:'p0_1',status:'passed',reason:'verified-step'},
        {review_id:'final-task',decision_step:1,plan_step_id:'whole_task',status:'unknown',reason:'remaining-requirement'}],
        step_verification:{review_id:'action-subtask',decision_step:1,plan_step_id:'p0_1',status:'passed'}});
      const old=document.createElement('div');
      old.innerHTML=localStepCard('legacy',{step_num:1,assessment:{action_status:'success'}});
      const snapshot=document.createElement('div');
      snapshot.innerHTML=localStepCard('snapshot',{step_num:1,step_verification:{status:'unknown',method:'code',
        action_transition:{effective_correct:'yes',answers:{correct:{label:'unknown'}}}}});
      return {reviews:holder.querySelectorAll('.review-card').length,head:holder.querySelector('.card-head').textContent,
        oldHead:old.querySelector('.card-head').textContent,snapshotHead:snapshot.querySelector('.card-head').textContent};
    }""")
    assert result['reviews'] == 2 and '整项任务待核查' in result['head']
    assert '子任务已完成' not in result['head']
    assert 'success' in result['oldHead']
    assert '动作已验证' in result['snapshotHead'] and '子任务待核查' in result['snapshotHead']


def test_jev_call_view_shows_real_state_questions_images_and_redacted_raw_output(review_page):
    from io import BytesIO
    import base64
    from PIL import Image
    from playwright.sync_api import expect
    page = review_page
    start_step(page, "真实SystemOne输入输出")
    buffer = BytesIO()
    Image.new('RGB', (12, 12), '#71a6cb').save(buffer, format='PNG')
    image = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    call = {'call_id': 'fixture:call:1', 'model': 'jev-test', 'latency_ms': 241,
        'request': {'model': 'jev-test', 'state': json.dumps({'image_order': ['after'],
            'after': {'nodes': [{'ref': '19', 'name': 'Observed title'}]},
            'api_key': 'never-display-key', 'nested': {'password': 'never-display-password'}}),
            'questions': {'correct': {'type': 'choice', 'instructions': 'Verify the actual click',
                'criteria': {'yes': 'correct', 'no': 'incorrect', 'unknown': 'missing evidence'}}},
            'images': [image]},
        'Authorization': 'Bearer never-display-header',
        'response': {'status': 'answered', 'answers': {'correct': {'label': 'unknown', 'top_label': 'yes',
            'probabilities': {'yes': .71, 'no': .19, 'unknown': .10}}},
            'raw_response': {'model': 'jev-test', 'answers': {'correct': {'choice': 'yes'}},
                'usage': {'input_tokens': 42}, 'token': 'never-display-token'}}}
    payload = {'kind': 'status', 'event': 'jev_call', 'step': 1, 'call': call, 'ts': 1}
    page.evaluate('payload => {addCard(payload);addCard(payload)}', payload)
    page.evaluate("""() => addCard({kind:'status',event:'jev_transition',step:1,
      transition:{effective_correct:'yes',answers:{correct:{label:'unknown'}}},ts:1})""")
    card = page.locator('.compact-step[data-step="1"]').last
    panel = card.locator('.jev-io')
    expect(panel.locator('summary').first).to_have_text('Jev 输入 / 输出 · 1 次')
    expect(panel).to_have_js_property('open', False)
    panel.evaluate('element => element.open = true')
    text = panel.text_content()
    assert 'Observed title' in text and 'Verify the actual click' in text
    assert 'raw_response' not in text  # Server body is separately labelled, not mixed with parsed output.
    assert '服务端原始输出' in text and 'input_tokens' in text and '"effective_correct": "yes"' in text
    for secret in ('never-display-key', 'never-display-password', 'never-display-header', 'never-display-token', image):
        assert secret not in text
    expect(panel.locator('img')).to_have_attribute('src', image)
    assert panel.locator('.jev-call-panel').count() == 1
    history = page.evaluate('call => localStepCard("fixture",{step_num:1,jev_calls:[call]})', call)
    assert 'Jev 输入 / 输出' in history and 'Observed title' in history and 'never-display-password' not in history


def test_default_and_debug_templates_switch_independently_without_model_calls(review_page):
    from playwright.sync_api import expect
    page = review_page
    page.evaluate('loadDefaultPrompt()')
    editor = page.locator('#run-system-prompt')
    expect(editor).to_have_value(re.compile('target_ref'))
    assert page.locator('#run-prompt-mode').input_value() == 'dom'
    dom_text = editor.input_value()
    page.locator('#run-prompt-mode').select_option('gui_browser', force=True)
    browser_text = editor.input_value()
    assert browser_text and browser_text != dom_text
    page.locator('#run-prompt-mode').select_option('gui_desktop', force=True)
    desktop_text = editor.input_value()
    assert desktop_text and desktop_text != browser_text
    page.locator('#run-prompt-mode').select_option('dom', force=True)
    assert editor.input_value() == dom_text
    page.evaluate("switchView('debug')")
    preview = page.locator('#debug-system-template')
    expect(preview).to_have_value(re.compile('target_ref'))
    preview.evaluate('element => element.closest("details").open = true')
    page.locator('#debug-prompt-mode').select_option('gui_desktop', force=True)
    edited = preview.input_value() + '\nTEST_EDITED_TEMPLATE'
    preview.fill(edited)
    page.locator('#debug-messages').fill(json.dumps([{'role': 'user', 'content': 'Keep this user message'}]))
    page.locator('#debug-use-template').evaluate('button => button.click()')
    messages = json.loads(page.locator('#debug-messages').input_value())
    assert messages[0] == {'role': 'system', 'content': edited}
    assert messages[1] == {'role': 'user', 'content': 'Keep this user message'}
    assert '尚未调用模型' in page.locator('#debug-prompt-state').text_content()


def test_phase_wall_time_is_separate_from_model_request_and_updates_in_place(review_page):
    from playwright.sync_api import expect
    page = review_page
    start_step(page, '阶段计时测试')
    page.evaluate("""() => addCard({kind:'status', event:'phase_timing', phase:'planning',
      phase_id:'plan-1',status:'running',started_at:Date.now()/1000,step:0,ts:Date.now()/1000})""")
    panel=page.locator('.phase-timing-panel').last
    expect(panel).to_contain_text('进行中')
    page.evaluate("""() => addCard({kind:'status', event:'phase_timing', phase:'planning',
      phase_id:'plan-1',status:'completed',elapsed_ms:22000,request_ms:20000,step:0,ts:Date.now()/1000})""")
    expect(panel.locator('.phase-timing-row')).to_have_count(1)
    expect(panel).to_contain_text('22.00 秒')
    expect(panel).to_contain_text('其中模型请求 20.00 秒')
    expect(panel).not_to_contain_text('进行中')
    assert page.locator('#nav-collect .nav-chip').count()==0
    assert page.locator('.sidebar-foot').count()==0
    assert page.locator('#studio-header .header-label').count()==0
    assert page.locator('#conn').count()==1


def test_verdict_and_cleanup_are_visible_without_ending_run_early(review_page):
    from playwright.sync_api import expect
    page = review_page
    start_step(page, '并行与清理展示测试')
    page.evaluate("""() => {
      addCard({kind:'status',event:'parallel_action_review',code_ms:1000,judge_ms:2000,elapsed_ms:2100});
      addCard({kind:'status',event:'verification_complete',success:true});
      addCard({kind:'status',event:'phase_timing',phase:'cleanup',phase_id:'cleanup',
        status:'running',started_at:Date.now()/1000});
    }""")
    expect(page.locator('#st-status')).to_contain_text('验收通过 · 正在清理')
    expect(page.locator('#st-result')).to_have_text('成功')
    assert page.evaluate('statusState.done') is False
    expect(page.locator('.phase-timing-panel').last).to_contain_text('环境与浏览器清理')
    assert '合计墙钟 2.10s' in page.locator('body').text_content()
    page.evaluate("""() => addCard({kind:'status',event:'task_end',success:true,result:0})""")
    assert page.evaluate('statusState.done') is True
