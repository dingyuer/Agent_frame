"""Real Studio -> FastAPI -> local System One fixture, without remote models."""
import json
import socket
import threading
import time

import pytest


def test_jev_debugger_and_runtime_selector(request, tmp_path):
    if not request.config.getoption('--run-browser-integration'):
        pytest.skip('pass --run-browser-integration to exercise Studio')
    import httpx
    import uvicorn
    from fastapi import Request
    from PIL import Image
    from playwright.sync_api import sync_playwright, expect
    from osworld_agent.viz.server import app
    from test_jev_debug import ANSWERS

    calls = []
    async def systemone(request: Request):
        payload = await request.json()
        calls.append(payload)
        assert request.headers['Authorization'] == 'Bearer test-only'
        answer = dict(ANSWERS[{'choice': 'status', 'noul': 'completed', 'score': 'rating'}[payload['questions']['status']['type']]])
        return {'model': 'actual-fixture-flash', 'answers': {'status': answer}, 'usage': {'input_tokens': 42, 'output_tokens': 0}}
    app.add_api_route('/__jev_fixture/v1/systemone', systemone, methods=['POST'])
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    runtime = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='error'))
    thread = threading.Thread(target=runtime.run, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{port}'
    errors = []
    try:
        with httpx.Client(trust_env=False) as client:
            for _ in range(100):
                try:
                    if client.get(base + '/').status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.05)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 1050})
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto(base)
            expect(page.locator('#run-model')).to_have_value('qwen3.7-flash')
            expect(page.locator('#run-model-provider')).to_have_value('openai')
            assert page.evaluate('decisionJudgeConfig().model') == 'vllm-sr/d3-flash'
            expect(page.locator('#judge-selection-policy')).to_have_value('maximum_probability')
            assert page.locator('#judge-threshold').is_disabled()
            assert page.evaluate('decisionJudgeConfig().selection_policy') == 'maximum_probability'
            page.locator('#judge-selection-policy').evaluate("el => {for(let p=el.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;}")
            page.locator('#judge-selection-policy').select_option('threshold')
            assert page.locator('#judge-threshold').is_enabled()
            page.reload()
            page.locator('#judge-selection-policy').evaluate("el => {for(let p=el.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;}")
            expect(page.locator('#judge-selection-policy')).to_have_value('threshold')
            assert page.evaluate('decisionJudgeConfig().selection_policy') == 'threshold'
            saved = page.evaluate("JSON.parse(localStorage.getItem(JUDGE_RUNTIME_STORAGE_KEY))")
            assert 'api_key' not in saved and 'jev-key' not in saved
            page.locator('#judge-selection-policy').select_option('maximum_probability')
            page.locator('#judge-model').select_option('')
            assert page.evaluate('decisionJudgeConfig()') == {}
            page.locator('#judge-model').select_option('vllm-sr/d3-flash')
            expect(page.locator('#judge-url')).to_have_value('http://114.212.10.51:8007/v1/systemone')
            assert page.evaluate('decisionJudgeConfig().review_mode') == 'exclusive'
            assert page.evaluate('decisionJudgeConfig().recheck_unknown') is True
            page.locator('#judge-recheck').uncheck()
            assert page.evaluate('decisionJudgeConfig().recheck_unknown') is False
            page.locator('#nav-chat').click()
            expect(page.locator('#view-chat')).to_be_visible()
            page.locator('[data-lab-mode=jev]').click()
            page.locator('#lab-connection').evaluate('el => el.open = true')
            page.locator('#lab-url').fill(base + '/__jev_fixture/v1/systemone')
            page.locator('#lab-key').fill('test-only')
            page.locator('#lab-input').fill('任务：检查页面已出现搜索结果。前图为空，后图为搜索结果。')
            png = tmp_path / 'before.png'
            Image.new('RGB', (800, 600), 'white').save(png)
            page.locator('#lab-files').set_input_files([str(png), str(png)])
            expect(page.locator('#lab-images img')).to_have_count(2)
            for index, kind in enumerate(['choice', 'noul', 'score']):
                question = {'type': kind, 'instructions': '判断任务完成状态'}
                if kind == 'choice':
                    question['criteria'] = {'passed': 'Yes', 'unknown': None}
                if kind == 'score':
                    question['criteria'] = ['No', 'Partly', 'Yes']
                page.locator('#lab-questions').fill(json.dumps({'status': question}))
                page.locator('#lab-send').click()
                expect(page.locator('.lab-turn')).to_have_count(index + 1)
                expect(page.locator('#lab-request-state')).to_contain_text('已返回')
                expect(page.locator('.lab-turn').last).to_contain_text('ms')
                expect(page.locator('.lab-turn').last).to_contain_text('actual-fixture-flash')
                assert 'test-only' not in page.evaluate("localStorage.getItem('yaero_model_lab_v1')")
            assert len(calls) == 3 and len(calls[0]['images']) == 2
            assert calls[0]['state'].startswith('任务：')
            page.screenshot(path=str(tmp_path / 'jev-desktop.png'), full_page=True)
            # Review latency is also rendered outside the manual debugger.
            html = page.evaluate("reviewCardHtml({method:'model',status:'passed',latency_ms:1234,reason:'ok'})")
            assert '耗时 1234 ms' in html
            html = page.evaluate("reviewCardHtml({method:'decision_model',status:'unknown',step_outcome:'progress',action_effect:{method:'code',status:'passed',reason:'输入已出现'},recheck:{reason:'无新证据'},checkpoint_ids:['p0_1:a1'],evidence_selection:{after:{nodes_selected:40,nodes_total:300,artifacts_selected:1,artifacts_total:12}}})")
            assert '子任务进行中' in html and '输入已出现' in html and '40 / 300' in html
            assert '无新证据' in html and 'p0_1:a1' in html
            html = page.evaluate("reviewCardHtml({method:'decision_model',status:'unknown',step_outcome:'progress',progress_hint:true,reason:'Jev 倾向有进展'})")
            assert '有进展倾向 · 未判完成' in html and '证据不足' not in html
            assert '待补充核查' in page.evaluate("reviewCardHtml({method:'code',status:'unknown',reason:'需要语义核查'})")
            page.evaluate("renderStatus({event:'step_verification',method:'decision_model',status:'unknown',step_outcome:'progress'})")
            expect(page.locator('#st-status')).to_contain_text('动作有进展')
            payload = {'event': 'framework_wait', 'reason': 'desktop_window', 'status': 'waiting',
                       'target': '资源管理器', 'elapsed_s': 5.25, 'timeout_s': 15,
                       'detail': '等待窗口进入前台', 'ts': 1}
            page.evaluate('s => renderStatus(s)', payload)
            expect(page.locator('#st-status')).to_contain_text('等待窗口就绪')
            html = page.evaluate('s => waitDetail(s)', payload)
            assert '资源管理器' in html and '5.3 秒' in html and '操作系统' in html
            assert '不消耗决策步数' in html
            for status, label in [('timeout', '超时'), ('cancelled', '取消'), ('task_timeout', '耗尽'), ('completed', '结束')]:
                page.evaluate('s => renderStatus(s)', dict(payload, status=status))
                expect(page.locator('#st-status')).to_contain_text(label)
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(tmp_path / 'jev-mobile.png'), full_page=True)
            assert not errors
            browser.close()
    finally:
        runtime.should_exit = True
        thread.join(timeout=10)
