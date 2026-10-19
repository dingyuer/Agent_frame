"""Actual Chromium presentation: replay, live fixture, export and failure path."""
import asyncio
import json
import socket
import threading
import time
from pathlib import Path

import pytest


def test_presentation_in_real_browser(request, tmp_path):
    if not request.config.getoption('--run-browser-integration'):
        pytest.skip('pass --run-browser-integration to exercise Studio')
    import httpx
    import uvicorn
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from playwright.sync_api import sync_playwright, expect
    from osworld_agent.viz.server import app

    calls, controls = [], {'fail': False}
    async def fixture(request: Request):
        data = await request.json()
        calls.append(data)
        if controls['fail']:
            return JSONResponse({'detail': 'fixture failure'}, status_code=503)
        await asyncio.sleep(.08)
        if 'readiness' in data['questions']:
            key = 'readiness'
            label = 'waiting' if len([c for c in calls if 'readiness' in c['questions']]) == 1 else 'ready'
        else:
            key = 'step'
            state = json.loads(data['state'])
            goal = state['step']['goal']
            label = 'progress' if '显示搜索结果' in goal else 'failed' if '综述' in goal else 'passed'
        keys = data['questions'][key]['criteria']
        return {'model': 'ui-fixture-9b', 'usage': {'input_tokens': 812, 'output_tokens': 0},
                'answers': {key: {'type': 'choice', 'choice': label,
                    'probabilities': {key: .97 if key == label else .03/(len(keys)-1) for key in keys}, 'confidence': .8}}}
    app.add_api_route('/__jev_demo_fixture/v1/systemone', fixture, methods=['POST'])
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='error'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{port}'
    try:
        for _ in range(80):
            try:
                if httpx.get(base, timeout=.2, trust_env=False).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(.05)
        with sync_playwright() as p:
            system_chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
            browser = p.chromium.launch(headless=True,
                **({'executable_path': str(system_chrome)} if system_chrome.is_file() else {}))
            page = browser.new_page(viewport={'width': 1600, 'height': 1100}, device_scale_factor=1)
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto(base+'/#jev-demo')
            expect(page.locator('#view-jev-demo')).to_be_visible()
            expect(page.locator('#jd-scenes button')).to_have_count(7)
            page.locator('.jd-config').evaluate('e => e.open = true')
            page.locator('#jd-pace').select_option('0')
            page.locator('#jd-replay').click()
            expect(page.locator('#jd-summary')).to_be_visible(timeout=15000)
            expect(page.locator('#jd-mode')).to_contain_text('REPLAY')
            expect(page.locator('#jd-mean')).to_have_text('—')
            expect(page.locator('#jd-calls')).to_have_text('0')
            expect(page.locator('#jd-code-count')).to_have_text('2 / 2')
            assert not calls
            with page.expect_download() as download:
                page.locator('#jd-export').click()
            exported = json.loads(Path(download.value.path()).read_text())
            assert exported['mode'] == 'replay' and exported['synthetic_observations']
            assert len(exported['results']) == 7 and not exported['trainable']
            assert all(r['model_latency_ms'] is None for r in exported['results'])

            page.locator('#jd-url').fill(base+'/__jev_demo_fixture/v1/systemone')
            page.locator('#jd-key').fill('private-demo-key')
            page.locator('#jd-live').click()
            expect(page.locator('#jd-summary')).to_be_visible(timeout=15000)
            expect(page.locator('#jd-mode')).to_contain_text('LIVE')
            expect(page.locator('#jd-calls')).to_have_text('5')
            expect(page.locator('#jd-code-count')).to_have_text('2 / 2')
            expect(page.locator('#jd-chart .jd-chart-row')).to_have_count(5)
            assert len(calls) == 5 and [len(c['images']) for c in calls] == [1, 1, 2, 2, 2]
            page.locator('#jd-scenes button').nth(3).click()
            expect(page.locator('#jd-outcome')).to_have_text('未通过')
            expect(page.locator('#jd-next-title')).to_have_text('纠正当前路径')
            expect(page.locator('#jd-probs .jd-prob-row')).to_have_count(5)
            page.locator('.jd-config').evaluate('e => e.open = false')
            page.locator('#view-jev-demo').evaluate('e => e.scrollTop = 0')
            page.screenshot(path='/tmp/jev-demo-desktop.png', full_page=True)
            with page.expect_download() as download:
                page.locator('#jd-export').click()
            exported = Path(download.value.path()).read_text()
            assert 'private-demo-key' not in exported
            assert json.loads(exported)['mode'] == 'live'

            controls['fail'] = True
            page.locator('#jd-live').click()
            expect(page.locator('#jd-error')).to_contain_text('未切换成模拟结果', timeout=15000)
            expect(page.locator('#jd-mode')).to_contain_text('LIVE')
            expect(page.locator('#jd-summary')).not_to_be_visible()
            expect(page.locator('#jd-replay')).to_be_enabled()

            page.locator('.jd-config').evaluate('e => e.open = true')
            page.locator('#jd-pace').select_option('5000')
            page.locator('#jd-replay').click()
            expect(page.locator('#jd-outcome')).to_have_text('处理中 · 等待', timeout=10000)
            page.locator('#jd-stop').click()
            expect(page.locator('#jd-live')).to_be_enabled(timeout=1000)
            expect(page.locator('#jd-outcome')).to_have_text('已停止演示')
            page.set_viewport_size({'width': 390, 'height': 844})
            page.locator('.jd-config').evaluate('e => e.open = false')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path='/tmp/jev-demo-mobile.png', full_page=True)
            assert page.locator('#view-jev-demo').evaluate('e => e.scrollWidth <= e.clientWidth'), page.locator('#view-jev-demo').evaluate("e => Array.from(e.querySelectorAll('*')).filter(n => n.getBoundingClientRect().right > e.getBoundingClientRect().right + 2).map(n => [n.tagName,n.id,n.className,Math.round(n.getBoundingClientRect().width)]).slice(0,20)")
            assert not errors
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
