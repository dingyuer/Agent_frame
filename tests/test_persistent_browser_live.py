"""Real local Chromium checks for persistent authentication and task-tab scope."""
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from osworld_agent.browser_session import BrowserOptions, BrowserSession, browser_thread
from test_browser_live import click, observe


@pytest.fixture
def persistent_browser(request, tmp_path, monkeypatch):
    if not request.config.getoption('--run-browser-integration'):
        pytest.skip('pass --run-browser-integration to run local Chromium fixtures')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    monkeypatch.setenv('no_proxy', '127.0.0.1,localhost')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith('/file'):
                content = b'Persistent fixture download.'
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Disposition', 'attachment; filename="fixture.txt"')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(b'''<!doctype html><title>Persistent fixture</title>
                <a id=download href=/file download>Download file</a>
                <a id=popup href=/child target=_blank>Open owned popup</a>
                <a id=popupdownload href=/file?popup target=_blank>Download in new tab</a>
                <button id=alert onclick="alert('Owned dialog')">Alert</button>''')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    from playwright.sync_api import sync_playwright
    state = {'base': f'http://127.0.0.1:{server.server_port}', 'port': port,
             'profile': tmp_path / 'profile', 'context': None, 'sessions': []}
    try:
        with sync_playwright() as playwright:
            def launch():
                kwargs = {}
                if not os.path.exists(playwright.chromium.executable_path):
                    system = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
                    if os.path.exists(system):
                        kwargs['executable_path'] = system
                state['context'] = playwright.chromium.launch_persistent_context(
                    str(state['profile']), headless=True, args=[f'--remote-debugging-port={port}'],
                    accept_downloads=True, locale='fr-FR', timezone_id='Asia/Shanghai', **kwargs)
                # This independent fixture connection must not auto-dismiss the
                # task connection's dialog before its explicit recovery call.
                state['context'].on('dialog', lambda dialog: None)
                return state['context']
            def session():
                result = BrowserSession(BrowserOptions(endpoint=f'http://127.0.0.1:{port}',
                    require_focus=False, session_mode='persistent'), workspace_dir=str(tmp_path / 'files'))
                state['sessions'].append(result)
                return result
            state.update(launch=launch, session=session)
            launch()
            yield state
    finally:
        for session in state['sessions']:
            session.close()
        if state['context'] is not None:
            try:
                state['context'].close()
            except Exception:
                pass
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def owner(session, callback):
    return browser_thread(lambda self: callback(self._pick_page(self._connect()[1])))(session)


def test_live_cookies_storage_settings_survive_tasks_sessions_and_browser_restart(persistent_browser):
    fixture = persistent_browser
    context, base = fixture['context'], fixture['base']
    outside = context.new_page()
    outside.goto(base)
    outside.evaluate("document.cookie='auth=retained; max-age=3600; path=/'; localStorage.setItem('setting','retained')")
    session = fixture['session']()
    first = session.begin_episode(start_url=base)
    assert first['profile']['storage'] == 'persistent' and first['isolation_scope'] == 'task_tabs'
    settings = owner(session, lambda page: page.evaluate("({cookie:document.cookie,storage:localStorage.getItem('setting'),locale:navigator.language,tz:Intl.DateTimeFormat().resolvedOptions().timeZone})"))
    assert settings == {'cookie': 'auth=retained', 'storage': 'retained', 'locale': 'fr-FR', 'tz': 'Asia/Shanghai'}
    assert len(session.list_tabs()['tabs']) == 1
    first_page = list(session._task_pages.values())[0]
    session.begin_episode(start_url=base)
    assert browser_thread(lambda self: first_page.is_closed())(session)
    assert not outside.is_closed()
    assert owner(session, lambda page: page.evaluate('document.cookie')) == 'auth=retained'
    session.close()
    assert not outside.is_closed()
    assert outside.evaluate("localStorage.getItem('setting')") == 'retained'
    second = fixture['session']()
    second.begin_episode(start_url=base)
    assert owner(second, lambda page: page.evaluate("localStorage.getItem('setting')")) == 'retained'
    second.close()
    context.close()
    fixture['launch']()
    third = fixture['session']()
    third.begin_episode(start_url=base)
    assert owner(third, lambda page: page.evaluate('document.cookie')) == 'auth=retained'
    assert owner(third, lambda page: page.evaluate("localStorage.getItem('setting')")) == 'retained'


def test_live_only_task_downloads_and_dialogs_are_observed(persistent_browser):
    fixture = persistent_browser
    outside = fixture['context'].new_page()
    outside.goto(fixture['base'])
    session = fixture['session']()
    session.begin_episode(start_url=fixture['base'])
    with outside.expect_download() as external_download:
        outside.locator('#download').click()
    external_download.value.save_as(fixture['profile'].parent / 'outside.txt')
    assert not session.event_metadata()['artifacts']
    assert click(session, 'download')['status'] == 'executed'
    deadline = time.monotonic() + 5
    events = session.event_metadata()
    while not events['artifacts'] and time.monotonic() < deadline:
        time.sleep(.02)
        events = session.event_metadata()
    assert len(events['artifacts']) == 1, events
    assert Path(events['artifacts'][0]['path']).read_bytes() == b'Persistent fixture download.'
    alert = click(session, 'alert')
    assert alert.get('dispatched') is True, alert
    assert observe(session).info['dialog']['message'] == 'Owned dialog'
    handled = session.handle_dialog(accept=False)
    assert handled['status'] == 'executed', handled
    session.close()
    assert not outside.is_closed()


def test_live_owned_popup_is_followed_and_closed_without_external_tabs(persistent_browser):
    fixture = persistent_browser
    outside = fixture['context'].new_page()
    outside.goto(fixture['base'])
    session = fixture['session']()
    session.begin_episode(start_url=fixture['base'])
    assert click(session, 'popup')['status'] == 'executed'
    current = observe(session)
    assert current.info['url'] == fixture['base'] + '/child'
    assert len(session.list_tabs()['tabs']) == 2
    fixture['context'].pages[0].wait_for_timeout(10)
    assert len(fixture['context'].pages) >= 3
    session.close()
    assert not outside.is_closed()


def test_live_download_opened_with_target_blank_is_owned(persistent_browser):
    fixture = persistent_browser
    session = fixture['session']()
    session.begin_episode(start_url=fixture['base'])
    assert click(session, 'popupdownload')['status'] == 'executed'
    deadline = time.monotonic() + 5
    events = session.event_metadata()
    while not events['artifacts'] and time.monotonic() < deadline:
        time.sleep(.02)
        events = session.event_metadata()
    assert len(events['artifacts']) == 1, events
    assert events['artifacts'][0]['url'].endswith('/file?popup')
