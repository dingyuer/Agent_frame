"""Session, browser-event, isolation and explicit-surface tests without a live browser."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

import pytest
from PIL import Image

from osworld_agent.actions import Action
from osworld_agent.adapters.windows_env import WindowsEnvironment
from osworld_agent.browser_session import BrowserOptions, BrowserSession
from osworld_agent.env import Observation
from test_browser_session_safety import FakeCDP, FakePage


class EventPage(FakePage):
    def __init__(self, context, page_id):
        super().__init__(page_id)
        self.context = context
        self.handlers = {}
        self.closed = False
        self.url = 'about:blank'

    def on(self, name, callback):
        self.handlers[name] = callback

    def close(self):
        self.closed = True
        self.context.pages.remove(self)


class EventContext:
    def __init__(self, browser):
        self.browser = browser
        self.pages = []
        self.handlers = {}
        self.closed = False
        self.request = SimpleNamespace(get=Mock())

    def on(self, name, callback):
        self.handlers[name] = callback

    def new_page(self):
        page = EventPage(self, 'page-' + str(len(self.browser.created)) + '-' + str(len(self.pages)))
        self.pages.append(page)
        if 'page' in self.handlers:
            self.handlers['page'](page)
        return page

    def new_cdp_session(self, page):
        return FakeCDP(page)

    def close(self):
        self.closed = True
        self.pages.clear()
        self.browser.contexts.remove(self)


class EventBrowser:
    def __init__(self):
        self.contexts = []
        self.created = []
        self.version = 'fake-Chromium-123'
        self.close = Mock()

    def new_context(self, **options):
        context = EventContext(self)
        self.contexts.append(context)
        self.created.append((context, options))
        return context


@pytest.fixture
def session(tmp_path):
    item = BrowserSession(BrowserOptions(require_focus=False), downloads_dir=str(tmp_path))
    yield item
    item.close()


def test_persistent_connection_runs_on_one_owner_thread_and_closes(monkeypatch, session):
    session.options.endpoint = "ws://127.0.0.1/devtools/browser/test"
    browser = EventBrowser()
    context = browser.new_context()
    context.new_page()
    connection_threads = []
    def connect(*args, **kwargs):
        connection_threads.append(threading.get_ident())
        return browser
    runtime = SimpleNamespace(chromium=SimpleNamespace(connect_over_cdp=Mock(side_effect=connect)), stop=Mock())
    start = AsyncMock(return_value=runtime)
    monkeypatch.setitem(sys.modules, 'playwright.async_api', SimpleNamespace(async_playwright=lambda: SimpleNamespace(start=start)))
    session.list_tabs()
    session.list_tabs()
    assert start.call_count == 1
    assert connection_threads == [session._worker_thread]
    assert session._worker_thread != threading.get_ident()
    browser.close.assert_not_called()
    session.close()
    browser.close.assert_called_once()
    runtime.stop.assert_called_once()


def test_episode_isolation_has_stable_initial_identity_and_unique_resources(session):
    browser = EventBrowser()
    session._connect = Mock(return_value=(None, browser))
    first = session.begin_episode(task_id='same-user-task')
    first_context = session._owned_context
    session._artifacts.append({'download_id': 'old'})
    second = session.begin_episode(task_id='same-user-task')
    assert first_context.closed
    assert session._owned_context is not first_context
    assert first['initial_state_id'] == second['initial_state_id']
    assert first['downloads_dir'] != second['downloads_dir']
    assert first['context_id'] != second['context_id']
    assert second['isolation_scope'] == 'browser_context'
    assert session.event_metadata()['artifacts'] == []
    assert session._owned_context.pages[0].url == 'about:blank'
    assert browser.created[-1][1]['accept_downloads'] is True


def test_isolation_failure_does_not_reuse_an_existing_context(session):
    browser = EventBrowser()
    browser.new_context = Mock(side_effect=RuntimeError('cannot isolate'))
    session._connect = Mock(return_value=(None, browser))
    with pytest.raises(RuntimeError, match='cannot isolate'):
        session.begin_episode()
    assert session.initial_state == {}
    assert session._owned_context is None


def test_browser_download_event_records_completed_artifact_once(session):
    browser = EventBrowser()
    context = browser.new_context()
    page = context.new_page()
    session._watch_page(page)
    download = SimpleNamespace(url='https://example.test/AAA.pdf', suggested_filename='../AAA.pdf',
        save_as=lambda path: Path(path).write_bytes(b'%PDF-1.7 test'))
    page.handlers['download'](download)
    session._drain_downloads()
    first = session.event_metadata()['artifacts']
    assert len(first) == 1
    artifact = first[0]
    assert artifact['status'] == 'completed' and artifact['completed'] is True
    assert artifact['transport'] == 'browser' and artifact['download_id']
    assert Path(artifact['path']).parent == Path(session.downloads_dir)
    session._drain_downloads()
    assert session.event_metadata()['artifacts'] == first


def test_failed_browser_download_never_becomes_completed_artifact(session):
    download = SimpleNamespace(url='https://example.test/AAA.pdf', suggested_filename='AAA.pdf',
        save_as=Mock(side_effect=RuntimeError('canceled')))
    session._queue_download(SimpleNamespace(url='https://example.test'), download)
    session._drain_downloads()
    assert session.event_metadata()['artifacts'] == []
    assert session.event_metadata()['download_events'][0]['completed'] is False


def test_javascript_dialog_preserves_reference_image_and_has_explicit_action(session):
    browser = EventBrowser()
    page = browser.new_context().new_page()
    session._connect = Mock(return_value=(None, browser))
    session._last_screenshot = Image.new('RGB', (800, 600))
    session._last_screenshot_metadata = {'page_id': page.page_id, 'screenshot_space': 'css_viewport'}
    dialog = SimpleNamespace(type='prompt', message='Your name?', default_value='Guest', accept=Mock(), dismiss=Mock())
    session._record_dialog(page, dialog)
    shot, snapshot, info = session.collect_observation()
    assert shot is session._last_screenshot and snapshot is None
    assert info['screenshot_stale'] is True
    assert info['dialog']['message'] == 'Your name?'
    result = session.handle_dialog(True, prompt_text='Alice', dialog_id=info['dialog']['id'])
    assert result['status'] == 'executed'
    dialog.accept.assert_called_once_with(prompt_text='Alice')
    assert session.event_metadata()['dialogs'] == []


def test_close_tab_restores_parent_without_selecting_ad_popup(session):
    browser = EventBrowser()
    context = browser.new_context()
    parent = context.new_page()
    popup = context.new_page()
    popup.opener_id = parent.page_id
    session._connect = Mock(return_value=(None, browser))
    assert session._pick_page(browser) is parent
    session.switch_tab(popup.page_id)
    result = session.close_tab()
    assert result['status'] == 'executed'
    assert session._pick_page(browser) is parent


def test_windows_begin_is_consumed_once_by_reset(monkeypatch):
    env = WindowsEnvironment()
    begin = Mock(return_value={'isolated': True, 'initial_state_id': 'blank-v1'})
    env._browser_session = SimpleNamespace(begin_episode=begin)
    monkeypatch.setattr(env, 'observe', lambda: Observation())
    env.begin_episode(task_id='rollout-1')
    env.reset('download AAA')
    assert begin.call_count == 1
    env.reset('another task')
    assert begin.call_count == 2


def test_windows_explicit_surface_switch_changes_image_space(monkeypatch):
    env = WindowsEnvironment()
    browser_shot, desktop_shot = Image.new('RGB', (800, 600)), Image.new('RGB', (1920, 1080))
    env._browser_session = SimpleNamespace(collect_observation=lambda: (browser_shot, None,
        {'screenshot_space': 'css_viewport', 'page_id': 'p', 'screenshot_size': [800, 600]}), event_metadata=lambda: {})
    monkeypatch.setattr(env, '_take_screenshot', lambda: desktop_shot)
    desktop = env.step(Action('observe_desktop', {})).observation
    assert desktop.screenshot is desktop_shot
    assert desktop.info['screenshot_space'] == 'screen_pixels'
    assert desktop.info['scene']['browser_use'] == 0
    webpage = env.step(Action('observe_browser', {})).observation
    assert webpage.screenshot is browser_shot
    assert webpage.info['screenshot_space'] == 'css_viewport'
    assert webpage.info['scene']['browser_use'] == 1


def test_modal_subtree_is_kept_even_after_normal_node_budget(session):
    page = FakePage()
    session.options.max_nodes = 1
    raw = [
        {'nodeId': 'root', 'backendDOMNodeId': 1, 'role': {'value': 'RootWebArea'}},
        {'nodeId': 'dialog', 'backendDOMNodeId': 2, 'parentId': 'root', 'role': {'value': 'dialog'}},
        {'nodeId': 'generic', 'backendDOMNodeId': 3, 'parentId': 'dialog', 'role': {'value': 'generic'}},
        {'nodeId': 'close', 'backendDOMNodeId': 4, 'parentId': 'generic', 'role': {'value': 'button'}, 'name': {'value': 'Close'}},
    ]
    original = FakeCDP(page)
    def send(command, params=None):
        if command == 'Accessibility.getFullAXTree':
            return {'nodes': raw}
        if command == 'DOMSnapshot.captureSnapshot':
            return {'strings': ['BUTTON', 'frame'], 'documents': [{'frameId': 1,
                'nodes': {'backendNodeId': [1, 2, 3, 4], 'nodeName': [0, 0, 0, 0], 'attributes': [[], [], [], []]},
                'layout': {'nodeIndex': [], 'bounds': []}}]}
        return original.send(command, params)
    snapshot = session._collect_page(page, SimpleNamespace(send=send), {}, 0)
    assert [n['role'] for n in snapshot['nodes']] == ['RootWebArea', 'dialog', 'button']
    assert snapshot['nodes'][-1]['name'] == 'Close'
    assert snapshot['nodes'][-1]['modal_context'] is True
    assert snapshot['active_modals'] == ['1']


def test_http_download_html_is_only_transport_evidence_and_preserves_source(session):
    browser = EventBrowser()
    context = browser.new_context()
    page = context.new_page()
    page.url = 'https://papers.example/AAA'
    response = SimpleNamespace(status=200, url='https://papers.example/login',
        headers={'content-type': 'text/html'}, body=lambda: b'<html>Login required</html>', dispose=Mock())
    context.request.get.return_value = response
    session._connect = Mock(return_value=(None, browser))
    result = session.download('https://papers.example/AAA.pdf')
    assert result['verified'] is True  # transport / local file only
    assert result['semantic_verified'] is False
    artifact = result['artifacts'][0]
    assert artifact['page_url'] == page.url
    assert artifact['url'] == 'https://papers.example/AAA.pdf'
    assert artifact['final_url'] == 'https://papers.example/login'
    assert artifact['content_type'] == 'text/html'


def test_same_size_cache_corruption_produces_new_download_identity(session):
    browser = EventBrowser()
    context = browser.new_context()
    context.new_page()
    response = SimpleNamespace(status=200, url='https://example.test/AAA.pdf',
        headers={}, body=lambda: b'abc', dispose=Mock())
    context.request.get.return_value = response
    session._connect = Mock(return_value=(None, browser))
    first = session.download(response.url)
    Path(first['path']).write_bytes(b'xyz')
    second = session.download(response.url)
    assert context.request.get.call_count == 2
    assert first['artifacts'][0]['download_id'] != second['artifacts'][0]['download_id']
    assert first['path'] != second['path']


def test_async_browser_download_never_blocks_observation_and_cancels_on_deadline(session):
    import asyncio
    import time
    from osworld_agent.browser_session import browser_thread
    owner_call = browser_thread(lambda owner, fn: fn())
    session.options.download_timeout_s = 0.02
    threads = []
    async def save_as(path):
        threads.append(threading.get_ident())
        await asyncio.Event().wait()
    download = SimpleNamespace(url='https://example.test/slow.pdf', suggested_filename='slow.pdf',
        save_as=save_as, cancel=AsyncMock())
    owner_call(session, lambda: session._await(asyncio.sleep(0)))
    before = time.monotonic()
    owner_call(session, lambda: session._queue_download(SimpleNamespace(url='https://example.test'), download))
    owner_call(session, session._drain_downloads)
    assert time.monotonic() - before < 0.2
    assert session.event_metadata()['download_events'][0]['status'] == 'in_progress'
    owner_call(session, lambda: session._await(asyncio.sleep(0.04)))
    download.cancel.assert_awaited_once()
    assert session.event_metadata()['artifacts'] == []
    assert session.event_metadata()['download_events'][0]['error'] == 'download deadline exceeded'
    assert threads == [session._worker_thread]


def test_async_download_success_records_artifact_without_waiting_for_next_action(session):
    import asyncio
    from osworld_agent.browser_session import browser_thread
    owner_call = browser_thread(lambda owner, fn: fn())
    async def save_as(path):
        await asyncio.sleep(0.001)
        Path(path).write_bytes(b'actual completed file')
    download = SimpleNamespace(url='https://example.test/paper.pdf', suggested_filename='paper.pdf',
        save_as=save_as, cancel=AsyncMock())
    owner_call(session, lambda: session._await(asyncio.sleep(0)))
    owner_call(session, lambda: session._queue_download(SimpleNamespace(url='https://example.test'), download))
    owner_call(session, lambda: session._await(asyncio.sleep(0.01)))
    assert session.event_metadata()['artifacts'][0]['completed'] is True
    assert session.event_metadata()['download_events'] == []
    download.cancel.assert_not_awaited()


def test_async_api_proxy_awaits_on_owner_thread_and_preserves_object_identity(session):
    from osworld_agent.browser_session import browser_thread
    owner_call = browser_thread(lambda owner, fn: fn())
    class Page:
        __module__ = 'playwright.async_api._generated'
        async def title(self):
            return threading.get_ident()
    raw = Page()
    wrapped = session._wrap_async(raw)
    assert session._wrap_async(raw) is wrapped
    assert owner_call(session, wrapped.title) == session._worker_thread
    assert session._worker_thread != threading.get_ident()


def test_browser_download_deadline_runs_while_policy_is_idle(session):
    import asyncio
    import time
    from osworld_agent.browser_session import browser_thread
    owner_call = browser_thread(lambda owner, fn: fn())
    session.options.download_timeout_s = 0.015
    async def save_as(path):
        await asyncio.Event().wait()
    canceled = threading.Event()
    async def cancel():
        assert threading.get_ident() == session._worker_thread
        canceled.set()
    download = SimpleNamespace(url='https://example.test/slow.pdf', suggested_filename='slow.pdf',
        save_as=save_as, cancel=cancel)
    owner_call(session, lambda: session._await(asyncio.sleep(0)))
    owner_call(session, lambda: session._queue_download(SimpleNamespace(url='https://example.test'), download))
    assert canceled.wait(1.0), 'owner loop must process deadlines even without another observation'
    assert session.event_metadata()['artifacts'] == []


def test_cdp_http_discovery_bypasses_environment_proxy_without_mutating_it(monkeypatch, session):
    import os
    import httpx
    monkeypatch.setenv('HTTP_PROXY', 'http://global-proxy.invalid:1234')
    monkeypatch.setenv('HTTPS_PROXY', 'http://global-proxy.invalid:1234')
    response = SimpleNamespace(raise_for_status=Mock(), json=lambda: {
        'webSocketDebuggerUrl': 'ws://127.0.0.1:9222/devtools/browser/known'})
    get = Mock(return_value=response)
    client = Mock()
    client.__enter__ = Mock(return_value=SimpleNamespace(get=get))
    client.__exit__ = Mock(return_value=False)
    factory = Mock(return_value=client)
    monkeypatch.setattr(httpx, 'Client', factory)
    assert session._resolve_cdp_endpoint() == 'ws://127.0.0.1:9222/devtools/browser/known'
    assert factory.call_args.kwargs['trust_env'] is False
    get.assert_called_once_with('http://127.0.0.1:9222/json/version/')
    assert os.environ['HTTP_PROXY'] == 'http://global-proxy.invalid:1234'
    assert os.environ['HTTPS_PROXY'] == 'http://global-proxy.invalid:1234'


def test_websocket_connection_failure_stops_driver_and_leaves_no_cached_connection(monkeypatch, session):
    session.options.endpoint = 'ws://127.0.0.1:9222/devtools/browser/missing'
    runtime = SimpleNamespace(chromium=SimpleNamespace(connect_over_cdp=Mock(side_effect=RuntimeError('refused'))), stop=Mock())
    monkeypatch.setitem(sys.modules, 'playwright.async_api', SimpleNamespace(
        async_playwright=lambda: SimpleNamespace(start=AsyncMock(return_value=runtime))))
    with pytest.raises(RuntimeError, match='refused'):
        session.list_tabs()
    runtime.stop.assert_called_once()
    assert session._browser is None and session._playwright is None


def test_cdp_discovery_failure_does_not_start_driver(monkeypatch, session):
    start = AsyncMock()
    monkeypatch.setitem(sys.modules, 'playwright.async_api', SimpleNamespace(
        async_playwright=lambda: SimpleNamespace(start=start)))
    monkeypatch.setattr(session, '_resolve_cdp_endpoint', Mock(side_effect=RuntimeError('discovery failed')))
    with pytest.raises(RuntimeError, match='discovery failed'):
        session.list_tabs()
    start.assert_not_called()
    assert session._browser is None


def test_sync_event_subscription_runs_inside_owner_loop_including_nested_callback(session):
    import asyncio
    from osworld_agent.browser_session import browser_thread
    owner_call = browser_thread(lambda owner, fn: fn())
    calls = []
    class Page:
        __module__ = 'playwright.async_api._generated'
        def on(self, event, callback):
            assert asyncio.get_running_loop() is session._loop
            assert threading.get_ident() == session._worker_thread
            calls.append(event)
            if event == 'page':
                callback(self)
    wrapped = session._wrap_async(Page())
    owner_call(session, lambda: wrapped.on('page', lambda page: page.on('dialog', lambda _: None)))
    assert calls == ['page', 'dialog']
