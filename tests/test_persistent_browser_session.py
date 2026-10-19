"""Persistent storage never expands the task's tab/event ownership."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from osworld_agent.adapters.windows_env import WindowsEnvironment
from osworld_agent.adapters.mac_env import MacEnvironment
from osworld_agent.browser_session import BrowserOptions, BrowserSession, browser_thread
from test_browser_lifecycle import EventBrowser


def owner(session, callback):
    return browser_thread(lambda self: callback())(session)


@pytest.fixture
def persistent_session(tmp_path):
    browser = EventBrowser()
    shared = browser.new_context()
    external = shared.new_page()
    external.url = 'https://outside.test/account'
    session = BrowserSession(BrowserOptions(require_focus=False, session_mode='persistent'),
                             downloads_dir=str(tmp_path))
    session._connect = Mock(return_value=(None, browser))
    yield session, browser, shared, external
    session.close()


def test_persistent_episode_reuses_storage_context_and_closes_only_task_tabs(persistent_session):
    session, browser, context, external = persistent_session
    first = session.begin_episode(start_url='https://task.test/')
    task = session._task_pages[next(iter(session._task_pages))]
    assert len(browser.created) == 1
    assert not first['isolated'] and first['isolation_scope'] == 'task_tabs'
    assert first['session_mode'] == 'persistent'
    assert first['profile']['storage'] == 'persistent'
    assert first['profile']['locale'] == 'inherited'
    assert set(external.handlers) == set()
    assert not context.handlers  # No global dialog listener on unrelated tabs.
    assert [t['url'] for t in session.list_tabs()['tabs']] == ['https://task.test/']
    second = session.begin_episode(start_url='https://task.test/again')
    assert task.closed and not external.closed and not context.closed
    assert len(browser.created) == 1
    assert first['context_id'] != second['context_id']
    current = list(session._task_pages.values())[0]
    session.close()
    assert current.closed and not external.closed and not context.closed
    assert context.pages == [external]


def test_external_pages_downloads_dialogs_and_responses_are_not_admitted(persistent_session):
    session, browser, context, external = persistent_session
    session.begin_episode()
    external_id = external.page_id
    assert session.switch_tab(external_id)['status'] == 'rejected'
    assert session.close_tab(external_id)['status'] == 'rejected'
    owner(session, lambda: session._watch_page(external))
    assert not external.handlers
    download = SimpleNamespace(url='https://outside.test/private', suggested_filename='private.txt',
                               cancel=Mock(), save_as=Mock())
    owner(session, lambda: session._queue_download(external, download))
    owner(session, lambda: session._record_dialog(external, Mock()))
    owner(session, lambda: session._record_response(external, Mock()))
    assert session.event_metadata()['artifacts'] == []
    assert session.event_metadata()['pending_downloads'] == []
    assert session.event_metadata()['dialogs'] == []
    assert session._main_responses == {}
    download.cancel.assert_not_called()
    download.save_as.assert_not_called()


def test_only_owned_opener_popup_event_admits_descendants(persistent_session):
    session, browser, context, external = persistent_session
    session.begin_episode()
    task = list(session._task_pages.values())[0]
    unrelated = context.new_page()
    child = context.new_page()
    owner(session, lambda: task.handlers['popup'](child))
    grandchild = context.new_page()
    owner(session, lambda: child.handlers['popup'](grandchild))
    listed = {t['tab_id'] for t in session.list_tabs()['tabs']}
    assert listed == {task.page_id, child.page_id, grandchild.page_id}
    session.close()
    assert child.closed and grandchild.closed and task.closed
    assert not unrelated.closed and not external.closed


def test_missing_default_context_fails_without_creating_empty_context(persistent_session):
    session, browser, _, _ = persistent_session
    browser.contexts.clear()
    with pytest.raises(RuntimeError, match='no default context'):
        session.begin_episode()
    assert len(browser.created) == 1
    assert session.initial_state == {}


@pytest.mark.parametrize('environment', [WindowsEnvironment, MacEnvironment])
def test_local_environment_passes_persistent_mode_and_accepts_truthful_manifest(environment):
    env = environment(browser_session_mode='persistent')
    session = env._get_browser_session()
    assert session.options.session_mode == 'persistent'
    session.begin_episode = Mock(return_value={'isolated': False, 'session_mode': 'persistent',
                                               'isolation_scope': 'task_tabs'})
    assert env.begin_episode()['isolation_scope'] == 'task_tabs'
    env.close()


@pytest.mark.parametrize('value', ['', 'reuse', None, 1, []])
def test_invalid_session_mode_rejected(value):
    with pytest.raises(ValueError, match='session_mode'):
        BrowserOptions(session_mode=value)
    with pytest.raises(ValueError, match='browser_session_mode'):
        WindowsEnvironment(browser_session_mode=value)
