"""A fixed task profile preserves storage without adopting somebody else's browser."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from osworld_agent import mac_chrome, persistent_chrome as chrome


@pytest.fixture(autouse=True)
def isolated_launcher_state(monkeypatch, tmp_path):
    chrome._LAUNCHES.clear()
    monkeypatch.setattr(chrome, '_settings_path', lambda: tmp_path / 'settings.json')
    yield
    chrome._LAUNCHES.clear()


def test_fixed_profile_reuses_cookie_and_preferences_bytes(tmp_path):
    profile = Path(chrome.prepare_persistent_profile(tmp_path / 'Task Profile'))
    (profile / 'Default').mkdir()
    cookies = profile / 'Default' / 'Cookies'
    preferences = profile / 'Default' / 'Preferences'
    cookies.write_bytes(b'persisted login database')
    preferences.write_text('{"intl":{"accept_languages":"de,en"}}')
    first_owner = chrome._marker(profile)
    assert chrome.prepare_persistent_profile(str(profile)) == str(profile)
    assert chrome._marker(profile) == first_owner
    assert cookies.read_bytes() == b'persisted login database'
    assert preferences.read_text() == '{"intl":{"accept_languages":"de,en"}}'


def test_refuses_existing_unowned_profile_and_daily_chrome(tmp_path):
    profile = tmp_path / 'unowned'
    profile.mkdir()
    (profile / 'Preferences').write_text('personal settings')
    with pytest.raises(ValueError, match='没有.*所有权标记'):
        chrome.prepare_persistent_profile(profile)
    assert not (profile / chrome._MARKER).exists()
    with pytest.raises(ValueError, match='日常'):
        chrome.prepare_persistent_profile(tmp_path / 'Google' / 'Chrome' / 'User Data')
    with pytest.raises(ValueError, match='绝对路径'):
        chrome.prepare_persistent_profile('relative/task')


def test_does_not_adopt_a_symlink(tmp_path):
    actual = tmp_path / 'actual'
    actual.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match='符号链接'):
        chrome.prepare_persistent_profile(alias)


def test_reuses_only_confirmed_profile_and_never_takes_its_ownership(tmp_path, monkeypatch):
    profile = chrome.prepare_persistent_profile(tmp_path / 'task')
    monkeypatch.setattr(chrome, '_port_open', lambda _: True)
    monkeypatch.setattr(chrome, '_endpoint_owned', lambda *args: True)
    monkeypatch.setattr(mac_chrome, '_cdp_ready', lambda _: True)
    assert chrome.ensure_persistent_chrome(9222, '/unused', profile_dir=profile, platform='mac') == (False, profile)
    assert chrome._LAUNCHES == {}
    chrome.close_persistent_chrome(9222, profile)
    assert Path(profile).is_dir()
    monkeypatch.setattr(chrome, '_endpoint_owned', lambda *args: False)
    with pytest.raises(RuntimeError, match='不会接管其他浏览器'):
        chrome.ensure_persistent_chrome(9222, '/unused', profile_dir=profile, platform='mac')


def test_new_profile_launch_does_not_force_incognito_or_locale(tmp_path, monkeypatch):
    executable = tmp_path / 'chrome'
    executable.touch()
    launched = []
    def popen(command, **kwargs):
        launched.append(command)
        return SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(chrome.subprocess, 'Popen', popen)
    monkeypatch.setattr(chrome, '_port_open', lambda _: False)
    monkeypatch.setattr(chrome, '_processes', lambda *args: (set(), set()))
    monkeypatch.setattr(chrome, '_endpoint_owned', lambda *args: True)
    monkeypatch.setattr(mac_chrome, '_cdp_ready', lambda _: True)
    monkeypatch.setattr(chrome.time, 'sleep', lambda _: None)
    created, profile = chrome.ensure_persistent_chrome(9222, str(executable), profile_dir=tmp_path / 'task', platform='mac')
    assert created and Path(profile).is_dir()
    assert not any(flag.startswith(('--incognito', '--lang=', '--disable-extensions', '--proxy-server')) for flag in launched[0])
    assert chrome._LAUNCHES[(profile, 9222)]['pid'] == 123


def test_rejects_same_profile_running_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setattr(chrome, '_port_open', lambda _: False)
    monkeypatch.setattr(chrome, '_processes', lambda *args: ({123}, set()))
    with pytest.raises(RuntimeError, match='其他端口'):
        chrome.ensure_persistent_chrome(9222, '/unused', profile_dir=tmp_path / 'task', platform='mac')


def test_close_keeps_profile_and_only_signals_current_launch(tmp_path, monkeypatch):
    profile = chrome.prepare_persistent_profile(tmp_path / 'task')
    cookie = Path(profile) / 'Cookies'
    cookie.write_bytes(b'login survives')
    chrome._LAUNCHES[(profile, 9222)] = {'pid': 123, 'owner_id': chrome._marker(profile)['owner_id'], 'platform': 'mac'}
    monkeypatch.setattr(chrome, '_processes', lambda *args: ({123}, set()))
    monkeypatch.setattr(chrome, '_endpoint_owned', lambda *args: False)
    monkeypatch.setattr(chrome.time, 'sleep', lambda _: None)
    signaled = []
    monkeypatch.setattr(chrome.os, 'kill', lambda pid, sig: signaled.append(pid))
    chrome.close_persistent_chrome(9222, profile)
    assert signaled == [123]
    assert cookie.read_bytes() == b'login survives'
    assert (profile, 9222) not in chrome._LAUNCHES


def test_close_refuses_reused_pid_and_changed_owner(tmp_path, monkeypatch):
    profile = chrome.prepare_persistent_profile(tmp_path / 'task')
    chrome._LAUNCHES[(profile, 9222)] = {'pid': 123, 'owner_id': chrome._marker(profile)['owner_id'], 'platform': 'mac'}
    monkeypatch.setattr(chrome, '_processes', lambda *args: ({456}, {456}))
    monkeypatch.setattr(chrome.os, 'kill', lambda *args: pytest.fail('must not signal a different browser'))
    chrome.close_persistent_chrome(9222, profile)
    chrome._LAUNCHES[(profile, 9222)] = {'pid': 456, 'owner_id': 'unrelated', 'platform': 'mac'}
    chrome.close_persistent_chrome(9222, profile)


def test_mac_process_matching_supports_application_support_spaces(tmp_path, monkeypatch):
    profile = tmp_path / 'Application Support' / 'AgentFrame' / 'TaskProfile'
    line = f'123 /Applications/Google Chrome --user-data-dir={profile} --remote-debugging-port=9222\n'
    line += f'456 /Applications/Google Chrome --user-data-dir={profile}-other --remote-debugging-port=9223\n'
    monkeypatch.setattr(mac_chrome.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=line))
    assert mac_chrome._profile_pids(str(profile)) == {123}


def test_studio_settings_save_mode_and_custom_task_directory(tmp_path):
    from fastapi.testclient import TestClient
    from osworld_agent.viz.server import app
    client = TestClient(app)
    initial = client.get('/browser/session').json()
    assert initial['browser_session_mode'] == 'persistent'
    profile = str(tmp_path / 'custom task browser')
    response = client.post('/browser/session', json={'browser_session_mode':'persistent', 'browser_profile_dir':profile})
    assert response.status_code == 200
    assert client.get('/browser/session').json()['effective_profile_dir'] == profile
    assert client.post('/browser/session', json={'browser_session_mode':'isolated'}).json()['browser_session_mode'] == 'isolated'
    assert client.get('/browser/session').json()['browser_profile_dir'] == profile
    assert client.post('/browser/session', json={'browser_session_mode':'isolated', 'browser_profile_dir':''}).json()['browser_profile_dir'] == profile
    assert client.post('/browser/session', json={'browser_session_mode':'daily'}).status_code == 422


def test_studio_passes_persistent_choice_into_platform_runner(tmp_path, monkeypatch):
    from osworld_agent.viz import server
    from osworld_agent import workspace_config
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    monkeypatch.setattr(workspace_config, '_settings_path', lambda: tmp_path / 'workspace.json')
    workspace_config.save_workspace_root(str(workspace))
    command = []
    monkeypatch.setattr(server.subprocess, 'Popen', lambda args, **kwargs: command.extend(args) or object())
    monkeypatch.setattr(server.threading, 'Thread', lambda **kwargs: SimpleNamespace(start=lambda:None))
    monkeypatch.setattr(server, '_agent_state', {'status':'idle'})
    profile = str(tmp_path / 'fixed task profile')
    assert server._start_local({'task':'打开网页', 'browser_session_mode':'persistent', 'browser_profile_dir':profile})['ok']
    assert command[command.index('--browser-session-mode') + 1] == 'persistent'
    assert command[command.index('--browser-profile-dir') + 1] == profile


def test_studio_isolated_run_preserves_fixed_profile_without_passing_it(tmp_path, monkeypatch):
    from osworld_agent.viz import server
    from osworld_agent import workspace_config
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    monkeypatch.setattr(workspace_config, '_settings_path', lambda: tmp_path / 'workspace.json')
    workspace_config.save_workspace_root(str(workspace))
    profile = str(tmp_path / 'signed-in task profile')
    chrome.save_browser_settings('persistent', profile)
    command = []
    monkeypatch.setattr(server.subprocess, 'Popen', lambda args, **kwargs: command.extend(args) or object())
    monkeypatch.setattr(server.threading, 'Thread', lambda **kwargs: SimpleNamespace(start=lambda:None))
    monkeypatch.setattr(server, '_agent_state', {'status':'idle'})
    assert server._start_local({'task':'打开网页', 'browser_session_mode':'isolated', 'browser_profile_dir':''})['ok']
    assert command[command.index('--browser-session-mode') + 1] == 'isolated'
    assert '--browser-profile-dir' not in command
    assert chrome.browser_settings()['browser_profile_dir'] == profile
    command.clear()
    assert server._start_local({'task':'打开网页', 'browser_session_mode':'persistent'})['ok']
    assert command[command.index('--browser-profile-dir') + 1] == profile


@pytest.mark.parametrize('failing', [None, 'model', 'grounding', 'environment', 'agent',
                                     'reset', 'log', 'run', 'env_close'])
def test_persistent_runner_closes_new_instance_after_success_or_errors(tmp_path, monkeypatch, failing):
    from test_incognito_lifecycle import _stub_main
    from osworld_agent import run_windows_local as runner
    profile = str(tmp_path / 'fixed profile')
    events, isolated_cleanup, error = _stub_main(monkeypatch, profile, failing=failing)
    sys.argv[sys.argv.index('--browser-session-mode') + 1] = 'persistent'
    sys.argv.extend(['--result-dir', str(tmp_path / 'runs')])
    monkeypatch.setattr(runner.time, 'sleep', lambda _:None)
    start = Mock(return_value=(True, profile))
    close = Mock(side_effect=lambda *args:events.append('persistent_close'))
    monkeypatch.setattr(chrome, 'ensure_persistent_chrome', start)
    monkeypatch.setattr(chrome, 'close_persistent_chrome', close)
    if failing:
        with pytest.raises(RuntimeError) as caught:
            runner.main()
        assert caught.value is error
    else:
        assert runner.main() == 0
    start.assert_called_once_with(9222, runner.CHROME_DEFAULT, profile_dir=None, platform='windows')
    close.assert_called_once_with(9222, profile)
    isolated_cleanup.assert_not_called()
    if failing not in {'model', 'grounding', 'environment'}:
        assert events.index('env_close') < events.index('persistent_close')


@pytest.mark.parametrize('failing', [None, 'run', 'env_close'])
def test_persistent_runner_does_not_close_reused_instance(tmp_path, monkeypatch, failing):
    from test_incognito_lifecycle import _stub_main
    from osworld_agent import run_windows_local as runner
    profile = str(tmp_path / 'fixed profile')
    _, isolated_cleanup, error = _stub_main(monkeypatch, profile, failing=failing)
    sys.argv[sys.argv.index('--browser-session-mode') + 1] = 'persistent'
    sys.argv.extend(['--result-dir', str(tmp_path / 'runs')])
    monkeypatch.setattr(runner.time, 'sleep', lambda _:None)
    monkeypatch.setattr(chrome, 'ensure_persistent_chrome', Mock(return_value=(False, profile)))
    close = Mock()
    monkeypatch.setattr(chrome, 'close_persistent_chrome', close)
    if failing:
        with pytest.raises(RuntimeError) as caught:
            runner.main()
        assert caught.value is error
    else:
        assert runner.main() == 0
    close.assert_not_called()
    isolated_cleanup.assert_not_called()


def test_manual_login_explicit_profile_becomes_next_studio_profile(tmp_path, monkeypatch):
    profile = str(tmp_path / 'new signed-in profile')
    monkeypatch.setattr(sys, 'argv', ['persistent_chrome', '--url', 'about:blank', '--browser-profile-dir', profile])
    monkeypatch.setattr(chrome, 'ensure_persistent_chrome', Mock(return_value=(True, profile)))
    close = Mock()
    monkeypatch.setattr(chrome, 'close_persistent_chrome', close)
    page = SimpleNamespace(goto=Mock(), close=Mock())
    browser = SimpleNamespace(contexts=[SimpleNamespace(new_page=lambda:page)], close=Mock())
    runtime = MagicMock()
    runtime.__enter__.return_value.chromium.connect_over_cdp.return_value = browser
    monkeypatch.setitem(sys.modules, 'playwright.sync_api', SimpleNamespace(sync_playwright=lambda:runtime))
    monkeypatch.setattr('osworld_agent.utils.cdp_websocket_endpoint', lambda *args:'ws://127.0.0.1:9222/owned')
    monkeypatch.setattr('builtins.input', lambda:'')
    assert chrome.main() == 0
    assert chrome.browser_settings()['browser_profile_dir'] == profile
    assert chrome.browser_settings()['browser_session_mode'] == 'persistent'
    close.assert_called_once_with(9222, profile)


def test_manual_failed_profile_preparation_does_not_change_saved_choice(tmp_path, monkeypatch):
    original = str(tmp_path / 'original profile')
    chrome.save_browser_settings('persistent', original)
    monkeypatch.setattr(sys, 'argv', ['persistent_chrome', '--browser-profile-dir', str(tmp_path / 'foreign')])
    monkeypatch.setattr(chrome, 'ensure_persistent_chrome', Mock(side_effect=ValueError('unowned profile')))
    with pytest.raises(ValueError, match='unowned'):
        chrome.main()
    assert chrome.browser_settings()['browser_profile_dir'] == original
