"""Mac adapter contracts: browser reuse, desktop focus, coordinates, host selection."""
import os
import signal
from types import SimpleNamespace

from osworld_agent.actions import Action
from osworld_agent.adapters.mac_env import MacEnvironment
from osworld_agent.adapters import mac_env
from osworld_agent import mac_chrome, mac_desktop, run_local
from osworld_agent.env import Observation


def _window(hwnd=1, foreground=True):
    return {'process_name': 'finder', 'pid': 42, 'hwnd': hwnd,
            'bounds': [0, 0, 800, 600], 'foreground': foreground,
            'visible': True}


def _state(hwnd=1):
    return {'applications': [_window(hwnd)], 'complete': True,
            'source': 'mac_quartz_window_server'}


def test_host_platform_auto_selection(monkeypatch):
    monkeypatch.setattr(run_local.sys, 'platform', 'darwin')
    assert run_local.host_platform() == 'mac'
    monkeypatch.setattr(run_local.sys, 'platform', 'win32')
    assert run_local.host_platform() == 'windows'


def test_mac_chrome_matches_var_alias_and_closes_without_cdp_listener(monkeypatch, tmp_path):
    real_temp = tmp_path / 'private'
    real_temp.mkdir()
    alias_temp = tmp_path / 'var'
    alias_temp.symlink_to(real_temp, target_is_directory=True)
    profile = alias_temp / 'osworld_agent_chrome_incognito_9222_task'
    profile.mkdir()
    monkeypatch.setattr(mac_chrome.tempfile, 'gettempdir', lambda: str(alias_temp))
    command = f'101 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --user-data-dir={profile}\n'
    monkeypatch.setattr(mac_chrome.subprocess, 'run', lambda *_args, **_kwargs: SimpleNamespace(stdout=command))
    assert mac_chrome._profile_pids(str(profile)) == {101}

    alive = {101}
    killed = []
    monkeypatch.setattr(mac_chrome, '_profile_pids', lambda _profile: set(alive))
    monkeypatch.setattr(mac_chrome, '_listener_pids', lambda _port: set())
    def kill(pid, sig):
        killed.append((pid, sig))
        alive.discard(pid)
    monkeypatch.setattr(mac_chrome.os, 'kill', kill)
    mac_chrome.close_test_chrome(9222, str(profile))
    assert killed == [(101, signal.SIGTERM)]
    assert not profile.exists()


def test_mac_chrome_keeps_foreign_listener(monkeypatch, tmp_path):
    profile = tmp_path / 'osworld_agent_chrome_incognito_9222_task'
    profile.mkdir()
    monkeypatch.setattr(mac_chrome.tempfile, 'gettempdir', lambda: str(tmp_path))
    alive = {101}
    monkeypatch.setattr(mac_chrome, '_profile_pids', lambda _profile: set(alive))
    monkeypatch.setattr(mac_chrome, '_listener_pids', lambda _port: {202})
    def kill(pid, _sig):
        assert pid == 101
        alive.discard(pid)
    monkeypatch.setattr(mac_chrome.os, 'kill', kill)
    mac_chrome.close_test_chrome(9222, str(profile))
    assert not profile.exists()


def test_mac_browser_keeps_shared_cdp_observation(monkeypatch):
    env = MacEnvironment()
    env._browser_session = SimpleNamespace(collect_observation=lambda: (
        b'png', {'kind': 'browser_ax', 'snapshot_id': 's'},
        {'screenshot_space': 'css_viewport', 'page_id': 'p', 'snapshot_id': 's'}))
    monkeypatch.setattr(mac_desktop, 'runtime_state', lambda: _state())
    obs = env.observe()
    assert env.os_name == 'mac'
    assert obs.screenshot == b'png' and obs.context[0]['snapshot_id'] == 's'
    assert obs.info['active_surface'] == 'browser'


def test_mac_desktop_rejects_changed_focus_before_input(monkeypatch):
    monkeypatch.setattr(mac_env, "sys", SimpleNamespace(platform="darwin"))
    env = MacEnvironment()
    env._observation_surface = 'desktop'
    env._desktop_capture = {'virtual_desktop': {'left': -100, 'top': 0, 'width': 900, 'height': 600},
                            'desktop_capture_id': 'shot'}
    env._last_observation_info = {'runtime_state': _state(), 'desktop_focus_stable': True}
    monkeypatch.setattr(mac_desktop, 'runtime_state', lambda: _state(hwnd=2))
    monkeypatch.setattr(env, 'observe', lambda: Observation())
    sent = []
    monkeypatch.setattr(mac_desktop, 'execute_action', lambda *args, **kwargs: sent.append(args))
    result = env.step(Action('click', {'x': 50, 'y': 40, 'coordinate_space': 'screen_pixels'}))
    assert result.info['error_code'] == 'desktop_focus_unconfirmed'
    assert result.info['dispatched'] is False and not sent


def test_mac_desktop_rejects_window_moved_after_screenshot(monkeypatch):
    observed = _state()
    moved = _state()
    moved['applications'][0]['bounds'] = [40, 20, 840, 620]
    try:
        mac_desktop.check_keyboard_focus(observed, moved)
    except mac_desktop.DesktopFocusError:
        pass
    else:
        raise AssertionError('moved desktop window must require a new screenshot')


def test_mac_desktop_uses_capture_origin_and_rejects_changed_layout(monkeypatch):
    monkeypatch.setattr(mac_env, "sys", SimpleNamespace(platform="darwin"))
    env = MacEnvironment()
    env._observation_surface = 'desktop'
    geometry = {'left': -100, 'top': 0, 'width': 900, 'height': 600}
    env._desktop_capture = {'virtual_desktop': geometry, 'desktop_capture_id': 'shot'}
    env._last_observation_info = {'runtime_state': _state(), 'desktop_focus_stable': True}
    monkeypatch.setattr(mac_desktop, 'runtime_state', lambda: _state())
    monkeypatch.setattr(mac_desktop, 'virtual_geometry', lambda: geometry)
    monkeypatch.setattr(mac_desktop, '_accessibility_trusted', lambda: True)
    monkeypatch.setattr(env, 'observe', lambda: Observation())
    sent = []
    monkeypatch.setattr(mac_desktop, 'execute_action', lambda action, origin, **kw: sent.append((action, origin)))
    result = env.step(Action('click', {'x': 50, 'y': 40, 'coordinate_space': 'screen_pixels'}))
    assert result.info['status'] == 'executed' and sent[0][1] == (-100, 0)
    monkeypatch.setattr(mac_desktop, 'virtual_geometry', lambda: {**geometry, 'width': 901})
    result = env.step(Action('click', {'x': 50, 'y': 40, 'coordinate_space': 'screen_pixels'}))
    assert result.info['status'] == 'rejected' and len(sent) == 1


def test_mac_desktop_reports_uncertain_after_dispatch_starts(monkeypatch):
    monkeypatch.setattr(mac_env, "sys", SimpleNamespace(platform="darwin"))
    env = MacEnvironment()
    env._observation_surface = 'desktop'
    geometry = {'left': 0, 'top': 0, 'width': 800, 'height': 600}
    env._desktop_capture = {'virtual_desktop': geometry, 'desktop_capture_id': 'shot'}
    env._last_observation_info = {'runtime_state': _state(), 'desktop_focus_stable': True}
    monkeypatch.setattr(mac_desktop, 'runtime_state', lambda: _state())
    monkeypatch.setattr(mac_desktop, 'virtual_geometry', lambda: geometry)
    monkeypatch.setattr(mac_desktop, '_accessibility_trusted', lambda: True)
    monkeypatch.setattr(env, 'observe', lambda: Observation())
    def partial_input(*args, **kwargs):
        raise mac_desktop.DesktopFocusError('焦点已变化')
    monkeypatch.setattr(mac_desktop, 'execute_action', partial_input)
    result = env.step(Action('click', {'x': 50, 'y': 40, 'coordinate_space': 'screen_pixels'}))
    assert result.info['status'] == 'uncertain'
    assert result.info['dispatched'] is True


def test_mac_display_geometry_preserves_negative_monitor_origin(monkeypatch):
    monkeypatch.setattr(mac_desktop, '_display_rects', lambda: [
        (1, 0, 0, 1440, 900), (2, -1920, -120, 1920, 1080)])
    assert mac_desktop.virtual_geometry() == {
        'left': -1920, 'top': -120, 'width': 3360, 'height': 1080}
