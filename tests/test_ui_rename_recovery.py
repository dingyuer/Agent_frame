"""GUI-only rename recovery: sparse PDF AX, real window identity, and path hints."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from osworld_agent.actions import Action, BASIC_ACTIONS
from osworld_agent import desktop_state
from osworld_agent.adapters.vmware_env import VMwareEnvironment, to_pyautogui
from osworld_agent.adapters.windows_env import WindowsEnvironment
from osworld_agent.browser_feedback import operational_facts, rename_ui_guidance
from osworld_agent.pipeline import StepState
from test_browser_operational_feedback import bound, pipe, pdf_obs
from test_pipeline_completion import observation


@pytest.fixture(autouse=True)
def no_viz(monkeypatch):
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)


def window(hwnd=123, foreground=False, title='Downloads', **kwargs):
    return dict(process_name='explorer.exe', hwnd=hwnd, title=title, class_name='CabinetWClass',
                foreground=foreground, **kwargs)


def runtime(monkeypatch, windows):
    monkeypatch.setattr(desktop_state, 'windows_runtime_state', lambda: {'applications': windows, 'complete': True})
    activate = Mock(return_value=(True, True))
    monkeypatch.setattr(desktop_state, '_activate_window', activate)
    return activate


def test_switch_targets_explorer_not_desktop_shell_or_random_window(monkeypatch):
    shell = dict(window(1), class_name='Progman')
    chrome = dict(window(2, True), process_name='chrome.exe')
    activate = runtime(monkeypatch, [shell, chrome, window(3)])
    result = desktop_state.switch_windows_application('File Explorer')
    activate.assert_called_once_with(3)
    assert result['status'] == 'executed' and result['dispatched'] is True


def test_already_foreground_does_not_toggle_away(monkeypatch):
    activate = runtime(monkeypatch, [window(1, True), window(2)])
    result = desktop_state.switch_windows_application('explorer')
    assert result['status'] == 'observed' and result['dispatched'] is False
    assert '不要反复切换' in result['reason']
    activate.assert_not_called()


def test_unopened_explorer_has_actionable_no_dispatch_feedback(monkeypatch):
    activate = runtime(monkeypatch, [dict(window(), class_name='Progman')])
    result = desktop_state.switch_windows_application('file_manager')
    assert result['error_code'] == 'application_not_found' and not result['dispatched']
    assert '["win","e"]' in result['reason']
    activate.assert_not_called()


def test_multiple_windows_require_explicit_visual_choice(monkeypatch):
    activate = runtime(monkeypatch, [window(1), window(2)])
    result = desktop_state.switch_windows_application('file_explorer')
    assert result['error_code'] == 'ambiguous_application' and not result['dispatched']
    activate.assert_not_called()


def test_failed_foreground_activation_is_not_success(monkeypatch):
    activate = runtime(monkeypatch, [window()])
    activate.return_value = True, False
    result = desktop_state.switch_windows_application('explorer.exe')
    assert result['status'] == 'uncertain' and result['error_code'] == 'foreground_not_confirmed'


def test_app_code_is_never_treated_as_command(monkeypatch):
    activate = runtime(monkeypatch, [window()])
    result = desktop_state.switch_windows_application('explorer.exe & del C:\\data.pdf')
    assert result['error_code'] == 'unknown_application'
    activate.assert_not_called()


def test_windows_adapter_uses_targeted_switch_only_after_desktop_observation(monkeypatch):
    env = WindowsEnvironment()
    monkeypatch.setattr(env, 'observe', lambda: observation())
    switch = Mock(return_value={'status': 'executed', 'dispatched': True, 'channel': 'desktop'})
    monkeypatch.setattr(desktop_state, 'switch_windows_application', switch)
    action = Action('switch_applications', {'app_code': 'file_explorer'})
    result = env.step(action)
    assert result.info['status'] == 'rejected' and not result.info['dispatched']
    switch.assert_not_called()
    env.step(Action('observe_desktop', {}))
    result = env.step(action)
    switch.assert_called_once_with('file_explorer')
    assert result.info['status'] == 'executed'


def test_generic_translator_does_not_lie_about_application_switch():
    assert to_pyautogui(Action('switch_applications', {'app_code': 'file_manager'})) == ''
    env = object.__new__(VMwareEnvironment)
    env.observe = observation
    result = env.step(Action('switch_applications', {'app_code': 'file_manager'}))
    assert result.info['error_code'] == 'application_switch_unsupported' and not result.info['dispatched']
    assert 'rename_artifact' not in BASIC_ACTIONS


@pytest.mark.parametrize('nodes', [
    [{'ref': '1', 'role': 'RootWebArea', 'direct': False}, {'ref': '2', 'role': 'EmbeddedObject', 'direct': True}],
    [{'ref': '1', 'role': 'RootWebArea', 'direct': False}, {'ref': '2', 'role': 'button', 'name': 'Container', 'direct': True}],
])
def test_pdf_sparse_ax_never_requires_fabricated_reference(nodes):
    p = pipe()
    p.env.route_action = lambda a, o: None
    obs = pdf_obs()
    obs.context[0]['nodes'] = nodes
    obs.info['exposed_refs'] = [n['ref'] for n in nodes]
    assert p._dom_reference_feedback(Action('click', {'target': 'PDF toolbar download button'}), StepState(obs=obs)) is None


def test_single_real_html_control_still_gets_dom_guidance():
    p = pipe()
    p.env.route_action = lambda a, o: None
    obs = observation()
    obs.context[0]['nodes'][0]['direct'] = True
    assert p._dom_reference_feedback(Action('type', {'target': 'Email', 'text': 'a'}), StepState(obs=obs))
    obs.context[0]['nodes'][0]['role'] = 'EmbeddedObject'
    assert p._dom_reference_feedback(Action('click', {'target': 'embedded canvas'}), StepState(obs=obs)) is None


def test_valid_pdf_dom_ref_is_still_routed():
    p = pipe()
    state = StepState(obs=pdf_obs())
    action = Action('click', {'target': 'Download', 'target_ref': '2'})
    assert p._dom_reference_feedback(action, state) is None
    p.ground(action, state)
    assert state.route['channel'] == 'browser_dom'


def rename_observation():
    obs = observation()
    obs.info['active_surface'] = 'desktop'
    path = r'C:\Agent Downloads\episode 7\2603.11558v3.pdf'
    obs.info['artifacts'] = [{'completed': True, 'path': path, 'bytes': 4156037}]
    report = {'checks': [{'id': 'check_1', 'kind': 'paper_download', 'status': 'failed',
                          'reason': 'downloaded filename does not match', 'path': path}]}
    return obs, report


def test_rename_guidance_uses_verified_actual_directory_not_assumed_downloads():
    obs, report = rename_observation()
    before = copy.deepcopy(obs.info)
    text = '\n'.join(operational_facts(obs, bound(), report, os_name='windows'))
    assert 'episode 7' in text and '2603.11558v3.pdf' in text and 'data.pdf' in text
    assert 'Ctrl+L' in text and 'F2' in text and '双重 .pdf' in text
    assert '无需再次下载' in text and 'win' in text
    assert obs.info == before  # hints do not mutate artifacts or verification


def test_frontmost_explorer_hint_advances_to_content_instead_of_switching():
    obs, report = rename_observation()
    obs.info['runtime_state'] = {'applications': [window(foreground=True)]}
    text = '\n'.join(operational_facts(obs, bound(), report, os_name='windows'))
    assert '系统观察到的前台窗口：explorer.exe' in text
    assert '已在前台，继续定位文件' in text


@pytest.mark.parametrize('case', ['wrong_paper', 'unregistered', 'missing_path', 'linux', 'already_passed', 'relative_path'])
def test_no_rename_recipe_without_matching_public_artifact(case):
    obs, report = rename_observation()
    os_name = 'windows'
    if case == 'wrong_paper':
        report['checks'][0]['reason'] = 'download transport does not identify the bound official paper'
    elif case == 'unregistered':
        obs.info['artifacts'] = []
    elif case == 'missing_path':
        report['checks'][0].pop('path')
    elif case == 'linux':
        os_name = 'linux'
    elif case == 'already_passed':
        report['checks'][0].update(status='passed', reason='completed')
    else:
        report['checks'][0]['path'] = r'folder\file.pdf'
    assert rename_ui_guidance(obs, bound(), report, os_name) == []


@pytest.mark.parametrize('class_name,expected', [('CabinetWClass', 'passed'), ('Progman', 'failed')])
def test_app_open_uses_same_alias_and_excludes_desktop_shell(class_name, expected):
    from osworld_agent.step_verifier import code_review
    obs = observation()
    obs.info['runtime_state'] = {'source': 'windows_window_api', 'complete': True,
                                  'applications': [dict(window(), class_name=class_name)]}
    review = code_review({'verification': [{'kind': 'app_open', 'app': 'file_explorer'}]}, obs, {}, {})
    assert review['status'] == expected


@pytest.mark.parametrize('focus_allowed,minimized', [(True, False), (True, True), (False, False)])
def test_native_focus_checks_actual_window_and_preserves_visible_size(monkeypatch, focus_allowed, minimized):
    import ctypes
    from ctypes import wintypes
    target = 0x123456789  # HWND must not be truncated to a 32-bit default int.
    native = SimpleNamespace(
        IsWindow=Mock(return_value=True), IsIconic=Mock(return_value=minimized),
        ShowWindow=Mock(), SetForegroundWindow=Mock(return_value=focus_allowed),
        GetForegroundWindow=Mock(return_value=target if focus_allowed else 99))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *a, **kw: native, raising=False)
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(desktop_state.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(desktop_state.time, 'sleep', lambda seconds: setattr(clock, 'now', clock.now + seconds))
    dispatched, focused = desktop_state._activate_window(target)
    assert dispatched and focused == focus_allowed
    native.SetForegroundWindow.assert_called_once_with(target)
    assert native.SetForegroundWindow.argtypes == [wintypes.HWND]
    assert native.GetForegroundWindow.restype == wintypes.HWND
    if minimized:
        native.ShowWindow.assert_called_once_with(target, 9)
    else:
        native.ShowWindow.assert_not_called()
