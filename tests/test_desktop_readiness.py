"""Slow Windows launch/focus races simulated without real user input."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from osworld_agent import desktop_readiness as readiness, desktop_state, desktop_capture
from osworld_agent.actions import Action
from osworld_agent.adapters import windows_env
from osworld_agent.adapters.windows_env import WindowsEnvironment
from osworld_agent.env import Observation
from osworld_agent.execution_queue import normalize_verification
from osworld_agent.step_verifier import code_review
from test_browser_lifecycle import session, EventBrowser


def window(**kw):
    return dict(dict(process_name='explorer.exe', pid=20, hwnd=2, title='Downloads',
        class_name='CabinetWClass', bounds=[-1500, -100, -100, 900], foreground=True,
        visible=True, minimized=False, cloaked=False, enabled=True, responsive=True), **kw)


def state(*windows):
    return {'applications': list(windows), 'complete': True, 'source': 'windows_window_api'}


class Clock:
    def __init__(self):
        self.value = 0
    def now(self):
        return self.value
    def sleep(self, seconds):
        self.value += seconds


def wait(read, **kw):
    clock = kw.pop('clock', Clock())
    return readiness.wait_for_window({'process_name': 'explorer.exe'}, read_state=read,
        clock=clock.now, sleep=clock.sleep, timeout_s=kw.pop('timeout_s', 10), **kw)


def test_slow_launch_waits_for_target_not_other_desktop_changes():
    clock, events = Clock(), []
    def read():
        return state(window() if clock.value >= 5 else window(process_name='chrome.exe', hwnd=50))
    result = wait(read, clock=clock, emit=events.append)
    assert result['status'] == 'completed' and result['elapsed_s'] == 5.25
    assert result['window']['bounds'][0] < 0
    assert len([e for e in events if e['status'] == 'waiting']) >= 5
    assert all(e['reason'] == 'desktop_window' for e in events)


def test_existing_explorer_activated_counts_without_requiring_new_hwnd():
    result = wait(lambda: state(window(hwnd=11, foreground=False), window(hwnd=12)))
    assert result['status'] == 'completed' and result['window']['hwnd'] == 12


@pytest.mark.parametrize('change', [
    {'class_name': 'Progman'}, {'class_name': 'Shell_TrayWnd'}, {'foreground': False},
    {'minimized': True}, {'cloaked': True}, {'enabled': False}, {'responsive': False},
    {'cloaked': None}, {'bounds': [0, 0, 0, 0]}, {'bounds': None},
])
def test_visible_or_existing_window_alone_is_not_readiness(change):
    assert wait(lambda: state(window(**change)), timeout_s=.5)['status'] == 'timeout'


def test_bounds_must_stabilize_and_foreground_must_be_unique():
    clock = Clock()
    def read():
        if clock.value < .5:
            return state(window(), window(hwnd=3))
        return state(window(bounds=[int(clock.value * 100), 0, 900, 900]) if clock.value < 1 else window())
    assert wait(read, clock=clock)['elapsed_s'] == 1.25


def test_wait_cancellation_and_task_deadline_are_bounded():
    clock = Clock()
    assert wait(lambda: state(), clock=clock, remaining=lambda: .1-clock.value)['status'] == 'task_timeout'
    assert clock.value == .1
    assert wait(lambda: state(), cancelled=lambda: True)['status'] == 'cancelled'


@pytest.mark.parametrize('current', [window(hwnd=3), window(pid=30), window(responsive=False)])
def test_keyboard_guard_rejects_focus_change_handle_reuse_and_hung_window(current):
    with pytest.raises(readiness.DesktopFocusError):
        readiness.check_keyboard_focus(state(window()), state(current))


def test_keyboard_guard_preserves_pending_target_and_capture_consistency():
    chrome = window(process_name='chrome.exe')
    with pytest.raises(readiness.DesktopFocusError):
        readiness.check_keyboard_focus(state(chrome), state(chrome), pending_target={'process_name': 'explorer.exe'})
    with pytest.raises(readiness.DesktopFocusError):
        readiness.check_keyboard_focus(state(window()), state(window()), capture_stable=False)
    assert readiness.check_keyboard_focus(state(window()), state(window()))['hwnd'] == 2


@pytest.fixture
def desktop(monkeypatch):
    env = WindowsEnvironment(window_timeout_s=.5)
    env._observation_surface = 'desktop'
    env._last_observation_info = {'runtime_state': state(window()), 'desktop_focus_stable': True}
    geometry = {'left': -1600, 'top': -100, 'width': 3200, 'height': 1100}
    env._desktop_capture = {'virtual_desktop': geometry, 'screen_origin': [-1600, -100],
                            'desktop_capture_id': 'shot', 'screenshot_size': [3200, 1100]}
    monkeypatch.setattr(windows_env, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.setattr(desktop_state, 'windows_runtime_state', lambda: state(window()))
    monkeypatch.setattr(desktop_capture, 'physical_dpi_context', nullcontext)
    monkeypatch.setattr(desktop_capture, 'virtual_geometry', lambda: geometry)
    env.observe = Mock(side_effect=lambda: Observation(info=dict(env._last_observation_info)))
    gui = SimpleNamespace(hotkey=Mock(), press=Mock(), click=Mock())
    monkeypatch.setitem(__import__('sys').modules, 'pyautogui', gui)
    writer = Mock()
    monkeypatch.setattr('osworld_agent.windows_text.type_unicode', writer)
    return env, gui, writer


def fake_wait(monkeypatch, read):
    real_wait = readiness.wait_for_window
    def impl(target, **kwargs):
        clock = Clock()
        return real_wait(target, read_state=read, clock=clock.now, sleep=clock.sleep, **kwargs)
    monkeypatch.setattr(readiness, 'wait_for_window', impl)


def test_launch_finishes_wait_before_screenshot_and_does_not_repeat_on_timeout(desktop, monkeypatch):
    env, gui, _ = desktop
    fake_wait(monkeypatch, lambda: state())
    events = []
    with env.framework_wait_context(events.append, lambda: 100):
        result = env.step(Action('hotkey', {'keys': ['win', 'e']}))
        assert result.info['status'] == 'uncertain' and result.info['framework_wait']['status'] == 'timeout'
        result = env.step(Action('hotkey', {'keys': ['win', 'e']}))
    gui.hotkey.assert_called_once_with('win', 'e')
    assert not result.info['dispatched']
    assert env.observe.call_count == 2 and events[-1]['status'] == 'timeout'
    assert env._window_wait_emit is None


def test_timeout_blocks_address_bar_input_in_wrong_app(desktop, monkeypatch):
    env, gui, _ = desktop
    chrome = window(process_name='chrome.exe')
    env._last_observation_info['runtime_state'] = state(chrome)
    env._pending_window = {'process_name': 'explorer.exe'}
    monkeypatch.setattr(desktop_state, 'windows_runtime_state', lambda: state(chrome))
    result = env.step(Action('hotkey', {'keys': ['ctrl', 'l']}))
    assert not result.info['dispatched'] and result.info['error_code'] == 'desktop_focus_unconfirmed'
    gui.hotkey.assert_not_called()


def test_type_click_focus_race_never_writes_or_overwrites(desktop, monkeypatch):
    env, gui, writer = desktop
    current = [window()]
    monkeypatch.setattr(desktop_state, 'windows_runtime_state', lambda: state(current[0]))
    gui.click.side_effect = lambda *_: current.__setitem__(0, window(process_name='chrome.exe', hwnd=88))
    result = env.step(Action('type', {'x': 100, 'y': 100, 'coordinate_space': 'screen_pixels',
        'text': 'C:\\工作区\\data.pdf', 'overwrite': True}))
    assert result.info['error_code'] == 'desktop_focus_changed_during_action'
    gui.click.assert_called_once()
    gui.hotkey.assert_not_called()
    gui.press.assert_not_called()
    writer.assert_not_called()


def test_stable_foreground_accepts_unicode_input(desktop):
    env, _, writer = desktop
    result = env.step(Action('type', {'coordinate_space': 'screen_pixels', 'text': 'C:\\工作区'}))
    assert result.info['status'] == 'executed'
    writer.assert_called_once_with('C:\\工作区')


def test_missing_switch_does_not_prevent_first_win_e_launch(desktop, monkeypatch):
    env, gui, _ = desktop
    monkeypatch.setattr(desktop_state, 'switch_windows_application', lambda *_: {'status': 'rejected', 'dispatched': False})
    env.step(Action('switch_applications', {'app_code': 'explorer'}))
    fake_wait(monkeypatch, lambda: state(window()))
    result = env.step(Action('hotkey', {'keys': ['win', 'e']}))
    assert result.info['framework_wait']['status'] == 'completed'
    assert env._pending_window is None
    gui.hotkey.assert_called_once_with('win', 'e')


def test_cancel_prevents_desktop_dispatch(desktop):
    env, gui, _ = desktop
    env.cancel_pending_wait()
    result = env.step(Action('hotkey', {'keys': ['win', 'e']}))
    assert not result.info['dispatched']
    gui.hotkey.assert_not_called()


def test_slow_activation_rechecks_same_hwnd_and_clears_provisional_error(desktop, monkeypatch):
    env, _, _ = desktop
    monkeypatch.setattr(desktop_state, 'switch_windows_application', lambda *_: {
        'status': 'uncertain', 'dispatched': True, 'window': window(foreground=False),
        'error_code': 'foreground_not_confirmed'})
    fake_wait(monkeypatch, lambda: state(window()))
    result = env.step(Action('switch_applications', {'app_code': 'explorer'}))
    assert result.info['status'] == 'executed' and 'error_code' not in result.info
    assert result.info['window']['foreground'] is True


def test_code_predicates_distinguish_exists_foreground_and_os_ready():
    def review(kind, app):
        pred = normalize_verification([{'kind': kind, 'value': 'explorer'}])
        return code_review({'verification': pred}, Observation(info={'runtime_state': state(app)}), {}, {})['status']
    assert review('app_open', window(foreground=False)) == 'passed'
    assert review('app_foreground', window(foreground=False)) == 'failed'
    assert review('app_foreground', window(responsive=False)) == 'passed'
    assert review('app_ready', window(responsive=False)) == 'failed'
    assert review('app_ready', window(cloaked=None)) == 'unknown'
    assert review('app_ready', window()) == 'passed'


def test_missing_os_readiness_never_delegates_to_a_visual_model():
    from test_browser_operational_feedback import pipe
    from osworld_agent.pipeline import StepState
    p = pipe()
    p.decision_judge = Mock()
    p.reviewer = Mock()
    report = {'method': 'code', 'status': 'unknown', 'checks': [{'kind': 'app_ready', 'status': 'unknown'}]}
    assert p._review_visual(StepState(), {}, {}, report, {})['status'] == 'unknown'
    p.reviewer.visual_review.assert_not_called()
    p.decision_judge.review.assert_not_called()


def test_jev_keeps_foreground_window_readiness_even_after_many_background_windows():
    from osworld_agent.judge_evidence import select_frame
    windows = [window(hwnd=i+100, foreground=False) for i in range(10)] + [window()]
    obs = Observation(info={'runtime_state': state(*windows)})
    selected = select_frame(obs)['runtime_state']['applications']
    assert len(selected) == 4 and selected[0]['foreground']
    assert selected[0]['responsive'] is True and selected[0]['cloaked'] is False


def test_episode_cleanup_keeps_foreign_context_pages_even_blank(session):
    browser = EventBrowser()
    foreign = browser.new_context()
    blank, user_page = foreign.new_page(), foreign.new_page()
    user_page.url = 'https://internal.example/'
    session._connect = Mock(return_value=(None, browser))
    session.begin_episode('first')
    own = session._owned_context
    session.begin_episode('second')
    assert own.closed
    assert not foreign.closed and not blank.closed and not user_page.closed


def test_pipeline_forwards_live_wait_without_additional_policy_calls(desktop, monkeypatch):
    from test_browser_operational_feedback import pipe
    env, _, _ = desktop
    fake_wait(monkeypatch, lambda: state(window()))
    p = pipe()
    p.env = env
    p._publish_diagnostic = Mock()
    result = p._step_environment(Action('hotkey', {'keys': ['win', 'e']}))
    assert result.info['framework_wait']['status'] == 'completed'
    assert p.model.calls == []
    assert p._publish_diagnostic.call_args.kwargs['status'] == 'completed'
    assert p._publish_diagnostic.call_args.args == ('framework_wait',)
