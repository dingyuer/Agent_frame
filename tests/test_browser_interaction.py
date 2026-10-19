"""Scroll evidence is about measured movement, including boundary/no-data cases."""
from osworld_agent.browser_interaction import scroll_feedback
from osworld_agent.browser_interaction import SCROLL_CHAIN, SCROLL_METRICS
import test_browser_session_safety as fixtures
from unittest.mock import Mock


def metric(top, *, maximum=700, ident='inner', kind='element', connected=True):
    return {'scrollTop': top, 'clientHeight': 200, 'scrollHeight': maximum + 200,
            'connected': connected, 'container': {'kind': kind, 'id': ident}}


def test_scroll_feedback_reports_the_ancestor_that_moved():
    before = [metric(700), metric(0, ident='', kind='page')]
    after = [metric(700), metric(120, ident='', kind='page')]
    result = scroll_feedback(before, after, 120)
    assert result['moved'] is True and result['noop'] is False
    assert result['target_container']['kind'] == 'page'
    assert result['actual_delta'] == 120
    assert result['containers'][0]['actual_delta'] == 0


def test_scroll_feedback_reports_bottom_noop_and_upward_movement():
    result = scroll_feedback([metric(700)], [metric(700)], 120)
    assert result['noop'] is True and result['at_bottom'] is True
    assert result['actual_delta'] == 0 and result['at_top'] is False
    upward = scroll_feedback([metric(20)], [metric(0)], -120)
    assert upward['actual_delta'] == -20 and upward['at_top'] is True


def test_missing_or_detached_scroll_metrics_are_not_a_noop():
    for before, after in [(None, None), ([metric(0)], [metric(0, connected=False)]),
                          ([metric(0)], []), ([metric(0)], [metric(float('nan'))])]:
        result = scroll_feedback(before, after, 120)
        assert result['available'] is False
        assert result['moved'] is None and result['noop'] is None


def test_scroll_poll_measures_the_same_objects_without_replaying_wheel():
    case = fixtures.BrowserSessionSafetyTests(methodName='runTest')
    case.setUp()
    reads = []
    try:
        class MeasuredCDP(fixtures.FakeCDP):
            def send(self, command, params=None):
                if command == 'Runtime.callFunctionOn' and params['functionDeclaration'] == SCROLL_CHAIN:
                    return {'result': {'objectId': 'captured-containers'}}
                if command == 'Runtime.callFunctionOn' and params['functionDeclaration'] == SCROLL_METRICS:
                    reads.append(params['objectId'])
                    return {'result': {'value': [metric(700)]}}
                return super().send(command, params)
        case.page.context.new_cdp_session = lambda _: MeasuredCDP(case.page)
        case.observe()
        result = case.session.execute(fixtures.action('scroll', amount=1, direction='down'), case.route())
        assert result['scroll_feedback']['noop'] is True
        assert reads == ['captured-containers'] * 4  # before + at most three short after reads
        case.page.mouse.wheel.assert_called_once_with(0, 120)
        case.page.mouse.click.assert_not_called()
    finally:
        case.doCleanups()


def test_scroll_metric_failure_after_dispatch_does_not_trigger_retry():
    case = fixtures.BrowserSessionSafetyTests(methodName='runTest')
    case.setUp()
    reads = []
    releases = []
    try:
        class FailedCDP(fixtures.FakeCDP):
            def send(self, command, params=None):
                if command == 'Runtime.callFunctionOn' and params['functionDeclaration'] == SCROLL_CHAIN:
                    return {'result': {'objectId': 'captured-containers'}}
                if command == 'Runtime.callFunctionOn' and params['functionDeclaration'] == SCROLL_METRICS:
                    reads.append(params['objectId'])
                    if len(reads) > 1:
                        raise RuntimeError('renderer changed after input')
                    return {'result': {'value': [metric(0)]}}
                if command == 'Runtime.releaseObject':
                    releases.append(params['objectId'])
                return super().send(command, params)
        case.page.context.new_cdp_session = lambda _: FailedCDP(case.page)
        case.observe()
        result = case.session.execute(fixtures.action('scroll', amount=1, direction='down'), case.route())
        assert result['status'] == 'executed' and result['dispatched'] is True
        assert result['scroll_feedback']['available'] is False
        assert result['scroll_feedback']['noop'] is None
        assert releases == ['captured-containers']
        case.page.mouse.wheel.assert_called_once_with(0, 120)
    finally:
        case.doCleanups()


def test_smooth_scroll_without_initial_displacement_is_unknown_not_noop():
    session = fixtures.BrowserSession(fixtures.BrowserOptions(require_focus=False))
    smooth = {**metric(0), 'smooth': True}
    cdp, page = Mock(), Mock()
    cdp.send.return_value = {'result': {'value': [smooth]}}
    result = session._finish_scroll_measurement(page, cdp,
        {'object_id': 'same-container', 'before': [smooth]}, 300, wheel=False)
    assert result['scroll_feedback']['available'] is False
    assert result['moved'] is None and result['noop'] is None
    assert page.wait_for_timeout.call_count == 3
    assert all(call.args == (50,) for call in page.wait_for_timeout.call_args_list)
