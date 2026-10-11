"""Local CDP discovery must not inherit web traffic proxies."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from osworld_agent import utils


def fake_session(monkeypatch, payload):
    response = SimpleNamespace(raise_for_status=Mock(), json=Mock(return_value=payload))
    session = MagicMock()
    session.__enter__.return_value = session
    session.get.return_value = response
    monkeypatch.setattr(utils.requests, 'Session', Mock(return_value=session))
    return session


def test_page_probe_uses_direct_session_and_preserves_global_proxy(monkeypatch):
    monkeypatch.setenv('HTTP_PROXY', 'http://proxy.invalid:1234')
    session = fake_session(monkeypatch, [{'id': 'page1'}])
    assert utils.list_cdp_pages('127.0.0.1', 9333) == [{'id': 'page1'}]
    assert session.trust_env is False
    session.get.assert_called_once_with('http://127.0.0.1:9333/json', timeout=4.0)
    session.__exit__.assert_called_once()
    assert os.environ['HTTP_PROXY'] == 'http://proxy.invalid:1234'


def test_probe_rejects_non_page_list(monkeypatch):
    fake_session(monkeypatch, {'error': 'not a page list'})
    assert utils.list_cdp_pages('127.0.0.1') == []


def test_sync_browser_helper_uses_discovered_websocket(monkeypatch):
    session = fake_session(monkeypatch, {'webSocketDebuggerUrl': 'ws://127.0.0.1:9222/devtools/browser/id'})
    runtime = SimpleNamespace(chromium=SimpleNamespace(connect_over_cdp=Mock(return_value='browser')), stop=Mock())
    monkeypatch.setitem(sys.modules, 'playwright.sync_api', SimpleNamespace(
        sync_playwright=lambda: SimpleNamespace(start=Mock(return_value=runtime))))
    assert utils._connect_browser('127.0.0.1', 9222) == (runtime, 'browser')
    runtime.chromium.connect_over_cdp.assert_called_once_with('ws://127.0.0.1:9222/devtools/browser/id', timeout=4000)
    assert session.trust_env is False


def test_invalid_discovery_does_not_start_a_driver(monkeypatch):
    fake_session(monkeypatch, {'webSocketDebuggerUrl': 'https://not-websocket'})
    start = Mock()
    monkeypatch.setitem(sys.modules, 'playwright.sync_api', SimpleNamespace(
        sync_playwright=lambda: SimpleNamespace(start=start)))
    with pytest.raises(ValueError, match='WebSocket'):
        utils._connect_browser('127.0.0.1', 9222)
    start.assert_not_called()
