"""Opt-in local Chromium tests. No public website or model endpoint is used."""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from osworld_agent.actions import Action
from osworld_agent.browser_session import BrowserOptions, BrowserSession
from osworld_agent.env import Observation
from osworld_agent.verifiers import CompletionVerifier
from test_paper_completion import pdf_file


PAGE = b"""<!doctype html><meta charset=utf-8><title>Paper fixture</title>
<h1>AAA</h1>
<a id=pdf href=/paper.pdf download>Download AAA PDF</a>
<a id=slow href=/slow.pdf download>Slow download</a>
<a id=stream href=/stream.pdf download>Streaming download</a>
<a id=private href=/private.pdf download>Private PDF</a>
<a id=fake href=/fake.pdf download>Fake PDF</a>
<button id=alert onclick="alert('Fixture dialog')">Show dialog</button>
<a id=popup href=/child target=_blank>Open child</a>
<button id=overlay onclick="document.getElementById('modal').hidden=false">Show modal</button>
<div id=modal role=dialog aria-label="Cookie notice" hidden style="position:fixed;inset:0;background:white;z-index:999">
<button id=close onclick="document.getElementById('modal').hidden=true">Close notice</button></div>"""


@pytest.fixture
def live_browser(request, tmp_path, monkeypatch):
    if not request.config.getoption("--run-browser-integration"):
        pytest.skip("pass --run-browser-integration to run local Chromium fixtures")
    # The fixture is entirely local, including the Node CDP transport.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    from playwright.sync_api import sync_playwright
    pdf = pdf_file(tmp_path / "fixture.pdf", "AAA", "Alice Smith", "Abstract", "Fixture text.").read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path in {"/paper.pdf", "/private.pdf", "/slow.pdf", "/stream.pdf", "/fake.pdf"}:
                if path == "/fake.pdf" or (path == "/private.pdf" and "session=test-session" not in self.headers.get("Cookie", "")):
                    body, mime = b"<html>Login required</html>", "text/html"
                else:
                    body, mime = pdf, "application/pdf"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Disposition", 'attachment; filename="AAA.pdf"')
                if path != '/stream.pdf':
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if path == "/slow.pdf":
                    self.wfile.write(body[:10])
                    self.wfile.flush()
                    time.sleep(3)
                    body = body[10:]
                if path == '/stream.pdf':
                    self.wfile.write(body[:1000])
                    self.wfile.flush()
                    time.sleep(3)
                    body = body[1000:]
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            if path != "/no-cookie":
                self.send_header("Set-Cookie", "session=test-session; Path=/")
            self.end_headers()
            if path == '/pdf/render-fixture':
                self.wfile.write(b"""<style>body{background:black;color:black;margin:0}</style>
                <script>setTimeout(() => {document.body.style.background='white';
                document.body.innerHTML=('<h2 style="margin:12px 20%">Rendered paper content</h2>').repeat(20)},4000)</script>""")
            else:
                self.wfile.write(PAGE if path != "/child" else b"<h1>Child tab</h1>")
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    session = BrowserSession(BrowserOptions(endpoint=f"http://127.0.0.1:{port}", require_focus=False),
                             downloads_dir=str(tmp_path / "downloads"))
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=[f"--remote-debugging-port={port}"])
            try:
                session.begin_episode(viewport={"width": 900, "height": 700})
                assert session.goto(base + "/")["status"] == "executed"
                yield session, base
            finally:
                session.close()
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def observe(session):
    shot, snapshot, info = session.collect_observation()
    return Observation(screenshot=shot, context=[snapshot] if snapshot else [], info=info)


def click(session, element_id):
    obs = observe(session)
    action = Action("click", {"element_id": element_id, "target": element_id})
    route = session.route(action, obs)
    assert route and route["channel"] == "browser_dom", (element_id, route)
    return session.execute(action, route)


def wait_for(session, predicate, timeout=4):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        obs = observe(session)
        if predicate(obs):
            return obs
        time.sleep(0.02)
    raise AssertionError("Local browser event did not arrive before deadline")


def test_real_dom_obstruction_recovery_and_browser_download(live_browser):
    session, _ = live_browser
    assert click(session, "overlay")["status"] == "executed"
    blocked = click(session, "pdf")
    assert blocked["status"] == "rejected" and blocked["error_code"] == "blocked"
    assert blocked.get("blocker") or blocked.get("active_modals")
    assert click(session, "close")["status"] == "executed"
    assert click(session, "pdf")["status"] == "executed"
    obs = wait_for(session, lambda o: bool(o.info.get("artifacts")))
    verifier = CompletionVerifier("请下载一篇叫做AAA的论文")
    verifier.observe(obs)
    assert verifier.evaluate()["status"] == "passed"


def test_real_javascript_dialog_has_explicit_recovery(live_browser):
    session, _ = live_browser
    click(session, "alert")
    obs = wait_for(session, lambda o: bool(o.info.get("dialog")))
    assert obs.info["dialog"]["message"] == "Fixture dialog"
    assert session.handle_dialog(accept=False)["status"] == "executed"
    obs = observe(session)
    assert not obs.info.get("dialog") and obs.screenshot is not None


def test_real_popup_requires_explicit_switch_then_returns_to_parent(live_browser):
    session, base = live_browser
    original = next(tab["tab_id"] for tab in session.list_tabs()["tabs"] if tab["active"])
    assert click(session, "popup")["status"] == "executed"
    wait_for(session, lambda o: len(o.info.get("tabs", [])) == 2)
    tabs = session.list_tabs()["tabs"]
    child = next(tab["tab_id"] for tab in tabs if tab["url"] == base + "/child")
    assert next(tab["tab_id"] for tab in tabs if tab["active"]) == original
    assert session.switch_tab(child)["status"] == "executed"
    assert observe(session).info["url"] == base + "/child"
    assert session.close_tab(child)["status"] == "executed"
    assert next(tab["tab_id"] for tab in session.list_tabs()["tabs"] if tab["active"]) == original


def test_real_ui_download_shares_browser_session(live_browser):
    session, base = live_browser
    result = click(session, "private")
    assert result["status"] == "executed"
    verifier = CompletionVerifier("请下载一篇叫做AAA的论文")
    verifier.observe(wait_for(session, lambda o: bool(o.info.get("artifacts"))), result)
    assert verifier.evaluate()["status"] == "passed"


def test_real_html_disguised_as_pdf_is_not_task_success(live_browser):
    session, base = live_browser
    result = click(session, "fake")
    verifier = CompletionVerifier("请下载一篇叫做AAA的论文")
    verifier.observe(wait_for(session, lambda o: bool(o.info.get("artifacts"))), result)
    assert verifier.evaluate()["status"] == "failed"


def test_real_slow_download_times_out_without_blocking_observation(live_browser):
    session, _ = live_browser
    session.options.download_timeout_s = 0.2
    started = time.monotonic()
    click(session, "slow")
    obs = wait_for(session, lambda o: any(e.get("status") == "failed" for e in o.info.get("download_events", [])))
    assert time.monotonic() - started < 2.5
    assert not obs.info.get("artifacts")


def test_real_episode_reset_isolates_session_and_downloads(live_browser):
    session, base = live_browser
    first = dict(session.initial_state)
    assert click(session, "private")["status"] == "executed"
    assert wait_for(session, lambda o: bool(o.info.get("artifacts"))).info["artifacts"]
    second = session.begin_episode(viewport={"width": 900, "height": 700})
    assert first["initial_state_id"] == second["initial_state_id"]
    assert first["context_id"] != second["context_id"]
    assert first["downloads_dir"] != second["downloads_dir"]
    obs = observe(session)
    assert not obs.info.get("artifacts") and obs.info["url"] == "about:blank"
    assert len(session.list_tabs()["tabs"]) == 1
    assert session.goto(base + "/no-cookie")["status"] == "executed"
    result = click(session, "private")
    verifier = CompletionVerifier("请下载一篇叫做AAA的论文")
    verifier.observe(wait_for(session, lambda o: bool(o.info.get("artifacts"))), result)
    assert verifier.evaluate()["status"] == "failed"  # No cookie leaked into the new context.


def test_real_pipeline_recovers_modal_and_verifies_paper(live_browser, tmp_path, monkeypatch):
    from osworld_agent.adapters.windows_env import WindowsEnvironment
    from osworld_agent.config import ModelConfig
    from osworld_agent.model.decision_model import ChatModel
    from osworld_agent.pipeline import Pipeline
    from osworld_agent.training_events import read_events
    from test_pipeline_completion import Model
    from test_pipeline_training import paper_plan

    session, _ = live_browser
    env = WindowsEnvironment(width=900, height=700)
    env._browser_session = session
    env._episode_prepared = True
    env.initial_state = dict(session.initial_state)
    monkeypatch.setattr("osworld_agent.pipeline._viz_log_status", None)
    model = Model([{"type": "click", "element_id": target, "target": target}
                   for target in ("overlay", "pdf", "close", "pdf")] +
                  [{"type": "request_finish", "answer": "请系统验收"}] * 2, paper_plan())
    model.vision = True
    model.build_vision_message = ChatModel(ModelConfig()).build_vision_message
    result = Pipeline(model=model, env=env, grounding=None, system_prompt="test", max_steps=6,
                      event_path=str(tmp_path / "events.jsonl")).run("请下载一篇叫做AAA的论文")
    assert result.success and result.evaluation_available and result.score == 1.0
    assert result.verification["checks"][0]["matched_title"] == "AAA"
    events = read_events(tmp_path / "events.jsonl")
    calls = [e["call"] for e in events if e["event_type"] == "model_call" and e["actor_role"] == "decision"]
    assert len(calls) >= 4
    assert all(any(isinstance(m["content"], list) and any(p["type"] == "image_url" for p in m["content"])
                   for m in c["messages"]) for c in calls)
    assert "先处理当前截图/AX中的遮挡" in str(calls[2]["messages"])
    assert events[-1]["event_type"] == "episode_end"


def test_real_browser_delayed_pdf_content_wait_and_semantic_dom(live_browser, monkeypatch):
    # A local PDF-like URL deliberately renders later; no external paper/model.
    from osworld_agent.adapters.windows_env import WindowsEnvironment
    from osworld_agent.pipeline import Pipeline, Execution, StepState
    from osworld_agent.visual_readiness import content_thumbnail
    from test_pipeline_completion import Model
    session, base = live_browser
    env = WindowsEnvironment(width=900, height=700)
    env._browser_session = session
    obs = env.observe()
    action = Action('click', {'target': 'Download AAA PDF link'})
    route = env.route_action(action, obs)
    assert route['channel'] == 'browser_dom'
    assert session.nodes[route['target_ref']]['id'] == 'pdf'
    assert session.goto(base + '/pdf/render-fixture')['status'] == 'executed'
    obs = env.observe()
    assert content_thumbnail(obs.screenshot) is None
    monkeypatch.setattr('osworld_agent.pipeline._viz_log_status', None)
    pipe = Pipeline(model=Model([]), env=env, grounding=None, system_prompt='test',
                    wait_config={'render_timeout_s': 8})
    state = StepState(obs=obs, step_id=1)
    result = pipe._settle_rendering(Execution(kind='browser_dom', obs=obs), state)
    assert result.obs.info['render_wait']['status'] == 'completed'
    assert content_thumbnail(result.obs.screenshot) is not None
    assert state.step_id == 1 and not pipe.model.calls


def test_real_slow_download_size_limit_cancels_before_body_finishes(live_browser):
    session, _ = live_browser
    session.options.download_max_bytes = 100
    started = time.monotonic()
    click(session, 'slow')
    obs = wait_for(session, lambda o: any(e.get('error_code') == 'download_size_limit' for e in o.info.get('download_events', [])))
    assert time.monotonic() - started < 2.5  # fixture delays its body for 3 seconds
    assert not obs.info.get('artifacts')


def test_real_sequential_same_url_download_is_canceled(live_browser):
    session, _ = live_browser
    assert click(session, 'pdf')['status'] == 'executed'
    first = wait_for(session, lambda o: bool(o.info.get('artifacts')))
    assert len(first.info['artifacts']) == 1
    # Direct session call intentionally bypasses policy pre-dispatch guard.
    click(session, 'pdf')
    after = wait_for(session, lambda o: any(e.get('status') == 'deduplicated' for e in o.info.get('download_events', [])))
    assert len(after.info['artifacts']) == 1 and not after.info['pending_downloads']


def test_real_stream_without_content_length_is_canceled_by_progress(live_browser):
    session, _ = live_browser
    session.options.download_max_bytes = 100
    started = time.monotonic()
    click(session, 'stream')
    obs = wait_for(session, lambda o: any(e.get('error_code') == 'download_size_limit' for e in o.info.get('download_events', [])))
    assert time.monotonic() - started < 2.5
    assert not obs.info.get('artifacts')
