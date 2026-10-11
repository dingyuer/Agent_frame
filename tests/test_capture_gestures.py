"""Opt-in Studio gesture boundaries; synthetic UI fixtures never become demonstrations.

Loads actual Studio assets through Playwright routes. No HTTP server, model endpoint,
collection backend, or exported training dataset is involved in this test.
"""
import base64
import io
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest


def test_capture_gestures_are_serial_semantic_steps(request):
    if not request.config.getoption("--run-browser-integration"):
        pytest.skip("pass --run-browser-integration to check browser gesture boundaries")
    from PIL import Image
    from playwright.sync_api import sync_playwright, expect

    static = Path(__file__).resolve().parents[1] / "viz" / "static"
    image = io.BytesIO()
    Image.new("RGB", (1280, 800), "#edf3ea").save(image, "PNG")
    screenshot = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
    requests, steps, errors = [], [], []
    observation_id = "fixture_observation_0"
    mock_dialog = None
    mock_null_observation = False

    def current():
        return {
            "ok": True,
            "session_id": "synthetic_ui_fixture_only",
            "status": "recording",
            "task": "Synthetic gesture fixture, not training data",
            "source": "test_fixture",
            "observation": None if mock_null_observation else {
                "id": observation_id, "screenshot": "" if mock_dialog else screenshot,
                "url": "https://fixture.invalid/", "width": 1280, "height": 800,
                "dialog": mock_dialog,
            },
            "steps": steps,
        }

    def respond(route):
        nonlocal observation_id, mock_dialog
        path = urlsplit(route.request.url).path
        if path == "/":
            route.fulfill(content_type="text/html", body=(static / "index.html").read_text())
            return
        if path in ("/static/studio.css", "/static/capture.js"):
            route.fulfill(content_type="text/css" if path.endswith(".css") else "text/javascript",
                          body=(static / Path(path).name).read_text())
            return
        if path == "/collect/status":
            body = {"ok": True, "status": "idle"}
        elif path in ("/collect/start", "/collect/frame"):
            body = current()
        elif path == "/collect/action":
            action = route.request.post_data_json
            assert action["session_id"] == "synthetic_ui_fixture_only"
            assert action["observation_id"] == observation_id
            requests.append(action)
            observation_id = "fixture_observation_" + str(len(requests))
            steps.append({"step": f"h_{len(requests)}", "source": "test_fixture",
                          "action": {"action": action["kind"], "args": {}},
                          "execution_status": "executed", "keep": None})
            if action["kind"] == "handle_dialog":
                mock_dialog = None
            body = current()
        elif path.startswith("/collect/export"):
            pytest.fail("Synthetic gesture fixtures must never be exported as demonstrations")
        elif path in ("/logs", "/runs", "/local-runs"):
            body = []
        else:
            # Existing run/debug controls are unrelated to collection and stay inert.
            body = {"ok": False, "error": "Not part of the synthetic fixture",
                    "tasks": {}, "strategies": [], "presets": []}
        route.fulfill(content_type="application/json", body=json.dumps(body))

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/*", respond)
            page.route_web_socket("**/ws", lambda socket: None)
            page.goto("http://studio-fixture.invalid/", wait_until="networkidle")
            page.locator("#nav-collect").click()
            page.locator("#capture-task").fill("Synthetic UI fixture")
            page.locator("#capture-start").click()
            expect(page.locator("#capture-text")).to_be_enabled()
            frame = page.locator("#capture-frame").bounding_box()
            x, y = frame["x"] + 100, frame["y"] + 120

            def wait_for_steps(count):
                expect(page.locator("#capture-step-count")).to_have_text(str(count))
                expect(page.locator("#capture-text")).to_be_enabled()

            # Hover/focus are context, never training actions.
            page.mouse.move(x, y)
            page.wait_for_timeout(350)
            assert not requests
            page.mouse.click(x, y)
            wait_for_steps(1)
            assert requests[-1]["kind"] == "click"

            # Two physical clicks produce one double_click, not two single steps.
            page.mouse.dblclick(x, y, delay=40)
            wait_for_steps(2)
            assert requests[-1]["kind"] == "double_click"
            page.mouse.click(x, y, button="right")
            wait_for_steps(3)
            assert requests[-1]["kind"] == "right_click"

            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + 75, y + 45, steps=6)
            page.mouse.up()
            wait_for_steps(4)
            assert requests[-1]["kind"] == "drag_and_drop"
            assert requests[-1]["x2"] > requests[-1]["x"]
            assert requests[-1]["y2"] > requests[-1]["y"]

            # Same-direction wheel input within 300 ms is one browser action.
            page.mouse.move(x, y)
            page.mouse.wheel(0, 180)
            page.wait_for_timeout(60)
            page.mouse.wheel(0, 220)
            wait_for_steps(5)
            assert requests[-1]["kind"] == "scroll"
            assert requests[-1]["amount"] == 1

            # A direction reversal flushes the first group; neither is dropped.
            page.mouse.wheel(0, 350)
            page.wait_for_timeout(70)
            page.mouse.wheel(0, -350)
            wait_for_steps(7)
            assert [action["direction"] for action in requests[-2:]] == ["down", "up"]

            # IME composition only stages text. Explicit submission makes one step.
            page.locator("#capture-text").fill("组合中文\n第二行")
            page.locator("#capture-text").dispatch_event("compositionstart")
            page.locator("#capture-text").dispatch_event("compositionend")
            page.wait_for_timeout(350)
            assert len(requests) == 7
            page.locator("#capture-text").press("Control+Enter")
            wait_for_steps(8)
            assert requests[-1]["kind"] == "type"
            assert requests[-1]["text"] == "组合中文\n第二行"
            page.locator("[data-capture-key=Enter]").click()
            wait_for_steps(9)
            assert requests[-1]["kind"] == "press"

            # A point outside the screenshot must not clamp to a clickable edge.
            page.locator("#capture-stage").evaluate("el => el.style.paddingTop = '40px'")
            stage = page.locator("#capture-stage").bounding_box()
            page.mouse.click(stage["x"] + 20, stage["y"] + 10)
            page.wait_for_timeout(350)
            assert len(requests) == 9

            # Native JS dialogs can be handled even when no screenshot is available.
            mock_dialog = {"type": "prompt", "message": "Fixture prompt"}
            page.locator("#capture-refresh").click()
            expect(page.locator("#capture-dialog-accept")).to_be_enabled()
            expect(page.locator("#capture-text")).to_be_disabled()
            page.locator("#capture-dialog-input").fill("中文对话框")
            page.locator("#capture-dialog-accept").click()
            wait_for_steps(10)
            assert requests[-1]["kind"] == "handle_dialog"
            assert requests[-1]["prompt_text"] == "中文对话框"

            # An explicitly unavailable observation cannot leave the old frame active.
            mock_null_observation = True
            page.locator("#capture-refresh").click()
            expect(page.locator("#capture-frame")).to_be_hidden()
            expect(page.locator("#capture-text")).to_be_disabled()
            assert len(requests) == 10
            assert len({action["request_id"] for action in requests}) == 10
            assert [action["kind"] for action in requests] == [
                "click", "double_click", "right_click", "drag_and_drop", "scroll",
                "scroll", "scroll", "type", "press", "handle_dialog",
            ]
            assert not errors
        finally:
            browser.close()
