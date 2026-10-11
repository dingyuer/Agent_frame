"""Opt-in UI-to-browser-to-SFT smoke test, entirely on localhost."""
import json
import socket
import threading
import time
import zipfile

import pytest


def test_studio_human_demonstration_from_buttons_to_dataset(request, tmp_path):
    if not request.config.getoption("--run-browser-integration"):
        pytest.skip("pass --run-browser-integration to exercise the real Studio UI")
    import httpx
    import uvicorn
    from fastapi.responses import HTMLResponse
    from playwright.sync_api import sync_playwright, expect
    from osworld_agent.viz import server

    controller = server.collection
    controller.root = tmp_path / "recordings"
    html = """<html><style>body{padding:30px;font:18px sans-serif}input,button{padding:12px;font:18px sans-serif}</style>
    <h1>Local search fixture</h1><label for=q>Paper title</label><input id=q aria-label="Paper title">
    <button id=search onclick="document.getElementById('result').textContent=document.getElementById('q').value">Search</button><p id=result></p></html>"""
    server.app.add_api_route("/__ui_fixture", lambda: HTMLResponse(html), methods=["GET"])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    runtime = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=runtime.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    errors = []
    try:
        deadline = time.monotonic() + 10
        with httpx.Client(trust_env=False) as client:
            while time.monotonic() < deadline:
                try:
                    if client.get(base + "/collect/status").status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.05)
            else:
                raise AssertionError("Studio did not start")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(base)
            page.locator("#nav-collect").click()
            page.locator("#capture-task").fill("演示在搜索框输入中文并搜索")
            page.locator("#capture-start-url").fill(base + "/__ui_fixture")
            page.locator("#capture-start").click()
            expect(page.locator("#capture-state")).to_have_text("正在采集", timeout=20000)
            expect(page.locator("#capture-send-text")).to_be_enabled()

            def click_element(element_id, expected_steps):
                before = controller.observation
                node = next(n for n in before.context[0]["nodes"] if n.get("id") == element_id)
                x, y, w, h = node["bounds"]
                scroll_x, scroll_y = before.info["scroll_offset"]
                shot_w, shot_h = before.info["screenshot_size"]
                image = page.locator("#capture-frame")
                image.scroll_into_view_if_needed()
                rect = image.bounding_box()
                # The stage owns pointer capture; the image intentionally has pointer-events:none.
                page.mouse.click(rect["x"] + (x + w / 2 - scroll_x) / shot_w * rect["width"],
                                 rect["y"] + (y + h / 2 - scroll_y) / shot_h * rect["height"])
                expect(page.locator("#capture-step-count")).to_have_text(str(expected_steps))
                expect(page.locator("#capture-send-text")).to_be_enabled()

            click_element("q", 1)
            page.locator("#capture-text").fill("人工示范测试")
            page.locator("#capture-send-text").click()
            expect(page.locator("#capture-step-count")).to_have_text("2")
            expect(page.locator("#capture-send-text")).to_be_enabled()
            assert next(n for n in controller.observation.context[0]["nodes"] if n.get("id") == "q")["value"] == "人工示范测试"
            click_element("search", 3)
            for index in range(1, 4):
                card = page.locator(f'details[data-step="h_{index}"]')
                if not card.evaluate("element => element.open"):
                    card.locator("summary").click()
                card.get_by_role("button", name="保留此步", exact=True).click()
                expect(card.locator(".step-review-status")).to_have_text("已保留")
            page.locator("#capture-stop").click()
            expect(page.locator("#capture-state")).to_have_text("已结束")
            with page.expect_download() as download:
                page.locator("#capture-export-zip").click()
            bundle = tmp_path / "dataset.zip"
            download.value.save_as(bundle)
            with zipfile.ZipFile(bundle) as archive:
                records = [json.loads(line) for line in archive.read("sft.jsonl").splitlines()]
                assert len(records) == 3
                assert all(record["human_reviewed"] and not record["reward_valid"] for record in records)
                assert all(image in archive.namelist() for record in records for image in record["images"])
            assert not errors
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(tmp_path / "studio-finished.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            browser.close()
    finally:
        controller.close()
        runtime.should_exit = True
        thread.join(timeout=10)
