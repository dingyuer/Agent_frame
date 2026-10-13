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

            # The same event renderer handles live websocket and reloaded history.
            page.set_viewport_size({"width": 1440, "height": 1050})
            page.emulate_media(reduced_motion="reduce")
            page.evaluate("""() => {
              addCard({kind:'status',event:'task_start',instruction:'核查测试',max_steps:8,ts:1});
              addCard({kind:'status',event:'step',step:1,ts:1});
              addCard({kind:'status',event:'task_analysis',analysis:{kind:'paper_download',contract_status:'partial',check_count:1,
                unresolved_requirements:['<script>bad()</script>']},ts:1});
              for (const count of [1,4]) addCard({kind:'status',event:'step_progress',report:{plan_step_id:'p0_1',
                goal:'搜索目标论文',status:count===4?'stalled':'watching',idle_decisions:count,limit:4,
                reason:'当前子任务没有新的可核实进展'},ts:1});
              addCard({kind:'status',event:'framework_recovery',status:'requested',ts:1});
              addCard({kind:'status',event:'decision_judgment',judgment:{model:'vllm-sr/d3',model_version:'fixture-version',latency_ms:123,
                answers:{task_type:{label:'mixed',confidence:.96,probabilities:{browser:.02,desktop:.01,mixed:.96,unknown:.01}}}},ts:1});
              addCard({kind:'status',event:'execution_queue',queue:{cursor:0,total:2,steps:[
                {id:'p0_1',goal:'搜索目标论文',status:'doing'}, {id:'p0_2',goal:'下载 PDF',status:'pending'}]},ts:1});
              for (const status of ['running','unknown']) addCard({kind:'status',event:'step_verification',
                review_id:'fixture-r1',plan_step_id:'p0_1',decision_step:1,method:'code',status,
                goal:'搜索目标论文',reason:'需要图像证据',evidence:[],ts:1});
              for (const status of ['running','passed']) addCard({kind:'status',event:'step_verification',
                review_id:'fixture-r1',plan_step_id:'p0_1',decision_step:1,method:'model',status,
                goal:'搜索目标论文',reason:'标题符合任务',evidence:['<script>bad()</script>'],ts:1});
            }""")
            assert page.locator('[data-kind="review"]').count() == 2  # pending cards update in place
            assert page.locator('[data-kind="execution-queue"]').count() == 1
            assert page.locator('[data-diagnostic-id="task_analysis:task"]').count() == 1
            assert '代码核查 + 语义审查' in page.locator('[data-diagnostic-id="task_analysis:task"]').inner_text()
            assert '<script>bad()</script>' in page.locator('[data-diagnostic-id="task_analysis:task"]').inner_text()
            assert page.locator('[data-diagnostic-id="step_progress:p0_1"]').count() == 1
            assert '4 / 4' in page.locator('[data-diagnostic-id="step_progress:p0_1"]').inner_text()
            assert '网页与桌面混合' in page.locator('#timeline').inner_text()
            assert '96.0%' in page.locator('#timeline').inner_text()
            assert 'fixture-version' in page.locator('#timeline').inner_text()
            # Both sizes share the optional config path; no endpoint is invoked by selection.
            page.locator('#nav-run').click()
            expect(page.locator('#local-download-max-mib')).to_have_value('50')
            page.locator('#local-workspace-dir').fill('C:\\Agent Workspace')
            expect(page.locator('#local-workspace-dir')).to_have_value('C:\\Agent Workspace')
            page.locator('#judge-model').locator('xpath=ancestor::details').evaluate('el => el.open = true')
            for model in ('vllm-sr/d3', 'vllm-sr/d3-flash'):
                page.locator('#judge-model').select_option(model)
                page.locator('#judge-url').fill('http://fixture.invalid:8899/v1')
                assert page.evaluate('decisionJudgeConfig().model') == model
            page.screenshot(path=str(tmp_path / 'studio-judge-settings.png'), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.set_viewport_size({"width": 1440, "height": 1050})
            page.locator('#judge-model').select_option('')
            assert page.evaluate('decisionJudgeConfig()') == {}
            assert page.locator('.review-passed').inner_text().find('独立模型核查') >= 0
            assert page.locator('.review-passed .review-evidence').inner_text() == '<script>bad()</script>'
            assert page.locator('#st-step').inner_text() == '1 / 8'
            page.locator('#nav-live').click()
            expect(page.locator('.review-passed')).to_be_visible()
            expect(page.locator('[data-kind="execution-queue"]')).to_be_visible()
            assert 'undefined' not in page.locator('#timeline').inner_text()
            page.evaluate("renderStatus({event:'framework_recovery',status:'exhausted'})")
            expect(page.locator('#st-status .failed')).to_have_text('恢复预算耗尽')
            page.evaluate("renderStatus({event:'step_verification',method:'decision_model',status:'running'})")
            expect(page.locator('#st-status')).to_have_text('Decision 3.0 核查 · 进行中')
            page.screenshot(path=str(tmp_path / 'studio-step-review.png'), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.evaluate("""() => {
              addCard({kind:'status',event:'operation_feedback',facts:['下载传输已完成：data.pdf，12345 字节','<script>bad()</script>'],ts:1});
              addCard({kind:'status',event:'framework_wait',reason:'pdf_render',status:'waiting',elapsed_s:3,ts:1});
              addCard({kind:'status',event:'duplicate_download',reason:'目标文件已下载，不再重复派发',path:'<script>bad()</script>',ts:1});
              addCard({kind:'status',event:'workspace_ready',workspace:{workspace_dir:'/workspace/<script>bad()</script>'},ts:1});
              addCard({kind:'status',event:'control_mode',mode:'desktop_takeover',ts:1});
            }""")
            assert '下载传输已完成：data.pdf' in page.locator('#timeline').inner_text()
            assert '目标文件已下载，不再重复派发' in page.locator('#timeline').inner_text()
            assert '/workspace/<script>bad()</script>' in page.locator('#timeline').inner_text()
            assert page.locator('#st-control').inner_text() == '桌面接管 · 真实输入'
            page.evaluate("addCard({kind:'status',event:'control_mode',mode:'background_browser',ts:1})")
            assert page.locator('#st-control').inner_text() == '后台浏览器 · CDP'
            assert '等待 PDF 渲染' in page.locator('#st-status').inner_text()
            page.evaluate("addCard({kind:'status',event:'framework_wait',reason:'pdf_render',status:'timeout',elapsed_s:20,ts:1})")
            assert 'PDF 渲染等待超时' in page.locator('#timeline').inner_text()
            assert '等待超时' in page.locator('#st-status').inner_text()
            assert not errors
            browser.close()
    finally:
        controller.close()
        runtime.should_exit = True
        thread.join(timeout=10)
