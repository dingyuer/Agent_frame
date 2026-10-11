"""Opt-in human-inspection checks against localhost Chromium only."""
import json
import time
from types import SimpleNamespace

import pytest

from osworld_agent.actions import Action
from osworld_agent.browser_session import browser_thread
import test_browser_live
from test_browser_live import live_browser, click


HUMAN_PAGE = b"""<!doctype html><meta charset=utf-8><title>Human fixture</title>
<style>body{font:16px sans-serif}label{display:block;margin:12px}input{width:240px}
#scrollbox{width:350px;height:120px;overflow:auto;border:1px solid black}
#scrollcontent{height:1200px;background:linear-gradient(white,lightblue)}</style>
<button id=save onclick="window.saveCount=(window.saveCount||0)+1"><span id=save_label>Save form</span></button>
<label>Email<input id=email name=email></label>
<label>Password<input id=password type=password value=fixture-secret-not-for-export></label>
<label>Verification code<input id=otp autocomplete=one-time-code value=123456></label>
<label>Card details<input id=card autocomplete=cc-number value=4111111111111111></label>
<div id=rich role=textbox aria-label="Rich text" contenteditable=true style="min-height:24px;border:1px solid gray"></div>
<div id=scrollbox role=region aria-label="Scrollable items"><div id=scrollcontent>Nested scrolling fixture</div></div>
<canvas id=canvas width=100 height=30></canvas>"""


@pytest.fixture
def human_browser(monkeypatch, request):
    monkeypatch.setattr(test_browser_live, "PAGE", HUMAN_PAGE)
    return request.getfixturevalue("live_browser")[0]


@browser_thread
def page_eval(session, script):
    _, browser = session._connect()
    page = session._pick_page(browser)
    return page.evaluate(script)


def center(session, selector):
    return page_eval(session, """(() => {
      const r=document.querySelector(%s).getBoundingClientRect();
      return [r.x+r.width/2,r.y+r.height/2];
    })()""" % json.dumps(selector))


def test_live_nested_span_hit_has_exact_button_ref_and_no_click(human_browser):
    session = human_browser
    _, snapshot, metadata = session.collect_observation()
    x, y = center(session, "#save_label")
    first = session.inspect_point(x, y, metadata)
    assert first["status"] == "inspected", first
    assert first["matched"] and first["element_id"] == "save"
    assert first["target_ref"] == session.resolve_id("save")
    assert first["target_label"] == "button Save form"
    assert first["snapshot_id"] == snapshot["snapshot_id"]
    assert session.snapshot is snapshot
    # A released object group must not invalidate nodes or a later inspection.
    assert session.inspect_point(x, y, metadata)["target_ref"] == first["target_ref"]
    assert page_eval(session, "window.saveCount || 0") == 0


def test_live_focused_input_and_sensitive_fields_never_return_values(human_browser):
    session = human_browser
    for element_id, sensitive in (("email", False), ("password", True), ("otp", True), ("card", True)):
        assert click(session, element_id)["status"] == "executed"
        _, _, metadata = session.collect_observation()
        focused = session.inspect_focused(metadata)
        assert focused["status"] == "inspected", focused
        assert focused["element_id"] == element_id, focused
        assert focused["editable"] is True
        assert focused["sensitive"] is sensitive, focused
        x, y = center(session, "#" + element_id)
        pointed = session.inspect_point(x, y, metadata)
        assert pointed["element_id"] == element_id, pointed
        assert pointed["sensitive"] is sensitive, pointed
        rendered = json.dumps([focused, pointed])
        assert all(secret not in rendered for secret in ("fixture-secret-not-for-export", "123456", "4111111111111111"))


def test_live_frozen_metadata_rejects_reload_and_canvas_has_no_fake_label(human_browser):
    session = human_browser
    _, _, metadata = session.collect_observation()
    x, y = center(session, "#canvas")
    result = session.inspect_point(x, y, metadata)
    assert result["reason"] == "canvas_target" and not result["matched"]
    assert result["target_label"] == "" and "target_ref" not in result
    page_eval(session, "location.reload()")
    time.sleep(0.1)
    rejected = session.inspect_focused(metadata)
    assert rejected["status"] == "rejected" and rejected["error_code"] == "stale"


def test_live_pointer_scroll_targets_nested_container(human_browser):
    session = human_browser
    _, _, metadata = session.collect_observation()
    x, y = center(session, "#scrollbox")
    result = session.execute_viewport_action(Action("scroll", {
        "direction": "down", "amount": 1, "x": x, "y": y}), metadata)
    assert result["status"] == "executed", result
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = page_eval(session, "({page:scrollY,container:document.querySelector('#scrollbox').scrollTop})")
        if state["container"] > 0:
            break
        time.sleep(0.02)
    assert state["container"] > 0 and state["page"] == 0
    stale = {**metadata, "scroll_offset": [0, 10]}
    assert session.execute_viewport_action(Action("scroll", {
        "direction": "down", "amount": 1, "x": x, "y": y}), stale)["error_code"] == "stale"


def test_live_scroll_inspection_maps_noninteractive_container_and_root(human_browser):
    session = human_browser
    _, snapshot, metadata = session.collect_observation()
    metadata["exposed_refs"] = list(session.nodes)
    x, y = center(session, "#scrollbox")
    result = session.inspect_scroll_target(x, y, metadata)
    assert result["status"] == "inspected", result
    assert result["root_scroll"] is False and result["semantic_unavailable"] is False, result
    assert result["target_ref"] == session.resolve_id("scrollbox")
    assert result["target_label"] == "region Scrollable items"
    assert session.nodes[result["target_ref"]]["direct"] is False
    assert session.snapshot is snapshot
    assert page_eval(session, "document.querySelector('#scrollbox').scrollTop") == 0
    root_x, root_y = center(session, "#save_label")
    root = session.inspect_scroll_target(root_x, root_y, metadata)
    assert root["root_scroll"] is True and root["semantic_unavailable"] is False, root
    assert "target_ref" not in root
    hidden = session.inspect_scroll_target(x, y, {**metadata, "exposed_refs": []})
    assert hidden["semantic_unavailable"] is True and "target_ref" not in hidden
    action = Action("scroll", {"direction": "down", "amount": 1, "target_ref": result["target_ref"]})
    route = session.route(action, SimpleNamespace(info=metadata))
    executed = session.execute(action, route)
    assert executed["status"] == "executed", executed
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        offset = page_eval(session, "document.querySelector('#scrollbox').scrollTop")
        if offset > 0:
            break
        time.sleep(0.02)
    assert offset == 120  # Matches the canonical DOM scroll action's amount.


def test_live_unmapped_scroll_container_has_no_invented_semantics(human_browser):
    session = human_browser
    page_eval(session, """(() => {
      const box=document.querySelector('#scrollbox'); box.removeAttribute('role');
      box.removeAttribute('aria-label');
    })()""")
    _, _, metadata = session.collect_observation()
    metadata["exposed_refs"] = list(session.nodes)
    x, y = center(session, "#scrollbox")
    result = session.inspect_scroll_target(x, y, metadata)
    assert result["root_scroll"] is False and result["semantic_unavailable"] is True, result
    assert result["target_label"] == "" and "target_ref" not in result


def test_live_focused_text_preserves_middle_caret_and_supports_chinese(human_browser):
    session = human_browser
    page_eval(session, """(() => {
      const e=document.querySelector('#email'); e.value='ABCD'; e.focus(); e.setSelectionRange(2,2);
    })()""")
    _, _, metadata = session.collect_observation()
    focused = session.inspect_focused(metadata)
    result = session.insert_focused_text("中文", metadata, expected_backend_id=focused["focused_backend_id"])
    assert result["status"] == "executed", result
    assert page_eval(session, "document.querySelector('#email').value") == "AB中文CD"
    _, _, metadata = session.collect_observation()
    focused = session.inspect_focused(metadata)
    result = session.insert_focused_text("新内容", metadata, overwrite=True,
                                         expected_backend_id=focused["focused_backend_id"])
    assert result["status"] == "executed", result
    assert page_eval(session, "document.querySelector('#email').value") == "新内容"


def test_live_focused_text_rejects_sensitive_field_and_changed_focus(human_browser):
    session = human_browser
    click(session, "email")
    _, _, metadata = session.collect_observation()
    email = session.inspect_focused(metadata)
    page_eval(session, "document.querySelector('#password').focus()")
    rejected = session.insert_focused_text("do-not-enter", metadata,
                                          expected_backend_id=email["focused_backend_id"])
    assert rejected["status"] == "rejected" and rejected["dispatched"] is False
    assert page_eval(session, "document.querySelector('#password').value") == "fixture-secret-not-for-export"
    _, _, metadata = session.collect_observation()
    password = session.inspect_focused(metadata)
    rejected = session.insert_focused_text("do-not-enter", metadata,
                                          expected_backend_id=password["focused_backend_id"])
    assert rejected["error_code"] == "blocked" and rejected["dispatched"] is False
    assert page_eval(session, "document.querySelector('#password').value") == "fixture-secret-not-for-export"


def test_live_editable_text_presence_is_boolean_without_content(human_browser):
    session = human_browser
    for selector, content in (("#email", ""), ("#email", "existing-private-text"),
                              ("#rich", ""), ("#rich", "existing-private-rich-content")):
        page_eval(session, """(() => {
          const e=document.querySelector(%s);
          if (e.isContentEditable) e.textContent=%s; else e.value=%s;
          e.focus();
        })()""" % (json.dumps(selector), json.dumps(content), json.dumps(content)))
        _, _, metadata = session.collect_observation()
        inspected = session.inspect_focused(metadata)
        assert inspected["status"] == "inspected" and inspected["editable"] is True, inspected
        assert inspected["editable_has_text"] is bool(content), inspected
        assert "value" not in inspected and "textContent" not in inspected
        assert "existing-private" not in json.dumps(inspected)
