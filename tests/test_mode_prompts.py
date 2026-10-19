"""A fresh observation controls the current menu and targeting instructions."""
import copy
import json
import re

from PIL import Image

from osworld_agent.actions import Action, BASIC_ACTIONS
from osworld_agent.decision_context import action_scope, scoped_system
from osworld_agent.decision_modes import decision_mode
from osworld_agent.env import Observation
from osworld_agent.pipeline import Pipeline
from osworld_agent.task_specs import task_profile


def test_live_prompt_metadata_and_menu_follow_current_surface_without_hiding_ax():
    from osworld_agent.pipeline import StepState
    from test_pipeline_completion import Env, Model

    obs = browser_observation()
    obs.context[0]["nodes"].append({"ref": "offscreen-target", "role": "link", "name": "Later result",
                                    "direct": True, "in_viewport": False, "href": "https://example.test/later"})
    pipe = Pipeline(model=Model([]), env=Env(), grounding=None, system_prompt="old static prompt")
    state = StepState(task="打开目标网页", obs=obs)
    state.routing_hint = {"next_mode": {"label": "gui"}}
    messages = pipe.build_input(state)
    assert state.prompt_metadata["prompt_profile"]["mode"] == "dom"
    assert state.prompt_metadata["selected_actions"] == pipe._active_actions
    assert "dom" in state.prompt_metadata["prompt_rules"]
    assert "gui" not in state.prompt_metadata["prompt_rules"]
    assert "offscreen-target" in state.prompt_metadata["exposed_refs"]
    assert state.prompt_metadata["system_bytes"] == len(messages[0]["content"].encode("utf-8"))
    assert '"policy_mode": "dom"' in messages[1]["content"]
    state.obs = desktop_observation()
    messages = pipe.build_input(state)
    assert state.prompt_metadata["prompt_profile"]["surface"] == "desktop"
    assert "gui" in state.prompt_metadata["prompt_rules"]
    assert "dom" not in state.prompt_metadata["prompt_rules"]
    assert "goto" not in pipe._active_actions and "switch_applications" in pipe._active_actions
    assert "old static prompt" not in messages[0]["content"]


def browser_observation(*, mode="browser_dom", snapshot="fresh-1", native_select=False):
    nodes = [{"ref": "19", "role": "link", "name": "Open item", "direct": True,
              "href": "https://example.test/item", "in_viewport": True}]
    if native_select:
        nodes.append({"ref": "27", "role": "combobox", "name": "Sort", "tag": "select",
                      "direct": True, "options": ["Title", "Date"]})
    scene = {"browser_use": 1, "structure_available": True, "mode": mode,
             "reason": "structured_page", "snapshot_id": snapshot, "page_id": "tab-1"}
    return Observation(screenshot=Image.new("RGB", (320, 240), "white"),
        context=[{"kind": "browser_ax", "snapshot_id": snapshot, "page_id": "tab-1",
                  "url": "https://example.test/", "nodes": nodes}],
        info={"scene": scene, "active_surface": "browser", "screenshot_space": "css_viewport",
              "snapshot_id": snapshot, "page_id": "tab-1", "exposed_refs": [n["ref"] for n in nodes]})


def desktop_observation():
    obs = browser_observation()
    # Residual browser AX must not decide the mode of a new desktop screenshot.
    obs.info.update(active_surface="desktop", screenshot_space="screen_pixels")
    return obs


def menu(obs, step=None):
    return action_scope(step, obs, profile=decision_mode(obs, step))


def test_mode_follows_fresh_dom_desktop_and_browser_observations():
    first = browser_observation()
    desktop = copy.deepcopy(first)
    desktop.info.update(active_surface="desktop", screenshot_space="screen_pixels")
    returned = browser_observation(snapshot="fresh-2")
    profiles = [decision_mode(obs) for obs in (first, desktop, returned)]
    assert [(p.mode, p.surface) for p in profiles] == [("dom", "browser"), ("gui", "desktop"), ("dom", "browser")]
    for profile in profiles:
        assert profile.to_dict() == {"mode": profile.mode, "surface": profile.surface,
                                     "reason": profile.reason, "source": profile.source,
                                     "fallback_reason": profile.fallback_reason}
        assert profile.reason


def test_desktop_surface_or_screen_pixel_space_each_override_residual_ax():
    for key, value in (("active_surface", "desktop"), ("screenshot_space", "screen_pixels")):
        obs = browser_observation()
        obs.info[key] = value
        assert (decision_mode(obs).mode, decision_mode(obs).surface) == ("gui", "desktop")
        assert "switch_applications" in menu(obs)
        assert "list_tabs" not in menu(obs)


def test_dom_requires_available_current_ax_with_an_enabled_direct_control():
    unavailable, stale, disabled = [browser_observation() for _ in range(3)]
    unavailable.info["scene"]["structure_available"] = False
    stale.info["scene"]["snapshot_id"] = "different-snapshot"
    disabled.context[0]["nodes"][0]["states"] = {"disabled": True}
    for obs in (unavailable, stale, disabled):
        assert decision_mode(obs).mode != "dom"
    assert decision_mode(browser_observation()).mode == "dom"


def test_browser_gui_preserves_tabs_and_only_real_native_select_capability():
    plain = browser_observation(mode="browser_gui")
    native = browser_observation(mode="browser_gui", native_select=True)
    assert decision_mode(native).mode == "gui"
    for obs in (plain, native):
        assert {"goto", "list_tabs"} <= set(menu(obs))
        assert not {"switch_tab", "close_tab"} & set(menu(obs))
        obs.info["tabs"] = [{"tab_id": "tab-1", "active": True}, {"tab_id": "tab-2", "active": False}]
        assert {"switch_tab", "close_tab"} <= set(menu(obs))
        assert "switch_applications" not in menu(obs)
    assert "select" not in menu(plain)
    assert "select" in menu(native)
    native.context[0]["nodes"][-1]["direct"] = False
    assert "select" not in menu(native)


def test_missing_stale_or_error_observation_exposes_only_safe_recovery_actions():
    missing, stale, error = [browser_observation() for _ in range(3)]
    missing.screenshot = None
    stale.info["screenshot_stale"] = True
    error.info["observation_error"] = "capture_failed"
    for obs in (None, missing, stale, error):
        assert decision_mode(obs).mode == "observe"
        actions = set(menu(obs))
        assert actions <= {"observe_browser", "observe_desktop", "wait", "fail"}
        assert {"wait", "fail"} <= actions


def test_native_dialog_takes_precedence_over_stale_image_and_ax():
    obs = browser_observation()
    obs.screenshot = None
    obs.info.update(dialog={"type": "confirm", "message": "Confirm?"}, screenshot_stale=True)
    assert decision_mode(obs).mode == "dialog"
    actions = set(menu(obs, {"actions": ["click", "goto", "request_finish", "wait"]}))
    assert {"handle_dialog", "list_tabs", "fail"} <= actions
    assert actions <= {"handle_dialog", "list_tabs", "observe_browser", "observe_desktop", "fail"}


def test_planner_suggestions_cannot_expand_dom_or_enable_code_capabilities():
    obs = browser_observation()
    actions = set(menu(obs, {"goal": "打开当前结果链接", "actions": [
        "switch_applications", "drag_and_drop", "run_in_terminal", "call_code_agent", "download", "CODE"]}))
    assert {"click", "type", "goto", "select"} <= actions
    assert not actions & {"switch_applications", "drag_and_drop", "run_in_terminal", "call_code_agent", "download", "CODE"}
    assert actions <= BASIC_ACTIONS


def test_dom_secondary_clicks_are_opt_in_by_current_goal_or_allowed_hint():
    obs = browser_observation()
    assert not {"right_click", "double_click"} & set(menu(obs, {"goal": "打开当前链接"}))
    assert "right_click" in menu(obs, {"goal": "右键打开当前链接的菜单"})
    assert "double_click" in menu(obs, {"goal": "双击打开当前条目"})
    assert "right_click" in menu(obs, {"goal": "打开当前链接", "actions": ["right_click"]})
    assert "double_click" in menu(obs, {"goal": "打开当前链接", "actions": ["double_click"]})


def test_desktop_gui_has_application_actions_without_browser_or_visual_select_menu():
    obs = desktop_observation()
    actions = set(menu(obs, {"actions": ["list_tabs", "get_page_source", "select", "goto"]}))
    assert {"click", "type", "drag_and_drop", "switch_applications"} <= actions
    assert not actions & {"list_tabs", "switch_tab", "close_tab", "get_page_source", "select", "goto"}


def test_current_menu_drives_actual_pipeline_validation_across_mode_switches():
    pipe = Pipeline.__new__(Pipeline)
    obs = browser_observation()
    pipe._active_actions = menu(obs)
    assert pipe._validate_action(Action("goto", {"url": "https://example.test/"})) is None
    assert pipe._validate_action(Action("switch_applications", {"app_code": "chrome"}))
    obs = desktop_observation()
    pipe._active_actions = menu(obs)
    assert pipe._validate_action(Action("switch_applications", {"app_code": "chrome"})) is None
    assert pipe._validate_action(Action("goto", {"url": "https://example.test/"}))
    assert pipe._validate_action(Action("list_tabs"))
    obs.screenshot = None
    pipe._active_actions = menu(obs)
    assert pipe._validate_action(Action("wait", {"seconds": 1})) is None
    assert pipe._validate_action(Action("click", {"target": "Old target"}))
    assert pipe._validate_action(Action("request_finish", {"answer": "done"}))
    assert pipe._validate_action(Action("run_in_terminal", {"command": "echo done"}))


def test_generic_dom_prompt_avoids_desktop_os_and_paper_specific_rules():
    obs = browser_observation()
    text = scoped_system(menu(obs), obs, "windows", profile=decision_mode(obs),
                         task="打开网站并填写表单", step={"goal": "填写邮箱"}, task_profile={"kind": "generic"})
    assert "next_action" in text and "summary" in text and "target_ref" in text
    assert not any(term in text for term in ("Finder", "资源管理器", "Windows 桌面", "macOS 桌面",
                                             "arXiv", "Scholar", "paper_results", "exact_title_positions"))
    assert "switch_applications" not in text and "drag_and_drop" not in text
    examples = []
    for value in re.findall(r'\{"next_action":\{[^{}]*\}[^{}\n]*\}', text):
        try:
            examples.append(json.loads(value))
        except json.JSONDecodeError:
            continue  # A field schema containing ellipses is not an action example.
    assert examples
    for example in examples:
        action = example["next_action"]
        if action.get("type") in {"click", "type", "select"}:
            assert action.get("target_ref") or action.get("element_id")


def test_paper_profile_adds_observed_identity_and_exact_title_rules_without_known_answers():
    task = "搜索论文 Attention Is All You Need 并下载 PDF"
    obs = browser_observation()
    text = scoped_system(menu(obs), obs, "windows", profile=decision_mode(obs), task=task,
                         step={"goal": "按原始标题搜索论文"}, task_profile=task_profile(task))
    assert "作者" in text and "ID" in text and "标题" in text
    assert any(term in text for term in ("完整标题", "精确标题", "exact_title_positions"))
    assert any(term in text for term in ("不能", "不要", "禁止", "不得"))
    assert "1706.03762" not in text and "Vaswani" not in text
    assert "资源管理器" not in text and "switch_applications" not in text
    # Legacy positional signatures remain callable for compatibility callers.
    assert isinstance(action_scope(None, obs, True), list)
    assert isinstance(scoped_system(menu(obs), obs, "linux"), str)


def test_dom_download_or_save_stage_keeps_ref_routed_right_click_reachable():
    obs = browser_observation()
    for step in ({"goal": "下载当前目标文件"}, {"goal": "Save the requested file"},
                 {"goal": "将目标另存到工程目录"}, {"goal": "取得全文文件", "phase": "download"}):
        assert decision_mode(obs, step).mode == "dom"
        actions = set(menu(obs, step))
        assert {"click", "right_click"} <= actions
        assert "switch_applications" not in actions and "drag_and_drop" not in actions


def test_drag_goal_on_a_structured_page_uses_browser_gui_in_the_same_viewport():
    obs = browser_observation()
    for goal in ("拖拽滑块到目标位置", "拖动当前卡片", "Drag the slider thumb to the right"):
        step = {"goal": goal}
        profile = decision_mode(obs, step)
        assert (profile.mode, profile.surface, profile.reason) == ("gui", "browser", "visual_gesture_required")
        assert "drag_and_drop" in menu(obs, step)
        assert "switch_applications" not in menu(obs, step)
        text = scoped_system(menu(obs, step), obs, "windows", profile=profile, step=step)
        assert "grounding" in text and "drag_and_drop" in text
    # Goal-driven targeting changes do not mutate the environment's AX evidence.
    assert obs.info["scene"]["mode"] == "browser_dom"
    assert obs.context[0]["nodes"][0]["direct"] is True
    assert decision_mode(obs, {"goal": "点击当前链接"}).mode == "dom"


def test_drag_goal_on_desktop_stays_in_desktop_gui_without_browser_navigation():
    obs = desktop_observation()
    step = {"goal": "拖动目标文件到另一个文件夹"}
    profile = decision_mode(obs, step)
    assert (profile.mode, profile.surface) == ("gui", "desktop")
    assert "drag_and_drop" in menu(obs, step)
    assert "switch_applications" in menu(obs, step)
    assert "goto" not in menu(obs, step) and "list_tabs" not in menu(obs, step)


def test_browser_gui_native_select_requires_exact_options_and_submission_rules():
    obs = browser_observation(mode="browser_gui", native_select=True)
    step = {"goal": "选择排序选项并刷新当前结果"}
    profile = decision_mode(obs, step)
    text = scoped_system(menu(obs, step), obs, "windows", profile=profile, step=step)
    assert "select" in menu(obs, step)
    assert "options" in text and "逐字" in text and "提交" in text
    assert any(term in text for term in ("target_ref", "真实引用", "AX引用"))
    assert any(term in text for term in ("Go", "Search", "Apply"))
