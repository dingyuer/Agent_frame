"""resolve_target 单元测试：语义匹配 / ref 提取 / allowed 过滤（纯内存，无需浏览器）。"""
import pytest

from osworld_agent.browser_session import BrowserSession, BrowserOptions


def _session():
    s = BrowserSession(options=BrowserOptions())
    s.snapshot = {"kind": "browser_ax", "snapshot_id": "abc", "nodes": []}
    s.nodes = {
        "1": {"ref": "1", "role": "menuitem", "name": "Remove Account", "value": "", "direct": True},
        "2": {"ref": "2", "role": "button", "name": "Submit", "value": "", "direct": True},
        "3": {"ref": "3", "role": "textbox", "name": "搜索", "value": "", "direct": True},
        "4": {"ref": "4", "role": "generic", "name": "Submit", "value": "", "direct": False},
    }
    return s


def test_quoted_exact_match():
    s = _session()
    assert s.resolve_target("The 'Remove Account' option in the menu") == "1"


def test_ref_in_target_extracted():
    # 模型把 ref 写进 target 描述里（如 "[3]"）→ 直接提取用
    s = _session()
    assert s.resolve_target("The search box [3]") == "3"


def test_exact_exposed_ref_or_html_id_in_target_resolves_without_grounding():
    s = _session()
    s.nodes["3"]["id"] = "query"
    assert s.resolve_target("3", allowed={"3"}) == "3"
    assert s.resolve_target("query", allowed={"3"}) == "3"
    assert s.resolve_target("query", allowed={"2"}) is None
    assert s.resolve_target("4", allowed={"4"}) is None  # non-interactive snapshot node


def test_hash_prefixed_observed_id_resolves_native_select_without_guessing_ref():
    s = _session()
    s.nodes['3'].update(id='searchtype', role='combobox')
    assert s.resolve_target('#searchtype', allowed={'3'}) == '3'
    assert s.resolve_target('#searchtype', allowed={'2'}) is None
    assert s.resolve_target('#searchtype', allowed=set()) is None
    assert s.resolve_target('#3', allowed={'3'}) is None  # ref 3 is not an HTML id
    assert s.resolve_target('#SEARCHTYPE', allowed={'3'}) is None  # HTML ids are case-sensitive


def test_hash_prefixed_id_requires_unique_direct_snapshot_node():
    s = _session()
    s.nodes['4']['id'] = 'container'
    assert s.resolve_target('#container', allowed={'4'}) is None
    s.nodes['2']['id'] = 'duplicate'
    s.nodes['3']['id'] = 'duplicate'
    assert s.resolve_target('#duplicate', allowed={'2'}) is None
    assert s.last_resolution_error == 'ambiguous'


@pytest.mark.parametrize('target', ['#searchtype option', '#searchtype>option', '#searchtype:focus',
                                  '#searchtype[value="x"]', '#searchtype,button', '#searchtype.foo',
                                  '#searchtype+button', '#searchtype~button', '#searchtype #child'])
def test_hash_target_never_executes_or_semantically_guesses_compound_css(target):
    s = _session()
    s.nodes['2']['name'] = target  # even a matching accessible name must not bypass the restriction
    s.nodes['3']['id'] = 'searchtype'
    assert s.resolve_target(target, allowed={'2', '3'}) is None


def test_non_direct_skipped():
    # 同名节点 "4" 是非 direct，应跳过，命中 direct 的 "2"
    s = _session()
    assert s.resolve_target("The 'Submit' element") == "2"


def test_allowed_filter():
    s = _session()
    assert s.resolve_target("The 'Remove Account' option", allowed={"2", "3"}) is None


def test_empty_allowed_list_cannot_resolve_any_node():
    s = _session()
    assert s.resolve_target("The 'Submit' button", allowed=set()) is None
    assert s.resolve_target("The Submit button [2]", allowed=set()) is None


def test_duplicate_labels_require_disambiguation():
    s = _session()
    s.nodes["5"] = dict(s.nodes["2"], ref="5")
    assert s.resolve_target("The 'Submit' button in the second row") is None
    assert s.last_resolution_error == "ambiguous"
    assert s.resolve_target("The 'Submit' button", allowed={"5"}) == "5"


def test_duplicate_html_ids_are_not_selected_arbitrarily():
    s = _session()
    s.nodes["1"]["id"] = "duplicate"
    s.nodes["2"]["id"] = "duplicate"
    assert s.resolve_id("duplicate") is None


def test_no_match_returns_none():
    s = _session()
    assert s.resolve_target("The canvas drawing area") is None


def test_empty_and_no_snapshot():
    s = _session()
    assert s.resolve_target("") is None
    s2 = BrowserSession(options=BrowserOptions())
    assert s2.resolve_target("The 'Submit' button") is None


def test_resolve_id():
    s = _session()
    s.nodes["2"]["id"] = "download-btn"
    assert s.resolve_id("download-btn") == "2"
    assert s.resolve_id("missing") is None


def test_bracket_phrase_match():
    # 方括号短语（如 "[PDF] xxx"）也应能命中 DOM
    s = _session()
    s.nodes["5"] = {"ref": "5", "role": "link", "name": "[PDF] ieeexplore.ieee.org",
                    "value": "", "direct": True}
    assert s.resolve_target("The [PDF] ieeexplore.ieee.org link next to the first search result") == "5"


def test_route_by_element_id():
    from osworld_agent.actions import Action
    from osworld_agent.env import Observation
    s = _session()
    s.nodes["2"]["id"] = "download-btn"
    obs = Observation(info={"exposed_refs": list(s.nodes.keys())})
    action = Action("click", {"target": "the download button", "element_id": "download-btn"})
    route = s.route(action, obs)
    assert route["channel"] == "browser_dom"
    assert route["target_ref"] == "2"


def test_target_ref_semantic_mismatch_rejected():
    # 拦截「点确定却路由到 语音转文字 复选框」的 target_ref 串号
    from osworld_agent.actions import Action
    from osworld_agent.env import Observation
    s = _session()
    s.nodes["64"] = {"ref": "64", "role": "checkbox", "name": "语音转文字", "value": "", "direct": True}
    s.nodes["33"] = {"ref": "33", "role": "StaticText", "name": "确定", "value": "", "direct": True}
    obs = Observation(info={"exposed_refs": list(s.nodes.keys())})
    wrong = s.route(Action("click", {"target": "确定", "target_ref": "64"}), obs)
    assert wrong["channel"] == "reobserve"
    assert wrong["error_code"] == "mismatch"
    # 正确 ref 走 DOM 定位
    ok = s.route(Action("click", {"target": "确定", "target_ref": "33"}), obs)
    assert ok["channel"] == "browser_dom"
    # 复选框自身 → 允许
    ck = s.route(Action("click", {"target": "语音转文字", "target_ref": "64"}), obs)
    assert ck["channel"] == "browser_dom"


def test_target_ref_semantic_substring_and_non_cjk_allowed():
    # 子串命中（关闭 → 关闭此对话框）与英文标签不触发中文串号拦截
    from osworld_agent.actions import Action
    from osworld_agent.env import Observation
    s = _session()
    s.nodes["10"] = {"ref": "10", "role": "button", "name": "关闭此对话框", "value": "", "direct": True}
    s.nodes["11"] = {"ref": "11", "role": "button", "name": "Submit", "value": "", "direct": True}
    obs = Observation(info={"exposed_refs": list(s.nodes.keys())})
    assert s.route(Action("click", {"target": "关闭", "target_ref": "10"}), obs)["channel"] == "browser_dom"
    assert s.route(Action("click", {"target": "Submit", "target_ref": "11"}), obs)["channel"] == "browser_dom"


def test_screen_to_viewport():
    class FakePage:
        def evaluate(self, js):
            return {"sx": 100, "sy": 80, "iw": 1200, "ih": 800, "oh": 900, "dpr": 1}
    # vx = 300/1 - 100 = 200; vy = 500/1 - 80 - (900-800) = 320
    assert BrowserSession._screen_to_viewport(FakePage(), 300, 500) == (200, 320)
