"""点击无效（no-effect）的确定性否认：动画页面上的死点击必须被识别。

场景来源：ui_20261015_125950 —— 模型对已选中的「热门方案」Tab 连点 8 次，
页面有公告轮播导致截图每步都变，屏幕差分从未触发，框架无法证明点击无效。
"""
from osworld_agent.env import Observation
from osworld_agent.judge_evidence import click_effect_denied, effect_review


def _obs(url='https://portal.test/#/', ticker='公告A', tab_focused=True, extra_nodes=0):
    nodes = [
        {'ref': '1', 'role': 'tab', 'name': '热门方案', 'id': '#tab-popular',
         'states': {'focused': tab_focused}},
        {'ref': '2', 'role': 'tab', 'name': '技术走读', 'id': '#tab-tech',
         'states': {'focused': not tab_focused}},
        {'ref': '3', 'role': 'button', 'name': '搜索', 'id': '#search'},
        {'ref': '4', 'role': 'StaticText', 'name': ticker},  # 公告轮播，每次观察文本都变
    ] + [{'ref': str(100 + i), 'role': 'StaticText', 'name': f'卡片{i}'} for i in range(extra_nodes)]
    return Observation(
        info={'url': url, 'page_id': 'p1', 'document_id': 'd1',
              'artifacts': [], 'pending_downloads': [], 'download_events': [],
              'artifact_events': [], 'dialogs': [], 'scroll_offset': [0, 0],
              'tabs': [{'tab_id': 't1', 'url': url, 'active': True}]},
        context=[{'kind': 'browser_ax', 'nodes': nodes}])


_CLICK = {'action': 'click', 'status': 'executed', 'dispatched': True, 'args': {'target_ref': '1'}}


def test_dead_click_on_selected_tab_is_denied():
    before = _obs()
    after = _obs(ticker='公告B')  # 只有轮播文本变了
    report = effect_review(before, after, _CLICK)
    assert report['status'] == 'unknown'          # 状态契约不变
    assert report.get('no_effect') is True        # 但带有确定性否认标记
    assert '未产生任何可观察变化' in report['reason']


def test_real_tab_switch_is_not_denied():
    before = _obs(tab_focused=True)
    after = _obs(tab_focused=False, ticker='公告B')  # 焦点移到另一个 Tab
    assert click_effect_denied(before, after, _CLICK) is None


def test_navigation_is_not_denied():
    before = _obs()
    after = _obs(url='https://portal.test/#/solution-center')
    assert click_effect_denied(before, after, _CLICK) is None


def test_content_growth_is_not_denied():
    before = _obs()
    after = _obs(extra_nodes=8)  # 「加载更多」追加了卡片
    assert click_effect_denied(before, after, _CLICK) is None


def test_new_download_is_not_denied():
    before = _obs()
    after = _obs()
    after.info['artifacts'] = [{'artifact_id': 'a1', 'completed': True}]
    assert click_effect_denied(before, after, _CLICK) is None


def test_undispatched_or_non_click_is_not_denied():
    before, after = _obs(), _obs()
    assert click_effect_denied(before, after, {**_CLICK, 'dispatched': False}) is None
    assert click_effect_denied(before, after, {**_CLICK, 'action': 'type'}) is None
    assert click_effect_denied(None, after, _CLICK) is None
