"""Small, framework-derived browser facts and conservative pre-dispatch checks."""
import re
from urllib.parse import urljoin

from .paper_resolver import arxiv_id_from_url
from .scene import browser_context


def exposed_nodes(obs):
    if obs is None or (obs.info or {}).get('active_surface') == 'desktop':
        return []
    exposed = (obs.info or {}).get('exposed_refs')
    return [n for b in browser_context(obs) for n in b.get('nodes', [])
            if exposed is None or n.get('ref') in exposed]


def active_paper(completion, step=None):
    if completion is None:
        return None
    goals = completion.paper_goals()
    ids = set((step or {}).get('check_ids', []))
    selected = [g for g in goals if g['check_id'] in ids]
    goals = selected or goals
    # Do not pick an arbitrary paper for a multi-paper task.
    return goals[0] if len(goals) == 1 and goals[0].get('binding') else None


def same_paper(identifier, expected):
    # Explicit versions remain constraints; an unversioned goal permits versions.
    if re.search(r'v\d+$', expected, re.I):
        return identifier.lower() == expected.lower()
    return re.sub(r'v\d+$', '', identifier, flags=re.I).lower() == expected.lower()


def observation_url(obs):
    if obs is None:
        return ''
    return (obs.info or {}).get('url') or next((b.get('url', '') for b in obs.context if b.get('kind') == 'browser_ax'), '')


def paper_action_error(action, obs, goal, route=None):
    """Reject an explicitly identifiable wrong paper, never invent a destination."""
    if not goal:
        return ''
    expected = goal['binding']['canonical_id']
    url = (route or {}).get('goto_url', '')
    if action.action == 'goto':
        url = action.args.get('url', '')
    elif action.action in {'click', 'double_click', 'right_click'}:
        ref = (route or {}).get('target_ref') or action.args.get('target_ref')
        eid = action.args.get('element_id')
        matches = [n for n in exposed_nodes(obs) if (ref and n.get('ref') == ref) or (eid and n.get('id') == eid)]
        if len(matches) == 1:
            url = urljoin(observation_url(obs), matches[0].get('href') or '')
    identifier = arxiv_id_from_url(url)
    if identifier and not same_paper(identifier, expected):
        return f'目标链接属于 arXiv {identifier}，任务已绑定 {expected}；请使用已核实论文的链接。'
    current = arxiv_id_from_url(observation_url(obs))
    keys = {str(k).lower() for k in (action.args.get('keys') or [])} if action.action == 'hotkey' else set()
    save_shortcut = action.action == 'hotkey' and bool(keys & {'ctrl', 'control', 'cmd', 'command'}) and 's' in keys
    if (current and not same_paper(current, expected) and action.action in {'click', 'double_click', 'right_click', 'hotkey', 'press'}
            and (re.search(r'download|save|下载|保存', str(action.args), re.I)
                 or save_shortcut)):
        return f'当前页面属于 arXiv {current}，不是已绑定的 {expected}；先返回正确论文页面，不要下载当前文件。'
    return ''


def operational_facts(obs, completion, report, step=None):
    """Only public binding + browser event evidence, never model-written facts."""
    lines = []
    goal = active_paper(completion, step)
    if goal:
        binding = goal['binding']
        lines.append(f"已核实论文：{binding['full_title']}；arXiv {binding['canonical_id']}。")
        lines.append(f"已核实 PDF 地址：{binding['pdf_url']}；可用 goto 打开后通过界面保存。")
        if goal.get('filename'):
            lines.append('交付文件名：' + goal['filename'])
        current = arxiv_id_from_url(observation_url(obs))
        if current and not same_paper(current, binding['canonical_id']):
            lines.append('当前打开的是其他论文，请回到上述已核实地址。')
    info = (obs.info or {}) if obs else {}
    pending = info.get('pending_downloads', [])
    if pending:
        lines.append('系统仍有下载进行中；等待框架确认，不要重复点击下载。')
    artifacts = [a for a in info.get('artifacts', []) if isinstance(a, dict) and a.get('completed') is True]
    for artifact in artifacts[-3:]:
        path = str(artifact.get('path') or artifact.get('filename') or artifact.get('suggested_filename') or '已登记文件')
        size = artifact.get('bytes')
        lines.append('下载传输已完成：' + path[:400] + (f'（{size} 字节）' if type(size) is int else '') + '；这不等于任务验收通过。')
        ids = [arxiv_id_from_url(artifact.get(k, '')) for k in ('url', 'final_url')]
        if goal and any(i and not same_paper(i, goal['binding']['canonical_id']) for i in ids):
            lines.append('该文件来源属于其他论文；不要继续点击它的下载按钮，也不要将它改名冒充目标论文。')
    # Use public_report; private evaluator values must never enter policy input.
    for check in (report or {}).get('checks', []):
        if check.get('kind') not in {'paper_download', 'download'}:
            continue
        reason = check.get('reason', '')
        if reason == 'download transport does not identify the bound official paper':
            lines.append('文件验收未通过：下载来源与已绑定论文不符，需打开正确 PDF 地址重新保存。')
        elif reason == 'downloaded filename does not match':
            lines.append('文件已收到，文件名仍不符合要求；通过文件管理器或另存为界面处理，不要反复点击原下载按钮。')
        elif artifacts and check.get('status') != 'passed':
            lines.append('文件验收仍待处理：' + str(reason)[:180])
    if artifacts and not pending:
        lines.append('请根据文件验收反馈推进后续步骤，不要仅因截图没有下载提示而重复下载。')
    rendering = info.get('render_wait', {})
    if rendering.get('status') == 'timeout':
        lines.append('PDF 渲染等待已超时，当前画面尚未确认就绪；请检查页面或切换观察，不要在空白画面反复保存。')
    return list(dict.fromkeys(lines))


def paper_review_error(obs, completion, report, step):
    """A screenshot reviewer cannot overrule a known paper-identity conflict."""
    goal = active_paper(completion, step)
    if not goal:
        return ''
    phase = step.get('phase')
    if phase in {'identify', 'locate_fulltext', 'download'}:
        current = arxiv_id_from_url(observation_url(obs))
        expected = goal['binding']['canonical_id']
        if current and not same_paper(current, expected):
            return f'当前页面 arXiv {current} 与已绑定目标 {expected} 不符，不能确认本步骤完成。'
    if phase in {'download', 'verify'}:
        for check in report.get('checks', []):
            if (check.get('id') == goal['check_id'] and check.get('status') == 'failed'
                    and check.get('reason') == 'download transport does not identify the bound official paper'):
                return '下载文件的来源与已绑定论文不符，不能确认下载步骤完成。'
    return ''
