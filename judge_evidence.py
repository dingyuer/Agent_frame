"""Bounded, task-relevant observations for the 9B judge; no policy claims as facts."""
import hashlib
import json
import re
from urllib.parse import urlsplit, urlunsplit

from .model.decision_model import encode_image

NODE_KEYS = ('ref', 'parent', 'id', 'role', 'name', 'value', 'states', 'href', 'in_viewport')
ARTIFACT_KEYS = ('artifact_id', 'download_id', 'status', 'completed', 'path', 'bytes', 'size_bytes', 'url', 'final_url')


def bounded(value, text_limit=240, items=8, depth=0):
    if depth > 4:
        return str(value)[:text_limit]
    if isinstance(value, str):
        return value[:text_limit]
    if isinstance(value, dict):
        return {str(k): bounded(v, text_limit, items, depth+1) for k, v in list(value.items())[:items]}
    if isinstance(value, (list, tuple)):
        return [bounded(v, text_limit, items, depth+1) for v in value[:items]]
    return value


def nodes(obs):
    return [n for b in (getattr(obs, 'context', None) or []) if b.get('kind') == 'browser_ax'
            for n in b.get('nodes', []) if isinstance(n, dict)]


def page_url(obs):
    return (getattr(obs, 'info', None) or {}).get('url') or next(
        (b.get('url', '') for b in (getattr(obs, 'context', None) or []) if b.get('kind') == 'browser_ax'), '')


def terms(value):
    text = json.dumps(value, ensure_ascii=False).casefold()
    words = re.findall(r'[a-z0-9_./:-]{3,}|[\u4e00-\u9fff]+', text)
    result = set()
    for word in words:
        if re.fullmatch(r'[\u4e00-\u9fff]+', word):
            result.update(word[i:i+2] for i in range(len(word)-1))
        else:
            result.add(word)
    return result


def _key(node):
    return ('id', node['id']) if node.get('id') else (node.get('role'), node.get('name', ''))


def _values(node):
    # Ignore changing refs, capture times and layout-only properties.
    return json.dumps({k: node.get(k) for k in ('name', 'value', 'states', 'href')}, sort_keys=True, ensure_ascii=False)


def target_identity(obs, args):
    candidates = nodes(obs)
    if args.get('target_ref') is not None:
        candidates = [n for n in candidates if str(n.get('ref')) == str(args['target_ref'])]
    elif args.get('element_id'):
        candidates = [n for n in candidates if n.get('id') == args['element_id']]
    elif args.get('target'):
        candidates = [n for n in candidates if n.get('name') == args['target']]
    else:
        return {}
    if len(candidates) != 1:
        return {}
    return {k: candidates[0][k] for k in ('id', 'role', 'name', 'href') if k in candidates[0]}


def select_frame(obs, step=None, execution=None, other=None, *, max_nodes=40):
    if obs is None:
        return None
    step, execution = step or {}, execution or {}
    info = getattr(obs, 'info', None) or {}
    all_nodes, other_nodes = nodes(obs), nodes(other)
    args = execution.get('args') or {}
    target = execution.get('target_node', {})
    focus = terms([step.get('goal'), step.get('expected_evidence'), step.get('verification'), args, target])
    other_values = {_key(n): _values(n) for n in other_nodes}
    ranked = []
    for index, node in enumerate(all_nodes):
        content = ' '.join(str(node.get(k, '')) for k in ('name', 'value', 'href', 'id')).casefold()
        score = min(12, sum(t in content for t in focus)) * 4
        if node.get('role') in {'dialog', 'alert', 'alertdialog', 'status'}:
            score += 35
        if node.get('states', {}).get('focused'):
            score += 20
        # Refs belong to the before snapshot; semantic/id matches survive reindexing.
        if (args.get('element_id') and str(node.get('id', '')) == str(args['element_id'])) or (args.get('target') and str(args['target']).casefold() == str(node.get('name', '')).casefold()):
            score += 65
        if target and _key(node) == _key(target):
            score += 80
        if other_values and other_values.get(_key(node)) != _values(node):
            score += 18
        if node.get('in_viewport') is False or node.get('states', {}).get('hidden'):
            score -= 12
        ranked.append((score, index, node))
    chosen = sorted(ranked, key=lambda item: (-item[0], item[1]))[:max_nodes]
    # Include nearby labels/parents around highest-ranked targets.
    indices = {index for _, index, _ in chosen[:max(1, max_nodes-8)]}
    refs = {str(n.get('ref')): i for i, n in enumerate(all_nodes)}
    for _, index, node in chosen[:4]:
        for near in (refs.get(str(node.get('parent'))), index-1, index+1):
            if near is not None and 0 <= near < len(all_nodes) and len(indices) < max_nodes:
                indices.add(near)
    for _, index, _ in chosen:
        if len(indices) < max_nodes:
            indices.add(index)
    selected = [{k: bounded(all_nodes[i][k], 180, 10) for k in NODE_KEYS if k in all_nodes[i]} for i in sorted(indices, key=lambda i: (-ranked[i][0], i))]
    artifacts = [a for a in info.get('artifacts', []) if isinstance(a, dict)]
    old_ids = {a.get('artifact_id') or a.get('download_id') or a.get('path')
               for a in (getattr(other, 'info', None) or {}).get('artifacts', []) if isinstance(a, dict)}
    def artifact_rank(pair):
        index, a = pair
        content = json.dumps(a, ensure_ascii=False).casefold()
        related = sum(t in content for t in focus)
        ident = a.get('artifact_id') or a.get('download_id') or a.get('path')
        return (related, bool(other is not None and ident not in old_ids), index)
    selected_artifacts = [a for _, a in sorted(enumerate(artifacts), key=artifact_rank, reverse=True)[:5]]
    runtime = info.get('runtime_state') or {}
    apps = sorted(runtime.get('applications', []), key=lambda a: not a.get('foreground'))[:4]
    runtime_view = {k: runtime[k] for k in ('source', 'complete') if k in runtime}
    runtime_view['applications'] = [{k: bounded(a[k], 180, 6) for k in (
        'process_name', 'pid', 'hwnd', 'title', 'class_name', 'foreground', 'bounds',
        'visible', 'minimized', 'cloaked', 'enabled', 'responsive') if k in a} for a in apps]
    frame = {'url': str(page_url(obs))[:1200], 'surface': info.get('active_surface'),
             'nodes': selected, 'artifacts': [{k: bounded(a[k], 500) for k in ARTIFACT_KEYS if k in a} for a in selected_artifacts],
             'pending_downloads': bounded(info.get('pending_downloads', []), 300, 4),
             'runtime_state': runtime_view,
             'desktop_focus_stable': info.get('desktop_focus_stable'),
             'desktop_window_wait': bounded(info.get('desktop_window_wait'), 200, 8),
             'workspace': bounded(info.get('initial_state'), 400, 6),
             'dialogs': bounded(info.get('dialogs', info.get('dialog', [])), 400, 4),
             'text': str(getattr(obs, 'text', '') or '')[:1600],
             'selection': {'nodes_total': len(all_nodes), 'nodes_selected': len(selected),
                           'artifacts_total': len(artifacts), 'artifacts_selected': len(selected_artifacts),
                           'truncated': len(selected) < len(all_nodes) or len(selected_artifacts) < len(artifacts)}}
    return frame


def effect_review(before, after, execution):
    """Narrow claims about dispatched inputs/navigation, never business completion."""
    result = {'method': 'code', 'status': 'unknown', 'reason': '尚无确定性动作效果证据', 'evidence': []}
    if execution.get('status') in {'failed', 'error', 'rejected'} or execution.get('dispatched') is False:
        return {**result, 'status': 'failed', 'reason': '动作未成功派发或执行器报告失败'}
    if execution.get('status') in {'pending', 'in_progress'}:
        return {**result, 'status': 'pending', 'reason': '系统报告操作仍在进行'}
    if execution.get('status') != 'executed' or before is None or after is None:
        return result
    action, args = execution.get('action'), execution.get('args') or {}
    if action == 'goto' and args.get('url'):
        def normalize(url):
            value = urlsplit(url)
            return urlunsplit((value.scheme.lower(), value.netloc.lower(), value.path or '/', value.query, value.fragment))
        if normalize(str(page_url(after))) == normalize(str(args['url'])):
            return {**result, 'status': 'passed', 'reason': '目标 URL 已打开；不代表页面业务已完成', 'evidence': [str(page_url(after))]}
    if action != 'type' or not isinstance(args.get('text'), str) or not args['text']:
        return result
    candidates = [n for n in nodes(before) if n.get('role') in {'textbox', 'searchbox', 'combobox'}]
    if args.get('target_ref') is not None:
        candidates = [n for n in candidates if str(n.get('ref')) == str(args['target_ref'])]
    elif args.get('element_id'):
        candidates = [n for n in candidates if n.get('id') == args['element_id']]
    elif args.get('target'):
        candidates = [n for n in candidates if n.get('name') == args['target']]
    else:
        candidates = [n for n in candidates if n.get('states', {}).get('focused')]
    if len(candidates) != 1 or page_url(before) != page_url(after):
        return result
    for key in ('page_id', 'document_id'):
        old_id, new_id = (before.info or {}).get(key), (after.info or {}).get(key)
        if old_id and new_id and old_id != new_id:
            return result
    old = candidates[0]
    matches = [n for n in nodes(after) if _key(n) == _key(old)]
    if len(matches) == 1 and isinstance(matches[0].get('value'), str):
        value, previous = matches[0]['value'], old.get('value', '')
        if value != previous and (value == args['text'] or value == str(previous) + args['text']):
            return {**result, 'status': 'passed', 'reason': '目标输入框已出现本次输入；子任务仍需单独核查',
                    'evidence': [str(old.get('name') or old.get('id') or '输入框') + ': ' + value[:160]]}
    return result


def evidence_fingerprint(obs, step, execution):
    frame = select_frame(obs, step, execution)
    def stable(value):
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items() if k not in {
                'ref', 'parent', 'snapshot_id', 'captured_at', 'timestamp', 'latency_ms', 'elapsed_s', 'selection'}}
        if isinstance(value, list):
            return [stable(v) for v in value]
        return value
    frame = stable(frame)
    if frame:
        frame['nodes'] = sorted(frame['nodes'], key=lambda n: json.dumps(n, sort_keys=True, ensure_ascii=False))
    encoded = encode_image(getattr(obs, 'screenshot', None))
    return hashlib.sha256((json.dumps(frame, ensure_ascii=False, sort_keys=True) + encoded).encode()).hexdigest()


def checkpoint(obs, step, report, decision_step):
    if obs is None or obs.screenshot is None or (obs.info or {}).get('screenshot_stale'):
        return None
    image = encode_image(obs.screenshot)
    if not image:
        return None
    return {'id': f"{step['id']}:a{decision_step}", 'goal': step.get('goal', ''),
            'decision_step': decision_step, 'review_method': report.get('method'),
            'frame': select_frame(obs, step, max_nodes=16), 'screenshot': image,
            'sha256': hashlib.sha256(image.encode()).hexdigest()}
