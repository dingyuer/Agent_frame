"""Chinese names for display; stable protocol IDs stay separate from labels."""
QUESTION_TITLES = {
    'readiness': '当前页面是否已经可以操作？',
    'replan': '现在是否需要重新规划？',
    'route': '下一步目标应该用哪种方式操作？',
    'identity': '当前打开的是任务要求的目标论文吗？',
    'next': '为了完成当前任务，下一步应该怎么做？',
    'step': '搜索子任务当前完成到哪一步？',
    'tab': '应该切换到哪个标签页？',
    'effect': '刚才的操作是否产生了预期效果？',
    'saved': '当前设置的全部改动是否已保存？',
    'progress': '这项任务目前达到哪个进度等级？',
}
LABEL_TITLES = {
    'ready': '已就绪', 'waiting': '仍在加载', 'unknown': '无法确定',
    'yes': '是', 'no': '否', 'dom': '通过 DOM 操作', 'gui': '通过 GUI 操作',
    'done': '下载已完成', 'wait': '继续等待', 'retry': '恢复或重试', 'inspect': '先检查证据',
    'dismiss': '先恢复会话', 'save': '直接点击保存',
    'passed': '子任务已完成', 'progress': '仅有局部进展', 'failed': '操作未成功',
    'correct': '输入目标正确', 'wrong_target': '输入了错误字段',
    'true': '已全部保存', 'false': '仍未保存',
    'tab_A': '标签页 A', 'tab_B': '标签页 B', 'tab_C': '标签页 C', 'tab_D': '标签页 D',
    '0': '尚未开始', '1': '已输入关键词', '2': '已显示搜索结果', '3': '已打开目标论文', '4': '已保存并验证 PDF',
}
OVERRIDES = {
    'replan': {'yes': '需要重新规划', 'no': '继续当前计划'},
    'identity': {'yes': '是目标论文', 'no': '不是目标论文'},
    'effect': {'yes': '预期效果已发生', 'no': '预期效果未发生'},
}


def question_ui(questions):
    result = {}
    for key, q in questions.items():
        labels = list(map(str, range(len(q['criteria'])))) if q['type'] == 'score' else list(q.get('criteria', {'false': '', 'true': ''}))
        names = {**LABEL_TITLES, **OVERRIDES.get(key, {})}
        result[key] = {'title': QUESTION_TITLES[key], 'options': {label: names[label] for label in labels}}
    return result
