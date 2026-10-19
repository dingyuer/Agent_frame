# Markdown 知识库与规划前筛选

## 存储和编辑

默认文件为 `knowledge/user_knowledge.md`。前端“知识库”左侧编辑或预览每条知识，右侧使用当前未保存草稿测试筛选。只有保存后的内容才用于实际任务。保存带文件内容摘要校验；其他编辑器已改动文件时返回冲突，避免覆盖。

每条知识是一个带 YAML front matter 的 Markdown 文档：

```markdown
---
id: campus
title: 校园官网
aliases: [学校官网, 校内门户]
keywords: [校园, 通知公告]
url: https://campus.example.org
enabled: true
---

# 适用场景
查询校园通知、报名信息。

## 操作提示
- 通知公告位于首页“通知公告”栏目。
- 报名时间与材料要求以通知正文为准。
```

`id` 唯一，只允许字母、数字、下划线、连字符。`title` 必填；`url` 与正文至少填写一个；`enabled: false` 会在所有筛选方法中排除该条。别名和关键词是列表。正文支持标题、列表、代码块、链接、表格和引用，预览不执行原始 HTML。

多个文档在同一个独立文件中使用单独一行 `<!-- knowledge-entry -->` 分隔，该保留分隔符不要写入正文。前端自动维护分隔符，无需手工操作。保留原有每条正文 3000 字符、全文件 2 MB、最多 2000 条的校验边界，不截断知识正文。

旧 `user_knowledge.yaml` 只在默认 Markdown 文件尚不存在时读取；第一次保存迁移到 Markdown，原 YAML 不删除。`AGENT_KNOWLEDGE_PATH` 仍可指定文件；显式指定 YAML 时保留兼容存储。

## 统一筛选接口

`knowledge_selection.py`：

```python
result = select_knowledge(task, snapshot, method='all', model=None, judge=None)
# snapshot = {'document': {'version': 1, 'entries': [...]}, 'revision': '...'}
# result: entries, selected_ids, method, latency_ms, candidate_count,
#         reason, calls, revision, status，以及模型相关的附加证据
```

三种独立函数通过 `SELECTORS` 注册表调度：

| 方法 | 函数 | 行为 |
| --- | --- | --- |
| `all`（默认） | `select_all` | 返回全部启用知识，无模型调用，不再进行旧关键词过滤或最多 4 条截取 |
| `decision` | `select_decision` | 完整任务与所有启用知识进入当前决策模型；只允许返回已有 ID，框架取回原文 |
| `jev` | `select_jev` | Jev System One 为每条知识回答相关性 choice；最高概率标签为 relevant 或 unknown 时保留 |

Jev 小知识库合并在一个 HTTP 请求中，多问题不等于一次模型前向。较大知识库按最多 32 问和状态长度分批串行处理，所有条目都会被覆盖；不偷偷截断。返回各条概率、调用记录和总耗时。知识相关性不是动作完成判定，分数也不是实际成功率。

增加新筛选方法时实现相同返回约定并加入注册表与配置选项即可；存储、规划器和前端结果结构不用重写。

## 实际任务的接入

运行页“模型与知识设置 → 规划知识”选择方法。默认 `all`。`decision` 复用决策模型的连接配置，使用独立客户端，不修改原决策对话或采样状态。`jev` 有独立端点设置，即使动作验证选择复用决策模型，也能用 Jev 筛选知识。

本机启动器、OSWorld 启动器和 `Agent` / `Pipeline` 均传递此配置：

- CLI：`--knowledge-method all|decision|jev`，`--knowledge-jev-url URL`。
- Python / agent 配置：`knowledge_method`、`knowledge_jev_url`。

初次规划前读取保存文件并筛选，选中原文加入规划提示词。重新规划复用本次任务的知识快照，不重复筛选，也不突然使用任务中途修改的文件。任务明确约束和系统规则优先，知识不是任务完成证据。

模型筛选失败时，实际任务显式记录 `status=fallback`、请求方法和错误类型，回退为全部启用知识；知识文件本身无效则记录读取失败并按原任务规划。测试页不回退，直接显示失败，避免误以为被测算法成功。

筛选耗时、原始模型调用及提供商 token 用量进入任务轨迹；调用角色为 `knowledge_selector`，不作为动作决策 SFT 样本。选择全部知识时没有额外模型 token，但知识会占用后续规划输入。

## 测试

```bash
python -m pytest -q tests/test_knowledge_selection.py tests/test_knowledge_base.py
```

覆盖 Markdown 往返与保存冲突、停用过滤、未知 ID 拒绝、Jev 全量分批、不确定保留、草稿测试不落盘、启动参数、默认全给、筛选失败回退及规划提示词实际注入。

若部署机器需要代理访问公网决策模型，可显式设置 `AGENT_KNOWLEDGE_HTTP_PROXY`。仅影响知识筛选的决策模型客户端，Jev 内网请求保持直连；未设置时沿用默认直连。代理地址和认证不写入轨迹或知识文件。本机验证使用已有主机代理访问 Qwen。

## 本机验证记录（2026-10-18）

- 相关后端回归 65 项通过；既有工作台 Chromium 回归 3 项通过。
- 全量回归为 1539 passed、97 skipped、9 failed（均为之前已记录的既有失败），随后新增的规划注入与回退测试也通过。
- 三条草稿知识（校园、采购、请假），任务为查校园通知：全部提供选入 3 条；真实 Qwen3.7-flash 和 51 服务器 Jev d3-flash 均仅选入校园条目。
- API 实测决策模型 1432.49 ms，Jev 211.50 ms；完整前端按钮复测分别 1101.96 ms、204.03 ms。样本很小，不作为普遍性能结论。
- 前端检查编辑/预览切换、草稿测试不落盘、HTML 不执行、方法设置与实际模型调用，无 JavaScript 错误。
- 测试不启动 GUI 任务，不进入历史轨迹；历史仍为 11 条多步骤场景记录。

实际界面截图位于 `/data1/chenyuning/GradRank/image/`：`agent-knowledge-markdown-editor.png`、`agent-knowledge-jev-test.png`、`agent-knowledge-decision-test.png`、`agent-settings-compact-knowledge.png`。

测试区主计时显示页面总等待（含网络），另外显示服务处理时间；原始结果的 `selection_ms` 为筛选函数耗时，`latency_ms` 包含模型客户端准备。Jev 分批中途失败也保留已发出的调用，避免漏计 token。
