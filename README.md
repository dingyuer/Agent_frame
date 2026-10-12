# Agent_frame

OSWorld GUI Agent 框架。

当前版本：**0.2.0**（[v0.2.0 Release](https://github.com/BAYMAXer/Agent_frame/releases/tag/v0.2.0)）。

源码直接位于仓库根目录，不同版本通过 Git 标签保存。

| 版本 | 对应源码 |
| --- | --- |
| [v0.1.0](https://github.com/BAYMAXer/Agent_frame/tree/v0.1.0) | 原 osworld_agent_1009 代码 |
| [v0.2.0](https://github.com/BAYMAXer/Agent_frame/tree/v0.2.0) | 原 osworld_agent_1010 代码 |

环境搭建：[本地环境搭建指南](docs/本地环境搭建指南.md)。

入口和测试使用 osworld_agent 包名。克隆时可将仓库目录命名为 osworld_agent：

~~~powershell
git clone https://github.com/BAYMAXer/Agent_frame.git osworld_agent
~~~

## 浏览器执行与系统验收

框架将规划、决策、定位、执行和验收分开。规划模型可以与决策模型共用端点，
也可以使用独立模型；定位模型保持固定，只参与执行，不采集 grounding SFT 或更新其权重。

任务必须先形成包含目标、操作前提和预期证据的计划。规划格式错误会修复重试一次；
仍失败则返回 `planning_failed`，不开始操作。完整计划、公开验收条件与当前截图是
受保护的决策输入，历史记忆与 AX 节点按预算裁剪。

持续卡住时，框架把实际失败和验收反馈交给规划器修订计划；默认最多重规划一次，
可通过 `Agent(max_replans=...)` 或配置的 `agent.max_replans` 调整，重规划不会重置环境。

### 例子：下载指定论文

对 `请下载一篇叫做AAA的论文`，系统通过保守的任务规则生成公开条件：

```json
{"checks": [{"id": "check_1", "kind": "paper_download", "title": "AAA", "count": 1, "format": "pdf"}]}
```

同时提供论文任务的流程提示：确认浏览器可用 → 用用户提供的名称在可访问的学术搜索入口检索
（例如 arXiv、Google Scholar 或出版机构）→ 核对文章身份 → 找到 PDF 全文 → 下载 → 读取系统验收结果。
搜索入口由规划和当前页面决定；框架不会假定某个外部站点始终可访问，也不会把摘要页当作下载完成。

单篇论文的短名和改名要求也可识别，例如 `请下载一篇叫做Roboclaw的论文，并重命名为Roboclaw`。
短名先作为查询保留，框架根据实际观察到的 arXiv 候选 URL，从官方元数据绑定完整标题和论文 ID。
适配器只接受完整标题或冒号前的完整别名；多个匹配、来源不可达或不支持的附加要求不会被静默忽略。
身份绑定单独追加到事件中，原始任务契约不变；官方来源与实际 PDF 交叉验收。
绑定后框架可直接下载对应 PDF 并交付为 `Roboclaw.pdf`，无需点击查看器工具栏。
自动身份解析目前支持 arXiv；其他来源仍可用显式完整标题合同和普通下载动作。
多篇或规则不能完整理解的指令应提供可信的 `completion_spec`，避免遗漏条件。
规划模型不能自行新增或放宽验收标准。领域流程提示也不能保证小模型在所有网站都能正确决策。

### 自动等待与预算

`max_steps` 约束模型决策轮次。下载发起后，框架读取浏览器事件并等待完成、失败或超时，
不反复调用模型输出 `wait`，不增加决策步数，也不将传输中的静止画面计为卡死。
最后一个决策步触发的下载同样先等待和验收，再判断是否耗尽步数。
没有真实下载迹象时仅给有界启动宽限；对话框需要处理时返回决策。不会因一直标为
`in_progress` 就无限续期，也不自动重发执行结果不确定的点击。

```python
agent = Agent(decision_model=model, grounding=grounding, env=env,
              wait_config={"operation_timeout_s": 300, "task_timeout_s": 1800,
                           "download_start_timeout_s": 2, "poll_interval_s": 0.25})
```

本机前端可设置下载等待上限和任务时限，等待状态显示“不消耗步数”，保留中断入口。
命令行对应 `--operation-timeout 300 --task-timeout 1800`，YAML 对应 `agent.wait_config`。
时间预算独立于决策预算；正在进行的外部模型请求仍受模型传输超时约束。
框架自动等待只记录事件；模型主动输出一次 `wait` 仍是一次真实采样，不能从训练记录删除。

### 截图、DOM 与弹窗

每轮决策结合截图、AX 和执行反馈；走 DOM 时仍提供当前截图。截图、AX、输入绑定到同一
标签页、文档和快照；视觉定位使用 CSS viewport 坐标，并检查尺寸、滚动偏移与时效。
AX 不足时可按需读取精简 DOM。优先使用 DOM 引用或唯一语义匹配，未派发的
`not_found/unsupported` 可在同页的新截图上尝试一次视觉定位；遮挡、歧义、已派发但结果不确定时先重新决策。

- 页面内弹窗：AX 中的 dialog/alert 节点优先保留，模型结合截图选择关闭、取消或其他适当操作，再恢复原任务。
- JavaScript alert/confirm/prompt：持久浏览器会话记录对话框事件，模型通过 `handle_dialog` 处理，可指定 `accept`、`prompt_text`、`dialog_id`。
- 浏览器工具栏、权限气泡或系统文件对话框：使用 `observe_desktop` 明确切换到桌面观察与操作；完成后用 `observe_browser` 回到页面。页面坐标不会被静默当作桌面坐标。
- 新标签页：通过 `list_tabs`、`switch_tab(tab_id)`、`close_tab` 管理，新弹出页面不会自动接管任务标签页。

视觉决策默认要求保留当前截图；预算不够时返回 `ContextBudgetError`，缺图则重新观察或停止，
不会为了塞入更多文本而静默删除最后一张图。JavaScript 对话框可能阻塞截图，此时允许依据新鲜的
对话框事件进行恢复；保留的旧截图明确标记为过期，不能据此执行页面坐标操作。
图像 token 成本仍是部署配置中的估计值，应按实际 processor 调整。

### 完成条件与证据

模型的 `request_finish` 只是结束申请。系统验收采用以下规则：

1. 有可信合同：逐项检查全部条件，条件之间为 AND。
2. 有独立环境评测器：任务结束时调用一次；同时存在合同时，环境评测与合同都必须通过。
3. 缺少证据或评测异常：保留 `unknown`，不会让模型自述代替验证。

没有可信条件或独立评测器的任务，结束申请得到 `success=false` 和
`termination_reason=completion_unverifiable`。训练轨迹对未知验收使用
`reward=null, reward_valid=false`；兼容字段 `score=0` 不能独自作为失败标签。

`completion_spec` 格式为 `{"checks": [...]}`。每项可填唯一 `id`，`scope` 默认为
`final`，表示结束时仍须成立。`milestone` 可记录中途已核实的页面状态；所有文件条件无论 scope
都在结束时重新读取文件，文件被删除、替换或变化会影响最终结果。

| kind | 必填字段 | 证据与可选约束 |
| --- | --- | --- |
| `url_equals` | `url` | 当前完整 URL；保留路径、查询串和锚点 |
| `element_state` | `role`, `name`，以及 `value` 或 `states` | 唯一 AX 节点的精确值/状态；可加 `element_id`, `url` |
| `text_present` | `text` | AX 完整文本经空白归一化后精确匹配；可加 `url` |
| `download` | `url` | 完成下载事件关联的实际文件；可加 `filename`, `min_bytes`, `sha256` |
| `paper_download` | `title` 或待绑定的 `query` | 完成事件、文件稳定性、真实 PDF 与首页标题；可加 `filename`, `count`, `format`, `author`, `year`, `doi`, `arxiv_id`, `url`, `min_bytes`, `sha256` |

未知字段或类型会报错。`paper_download` 当前仅支持 `pdf`；系统检查实际字节、SHA256、PDF
可解析性与页数，并在首页开头区域匹配完整标题。标题只做 Unicode、大小写和空白归一化，保留标点；
文件名、PDF metadata、模型声称的标题、正文或参考文献中提到的论文都不足以通过。
有作者、年份、DOI 或 arXiv ID 条件时，还须提供对应身份依据；arXiv 页边标识和绑定后的
官方 PDF 来源会专门核验，正文或参考文献中出现某个 ID 不能单独通过。

`download(url, filename=...)` 可直接指定交付文件名。后续改名使用
`rename_artifact(artifact_id, filename)`，仅处理本轮已登记文件，不覆盖现有目标。
文件 ID、初始摘要和改名链保持不变；内容、论文身份与最终文件名必须由同一份文件满足。
手工在框架之外移动文件不等同于登记的改名事件。

这是刻意保守的验证器。扫描件缺少可提取文字、加密 PDF、缺少 `pypdf`、超过 100 MiB 的文件、
提取顺序异常或标题/附加身份无法精确确定时可能返回 `unknown`；损坏的 PDF 或非 PDF 字节会失败。
它没有 OCR 或模糊匹配补判，也不保证仅凭同名标题就能消除论文身份歧义。任务可提供额外身份条件，
固定评测数据可提供私有摘要；不能把 `unknown` 自动改成成功或负训练样本。

### 公开任务与私有评测

公开条件只描述用户需要实现的目标；私有评测可保存固定测试集的真实 URL、文件摘要或其他标准。
私有条件通过 `private_completion_spec` 单独传入，不发送给规划器或决策模型。
模型只获得汇总的私有验证状态；完整答案和证据只进入归档/评测数据，不进入 SFT/RL 的 policy prompt。

例如公开条件使用上面的 `check_1`，私有文件 `private.json` 可以写成：

```json
{
  "checks": [{
    "kind": "download",
    "bind_to": "check_1",
    "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
  }]
}
```

将示例中的 64 个 `0` 替换为评测集真实目标文件的 SHA256。`bind_to` 仅允许出现在私有文件条件中，
要求公开标题条件和私有摘要由**同一个实际文件**满足；分别下载一个标题符合的文件和另一个摘要符合的文件不能通过。
普通独立下载目标必须由不同实际文件满足，不能用改名或硬链接重复计数。

Python API 支持：

```python
agent = Agent(decision_model=model, grounding=grounding, env=env,
              planner_model=planner, completion_spec=public_spec,
              private_completion_spec=private_spec, event_path="runs/{run_id}/events.jsonl")
result = agent.run(task, max_steps=30)
```

也可在 `agent.run(...)` 中覆盖 `completion_spec`、`private_completion_spec`、`event_path`、`run_id`。
OSWorld 任务 JSON 的 `completion` 可提供公开合同；存在环境评测器时仍执行两者的联合判定。

## 连接远程模型运行

下列命令在 `osworld_agent` 的父目录运行，本机须有 Windows 浏览器运行依赖及带 remote debugging
的 Chrome。模型均可位于其他机器；`AAA` 和模型名称、端点需替换为实际值。

```powershell
python -m osworld_agent.run_windows_local --task "请下载一篇叫做AAA的论文" --model DECISION_MODEL --model-url http://MODEL_HOST:9028/v1 --planner-model PLANNER_MODEL --planner-url http://PLANNER_HOST:9028/v1 --ground-type ui-tars --ground-model UI-TARS-1.5-7B --ground-url http://GROUND_HOST:49999/v1 --max-steps 30 --result-dir ./runs/paper-001
```

不传 `--planner-model` 时，规划使用决策模型。独立规划器在 RL 中明确视为固定 teacher；同一模型
承担规划和决策时，两类采样都进入 on-policy 数据。两者均可导出经过筛选的规划 SFT 样本。

可加 `--completion-spec public.json --private-completion-spec private.json`；单篇标题自动规则可省略
公开文件。多文件示例见 [config/completion_example.json](config/completion_example.json)。
每次使用新的 `--result-dir`，事件文件不会覆盖已有运行。

现有 UI-Venus 默认值保留，切换 UI-TARS 时同时指定 `--ground-type`、模型名和端点。
UI-TARS 适配器默认按输入图像像素解释输出；若部署端返回其他尺寸下的坐标，构造
`RealUItarsGrounding` 时显式提供 `grounding_width/grounding_height`。这两个定位器都作为固定组件使用。

## 轨迹采集与 SFT/RL

### 持续事件记录

指定 `event_path` 或本地入口的 `--result-dir` 后，`events.jsonl` 在运行过程中追加写入，
包含 episode、模型调用、观察、路由、定位结果、执行、验收与结束/异常事件。
`run_id/event_id/parent_decision_id/observation_id/actor_role/source` 连接事件、调用和 transition；
DOM 失败后刷新和视觉恢复各自保留记录，不会只剩最后一次动作。

每条事件立即 flush/fsync；图片先按内容摘要写入 `assets/` 再引用，重复图片去重。
即使 episode 中途异常，已写事件仍可读取。正常结束生成的 `trajectory.json` 使用 schema 3.0，
保存真实 prompt/response、前后观察、动作、执行结果、模型自评与系统验收来源，以及配置/模型版本 manifest。
可视化日志是另一条允许丢弃的异步通道，不承担训练数据持久化。

当前持久化不包含整个操作系统的可重放快照；能够恢复模型输入，不代表能确定性重放任意互联网网页。
私有验收归档和公开训练样本应分开管理；导出器不会把后验答案补进 prompt。

### 两类 SFT 导出

```bash
python -m osworld_agent.trajectory runs/paper-001/trajectory.json --mode sft-planner --output planner.jsonl
python -m osworld_agent.trajectory runs/paper-001/trajectory.json --mode sft-decision --image-paths --output decision.jsonl
```

默认仅选择独立验收成功的 episode；规划调用须有有效计划标记。
决策正样本排除明确拒绝、失败、不确定和提前申请结束的步骤；原始失败事件仍保留，可用于错误分析或其他训练目标。
最终成功也不能证明每一步都是最优示范，大规模训练前仍应检查样本质量，并按任务/网站模板/论文实体划分训练与测试集。

`--image-paths` 输出图片占位符和有序 `images` 绝对路径，便于训练侧的多模态数据适配器读取。
不传该选项则恢复原始消息中的 data URL。`loss_scope=completion_only` 表达监督范围；训练端仍需
用固定版本的 tokenizer/processor/chat template 构造张量及真实损失掩码。这是数据交换格式，
不能声称 JSONL 对任意 TRL 或其他训练器都能直接使用。没有 grounding SFT exporter。

### RL 采集与服务端元数据

单次运行可打开 `--capture-rollout`，并提供以下版本参数：

```text
--policy-revision CHECKPOINT_REVISION
--tokenizer-revision TOKENIZER_REVISION
--processor-revision PROCESSOR_REVISION
--chat-template-revision CHAT_TEMPLATE_REVISION
```

`ModelConfig` 中的对应字段是 `capture_rollout`、`policy_revision`、`tokenizer_revision`、
`processor_revision`、`chat_template_revision`。开启采集后，请求 `logprobs` 与 vLLM 的
`return_token_ids`，读取服务端实际的 prompt/completion token IDs；不会在客户端重新 tokenize 回复来假造采样记录。
标准字段细节见 [vLLM Chat Completion 协议](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/openai/chat_completion/protocol/)。

除原生 token 字段外，训练网关可在 OpenAI 兼容响应顶层提供本框架约定的扩展：

```json
{
  "rollout_metadata": {
    "prompt_token_ids": [10, 11],
    "token_ids": [12, 13],
    "token_logprobs": [-0.3, -1.2],
    "policy_revision": "ACTUAL_SERVED_CHECKPOINT",
    "tokenizer_revision": "ACTUAL_TOKENIZER",
    "processor_revision": "ACTUAL_PROCESSOR",
    "chat_template_revision": "ACTUAL_CHAT_TEMPLATE"
  }
}
```

这里只展示字段形状；数值必须来自该次真实采样。`rollout_metadata` 是本框架的网关契约，不能假定
原生 vLLM 自动返回全部字段。尤其 `policy_revision` 必须由服务端声明实际加载的不可变权重版本
（也可使用响应顶层同名字段）；客户端配置只是预期版本，不能作为证明。其余处理版本必须与实际服务配置一致。
缺 token、缺 logprobs、版本不符或校验失败会标记 `rl_ready=false`，保留原始数据但拒绝 on-policy 导出。

批量 collector 要求环境提供 `begin_episode(task_id=..., isolated=True)`，返回有效初态证明。
Windows 环境使用每轮独立 BrowserContext、空白页和下载目录；这隔离的是浏览器上下文，
不等于整个 Windows 桌面或外部网站数据库回滚。还可通过 Python `env_factory` 提供新环境。
仅调用 `reset()` 而不能证明隔离的环境会被拒绝。

在可导入的 `training_setup.py` 中提供 `create_agent()`，返回已配置上述采集字段、固定定位器和
支持隔离环境的 `Agent`；`tasks.jsonl` 每行一个对象：

```json
{"instruction":"请下载一篇叫做AAA的论文","private_completion_spec":{"checks":[{"kind":"download","bind_to":"check_1","sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]}}
```

将摘要替换为真实评测值，然后运行：

```bash
python -m osworld_agent.rl.rollout --factory training_setup:create_agent --tasks tasks.jsonl --output-dir runs/rollouts --n 4 --max-steps 30 --policy-revision CHECKPOINT_REVISION
python -m osworld_agent.trajectory runs/rollouts/EPISODE_ID/trajectory.json --mode rl --policy-revision CHECKPOINT_REVISION --output rollout.jsonl
```

collector 正确传递每轮步数预算，并将私有条件仅交给验收器。未知 reward 不得进入负奖励统计；
credit assignment 还要求任务、验收规范 hash、初态和策略版本一致。
语义树 credit 是可选的归因估计，不是局部动作成功的真值，也不能消除动态网页随机性。
同策略 planner 与 decision 的实际采样都记录；固定 teacher 和定位器不更新。
导出按模型调用事件的实际追加顺序排列，保留中途重规划的位置。
当前代码子 Agent 的内部采样尚未纳入这一契约，包含 `call_code_agent` 的 episode 会被 RL 导出拒绝。
OCR 定位兜底仅使用唯一精确文字匹配，不会暗中额外调用决策模型。

本仓库提供执行、验收、持续采集和导出，尚未集成梯度更新、PPO/GRPO 优化器或分布式训练调度。
使用训练器前还须核实其采样分布、约束解码与行为 logprobs 的含义一致。

## 测试

默认离线测试使用模拟模型、CDP 与临时文件，不连接模型端点或启动真实浏览器：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q
```

`requirements-test.txt` 包含严格 PDF 验收所需的 `pypdf`。Windows 使用 `.venv\Scripts\python`；
测试支持任意克隆目录名。

本地 Chromium 集成测试通过 `--run-browser-integration` 显式启用，需要另外安装 Playwright
和 Chromium；它使用受控本地页面验证浏览器机制，不评估远程模型的任务能力：

```bash
.venv/bin/python -m pip install playwright
.venv/bin/python -m playwright install chromium
.venv/bin/python -m pytest -q --run-browser-integration
```

真实网站布局、访问限制、浏览器原生 UI 与远程模型/processor 协议仍需在部署环境回归。

## 人工演示采集

Studio 的「SFT 采集」用于人工执行浏览器任务并审阅示范，无须部署决策模型或定位模型。
在名为 `osworld_agent` 的仓库目录的父目录，使用选定的 Python 环境运行：

```bash
python -m pip install -r osworld_agent/requirements-viz.txt
python -m playwright install chromium
python -m osworld_agent.viz.server
```

打开 `http://127.0.0.1:8088`，进入「SFT 采集」。端口可通过 `AGENTS_VIZ_PORT` 环境变量调整，
数据目录可通过 `OSWORLD_SFT_DIR` 调整，默认写入仓库的 `human_runs/SESSION_ID/`。
默认启动独立 Chromium，也可连接已开启远程调试的浏览器；会话使用独立 BrowserContext 和下载目录。
浏览器运行在服务所在的机器上；无桌面环境时使用后台浏览器，通过 Studio 中的画面操作。

1. 填写任务和可选起始网址；需要确定完成条件时，填写可信的 `completion_spec`。
2. 在采集画面中点击、输入、滚动或拖拽，使用工具栏处理导航、按键、标签页和 JavaScript 对话框。
3. 在操作时间线中查看前后截图，逐步选择保留或丢弃；必要时补充目标、意图和预期效果。
4. 结束采集后导出，优先选择「图片 + ZIP」，其中包含 `sft.jsonl` 和 `assets/`。

步骤边界按一次完整的操作确定：

| 人工操作 | 记录规则 |
| --- | --- |
| 单击 / 右击 | 一次点击记一步；单击等待 250 ms，以区分双击 |
| 双击 | 合并为一个 `double_click`，不附加两个单击 |
| 中文输入 / 粘贴 | 在输入区完成组字后统一提交，一整段记一个 `type` |
| Enter / 快捷键 | 每次完整按键或组合键记一步，与输入文字分开 |
| 滚轮 | 同位置、同方向的连续滚动在 300 ms 内合并，改变方向另起一步 |
| 拖拽 | 从按下到松开记一个 `drag_and_drop`，移动过程不拆步 |
| 刷新观察 / 鼠标移动 / 输入法组字 | 不计操作步骤 |

例如“点击搜索框 → 提交完整标题 → 按 Enter”是三步。起始页的初始化不计步，之后通过地址栏
导航会记录为 `goto`。每一步完成并返回新画面后才允许下一步；重复请求不会重复执行。
图像失效时先刷新观察，不把旧图的坐标应用到新页面。

采集仅记录通过面板提交的操作；在外部浏览器窗口或桌面直接操作不会自动成为示范。
登录时应暂停采集，在外部完成登录后继续；密码输入默认被拒绝，显式标记敏感的记录会脱敏并排除训练。
保留某一步就是人工审阅批准，无须额外的安全确认；但不能覆盖失败、敏感内容或缺失证据等排除规则。

每次动作前冻结任务、当前截图、AX 和先前操作，执行后保存真实结果与后图。
审阅时补充的目标、意图或预期效果不会倒灌进动作前输入；后图只用于审阅，不作为该步训练输入。
纯坐标点击必须有 DOM 语义标签或人工补充的目标描述才能成为决策正样本；未出现在实际 prompt
中的 DOM 引用会被移除。只有一般意图、没有目标描述的坐标点击仍不能导出。
嵌套滚动、滑块定位或依赖已有文字光标位置的局部编辑等无法由当前语义动作等价表达的操作，控制器会标记排除原因；
原始操作仍归档，人工保留不能将其转为正样本。
文本示范优先使用空输入框，或勾选替换原内容后提交完整文本。
缺失或过期的动作前截图可以保留为恢复事件，但不能导出为视觉决策正样本。

点击结束或人工声称完成不代表任务成功。`CompletionVerifier` 独立执行系统验收：
验收通过时标记 `independent_verified=true`；验收明确失败时不导出正样本；验收未知时，
已人工保留、执行确认有效的步骤仍可作为人工 SFT 示范导出，但保持
`independent_verified=false, reward=null, reward_valid=false`。
未审阅、已丢弃、执行失败或不确定的步骤均排除。Python API 的
`export_sft(require_verified=True)` 可进一步要求整个任务通过系统验收。

核心接口为 `HumanDemonstrationSession`：控制器先调用 `prepare_observation(before)`，
执行后将同一份准备标识和冻结输入交给 `record_transition(...)`；随后使用
`review_step(...)`、`stop(...)`、`export_sft(...)` 和 `close()`。
`events.jsonl` 持续落盘，截图按摘要去重；结束时归档 `trajectory.json`，私有验收只进入归档。
UI 仅获取公开验收摘要和同会话图片，避免把私有目标加入训练输入。

人工导出只监督决策动作，采用现有 `next_action` schema，不合成模型思维链或规划样本，
也不生成 grounding SFT。数据明确标记 `actor/source=human` 和外部行为策略，on-policy RL 导出拒绝接收。
图片路径相对于 JSONL 所在目录；单独下载 JSONL 时需同时保留对应 `assets/`，ZIP 已打包二者。
训练侧仍需使用目标模型的 processor/chat template 读取图片并构造 completion-only loss，
这些导出文件不是已集成的训练器。
