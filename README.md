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
仍失败则返回 `planning_failed`，不开始操作。完整计划保存在框架执行队列中；每轮决策仅发送原始任务、
当前子目标与本步骤预期、一条简短系统反馈、近期行动摘要、截图及按预算选择的 AX 节点。
相邻步骤、完整队列、验收契约和核查证据保留在框架/核查员/前端/轨迹中，不再重复发给操作模型。
仅在任务配置中给出的当前字段目标值/目标网址会转为简短操作参数，防止精简时丢失实际要填写的内容。
规划按可独立核查的目标拆分，不要求固定数量；一个子目标允许多个动作。比如“输入框聚焦、输入关键词、
回车”合并为“搜索并显示结果”。解析器最多接受 32 个子目标以限制异常输出，该上限不是建议数量。
原始对象不能被改成相关主题，例如搜索短名论文时必须先核实身份，不能选任意相关论文代替。

决策输出简化为 `{"next_action":{"type":"click","target":"Search button"},"summary":"已看到搜索按钮，点击以显示结果。"}`。
`summary` 是一句话观察与行动计划；最近最多 8 条按时间组成记忆，超预算优先保留最新完整句子。
框架附上“已派发/被拒绝/结果不确定”等状态，避免把计划当成已完成事实；完整历史仍保存在轨迹中。
兼容旧输出字段，但不再要求 `state_update`，无效辅助记忆补丁不阻止合法动作；模型不能改写队列或完成状态。

格式纠错表示本次候选动作尚未执行，不是上一步操作失败。轨迹记录 `validation_error` 和错误类别，
前端显示具体校验原因；每次重试只附一对有界纠错消息，不累计长篇错误回复。动作参数仍严格校验。

截图按原始尺寸编码发送，客户端不缩放；浏览器使用 CSS 视口截图，桌面使用实际屏幕截图。
有 AX 结构时默认给操作模型 1 张当前图，纯视觉时最多 3 张；核查员独立取图。
每步 `prompt_metadata.image_sizes` 记录实际发送尺寸，`image_resize=none_client_side` 标明客户端处理。
`image_tokens=2048` 只是上下文预算估计，不是图片尺寸/真实视觉 token 数；服务端 processor 仍可能缩放，需查远程配置。

### 基础动作、执行队列与逐步核查

操作模型只能调用基础 GUI/DOM 动作（点击、输入、选择、拖拽、按键、滚动、导航等），以及
观察表面切换、标签页和对话框处理。`download`、`rename_artifact`、终端、代码代理和 UNO
文件写入均已从策略动作空间移除，执行入口也会拒绝；下载、另存、改名通过界面完成。
`completion_spec` 中的 `download` 是只读验收条件，不是操作动作。

规划步骤格式示例：

```json
{"steps":[{"goal":"打开 arXiv 首页","expected_evidence":"当前页面位于 arxiv.org",
  "verification":[{"kind":"url_host","host":"arxiv.org"}],"actions":["goto"],"check_ids":[]},
 {"goal":"搜索用户指定论文并核对结果","expected_evidence":"结果中的论文标题、作者与任务匹配",
  "verification":[{"kind":"visual"}],"actions":["click","type","press"],"check_ids":[]}],
 "keywords":["用户指定的论文名"],"final_check":"原始任务的全部要求得到满足"}
```

每个动作结束、异步操作等待完成后，先用代码核对当前步骤。支持 `url_equals`、`url_host`、
`text_present`、`element_state`、`app_open` 和已有可信条件的 `check_ids`；应用检查使用操作系统
窗口/进程证据，目前 Windows 提供该元数据，其他环境证据缺失时转入视觉核查。
对不能由代码充分核实的目标，独立核查员接收动作后的截图、AX、系统证据和原始任务，输出
`passed/failed/unknown`、理由和证据。明确失败的代码条件不能由模型覆盖；失败或未知时不推进队列。
一个规划步骤可以需要多次基础动作，只有核查通过才移到下一个步骤。队列全部核查通过后，还要
检查整项原始任务，防止初始规划遗漏要求。

规划谓词兼容单参数 `value` 别名：`url_host → host`、`url_equals → url`、
`text_present → text`、`app_open → app`。原始模型回复保留，执行队列使用规范字段。
别名仍经过类型、长度与原有语义校验；同时给出规范字段和 `value` 时必须一致。
`element_state.value` 有歧义，不会猜成元素名或自动降级为视觉检查；错误反馈会指出具体子目标。

前端实时监控显示执行队列、代码核查和独立模型核查的进行中/通过/未通过/证据不足状态，历史
步骤保留两种核查的理由与证据。核查和自动等待不增加操作步数。

默认核查员复用决策模型的端点配置，但使用独立客户端、独立提示和独立调用记录（温度 0、
最多输出 768 token、单次超时不超过 45 秒及剩余任务预算）。可在 Python 中传入
`Agent(..., verifier_model=review_model)`，本机 CLI 可用 `--verifier-model` 和 `--verifier-url`。
核查调用保存在 `review_calls` 和事件日志中，标记 `actor_role=verifier, policy_trainable=false`，
不会混入操作/规划的 SFT 或策略 RL 样本。轨迹新增 `step_verification`、`step_reviews`、`execution_queue`。
最终语义核查同时接收规划器的 `final_check` 提示和原始未解决要求；提示不是证据，不能覆盖原始任务。
只有确定性条件全部通过后，剩余语义要求才允许模型保底；模型验收结果单独标记，不冒充确定性 RL 奖励。


持续卡住时，框架把实际失败和验收反馈交给规划器修订计划；默认最多重规划一次，
可通过 `Agent(max_replans=...)` 或配置的 `agent.max_replans` 调整，重规划不会重置环境。

任务意图与验收编译独立：`搜索论文roboclaw并下载，然后改名为data` 等常见语序会编译为
论文查询与 `data.pdf` 文件名条件；未知句式、保存目录、收件人等不能核实的要求保留原文。
即使合同没有论文检查项，识别到论文下载意图仍会给规划器相应流程，不能因此丢失下载/验收阶段。
规划器可在同一次规划调用中输出 `requirements`（`kind=paper_download/rename/other`、
`source_text`、`value`）；框架检查原文引用与字段，记录为未受信候选，不把模型提出的论文 ID、
URL 或“完成标准”写入可信合同。引用存在不证明语义正确或需求覆盖完整，最终仍核对原始任务。
前端显示任务类型、代码/语义验收方式和未解析要求，事件日志与 manifest 同步保存任务分析。

除原有动作重复检测外，框架按当前子任务累计无进展决策，默认连续 4 次触发恢复。
按键、换目标或画面闪动都不会单独清零；真实输入/选择值变化、新通过的代码检查项和完成下载
可以证明阶段性进展，但不等于任务完成。新页面/滚动发现结构内容最多获得 2 次探索宽限，
避免跳转循环无限续期。真实 pending 操作暂停计数，模型没有待处理操作却反复输出 wait 仍计数。
桌面纯视觉场景缺少结构证据时，该信号可能更保守；可调整
`Agent(..., progress_config={"no_progress_limit": 6, "exploration_allowance": 2})`，或 YAML
`agent.progress_config`。它用于触发重规划，不作为成功/失败训练标签；前端和轨迹保存
`step_progress`，包括当前步骤、连续无进展次数、证据和最近尝试。

### Decision 3.0 远程评估器（可选）

支持 [Decision 3.0 的 System One 接口](https://huggingface.co/vllm-sr/d3)，可选择
`vllm-sr/d3`（27B）或 `vllm-sr/d3-flash`（9B）。该接口是 `/v1/systemone`，不是
`/chat/completions`；部署时使用模型自带服务或官方支持的 Engine。框架只发送 HTTP 请求，
不在本机加载模型。前端“连接模型 → Decision 3.0 评估器”选择型号并填写远程服务地址即可启用，
默认关闭。两种型号共用接口，可以对同一组任务分别运行以比较误判率、耗时与任务成功率。

```python
agent = Agent(decision_model=model, grounding=grounding, env=env,
    decision_judge={"url": "http://JUDGE_HOST:8899/v1", "model": "vllm-sr/d3-flash",
                    "revision": "YOUR_DEPLOYED_REVISION", "threshold": 0.9,
                    "margin": 0.15, "timeout": 10})
```

本机/前端 VM runner 支持 `--judge-url`、`--judge-model`、`--judge-revision`、
`--judge-threshold`、`--judge-margin`、`--judge-timeout`；YAML 对应 `agent.decision_judge`。
需要认证时设置服务端环境变量 `DECISION_JUDGE_API_KEY`；凭据不进入 manifest 或模型调用记录。

- 规划前分类网页、桌面、混合或未知任务；分类作为提示，实际 DOM 可用性、截图来源与坐标空间仍由环境决定。
- 每次动作先做代码核查；需要视觉审查时，Decision 接收动作前后截图、当前步骤、预期效果和精简的系统证据，
  分别输出子任务完成、动作效果、当前操作表面的概率。动作有效不等于子任务完成。
- 代码明确失败或真实操作仍 pending 时不交给 Decision 覆盖。低分、选项分数接近、超时、缺图、非法返回均保留 unknown，
  回退原独立视觉核查。模型选择 pending 不会凭空启动框架等待；等待需要真实系统事件。
- 整项任务的最终语义核查仍检查原始任务和硬性验收条件，局部 Decision 判定不能直接给出任务奖励。
- 前端显示分类、各选项概率、耗时和版本。请求/结果记录为 `actor_role=verifier, policy_trainable=false`，
  与操作策略的 SFT/RL 样本分离。服务未返回权重版本时，用 `revision` 记录部署版本。

默认阈值是可调整的初始值，未经过本任务分布校准；模型概率不能解释为真实成功率。
本仓库测试验证协议、回退和验收边界，不替代远程 27B/9B 的实际准确率及延迟评测。

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
绑定后模型通过页面链接或查看器下载按钮保存 PDF，通过另存为/文件管理器界面满足文件名要求。
自动身份解析目前支持 arXiv；其他来源仍可用显式完整标题合同和界面下载。
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
                           "download_start_timeout_s": 2, "poll_interval_s": 0.25, "render_timeout_s": 20})
```

PDF 页面另有默认 20 秒的渲染等待：框架轮询正文区域，连续非空白且稳定后再交给模型，等待不增加决策步数。
超时会明确提示模型检查；同一文档仍为空白时不会每轮重新等待。此检测是就绪启发式，不代表任务完成。
前端分别展示 PDF 渲染等待和下载等待，轨迹保留等待结果。

已绑定的论文标题、arXiv ID、官方 PDF 地址，以及下载完成的路径/字节数会进入精简决策上下文；
传输完成与论文身份、文件名验收分开。明显指向其他 arXiv 论文的链接会在派发前拒绝。
框架将论文计划的 download 阶段编译为 `download_complete(check_id)`：登记的非空文件完整落盘且
内容未变化后，先推进到独立的 `download_valid(check_id)` 核查步骤；不依赖视觉核查，也不等最终验收。
即使下错文件，传输步骤也会完成，但内容/论文身份核查不会通过；当前子目标变为核对和纠正文件，
不再留在下载阶段空点。已有正确文件时，这项代码核查自动通过，不额外调用决策模型。
有文件名要求时再进入独立的界面改名步骤；最终合同仍检查全部要求。
历史错文件不会覆盖后来正确文件的验收状态。同一 URL 对应文件仍完整存在时，明确的重复下载动作
在派发前拒绝；无法从动作确定 URL 的点击由浏览器下载事件兜底取消，避免生成重复副本。
改名后仍按同一产物去重；文件删除或修改后允许重新下载。不将不同签名 URL 擅自视为同一资源。
DOM 可用但缺引用时先进行一次候选动作纠正；多个元素匹配同一描述时要求明确引用，不直接猜选坐标。
PDF 工具栏等无 AX 元素的界面仍可通过桌面观察和视觉定位操作，保存和改名仍由界面完成。
PDF 文档不触发“缺少 DOM 引用”的候选纠错；已有有效引用仍照常走 DOM，不以 AX 节点数量猜控件可用性。

Windows 的 `switch_applications` 按应用/进程识别已打开窗口并核对实际前台窗口，不再用盲按 Alt+Tab 代替。
资源管理器不存在时会提示 `Win+E`；多窗口有歧义或激活未确认时明确返回失败/不确定。
VMware 当前没有精确窗口激活实现，会提示改用桌面任务栏或系统快捷键。
文件名验收不符时，决策输入和前端会显示已登记文件的真实目录、原文件名及 `F2` 界面改名指引，
提醒处理隐藏扩展名；同一文件改名后的追踪仍由框架只读核验，不恢复 `rename_artifact`。

本机前端可设置下载等待上限和任务时限，等待状态显示“不消耗步数”，保留中断入口。
命令行对应 `--operation-timeout 300 --task-timeout 1800 --download-max-mib 50`，YAML 对应 `agent.wait_config`。
单文件默认资源上限为 50 MiB，可在前端/命令行调整，0 表示不限；Python/YAML 使用 `download_max_bytes`。
Chromium 响应长度/下载进度超过上限时取消传输，保存完成后再次检查实际大小；超限不代表文件身份错误。
时间预算独立于决策预算；正在进行的外部模型请求仍受模型传输超时约束。
框架自动等待只记录事件；模型主动输出一次 `wait` 仍是一次真实采样，不能从训练记录删除。

### 默认操作目录与桌面观察

Windows 任务在规划前建立默认操作目录，`begin_episode` 沿用同一目录，整轮不会改变。
默认根目录是项目下 `workspace/`；可在前端配置，或传入 `--workspace-dir PATH`
（兼容 `--download-dir`），也可设置 `OSWORLD_WORKSPACE_DIR`。每次运行创建独立子目录，
以免把上次运行的文件当成新成果；实际绝对路径写入 `initial_state.workspace_dir`、manifest、
规划输入、操作反馈和前端。下载文件落在该目录，模型找文件、改名、另存默认使用该路径；
原始任务明确指定其他文件或路径时才改用指定位置。这是操作默认值，不是文件系统沙箱。
远程 VM 的控制器产物目录会标注为 `controller_host`，不能假设其在 VM 文件管理器里可见。

后台浏览器继续使用 CDP 视口截图和虚拟输入，不依赖前台焦点，也不会在截图失败时自动截桌面。
显式 `observe_desktop` 使用 Pillow 的 Windows `all_screens` 截图及 Win32 虚拟屏幕度量，
保留负原点、实际尺寸与截图 ID。定位器返回图内坐标，执行器验证布局后转换为物理桌面坐标；
显示器布局发生变化或坐标越界时拒绝执行，要求重新观察。截图与输入均使用线程级 DPI 上下文。
中文路径与文件名通过 Windows Unicode 键盘输入完成，不借助剪贴板或后台改文件。

前端明确显示“后台浏览器”或“桌面接管”。桌面接管使用真实输入设备，不能承诺用户同时操作
桌面仍互不干扰；需要全程隔离桌面操作时，应使用独立 VM/桌面会话。`observe_browser` 返回后台网页模式。
实现依据：[Pillow 全屏截图](https://pillow.readthedocs.io/en/stable/reference/ImageGrab.html)、
[Windows 线程 DPI 上下文](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setthreaddpiawarenesscontext)。

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
3. 没有完整确定性合同，或存在明确的语义要求时：独立核查员检查原始任务的所有要求。
   它不能覆盖任何失败的硬性条件或私有评测条件。证据不足、调用失败或输出非法时保留 `unknown`。

只有独立模型核查通过的任务使用 `termination_reason=verified_by_model`，并明确记录
`source=independent_model_review`；任务可以显示完成，但 `evaluation_available=false`、
`reward=null, reward_valid=false`，不会冒充确定性训练奖励。无核查模型也无可信证据时仍返回
`completion_unverifiable`。兼容字段 `score=0` 不能独自作为失败标签。

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

下载事件被动登记浏览器实际保存的文件。通过文件管理器界面改名后，框架只读扫描本任务下载目录，
在文件身份、大小、内容摘要都一致且无歧义时登记改名链；不自行移动或修改文件。
跨目录移动、复制替换或内容变化不会被自动认作原文件。内容、论文身份与最终文件名须由同一份文件满足。

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
当前策略不提供代码子 Agent 动作；历史上包含 `call_code_agent` 的 episode 仍被 RL 导出拒绝。
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
