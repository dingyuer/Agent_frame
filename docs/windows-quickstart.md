# Windows 快速开始

使用 Windows 10/11、Python 3.12 和 Google Chrome。在 PowerShell 中进入你准备放项目的父目录。仓库目录需要命名为 `osworld_agent`（Python 包名），虚拟环境放在它旁边。

## 首次安装

```powershell
git clone https://github.com/dingyuer/Agent_frame.git osworld_agent
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r osworld_agent\requirements-windows.txt
.\.venv\Scripts\python.exe -m playwright install chromium
if (-not (Test-Path osworld_agent\.env)) { Copy-Item osworld_agent\.env.example osworld_agent\.env }
notepad osworld_agent\.env
```

在 `.env` 填写 `DASHSCOPE_API_KEY=你的密钥`，保存后启动：

```powershell
powershell -ExecutionPolicy Bypass -File osworld_agent\start_studio_windows.ps1
```

浏览器打开 **http://127.0.0.1:8090**。在运行任务页重新选择 Windows 上的工程目录，并配置决策、定位、验证模型。可以继续使用 Qwen 决策/定位及远程 Jev 9B；Jev 服务地址需要能从该 Windows 电脑访问。认证可使用 `env:DASHSCOPE_API_KEY`，不要将密钥写入任务描述。

## 更新已有安装

先关闭旧 Studio 与正在运行的任务。在项目父目录执行：

```powershell
git -C osworld_agent status --short
git -C osworld_agent pull --ff-only
.\.venv\Scripts\python.exe -m pip install -r osworld_agent\requirements-windows.txt
.\.venv\Scripts\python.exe -m playwright install chromium
powershell -ExecutionPolicy Bypass -File osworld_agent\start_studio_windows.ps1
```

如有本地代码修改，先备份或提交后再拉取，勿强制覆盖。已有 `.env` 和本机模型设置保留；它们不在仓库中。若原目录名为 `Agent_frame`，关闭程序后将其改名为 `osworld_agent`，并确保 `.venv` 在父目录。更新后浏览器按 **Ctrl+F5** 刷新前端资源。

## 这次更新

- GUI / DOM / CLI 执行与模型调试、Markdown 知识库筛选。
- 监控与回看共用三栏视图；单行进度、实际执行模式、分版本动作记录与完整 Markdown 结果展示。
- 下载回执、文件读回证据、等待及规划恢复逻辑改进。
- Markdown 解析器随仓库提供，前端运行无需加载外部 CDN。

## 测试范围与已知限制

已在 Linux 上运行相关自动化回归，并在远程 Mac 做实际浏览器和文件操作测试；本次发布尚未在真实 Windows 桌面完整验证。

整体模型验收通过不代表每个历史子任务都记录了独立完成核查；前端不会补造这些记录。规划器仍可能生成额外的文件正文或指定编辑器条件，复杂网站检索也可能绕路。文件 `open` 操作成功回执证明请求已派发，不单独证明应用窗口可见。
