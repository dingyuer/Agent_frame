"""Windows 本机环境：pyautogui 控制桌面 + CDP 控制浏览器。

用于在 Windows 本机（非 VM）上跑 GUI agent 任务，重点支持 browser use：
- 观察：pyautogui 截图 + browser_ax 快照（焦点在浏览器时经 BrowserSession 抓取）；
- 执行：桌面 GUI 动作（click/type/hotkey 等）走 pyautogui（坐标定位）；
        浏览器 DOM 动作（带 target_ref / 语义兜底命中）走 CDP（DOM 定位，跳过视觉定位）；
- 焦点检测走 Windows GetForegroundWindow。

与 VMwareEnvironment 的对齐点：browser_ax 快照进 context、target_ref 路由 + 语义兜底、
execute_routed 走 CDP。差异点：截图用 pyautogui、CDP host 是 localhost、桌面无 AT-SPI。
"""
from __future__ import annotations

import re
import subprocess
import sys
from typing import Any, Dict, Optional

from ..env import Environment, Observation, StepResult
from ..actions import Action, get_action_spec, BASIC_ACTIONS
from .vmware_env import to_pyautogui


class WindowsEnvironment(Environment):
    """Windows 本机环境（GUI + browser use）。"""

    evaluation_available = False

    def __init__(self, browser_port: int = 9222, width: int = 1920, height: int = 1080,
                 downloads_dir: Optional[str] = None):
        self.browser_port = browser_port
        self.width = width
        self.height = height
        self.os_name = "windows"
        self.browser_only = True  # 浏览器专用：用精简 system prompt + 强化 DOM 定位
        self.downloads_dir = downloads_dir
        self._browser_session = None
        self._last_observation_info = {}
        self._observation_surface = "browser"
        self._episode_prepared = False
        self.initial_state = {}

    def begin_episode(self, task_id=None, isolated=True):
        session = self._get_browser_session()
        self.initial_state = session.begin_episode(task_id=task_id, isolated=isolated,
            viewport={"width": self.width, "height": self.height})
        if not self.initial_state.get("isolated"):
            raise RuntimeError("Browser episode isolation was not established")
        self._episode_prepared = True
        self._observation_surface = "browser"
        self._last_observation_info = {}
        return dict(self.initial_state)

    # ---- 观察 ----
    def _take_screenshot(self):
        """Desktop fallback; browser screenshots are collected atomically with AX."""
        import pyautogui
        return pyautogui.screenshot()

    def _focus_is_browser(self) -> bool:
        """本机浏览器模式：以「CDP 有可用页面」为准，不依赖前台焦点。

        用户可同时用电脑（切到别的窗口），智能体照常通过 CDP 虚拟控制浏览器。
        """
        from ..utils import list_cdp_pages
        try:
            pages = list_cdp_pages("localhost", self.browser_port)
            return any(p.get("url") and not p.get("url", "").startswith(
                ("devtools://", "chrome-extension://")) for p in pages)
        except Exception:  # noqa: BLE001
            return False

    def _get_browser_session(self):
        """懒加载 BrowserSession（保持快照，跨步复用）。连不上返回 None。

        require_focus=False：本机模式用 CDP 虚拟输入，不要求浏览器前台聚焦——
        这样智能体操作浏览器时，你可以照常用自己的鼠标键盘。
        """
        if self._browser_session is None:
            from ..browser_session import BrowserSession, BrowserOptions
            self._browser_session = BrowserSession(
                options=BrowserOptions(endpoint=f"http://localhost:{self.browser_port}",
                                       require_focus=False),
                foreground_check=None,
                downloads_dir=self.downloads_dir,
            )
        return self._browser_session

    def _collect_browser_snapshot(self):
        """抓当前浏览器页面的结构化 AX/DOM 快照（browser_ax）。非浏览器场景返回 None。"""
        if not self._focus_is_browser():
            return None
        try:
            session = self._get_browser_session()
            if session is None:
                return None
            return session.collect()
        except Exception:  # noqa: BLE001
            return None

    def _build_observation(self) -> Observation:
        """Collect screenshot and AX on one page, with an explicit coordinate space.

        浏览器模式（后台独立浏览器）只取 CDP 截图，绝不回退到 pyautogui 截取用户整屏；
        pyautogui 整屏截图仅用于显式 observe_desktop。这样即使用户把浏览器窗口挡在后面，
        截图仍是该浏览器页面的完整内容，不受遮挡/前台影响。
        """
        from ..model.grounding import screenshot_size
        from ..scene import Scene

        screenshot, snapshot, metadata = None, None, {}
        error = ""
        if self._observation_surface == "browser":
            # 页面仍在导航/渲染时截图会瞬时失败，重试几次再放弃。
            import time as _time
            for _ in range(4):
                try:
                    session = self._get_browser_session()
                    if session is not None:
                        screenshot, snapshot, metadata = session.collect_observation()
                        metadata = dict(metadata or {})
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)
                if screenshot is not None and metadata.get("screenshot_space") == "css_viewport":
                    break
                _time.sleep(0.25)
        else:
            try:
                session = self._get_browser_session()
                if session is not None and hasattr(session, "event_metadata"):
                    metadata = session.event_metadata()
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

        if screenshot is not None and metadata.get("screenshot_space") == "css_viewport":
            info = metadata
            info["scene"] = Scene(
                browser_use=1, structure_available=bool(snapshot),
                mode="browser_dom" if snapshot else "visual",
                reason="structured_page" if snapshot else "no_structured_page",
                url=(snapshot or {}).get("url", metadata.get("url", "")),
                page_id=metadata.get("page_id", ""),
                snapshot_id=metadata.get("snapshot_id", ""),
            ).to_dict()
            context = [snapshot] if snapshot else []
        elif self._observation_surface == "browser":
            # 浏览器模式：CDP 截图失败 → 返回错误观测（无截图），不截取用户桌面。
            info = {"screenshot_space": "css_viewport",
                    "scene": Scene(browser_use=1, structure_available=bool(snapshot),
                                   mode="browser_dom" if snapshot else "visual",
                                   reason="browser_observation_failed").to_dict(),
                    "observation_error": error or str(metadata.get("error") or "浏览器截图/快照采集失败")}
            for key in ("artifacts", "download_events", "pending_downloads", "dialogs", "dialog",
                        "initial_state", "tabs", "artifact_events"):
                if key in metadata:
                    info[key] = metadata[key]
            context = [snapshot] if snapshot else []
        else:
            # 桌面模式：显式 observe_desktop 才用 pyautogui 截取整屏。
            screenshot = self._take_screenshot()
            size = screenshot_size(screenshot)
            info = {"screenshot_space": "screen_pixels",
                    "screenshot_size": list(size) if size else None,
                    "scene": Scene(browser_use=0, structure_available=False,
                                   mode="visual", reason="desktop_fallback").to_dict()}
            for key in ("artifacts", "download_events", "pending_downloads", "dialogs", "dialog",
                        "initial_state", "tabs", "artifact_events"):
                if key in metadata:
                    info[key] = metadata[key]
            if error or metadata.get("error") or metadata.get("observation_errors"):
                info["observation_error"] = error or metadata.get("error") or "; ".join(metadata["observation_errors"])
            context = []
        self._last_observation_info = dict(info)
        info["active_surface"] = "browser" if info.get("screenshot_space") == "css_viewport" else "desktop"
        info["requested_surface"] = self._observation_surface
        from ..desktop_state import windows_runtime_state
        info["runtime_state"] = windows_runtime_state()
        return Observation(screenshot=screenshot, text="", context=context, info=info)

    def _derive_scene(self, snapshot):
        """算显式场景：有快照→browser_dom；浏览器但无结构→visual；非浏览器→visual/desktop。"""
        from ..scene import Scene
        if snapshot:
            return Scene(browser_use=1, structure_available=True, mode="browser_dom",
                         reason="structured_page", url=snapshot.get("url", ""),
                         snapshot_id=snapshot.get("snapshot_id", "")).to_dict()
        if self._focus_is_browser():
            return Scene(browser_use=1, structure_available=False, mode="visual",
                         reason="no_structured_page").to_dict()
        return Scene(browser_use=0, structure_available=False, mode="visual",
                     reason="not_browser").to_dict()

    def _fresh_obs_after_action(self) -> Observation:
        """动作执行后重新观察：新截图 + 新 browser 快照。"""
        return self.observe()

    # ---- 路由 + 执行（browser_dom）----
    # 浏览器场景里全部走 CDP 的动作（虚拟鼠标键盘，不碰真实外设）
    _BROWSER_CDP_ACTIONS = ("goto", "download", "click", "double_click", "right_click",
                            "type", "select", "scroll", "press", "hotkey")

    def route_action(self, action, obs):
        """浏览器场景的动作 → browser_dom（CDP 虚拟输入）；结构解析失败 → None（视觉定位）。

        流程：DOM（element_id / target_ref / 语义匹配）→ 命中走 CDP；都没命中返回 None，
        交给 pipeline 用 grounding 模型定坐标，再由 step() 用 CDP 虚拟坐标点击（仍不动真实鼠标）。
        """
        if obs is None or action.action not in self._BROWSER_CDP_ACTIONS:
            return None
        # Route against the actual observation, not a fresh CDP page guess.
        from ..scene import observation_scene
        if not observation_scene(obs).get("browser_use"):
            return None
        session = self._get_browser_session()
        if session is None:
            return None
        # 地址栏输入 URL → 转 goto（地址栏是浏览器 chrome UI，不在 DOM）
        url = self._address_bar_url(action)
        if url:
            return {"channel": "browser_dom", "goto_url": url, "coordinate_space": "css_viewport"}
        # A scroll with an explicit node refers to that container, not the page.
        if action.action == "scroll" and (action.args.get("target_ref") or action.args.get("element_id")):
            return session.route(action, obs)
        # goto / scroll / press / hotkey：无 target，直接 browser_dom
        if action.action in ("goto", "download", "scroll", "press", "hotkey"):
            return {"channel": "browser_dom", "target": action.args.get("target"),
                    "coordinate_space": "css_viewport", **self._route_identity(obs)}
        # click / type / select / double / right：优先 DOM 定位
        try:
            route = session.route(action, obs)  # 处理 target_ref / element_id
        except Exception:  # noqa: BLE001
            route = None
        if route is not None:
            return route  # browser_dom 或 reobserve
        # 语义兜底：target 描述 → ref（仍在结构层）
        target = str(action.args.get("target") or "")
        if target:
            exposed = obs.info.get("exposed_refs") if isinstance(obs.info, dict) else None
            if exposed is not None and not exposed:
                return None
            allowed = set(exposed) if exposed else None
            try:
                ref = session.resolve_target(target, allowed=allowed)
            except Exception:  # noqa: BLE001
                ref = None
            if ref:
                action.args["target_ref"] = ref
                return session.route(action, obs)
        # 结构解析失败：返回 None → 视觉定位（grounding 定坐标 → CDP 虚拟点击）
        return None

    @staticmethod
    def _route_identity(obs):
        return {key: obs.info[key] for key in ("page_id", "document_id", "snapshot_id")
                if key in obs.info}

    @staticmethod
    def _address_bar_url(action):
        """模型往地址栏 type 网址（常见误操作）→ 提取 URL 用于 goto；否则返回 None。

        地址栏是浏览器 chrome UI、不在页面 DOM 里，type/click 定位不到；识别后直接转 goto。
        """
        if action.action != "type":
            return None
        text = str(action.args.get("text", "")).strip()
        target = str(action.args.get("target", "")).lower()
        if not re.search(r"^(https?://)?[\w-]+(\.[\w-]+)+(/\S*)?$", text):
            return None
        if any(k in target for k in ("address bar", "omnibox", "url bar", "地址栏", "导航栏", "网址栏")):
            return text
        return None

    def execute_routed(self, action, route):
        """执行已路由的 browser_dom 动作：goto / DOM(ref) / scroll/press/hotkey 全部走 CDP。"""
        if action.action in {"download", "rename_artifact", "run_in_terminal", "call_code_agent", "set_cell_values", "save"}:
            return StepResult(self.observe(), info={"status": "rejected", "dispatched": False,
                "error_code": "non_basic_action", "action": action.action})
        session = self._get_browser_session()
        if session is None:
            return StepResult(observation=self.observe(), info={
                "status": "rejected", "dispatched": False, "channel": "browser_dom",
                "error": "browser session unavailable"})
        if action.action == "goto" or route.get("goto_url"):
            result = session.goto(str(action.args.get("url") or route.get("goto_url")))
        elif route.get("target_ref"):
            result = session.execute(action, route)
        elif action.action in ("scroll", "press", "hotkey"):
            result = session.execute_viewport_action(
                action, metadata=action.args.get("observation_metadata") or self._last_observation_info)
        else:
            result = {"status": "rejected", "dispatched": False, "channel": "browser_dom",
                      "error": "no routable target"}
        obs = self._fresh_obs_after_action()
        info = dict(result)
        info["artifacts"] = obs.info.get("artifacts", info.get("artifacts", []))
        return StepResult(observation=obs, info=info)

    # ---- Environment 接口 ----
    def reset(self, task: str) -> Observation:
        if not self._episode_prepared:
            self.begin_episode()
        self._episode_prepared = False
        return self.observe()

    def observe(self) -> Observation:
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        name = action.action
        if name not in BASIC_ACTIONS:
            return StepResult(self.observe(), info={"status": "rejected", "dispatched": False,
                "error_code": "non_basic_action", "action": name})
        spec = get_action_spec(name)
        kind = spec.kind if spec else "env"
        if name in {"observe_desktop", "observe_browser"}:
            self._observation_surface = "desktop" if name == "observe_desktop" else "browser"
            obs = self.observe()
            return StepResult(observation=obs, info={"status": "executed", "dispatched": False,
                "action": name, "artifacts": obs.info.get("artifacts", [])})
        if name in {"handle_dialog", "list_tabs", "switch_tab", "close_tab"}:
            session = self._get_browser_session()
            kwargs = {k: v for k, v in action.args.items() if k in {
                "accept", "prompt_text", "dialog_id", "tab_id", "artifact_id", "filename", "url"}}
            result = getattr(session, name)(**kwargs)
            obs = self.observe()
            result["artifacts"] = obs.info.get("artifacts", [])
            return StepResult(observation=obs, info=result)

        coordinate_space = action.args.get("coordinate_space")
        coordinate_actions = ("click", "double_click", "right_click", "type", "select", "drag_and_drop")
        if action.args.get("grounding_failed"):
            return StepResult(observation=self.observe(), info={
                "status": "rejected", "dispatched": False, "error": "grounding failed"})

        if coordinate_space == "css_viewport" and kind == "env" and name != "wait":
            session = self._get_browser_session()
            if session is None:
                result = {"status": "rejected", "dispatched": False,
                          "channel": "browser_cdp", "error": "browser session unavailable"}
            else:
                result = session.execute_viewport_action(
                    action, metadata=action.args.get("observation_metadata") or self._last_observation_info)
            obs = self.observe()
            result = dict(result)
            result["artifacts"] = obs.info.get("artifacts", [])
            return StepResult(observation=obs, info=result)

        if name in coordinate_actions and coordinate_space != "screen_pixels":
            return StepResult(observation=self.observe(), info={
                "status": "rejected", "dispatched": False,
                "error": "coordinate action requires css_viewport or screen_pixels"})

        if kind == "browser":
            result = self.browser_action(name, action.args)
            return StepResult(observation=self._build_observation(), info={"browser_result": result})

        if kind == "shell":
            output = self.run_command(str(action.args.get("command", "")))
            return StepResult(observation=self._build_observation(), info={"terminal_result": output})

        # 浏览器表面使用 CDP；显式切到桌面并重新观察后才允许桌面输入。
        if self.browser_only and self._observation_surface != "desktop" and name != "wait":
            return StepResult(observation=self.observe(), info={
                "status": "rejected", "dispatched": False, "channel": "desktop",
                "error": "请先 observe_desktop 获取桌面截图后再执行 GUI 操作；网页操作使用 DOM 或 observe_browser"})
        # 坐标化后执行 pyautogui 命令
        cmd = to_pyautogui(action)
        if cmd:
            try:
                exec(cmd, {"__name__": "__main__"})  # noqa: S102  执行 pyautogui 命令字符串
            except Exception as exc:  # noqa: BLE001
                return StepResult(observation=self._build_observation(),
                                  info={"status": "failed", "dispatched": True,
                                        "channel": "desktop", "error": f"执行异常: {exc}"})
            return StepResult(observation=self.observe(), info={
                "status": "executed", "dispatched": True, "channel": "desktop"})
        return StepResult(observation=self.observe(), info={
            "status": "rejected", "dispatched": False, "channel": "desktop",
            "error": "unsupported action or invalid coordinates"})

    # ---- 浏览器操作 ----
    def browser_action(self, action_name: str, args: Dict[str, Any]) -> str:
        """执行 browser kind 动作（当前只有 get_page_source）。"""
        if action_name == "get_page_source":
            return self.get_page_source(target_ref=args.get("target_ref"), max_chars=args.get("max_chars", 20000))
        return f"未知浏览器动作 {action_name}"

    def get_page_source(self, target_ref=None, max_chars=20000) -> str:
        """Read the same task page used by observations and actions."""
        session = self._get_browser_session()
        if session is None:
            return "浏览器会话不可用，请重新观察。"
        return session.get_page_source(target_ref=target_ref, max_chars=max_chars)

    def run_command(self, command: str) -> str:
        stripped = (command or "").strip()
        # `start` 用于启动 GUI 程序/打开文件：分离进程、不捕获输出、立即返回。
        # 否则 subprocess.run(capture_output=True) 会等被启动的 GUI 进程关闭其
        # 继承的 stdout/stderr 管道才结束，导致 run_in_terminal 卡到 120s 超时。
        if re.match(r'^start\b', stripped, re.IGNORECASE):
            try:
                subprocess.Popen(["cmd", "/c", command],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 stdin=subprocess.DEVNULL,
                                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                return "已启动（后台进程，不等待退出）：" + command[:120]
            except Exception as exc:  # noqa: BLE001
                return f"命令执行失败: {exc}"
        try:
            r = subprocess.run(["cmd", "/c", command], capture_output=True, text=True,
                               timeout=120, encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            return f"命令执行失败: {exc}"
        return (r.stdout or "").strip() or (r.stderr or "").strip() or "(无输出)"

    def run_python(self, code: str) -> str:
        try:
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               timeout=120, encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            return f"python 执行失败: {exc}"
        return (r.stdout or "").strip() or (r.stderr or "").strip() or "(无输出)"

    def evaluate(self) -> float:
        # 本地手动任务没有独立评测器；框架通过任务契约和执行证据验证。
        return 0.0

    def close(self) -> None:
        if self._browser_session is not None:
            try:
                self._browser_session.close()
            except Exception:  # noqa: BLE001
                pass
            self._browser_session = None
