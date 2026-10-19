"""viz 服务器：前端启动 / 实时看过程 / 看历史。

适配新框架（osworld_agent）：
- 核心端点 /log /history /ws /run/* /runs /tasks 保留，契约与原 Agent-S3 viz 一致；
- /experiments/* 和 /debug/* 是原框架的进阶功能，这里做最小 stub，避免前端报错；
- /run/start 启动的是新框架的 osworld_agent.viz.runner（跑 VMware 真实任务）。

运行：python -m osworld_agent.viz.server
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="osworld_agent Viz")


@app.get('/runtime/info')
async def runtime_info():
    """Identify the actual executor host, including through an SSH tunnel."""
    return {'service': 'yaero-studio', 'platform': sys.platform,
            'local_execution': True}


from .jev_debug import router as jev_router
app.include_router(jev_router)
from .jev_demo import router as jev_demo_router
from .jev_compare import router as jev_compare_router
app.include_router(jev_compare_router)
app.include_router(jev_demo_router)
from .paper_search_fixture import router as paper_search_router
app.include_router(paper_search_router)
from .workspace_picker import router as workspace_router
app.include_router(workspace_router)
from .knowledge import router as knowledge_router
app.include_router(knowledge_router)


@app.get('/browser/session')
async def get_browser_session_settings():
    from osworld_agent.persistent_chrome import browser_settings
    return {'ok': True, **browser_settings()}


@app.post('/browser/session')
async def set_browser_session_settings(request: Request):
    from fastapi import HTTPException
    from osworld_agent.persistent_chrome import save_browser_settings
    try:
        value = await request.json()
        if not isinstance(value, dict):
            raise ValueError('浏览器设置必须为对象')
        from osworld_agent.persistent_chrome import browser_settings
        saved = browser_settings()
        mode = value.get('browser_session_mode', 'persistent')
        profile = value.get('browser_profile_dir', saved['browser_profile_dir'])
        if mode == 'isolated' and not profile:
            profile = saved['browser_profile_dir']
        settings = save_browser_settings(mode, profile)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {'ok': True, **settings}

# ---- 配置（环境变量覆盖）----
_PKG = Path(__file__).resolve().parent.parent          # osworld_agent 包目录
_PROJECT_ROOT = _PKG.parent                            # 项目根（含 osworld_agent 与 AgentS）
AGENT_ROOT = os.getenv("OSWORLD_AGENT_ROOT", str(_PROJECT_ROOT))  # 跑 runner 的 cwd
EXAMPLES_DIR = os.getenv("OSWORLD_EXAMPLES_DIR", "")
DESKTOP_ENV_PATH = os.getenv("OSWORLD_DESKTOP_ENV_PATH", "")
VM_PATH = os.getenv("OSWORLD_VM_PATH", "")
RESULTS_DIR = os.getenv("AGENTS_RESULTS_DIR", str(_PKG / "viz" / "runs"))
LOCAL_RUNS_DIR = os.getenv("OSWORLD_LOCAL_RUNS_DIR", str(_PKG / "local_runs"))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _runner_python() -> str:
    """确定跑 runner 的 Python 解释器。

    desktop_env 的依赖（gymnasium 等）装在 AgentS 的 .venv 里，而 viz server 自身可能
    用系统 Python（只装了 fastapi/uvicorn）启动。若 runner 直接复用 sys.executable，会在
    import desktop_env 时崩（ModuleNotFoundError: gymnasium）。因此优先用 AgentS 的 .venv。
    """
    env_py = os.getenv("OSWORLD_RUNNER_PYTHON")
    if env_py:
        return env_py
    if DESKTOP_ENV_PATH:
        for rel in (os.path.join(".venv", "Scripts", "python.exe"),   # Windows
                    os.path.join(".venv", "bin", "python")):           # POSIX
            venv_py = os.path.join(DESKTOP_ENV_PATH, rel)
            if os.path.isfile(venv_py):
                return venv_py
    return sys.executable


def _detect_vm_path() -> str:
    """自动探测已解压的 VMware VM（.vmx）。

    OSWorld 的 vmware manager 用 registry（.vmware_vms）记录空闲 VM，registry 一旦为空
    就会走 _install_vm 从 HF 重新下载 Ubuntu-x86.zip（被内网代理 SSL 拦截）。而 VM 本体
    早已解压在 {DESKTOP_ENV_PATH}/vmware_vm_data/ 下，直接把 .vmx 路径传给 DesktopEnv
    （path_to_vm 非空即跳过 registry/下载），即可复用。
    """
    if not DESKTOP_ENV_PATH:
        return ""
    vms_dir = os.path.join(DESKTOP_ENV_PATH, "vmware_vm_data")
    if not os.path.isdir(vms_dir):
        return ""
    for name in sorted(os.listdir(vms_dir)):
        d = os.path.join(vms_dir, name)
        vmx = os.path.join(d, name + ".vmx")
        if os.path.isdir(d) and os.path.isfile(vmx):
            return vmx
    return ""


MAX_LOGS = 200
logs = []
clients = []
_workbench_cache = OrderedDict()
_workbench_sequence = 0
_workbench_run = None


def _cache_workbench(payload):
    """Latest reconstructable run state, independent of the 200 debug-log cap.

    Reconnects replay only changed records, not every screenshot on each poll.
    """
    global _workbench_sequence, _workbench_run
    event = payload.get('event')
    if event == 'workbench_start':
        _workbench_cache.clear()
        _workbench_run = payload.get('run_id')
    if not _workbench_run or payload.get('run_id') != _workbench_run:
        return
    relevant = {'workbench_start', 'workbench_record', 'execution_queue', 'step_verification',
                'report_start', 'report_delta', 'report_complete', 'report_error', 'report_verification',
                'run_metrics', 'live_frame', 'framework_wait', 'phase_timing'}
    if event not in relevant:
        return
    key = event
    if event == 'workbench_record':
        record = payload.get('record') or {}
        kind = record.get('event_type')
        ident = ((record.get('call') or {}).get('call_id') if kind == 'model_call' else
                 record.get('revision') if kind == 'plan_revision' else
                 record.get('parent_decision_id') or record.get('step'))
        key = (event, kind, ident or record.get('event_id'))
    elif event == 'step_verification':
        key = (event, payload.get('review_id'), payload.get('method'))
    elif event == 'execution_queue':
        key = (event, (payload.get('queue') or {}).get('revision'))
    elif event == 'framework_wait':
        key = (event, payload.get('step'))
    _workbench_sequence += 1
    payload['_wb_seq'] = _workbench_sequence
    _workbench_cache[key] = dict(payload)


# 内网模型端点直连，不走华为 HIS 代理
_no_proxy = os.environ.get("NO_PROXY", "")
for _seg in ("7.0.0.0/8", "192.168.0.0/16", "10.0.0.0/8", "localhost", "127.0.0.1"):
    if _seg not in _no_proxy:
        _no_proxy = _seg if not _no_proxy else _no_proxy + "," + _seg
os.environ["NO_PROXY"] = _no_proxy
os.environ["no_proxy"] = _no_proxy

_agent_state = {
    "status": "idle", "proc": None, "task_name": None, "config": None,
    "start_time": None, "output": [], "current_step": 0, "max_steps": 0, "task_total": 0,
    "last_error": "",
}

# Human demonstrations use their own browser context and never invoke a model.
from .collection import CollectionController, collection_router
import atexit

collection = CollectionController(os.getenv("OSWORLD_SFT_DIR", str(_PKG / "human_runs")),
                                 agent_busy=lambda: _agent_state["status"] == "running")
app.include_router(collection_router(collection))
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
atexit.register(collection.close)


def _reader_thread(proc, timing_path=None, started=None):
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                _agent_state["output"].append(line)
                if len(_agent_state["output"]) > 500:
                    _agent_state["output"] = _agent_state["output"][-500:]
    except Exception:
        pass
    finally:
        rc = proc.wait()
        elapsed = round(time.perf_counter() - started, 3) if started is not None else None
        if timing_path and elapsed is not None:
            try:
                path = Path(timing_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps({'elapsed_s': elapsed, 'scope': 'process', 'exit_code': rc}))
                os.replace(temporary, path)
            except OSError:
                pass
        if _agent_state.get('proc') is proc:
            _agent_state['elapsed_final'] = elapsed
        if _agent_state.get('proc') is proc and _agent_state["status"] == "running":
            _agent_state["status"] = "done" if rc == 0 else "failed"
            if rc != 0:
                _agent_state["last_error"] = next(
                    (line for line in reversed(_agent_state["output"])
                     if re.match(r"^[\w.]+(?:Error|Exception):", line)),
                    f"任务进程退出，代码 {rc}。可查看运行输出。")


# ================= 实时日志 ================= #
def _safe_jev_ui_value(value):
    """Redact credentials, including JSON encoded inside SystemOne state text."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = re.sub(r'[^a-z0-9]', '', str(key).lower())
            if (normalized in {'authorization', 'token', 'password', 'passwd', 'secret', 'cookie', 'setcookie'}
                    or normalized.endswith(('apikey', 'password', 'token', 'clientsecret'))):
                continue
            result[key] = _safe_jev_ui_value(item)
        return result
    if isinstance(value, list):
        return [_safe_jev_ui_value(item) for item in value]
    if isinstance(value, str) and not value.startswith('data:image/'):
        if value.lstrip().startswith(('{', '[')):
            try:
                return json.dumps(_safe_jev_ui_value(json.loads(value)), ensure_ascii=False)
            except (ValueError, TypeError):
                pass
        value = re.sub(r'(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+', 'Bearer [redacted]', value)
        value = re.sub(r'\bsk-[A-Za-z0-9]{16,}', '[redacted]', value)
    return value


def _jev_call_for_ui(call):
    request = call.get('request') if isinstance(call, dict) else None
    if not isinstance(request, dict) or 'state' not in request or not isinstance(request.get('questions'), dict):
        return None
    return _safe_jev_ui_value(call)


def _paper_identity_call_for_ui(call):
    if (not isinstance(call, dict) or call.get('call_type') != 'paper_identity'
            or not isinstance(call.get('selection_result'), dict)):
        return None
    return _safe_jev_ui_value({key: call[key] for key in (
        'call_id', 'call_type', 'decision_step', 'model', 'messages', 'response',
        'selection_result', 'latency_ms', 'status', 'error') if key in call})


def _paper_identity_entry_for_ui(call):
    call = _paper_identity_call_for_ui(call)
    if call is None:
        return None
    query = ''
    for message in call.get('messages') or []:
        if not isinstance(message, dict) or message.get('role') != 'user':
            continue
        try:
            payload = json.loads(message.get('content', ''))
            if isinstance(payload, dict):
                query = str(payload.get('original_query') or '')
        except (TypeError, ValueError):
            continue
    return {'query': query, 'report': call['selection_result'], 'call': call}


@app.post("/log")
async def receive_log(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return {"ok": False, "error": "invalid json"}
    if payload.get('event') == 'jev_call':
        payload = {**payload, 'call': _jev_call_for_ui(payload.get('call'))}
    elif payload.get('event') == 'paper_identity_review':
        payload = _safe_jev_ui_value(payload)
        if payload.get('call') is not None:
            payload['call'] = _paper_identity_call_for_ui(payload['call'])
        if isinstance(payload.get('calls'), list):
            payload['calls'] = [call for item in payload['calls']
                                if (call := _paper_identity_call_for_ui(item)) is not None]
    payload.setdefault("ts", time.time())
    _cache_workbench(payload)
    stored = dict(payload)
    # 保留 screenshot：前端 poll() 从 /history 拿日志渲染，去掉截图会导致实时监控截图缺失
    # （poll 抢先消费事件、WebSocket 的有截图事件被 seenKeys 去重跳过）。
    if payload.get('event') == 'live_frame':
        # A preview is ephemeral; keep only the latest frame per active run.
        logs[:] = [e for e in logs if not (e.get('event') == 'live_frame' and e.get('run_id') == payload.get('run_id'))]
    logs.append(stored)
    if len(logs) > MAX_LOGS:
        del logs[0:len(logs) - MAX_LOGS]
    if payload.get("kind") == "status":
        ev = payload.get("event")
        if ev == "step" and payload.get("step") is not None:
            _agent_state["current_step"] = max(_agent_state["current_step"] or 0, int(payload["step"]))
        elif ev == "task_start":
            _agent_state["current_step"] = 0
    msg = json.dumps(payload, ensure_ascii=False)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.remove(ws)
    return {"ok": True, "total": len(logs)}


@app.get("/history")
async def history(since: int = 0):
    if _workbench_cache:
        return {'logs': sorted((v for v in _workbench_cache.values() if v['_wb_seq'] > since), key=lambda v:v['_wb_seq']),
                'cursor': _workbench_sequence}
    return {"logs": logs, 'cursor': 0}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        pass
    finally:
        if ws in clients:
            clients.remove(ws)


@app.get("/")
async def index():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ================= 任务列表 & 历史 ================= #
@app.get("/tasks")
async def list_tasks(test_all_meta_path: str = "evaluation_examples/test_small.json"):
    """返回所选测试集内的任务，按域名分组。

    优先读测试集 meta（如 evaluation_examples/test_small.json，相对 DESKTOP_ENV_PATH 即 AgentS 根目录），
    只返回该测试集的任务子集；meta 读不到时回退到扫描全部 examples。
    """
    if DESKTOP_ENV_PATH:
        meta_path = os.path.abspath(os.path.join(DESKTOP_ENV_PATH, test_all_meta_path))
        try:
            if meta_path.endswith(".json"):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                if isinstance(meta, dict):
                    return {"tasks": {
                        domain: ids for domain, ids in sorted(meta.items())
                        if isinstance(ids, list)
                    }}
        except Exception:
            pass

    # 回退：扫描全部 examples
    tasks = {}
    if EXAMPLES_DIR and os.path.isdir(EXAMPLES_DIR):
        for domain in sorted(os.listdir(EXAMPLES_DIR)):
            domain_dir = os.path.join(EXAMPLES_DIR, domain)
            if not os.path.isdir(domain_dir):
                continue
            ids = sorted(f[:-5] for f in os.listdir(domain_dir) if f.endswith(".json"))
            if ids:
                tasks[domain] = ids
    if not tasks and EXAMPLES_DIR and os.path.isdir(EXAMPLES_DIR):
        flat = sorted(f[:-5] for f in os.listdir(EXAMPLES_DIR) if f.endswith(".json"))
        if flat:
            tasks["all"] = flat
    return {"tasks": tasks}


@app.get("/runs")
async def list_runs():
    runs = []
    base = os.path.join(RESULTS_DIR, "pyautogui", "screenshot")
    if os.path.isdir(base):
        for model in os.listdir(base):
            model_dir = os.path.join(base, model)
            if not os.path.isdir(model_dir):
                continue
            for domain in os.listdir(model_dir):
                domain_dir = os.path.join(model_dir, domain)
                if not os.path.isdir(domain_dir):
                    continue
                for task_id in os.listdir(domain_dir):
                    task_dir = os.path.join(domain_dir, task_id)
                    if not os.path.isdir(task_dir):
                        continue
                    instruction = result = None
                    try:
                        with open(os.path.join(task_dir, "instruction.txt"), encoding="utf-8") as f:
                            instruction = f.read().strip()
                    except Exception:
                        pass
                    try:
                        with open(os.path.join(task_dir, "result.txt")) as f:
                            result = float(f.read().strip())
                    except Exception:
                        pass
                    shots = [f for f in os.listdir(task_dir)
                             if f.startswith("step_") and f.endswith(".png")]
                    example_id, run_ts = task_id, ""
                    if "__" in task_id:
                        example_id, run_ts = task_id.rsplit("__", 1)
                    runs.append({
                        "model": model, "domain": domain, "task_id": task_id,
                        "example_id": example_id, "run_ts": run_ts,
                        "instruction": instruction or "", "result": result,
                        "steps": len(shots), "mtime": os.path.getmtime(task_dir),
                        "error": None, "experiment": {},
                    })
    runs.sort(key=lambda r: r.get("mtime", 0), reverse=True)
    return {"runs": runs}


# ================= 本地运行（独立 local_runs；兼容旧工程日期目录）================= #
_LOCAL_SUMMARY_CACHE_LIMIT = 512
_local_summary_cache = OrderedDict()
_local_summary_cache_lock = threading.Lock()
_TOKEN_CACHE_LIMIT = 512
_token_cache = OrderedDict()
_token_cache_lock = threading.Lock()


def _local_run_summary_from_events(events_path):
    """从运行中实时写入的 events.jsonl 提取摘要（失败/中断的运行没有 trajectory.json）。"""
    empty = {"instruction": "", "score": None, "success": None, "steps": 0,
             "termination_reason": "failed_no_log", "failure_reason": ""}
    try:
        task = ""
        reason = ""
        success = None
        steps = 0
        with open(events_path, encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                t = ev.get("event_type")
                if t == "episode_start":
                    task = (ev.get("manifest") or {}).get("task", "")
                elif t == "execution":
                    steps += 1
                elif t == "episode_end":
                    o = ev.get("outcome") or {}
                    reason = o.get("termination_reason") or ""
                    success = o.get("success")
                elif t == "episode_error":
                    reason = ev.get('error_type') or 'episode_error'
                    success = False
        if not task and not reason and not steps:
            return empty
        return {"instruction": task, "score": None, "success": success,
                "steps": steps, "termination_reason": reason or "interrupted",
                "failure_reason": ""}
    except Exception:  # noqa: BLE001
        return empty


def _events_path_for(traj_path, directory, prefix):
    """给定轨迹路径，推导同一次运行的 events.jsonl 路径（平铺/子目录两种布局）。"""
    if traj_path is None:
        return None
    if traj_path.name.endswith(".jsonl"):
        return traj_path  # 已经是 events.jsonl（无 trajectory 的兜底记录）
    return directory / (prefix + "events.jsonl")


def _run_token_usage(events_path):
    """One canonical event stream; missing usage remains explicitly partial."""
    from ..run_metrics import summarize_events
    if not events_path or not os.path.isfile(events_path):
        return summarize_events([])
    stat = os.stat(events_path)
    key, version = str(os.path.abspath(events_path)), (stat.st_mtime_ns, stat.st_size)
    with _token_cache_lock:
        cached = _token_cache.get(key)
        if cached and cached[0] == version:
            return dict(cached[1])
    def events():
        with open(events_path, encoding='utf-8') as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
                except ValueError:
                    continue
    result = summarize_events(events())
    with _token_cache_lock:
        _token_cache[key] = (version, dict(result))
        _token_cache.move_to_end(key)
        while len(_token_cache) > _TOKEN_CACHE_LIMIT:
            _token_cache.popitem(last=False)
    return result


def _local_metrics(events_path, directory, prefix='', fallback=None):
    metrics = _run_token_usage(events_path)
    if (not events_path or not Path(events_path).exists()) and fallback:
        metrics = dict(fallback)
    # The launcher records the whole process span, including browser boot,
    # export and cleanup. Pipeline time is retained as a separate scope.
    try:
        timing = json.loads((directory / (prefix + 'run_timing.json')).read_text())
        elapsed = timing.get('elapsed_s')
        if isinstance(elapsed, (int, float)) and 0 <= elapsed < 1e9:
            metrics = {**metrics, 'pipeline_elapsed_s': metrics.get('elapsed_s'),
                       'elapsed_s': elapsed, 'timing_source': 'process', 'complete': True}
    except (OSError, ValueError, TypeError):
        pass
    return metrics



_SUMMARY_HEAD_LIMIT = 4 * 1024 * 1024  # task/outcome 都很小且位于 steps 之前


def _summary_head_parse(traj_path):
    """只解析 trajectory.json 头部（version/task/outcome），跳过巨型 steps 数组。

    trajectory.json 顶层顺序为 version, task, outcome, steps, ...；task/outcome 在
    steps 之前且都很小。定位到顶层 "steps" 键即截断、补闭合并解析，避免对大文件做
    整包 json.load。解析失败返回 None，调用方回退到完整解析。
    """
    try:
        with open(traj_path, encoding="utf-8") as f:
            head = f.read(_SUMMARY_HEAD_LIMIT)
    except OSError:
        return None
    match = re.search(r'"steps"\s*:', head)
    if not match or not head.lstrip().startswith('{'):
        return None
    prefix = head[:match.start()].rstrip().rstrip(',') + '}'
    try:
        return json.loads(prefix)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _local_run_summary(traj_path):
    """Cache only small list metadata; a changed trajectory is read again."""
    empty = {"instruction": "", "score": None, "success": None, "steps": 0,
             "termination_reason": "", "failure_reason": ""}
    if traj_path is None:
        return {"instruction": "", "score": None, "success": False, "steps": 0,
                "termination_reason": "failed_no_log", "failure_reason": ""}
    try:
        before = traj_path.stat()
        path_key = str(traj_path.resolve())
        version = (before.st_mtime_ns, before.st_size)
        with _local_summary_cache_lock:
            cached = _local_summary_cache.get(path_key)
            if cached is not None and cached[0] == version:
                _local_summary_cache.move_to_end(path_key)
                return dict(cached[1])
        if str(traj_path).endswith("events.jsonl"):
            summary = _local_run_summary_from_events(traj_path)
        else:
            # 快路径：只读头部（version/task/outcome），避免整包解析巨型 steps
            traj = _summary_head_parse(traj_path)
            if traj is not None and (traj.get("outcome") or {}).get("num_steps") is None:
                traj = None  # num_steps 缺失，需完整解析数步数
            if traj is None:
                with open(traj_path, encoding="utf-8") as f:
                    traj = json.load(f)
            outcome = traj.get("outcome") or {}
            reason = outcome.get("termination_reason") or ""
            num_steps = outcome.get("num_steps")
            if num_steps is None:
                num_steps = len(traj.get("steps") or [])
            failure_reason = ""
            if reason == "planning_failed":
                if "planning_calls" not in traj:
                    with open(traj_path, encoding="utf-8") as f:
                        traj = json.load(f)
                failure_reason = next((str(c.get("validation_error")) for c in reversed(
                    traj.get("planning_calls") or []) if c.get("validation_error")), "")
            summary = {"instruction": (traj.get("task") or {}).get("instruction", ""),
                       "score": outcome.get("score"), "success": outcome.get("success"),
                       "steps": num_steps,
                       "termination_reason": reason,
                       "failure_reason": failure_reason, "run_metrics": outcome.get("run_metrics", {})}
        after = traj_path.stat()
        if version == (after.st_mtime_ns, after.st_size):
            with _local_summary_cache_lock:
                _local_summary_cache[path_key] = (version, dict(summary))
                _local_summary_cache.move_to_end(path_key)
                while len(_local_summary_cache) > _LOCAL_SUMMARY_CACHE_LIMIT:
                    _local_summary_cache.popitem(last=False)
        return summary
    except Exception:  # noqa: BLE001
        # An in-progress or invalid JSON file must not poison the cache.
        return empty


def _read_local_runs():
    runs, shots_by_directory = [], {}
    for run_id, traj_path, directory, prefix in _local_run_files():
        summary = _local_run_summary(traj_path)
        summary["tokens"] = _local_metrics(_events_path_for(traj_path, directory, prefix), directory, prefix, summary.get("run_metrics"))
        try:
            mtime = (traj_path.stat().st_mtime if traj_path is not None
                     else directory.stat().st_mtime)
            if directory not in shots_by_directory:
                shots_by_directory[directory] = [name for name in os.listdir(directory)
                                                 if name.endswith(".png")]
            shots = sum(name.startswith(prefix + "step_")
                        for name in shots_by_directory[directory])
        except OSError:
            continue  # A run may be removed while the directory is being scanned.
        runs.append({"id": run_id, **summary, "shots": shots, "mtime": mtime})
    runs.sort(key=lambda item: item['mtime'], reverse=True)
    return {"runs": runs}


@app.get("/local/runs")
async def local_runs():
    """列出独立运行日志，同时兼容工程日期目录中的旧轨迹。"""
    return await asyncio.to_thread(_read_local_runs)


def _local_run_files():
    """Read new per-run logs and legacy flat workspace logs without moving files."""
    from osworld_agent.workspace_config import default_workspace_root
    root = Path(default_workspace_root())
    if root.is_dir():
        for day in root.iterdir():
            if not day.is_dir() or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', day.name):
                continue
            for path in day.glob('ui_*_trajectory.json'):
                run_id = path.name[:-len('_trajectory.json')]
                if re.fullmatch(r'ui_\d{8}_\d{6}_[0-9a-f]{8}', run_id):
                    yield run_id, path, day, run_id + '_'
    legacy = Path(LOCAL_RUNS_DIR)
    if legacy.is_dir():
        for directory in legacy.iterdir():
            if not directory.is_dir():
                continue
            standard = directory / 'trajectory.json'
            if standard.is_file():
                yield directory.name, standard, directory, ''
                continue
            candidates = sorted(directory.glob('*_trajectory.json'))
            for path in candidates:
                prefix = path.name[:-len('trajectory.json')]
                run_id = directory.name if len(candidates) == 1 else directory.name + '__' + path.stem
                yield run_id, path, directory, prefix
            # 兜底：没有 trajectory.json 但写了 events.jsonl（stalled/中断的运行也保留）
            events = directory / 'events.jsonl'
            if events.is_file():
                yield directory.name, events, directory, ''
            # 更早失败（连 events.jsonl 都没写）也保留一条"失败"记录
            elif not any(directory.iterdir()):
                yield directory.name, None, directory, ''


def _local_run_path(run):
    if not isinstance(run, str) or '/' in run or '\\' in run or run in {'', '.', '..'}:
        return None
    for run_id, path, directory, prefix in _local_run_files():
        if run_id == run:
            return path, directory, prefix
    return None


def _local_traj_to_frontend(traj, prefix=''):
    """把标准 trajectory.json 的 steps 转成前端 traj.jsonl 格式（复用历史视图渲染）。"""
    lines = []
    calls_by_step = {}
    identity_by_step = {}
    identity_diagnostics = {}
    for event in traj.get('events') or []:
        if not isinstance(event, dict):
            continue
        if event.get('event_type') == 'paper_identity_review' and isinstance(event.get('report'), dict):
            parent = re.fullmatch(r'a_(\d+)', str(event.get('parent_decision_id') or ''))
            step_id = parent.group(1) if parent else event.get('decision_step', event.get('step'))
            if re.fullmatch(r'\d+', str(step_id or '')):
                event_call = event.get('call') or next((call for call in reversed(event.get('calls') or [])
                    if _paper_identity_call_for_ui(call) is not None), None)
                identity_diagnostics.setdefault(str(int(step_id)), []).append(_safe_jev_ui_value({
                    'query': event.get('query', ''), 'report': event['report'],
                    'call': _paper_identity_call_for_ui(event_call)}))
        if event.get('event_type') not in {'model_call', 'jev_call'}:
            continue
        identity = _paper_identity_entry_for_ui(event.get('call'))
        if identity is not None:
            parent = re.fullmatch(r'a_(\d+)', str(event.get('parent_decision_id') or ''))
            step_id = parent.group(1) if parent else identity['call'].get('decision_step')
            if re.fullmatch(r'\d+', str(step_id or '')):
                identity_by_step.setdefault(str(int(step_id)), []).append(identity)
        call = _jev_call_for_ui(event.get('call'))
        if call is None:
            continue
        parent = re.fullmatch(r'a_(\d+)', str(event.get('parent_decision_id') or ''))
        step_id = parent.group(1) if parent else call.get('decision_step')
        if re.fullmatch(r'\d+', str(step_id or '')):
            calls_by_step.setdefault(str(int(step_id)), []).append(call)
    for call in traj.get('review_calls') or []:
        identity = _paper_identity_entry_for_ui(call)
        step_id = (identity or {}).get('call', {}).get('decision_step')
        if identity is not None and re.fullmatch(r'\d+', str(step_id or '')):
            identity_by_step.setdefault(str(int(step_id)), []).append(identity)
    for s in traj.get("steps") or []:
        action = s.get("action")
        action_str = json.dumps(action, ensure_ascii=False) if isinstance(action, dict) else str(action or "")
        reasoning = s.get("reasoning") or {}
        assessment = s.get("assessment") or {}
        grounding = s.get("grounding") or {}
        resolved_action = s.get('resolved_action') or {}
        if isinstance(resolved_action, str):
            try:
                resolved_action = json.loads(resolved_action)
            except (TypeError, ValueError):
                resolved_action = {}
        timing = s.get("timing_ms") or {}
        gs = []
        if grounding and grounding.get("coord"):
            gs.append({"query": grounding.get("target", ""),
                       "coord": grounding.get("coord"),
                       "elapsed": timing.get("ground")})
        # 决策耗时：decision_calls 最后一条的 latency_ms（LoggingChatModel 转发后才有）
        decision_elapsed = None
        dc = s.get("decision_calls") or []
        if dc:
            decision_elapsed = dc[-1].get("latency_ms")
        # 子 Agent 调用：从 execution_info 里提取 call_code_agent
        ei = s.get("execution_info") or {}
        jev_calls, seen_calls = [], set()
        for raw_call in [*(s.get('jev_calls') or []), *calls_by_step.get(str(s.get('step')), [])]:
            call = _jev_call_for_ui(raw_call)
            if call is None:
                continue
            ident = call.get('call_id') or json.dumps(call, ensure_ascii=False, sort_keys=True)
            if ident not in seen_calls:
                seen_calls.add(ident)
                jev_calls.append(call)
        identity_reviews, seen_identity_calls = [], set()
        identity_entries = [*identity_by_step.get(str(s.get('step')), []),
                            *(_paper_identity_entry_for_ui(call) for call in s.get('review_calls') or [])]
        for identity in identity_entries:
            if identity is None:
                continue
            call = identity['call']
            ident = call.get('call_id') or json.dumps(call, ensure_ascii=False, sort_keys=True)
            if ident not in seen_identity_calls:
                seen_identity_calls.add(ident)
                identity_reviews.append(identity)
        for identity in identity_diagnostics.get(str(s.get('step')), []):
            call = identity.get('call') or {}
            if (call.get('call_id') in seen_identity_calls or any(
                    old.get('query') == identity.get('query') and old.get('report') == identity.get('report')
                    for old in identity_reviews)):
                continue
            if call.get('call_id'):
                seen_identity_calls.add(call['call_id'])
            identity_reviews.append(identity)
        subagent = None
        if ei.get("action") == "call_code_agent":
            subagent = {"task": ei.get("task", ""), "result": ei.get("result", "")}
        lines.append({
            "step_num": s.get("step"),
            "plan": reasoning.get("screen_analysis", ""),
            "expected_effect": reasoning.get("expected_effect", ""),
            "action": action_str,
            "reward": s.get("reward"),
            "done": bool(s.get("done", False)),
            "screenshot_file": f"{prefix}step_{s.get('step')}.png",
            "grounding": gs,
            "resolved_action": resolved_action,
            "routing": s.get("routing") or {},
            "operation_mode": s.get("operation_mode"),
            "prompt_metadata": s.get("prompt_metadata") or {},
            "execution_info": {key: ei[key] for key in
                               ("status", "dispatched", "channel", "resolved", "error_code", "reason", "coord", "coordinate_space")
                               if key in ei},
            "outcome": (s.get("execution") or {}).get("outcome", ""),
            "assessment": assessment,
            "step_verification": s.get("step_verification"),
            "step_reviews": s.get("step_reviews", []),
            "jev_calls": jev_calls,
            "paper_identity_reviews": identity_reviews,
            "execution_queue": s.get("execution_queue"),
            "prompt": s.get("prompt_text", ""),
            "timing_ms": timing,
            "decision_elapsed_ms": decision_elapsed,
            "subagent": subagent,
        })
    return lines


def _read_local_traj(run):
    found = _local_run_path(run)
    if found is None:
        return {"ok": False, "error": "轨迹不存在"}
    traj_path, directory, prefix = found
    if traj_path.name.endswith('.jsonl'):
        summary = _local_run_summary_from_events(traj_path)
        from .workbench import project_event_workbench
        from ..training_events import read_events
        metrics = _local_metrics(traj_path, directory, prefix)
        try:
            workbench = project_event_workbench(read_events(traj_path), summary['instruction'], metrics)
        except (OSError, ValueError):
            return {'ok': False, 'error': '事件记录损坏，无法可靠重建轨迹'}
        return {'ok': True, 'traj': [], 'instruction': summary['instruction'], 'workbench': workbench,
                'outcome': {**summary, 'final_answer': '运行未导出完整轨迹；展示已落盘事件。'},
                'tokens': metrics}
    try:
        with open(traj_path, encoding="utf-8") as f:
            traj = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"轨迹解析失败：{exc}"}
    from .workbench import project_workbench
    metrics = _local_metrics(_events_path_for(traj_path, directory, prefix), directory, prefix, (traj.get('outcome') or {}).get('run_metrics'))
    workbench = project_workbench(traj)
    workbench['metrics'] = metrics
    return {"ok": True, "workbench": workbench, "traj": _local_traj_to_frontend(traj, prefix),
            "instruction": (traj.get("task") or {}).get("instruction", ""),
            "plan": traj.get("plan"),
            "outcome": traj.get("outcome") or {},
            "planning_errors": [{"attempt": i + 1, "reason": c.get("validation_error")}
                                for i, c in enumerate(traj.get("planning_calls") or [])
                                if c.get("validation_error")],
            "tokens": metrics,
            "knowledge_retrieval": (traj.get("run_manifest") or {}).get("knowledge_retrieval"),
            "replans": [{"event": e.get("event_type"), **{k: e[k] for k in ("brief", "strategy_change", "revision") if k in e}}
                        for e in traj.get("events", []) if e.get("event_type") in {"replan_context", "replan_strategy"}]}


@app.get("/local/traj")
async def local_traj(run: str):
    """返回某个本地运行的轨迹；磁盘扫描与解析在后台线程执行。"""
    return await asyncio.to_thread(_read_local_traj, run)


class ResultFileAction(BaseModel):
    run: str
    artifact_id: str
    operation: Literal['open', 'reveal']
    scope: Literal['history', 'live'] = 'history'


@app.post('/local/result-file')
async def result_file_action(body: ResultFileAction):
    from .result_files import open_download
    if body.scope == 'live':
        if body.run != _workbench_run:
            return {'ok': False, 'error': '当前运行已切换，请从本机轨迹打开原任务。'}
        outcome = next((p['record'].get('outcome', {}) for p in reversed(list(_workbench_cache.values()))
                        if p.get('event') == 'workbench_record'
                        and p.get('record', {}).get('event_type') == 'episode_end'), {})
    else:
        data = await asyncio.to_thread(_read_local_traj, body.run)
        if not data.get('ok'):
            return {'ok': False, 'error': '轨迹不存在，无法核实下载文件。'}
        outcome = data.get('workbench', {}).get('outcome', {})
    return await asyncio.to_thread(open_download, outcome, body.artifact_id, body.operation)


@app.get("/local/shot")
async def local_shot(run: str, filename: str):
    """返回某个本地运行的截图（step_N.png）。"""
    filename = os.path.basename(filename)  # 防目录穿越
    found = _local_run_path(run)
    if found is None:
        return Response(status_code=404)
    _, directory, prefix = found
    if not re.fullmatch(re.escape(prefix) + r'step_\d+\.png', filename):
        return Response(status_code=404)
    shot = directory / filename
    if shot.is_file():
        return FileResponse(str(shot))
    return Response(status_code=404)


@app.get('/local/jev-image')
async def local_jev_image(run: str, filename: str):
    """Serve only content-addressed screenshot assets from the selected run."""
    if not re.fullmatch(r'assets/[0-9a-f]{64}\.(?:png|jpg|webp)', filename):
        return Response(status_code=404)
    found = _local_run_path(run)
    if found is None:
        return Response(status_code=404)
    _, directory, _ = found
    asset_root = (directory / 'assets').resolve()
    path = (directory / filename).resolve()
    if path.parent == asset_root and path.is_file():
        return FileResponse(str(path))
    return Response(status_code=404)


@app.post("/local/delete")
async def local_delete(request: Request):
    """删除某个本地运行。local_runs 独立子目录整目录删；工程日期目录只删该 run 的文件。"""
    import shutil
    try:
        cfg = await request.json()
    except Exception:
        return {"ok": False, "error": "invalid json"}
    run = cfg.get("run")
    if not isinstance(run, str) or not run:
        return {"ok": False, "error": "missing run id"}
    found = _local_run_path(run)
    if found is None:
        return {"ok": False, "error": "not found"}
    _, directory, prefix = found
    try:
        try:
            legacy_root = os.path.realpath(LOCAL_RUNS_DIR)
            real_dir = os.path.realpath(directory)
            in_legacy = real_dir != legacy_root and os.path.commonpath([legacy_root, real_dir]) == legacy_root
        except (ValueError, OSError):
            in_legacy = False
        if in_legacy:
            shutil.rmtree(directory)
        elif prefix:
            removed = 0
            for p in directory.glob(prefix + '*'):
                try:
                    if p.is_file():
                        p.unlink()
                        removed += 1
                except OSError:
                    continue
            if not removed:
                return {"ok": False, "error": "未找到可删除的文件"}
        else:
            return {"ok": False, "error": "无法识别的轨迹布局"}
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.get("/run/traj")
async def get_traj(model: str, domain: str, task_id: str):
    base = os.path.join(RESULTS_DIR, "pyautogui", "screenshot", model, domain, task_id)
    lines = []
    traj_path = os.path.join(base, "traj.jsonl")
    if os.path.exists(traj_path):
        with open(traj_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except Exception:
                        pass
    instruction = ""
    ip = os.path.join(base, "instruction.txt")
    if os.path.exists(ip):
        with open(ip, encoding="utf-8") as f:
            instruction = f.read().strip()
    return {"traj": lines, "instruction": instruction, "config": {}, "system_prompt": ""}


@app.get("/run/trajectory")
async def get_trajectory(model: str, domain: str, task_id: str):
    """返回标准训练轨迹（trajectory.json，供前端展示 + 训练管线取用）。"""
    base = os.path.join(RESULTS_DIR, "pyautogui", "screenshot", model, domain, task_id)
    traj_path = os.path.join(base, "trajectory.json")
    if os.path.exists(traj_path):
        try:
            with open(traj_path, encoding="utf-8") as f:
                return {"ok": True, "trajectory": json.load(f)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"轨迹解析失败：{exc}"}
    return {"ok": False, "error": "该运行未导出标准轨迹（trajectory.json 不存在）"}


@app.get("/run/shot")
async def get_shot(model: str, domain: str, task_id: str, filename: str):
    shot_path = os.path.join(RESULTS_DIR, "pyautogui", "screenshot", model, domain, task_id, filename)
    if os.path.exists(shot_path):
        return FileResponse(shot_path)
    return Response(status_code=404)


@app.post("/run/delete")
async def delete_run(request: Request):
    import shutil
    try:
        cfg = await request.json()
    except Exception:
        return {"ok": False, "error": "invalid json"}
    model, domain, task_id = cfg.get("model"), cfg.get("domain"), cfg.get("task_id")
    if not (model and domain and task_id):
        return {"ok": False, "error": "missing params"}
    task_dir = os.path.join(RESULTS_DIR, "pyautogui", "screenshot", model, domain, task_id)
    if os.path.isdir(task_dir):
        shutil.rmtree(task_dir)
        return {"ok": True}
    return {"ok": False, "error": "not found"}


# ================= 进程管理 ================= #
@app.post("/run/test-connection")
async def test_connection(request: Request):
    import base64
    import io
    import secrets
    import requests as req
    from PIL import Image, ImageDraw
    from osworld_agent.config import resolve_secret
    try:
        cfg = await request.json()
    except Exception:
        cfg = {}
    model = cfg.get("model", "Qwen3.6-27B")
    url = cfg.get("model_url", "http://7.246.80.237:48888/v1")
    try:
        api_key = resolve_secret(cfg.get("model_api_key") or "EMPTY")
    except (RuntimeError, TypeError) as exc:
        return {"ok": False, "status": -1, "error": str(exc)}
    challenge = secrets.token_hex(3).upper()
    image = Image.new("RGB", (260, 90), "white")
    ImageDraw.Draw(image).text((35, 25), challenge, fill="black", stroke_width=1)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    payload = {"model": model, "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "读取图片里的六位字符，只输出 JSON 对象，字段名为 code。"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]},
    ], "max_tokens": 32, "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False}}
    try:
        r = req.post(f"{url}/chat/completions", json=payload,
                     headers={"Authorization": f"Bearer {api_key}"}, timeout=30,
                     proxies={"http": None, "https": None})
        if r.status_code == 200:
            content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            try:
                recognized = str(json.loads(content).get("code") or "").strip().upper()
            except (TypeError, ValueError, json.JSONDecodeError):
                recognized = ""
            if recognized != challenge:
                return {"ok": False, "status": 200,
                        "error": "接口可调用，但未正确读取随机图片内容；不能作为截图决策模型"}
            return {"ok": True, "status": 200, "content": "视觉与 JSON 输出均通过",
                    "visual_support": "随机图片内容识别通过"}
        return {"ok": False, "status": r.status_code, "error": r.text[:300]}
    except Exception as e:
        return {"ok": False, "status": -1, "error": str(e)[:300]}


def _start_local(cfg: dict, viz_port: int | None = None):
    """启动当前主机平台的本机任务，复用 _reader_thread + /log + /ws。"""
    task = str(cfg.get("task") or "").strip()
    if not task:
        return {"ok": False, "error": "请输入任务描述"}
    model = cfg.get("model", "Qwen3.6-27B")
    model_url = cfg.get("model_url", "http://7.246.80.237:48888/v1")
    model_api_key = cfg.get("model_api_key") or "EMPTY"
    ground_model = cfg.get("ground_model", "UI-Venus-2-9B")
    ground_url = cfg.get("ground_url", "http://7.246.80.189:8001/v1")
    ground_api_key = cfg.get("ground_api_key") or "EMPTY"
    max_steps = int(cfg.get("max_steps", 15))
    from osworld_agent.persistent_chrome import browser_settings, save_browser_settings
    saved_browser = browser_settings()
    session_mode = cfg.get('browser_session_mode', saved_browser['browser_session_mode'])
    profile_dir = cfg.get('browser_profile_dir', saved_browser['browser_profile_dir']) or ''
    try:
        if session_mode == 'isolated' and not profile_dir:
            profile_dir = saved_browser['browser_profile_dir']
        browser_choice = save_browser_settings(session_mode, profile_dir)
    except ValueError as exc:
        return {'ok': False, 'error': str(exc)}
    try:
        from osworld_agent.workspace_config import default_workspace_root, normalize_workspace_root
        workspace = normalize_workspace_root(cfg.get('workspace_dir') or default_workspace_root())
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    from osworld_agent.workspace_config import allocate_local_result_dir
    run_prefix = None
    result_dir = cfg.get('result_dir') or allocate_local_result_dir(LOCAL_RUNS_DIR)

    cmd = [
        _runner_python(), "-m", "osworld_agent.run_local",
        "--task", task,
        "--model", model,
        "--model-url", model_url,
        "--model-api-key", model_api_key,
        "--ground-model", ground_model,
        "--ground-type", str(cfg.get("ground_type", "qwen-plus")),
        "--ground-url", ground_url,
        "--ground-api-key", ground_api_key,
        "--max-steps", str(max_steps),
        "--browser-session-mode", session_mode,
        "--result-dir", result_dir,
    ]
    if cfg.get('cli_only') is True:
        cmd += ['--cli-only']
    if session_mode == 'persistent' and profile_dir:
        cmd += ['--browser-profile-dir', profile_dir]
    cfg.update(browser_session_mode=session_mode, browser_profile_dir=profile_dir,
               effective_profile_dir=browser_choice['effective_profile_dir'])
    if run_prefix:
        cmd += ['--run-prefix', run_prefix]
    if cfg.get("browser_start_url"):
        from urllib.parse import urlsplit
        start_url = str(cfg["browser_start_url"])
        parsed = urlsplit(start_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}
                or parsed.path.rstrip("/") != "/demo/paper-search" or parsed.query or parsed.fragment):
            return {"ok": False, "error": "演示起始页只能使用本机论文搜索回放页"}
        cmd += ["--browser-start-url", start_url]
    try:
        cmd += _wait_cli_args(cfg)
        from osworld_agent.prompt_settings import prompt_cli_args
        cmd += prompt_cli_args(cfg.get('mode_prompts'), Path(LOCAL_RUNS_DIR) / '.prompt_configs')
        cmd += ['--workspace-dir', workspace]
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if cfg.get("completion_spec") is not None:
        from osworld_agent.verifiers import CompletionVerifier
        try:
            contract = CompletionVerifier(task, cfg["completion_spec"]).spec
        except ValueError as exc:
            return {"ok": False, "error": f"验收条件无效：{exc}"}
        os.makedirs(result_dir, exist_ok=True)
        contract_name = (run_prefix + '_' if run_prefix else '') + 'completion_spec.json'
        contract_path = os.path.abspath(os.path.join(result_dir, contract_name))
        with open(contract_path, "w", encoding="utf-8") as f:
            json.dump(contract, f, ensure_ascii=False, indent=2)
        cmd += ["--completion-spec", contract_path]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["AGENTS_VIZ_SERVER_URL"] = f"http://127.0.0.1:{viz_port or os.getenv('AGENTS_VIZ_PORT', '8088')}"
    env["AGENTS_RESULTS_DIR"] = RESULTS_DIR
    env["NO_PROXY"] = "192.168.0.0/16,10.0.0.0/8,7.0.0.0/8,localhost,127.0.0.1"
    env["no_proxy"] = env["NO_PROXY"]

    _agent_state["task_name"] = task[:60]
    _agent_state["task_total"] = 1
    _agent_state["config"] = cfg
    _agent_state["output"] = []
    _agent_state["last_error"] = ""
    _agent_state["current_step"] = 0
    _agent_state["max_steps"] = max_steps
    _agent_state["start_time"] = time.time()
    _agent_state["started_monotonic"] = time.perf_counter()
    _agent_state["elapsed_final"] = None
    _agent_state["status"] = "starting"
    logs.clear()
    try:
        proc = subprocess.Popen(cmd, cwd=AGENT_ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        _agent_state["proc"] = proc
        _agent_state["status"] = "running"
        threading.Thread(target=_reader_thread, args=(proc, os.path.join(result_dir, "run_timing.json"), _agent_state["started_monotonic"]), daemon=True).start()
        return {"ok": True, "task": _agent_state["task_name"], "task_total": 1}
    except Exception as e:
        _agent_state["status"] = "failed"
        return {"ok": False, "error": str(e)}


def _wait_cli_args(cfg):
    import math
    from osworld_agent.knowledge_selection import selection_cli_args
    result = _judge_cli_args(cfg) + selection_cli_args(cfg)
    for field, flag, default in (("operation_timeout_s", "--operation-timeout", 300),
                                  ("task_timeout_s", "--task-timeout", 1800)):
        try:
            value = float(cfg.get(field, default))
        except (TypeError, ValueError):
            raise ValueError("等待时限必须是大于零的秒数") from None
        if isinstance(cfg.get(field), bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("等待时限必须是大于零的秒数")
        result += [flag, str(value)]
    if 'download_max_mib' in cfg:
        value = cfg['download_max_mib']
        if type(value) is not int or value < 0:
            raise ValueError('下载大小上限必须为非负整数 MiB，0 表示不限')
        result += ['--download-max-mib', str(value)]
    return result


def _judge_cli_args(cfg):
    """Both runners use the same validated optional System One configuration."""
    from osworld_agent.model.decision_judge import DecisionJudgeConfig
    data = cfg.get('decision_judge') or {}
    if not isinstance(data, dict):
        raise ValueError('Decision 评估器配置必须是对象')
    if data.get('api_key'):
        raise ValueError('Decision 认证请在服务端设置 DECISION_JUDGE_API_KEY')
    explicit_selection = 'selection_policy' in data
    data = dict(data)
    selection_policy = data.pop('selection_policy', 'maximum_probability')
    if not isinstance(selection_policy, str) or selection_policy not in {'threshold', 'maximum_probability'}:
        raise ValueError('Jev 标签选择必须为 threshold 或 maximum_probability')
    try:
        value = DecisionJudgeConfig(**data)
    except (TypeError, ValueError) as exc:
        raise ValueError('Decision 评估器配置无效') from exc
    if not value.url:
        return []
    # New runners default to maximum_probability. Legacy configurations without
    # this extension remain parseable by the original base CLI argument helper.
    selection_args = ['--judge-selection-policy', selection_policy] if explicit_selection else []
    return ['--judge-url', value.url, '--judge-model', value.model, '--judge-revision', value.revision,
            '--judge-threshold', str(value.threshold), '--judge-margin', str(value.margin), '--judge-timeout', str(value.timeout),
            '--judge-review-mode', value.review_mode] + selection_args + ([] if value.recheck_unknown else ['--judge-no-recheck'])


@app.post("/run/start")
async def start_run(request: Request):
    if collection.active:
        return {"ok": False, "error": "人工采集进行中，请先结束采集。"}
    if _agent_state["status"] == "running":
        return {"ok": False, "error": "已有任务在运行中，请先停止"}
    try:
        cfg = await request.json()
    except Exception:
        return {"ok": False, "error": "invalid json"}

    from osworld_agent.prompt_settings import validate_mode_prompts
    try:
        cfg['mode_prompts'] = validate_mode_prompts(cfg.get('mode_prompts'))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if cfg.get("mode") == "local":
        return _start_local(cfg, request.url.port)

    domain = cfg.get("domain", "all")
    task_id = cfg.get("task_id")
    if not task_id:
        return {"ok": False, "error": "请选择一个具体任务 task_id"}
    if not EXAMPLES_DIR:
        return {"ok": False, "error": "未配置 OSWORLD_EXAMPLES_DIR（evaluation_examples/examples）"}
    if not DESKTOP_ENV_PATH:
        return {"ok": False, "error": "未配置 OSWORLD_DESKTOP_ENV_PATH（包含 desktop_env 的目录）"}

    model = cfg.get("model", "Qwen3.6-27B")
    model_url = cfg.get("model_url", "http://7.246.80.237:48888/v1")
    model_api_key = cfg.get("model_api_key") or "EMPTY"
    ground_model = cfg.get("ground_model", "UI-Venus-2-9B")
    ground_url = cfg.get("ground_url", "http://7.246.80.189:8001/v1")
    ground_api_key = cfg.get("ground_api_key", "EMPTY")
    ground_type = cfg.get("ground_type", "ui-venus2")
    max_steps = int(cfg.get("max_steps", 20))

    cmd = [
        _runner_python(), "-m", "osworld_agent.viz.runner",
        "--task-id", task_id,
        "--examples-dir", EXAMPLES_DIR,
        "--domain", domain,
        "--model", model,
        "--model-url", model_url,
        "--model-api-key", model_api_key,
        "--ground-model", ground_model,
        "--ground-url", ground_url,
        "--ground-api-key", ground_api_key,
        "--ground-type", ground_type,
        "--max-steps", str(max_steps),
        "--desktop-env-path", DESKTOP_ENV_PATH,
        "--cache-dir", os.path.join(DESKTOP_ENV_PATH, "cache"),
    ]
    try:
        cmd += _wait_cli_args(cfg)
        from osworld_agent.prompt_settings import prompt_cli_args
        cmd += prompt_cli_args(cfg.get('mode_prompts'), Path(LOCAL_RUNS_DIR) / '.prompt_configs')
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    vm_path = VM_PATH or _detect_vm_path()
    if vm_path:
        cmd += ["--vm-path", vm_path]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["AGENTS_VIZ_SERVER_URL"] = f"http://127.0.0.1:{request.url.port or os.getenv('AGENTS_VIZ_PORT', '8088')}"
    env["AGENTS_RESULTS_DIR"] = RESULTS_DIR
    env["NO_PROXY"] = "192.168.0.0/16,10.0.0.0/8,7.0.0.0/8,localhost,127.0.0.1"
    env["no_proxy"] = env["NO_PROXY"]
    # 本地 proxy 任务的上游代理：用 HIS 活 IP，否则 _proxy_setup 起不来 tinyproxy
    env["OSWORLD_UPSTREAM_PROXY"] = os.environ.get("OSWORLD_UPSTREAM_PROXY") or "http://172.18.94.109:8080"

    _agent_state["task_name"] = f"{domain}/{task_id}"
    _agent_state["task_total"] = 1
    _agent_state["config"] = cfg
    _agent_state["output"] = []
    _agent_state["last_error"] = ""
    _agent_state["current_step"] = 0
    _agent_state["max_steps"] = max_steps
    _agent_state["start_time"] = time.time()
    _agent_state["started_monotonic"] = time.perf_counter()
    _agent_state["elapsed_final"] = None
    _agent_state["status"] = "starting"
    logs.clear()
    try:
        proc = subprocess.Popen(cmd, cwd=AGENT_ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        _agent_state["proc"] = proc
        _agent_state["status"] = "running"
        threading.Thread(target=_reader_thread, args=(proc, None, _agent_state["started_monotonic"]), daemon=True).start()
        return {"ok": True, "task": _agent_state["task_name"], "task_total": 1}
    except Exception as e:
        _agent_state["status"] = "failed"
        return {"ok": False, "error": str(e)}


@app.get("/run/status")
async def run_status():
    elapsed = _agent_state.get("elapsed_final")
    if elapsed is None:
        elapsed = time.perf_counter() - _agent_state["started_monotonic"] if _agent_state.get("started_monotonic") else 0
    return {
        "status": _agent_state["status"],
        "task": _agent_state["task_name"],
        "current_step": _agent_state["current_step"],
        "max_steps": _agent_state["max_steps"],
        "elapsed": round(elapsed, 1),
        "pid": _agent_state["proc"].pid if _agent_state["proc"] else None,
        "error": _agent_state["last_error"],
    }


@app.post("/run/stop")
async def stop_run():
    proc = _agent_state["proc"]
    if proc and proc.poll() is None:
        if os.name == 'nt':
            try:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                               capture_output=True, timeout=15)
            except Exception:
                proc.terminate()
        else:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    _agent_state["status"] = "idle"
    _agent_state["proc"] = None
    _agent_state["task_name"] = None
    return {"ok": True}


@app.get("/run/output")
async def run_output():
    return {"output": _agent_state["output"][-200:]}


@app.get("/model-presets")
async def model_presets():
    """返回模型预设（决策模型 + grounding 模型），供前端启动界面下拉选择。"""
    import yaml
    presets_path = _PKG / "config" / "model_presets.yaml"
    try:
        with open(presets_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return {
            "decision_models": data.get("decision_models") or [],
            "grounding_models": data.get("grounding_models") or [],
        }
    except Exception as exc:  # noqa: BLE001
        return {"decision_models": [], "grounding_models": [], "error": str(exc)}


@app.post("/run/manual-eval")
async def manual_eval():
    return {"ok": False, "error": "新框架暂不支持手动模式"}


# ================= 实验/调试 stub（防止前端报错）================= #
@app.get("/experiments/default-prompt")
async def get_default_prompt(backend: str = "native", browser_only: bool = False, os_name: str = ''):
    try:
        from osworld_agent.prompts import build_mode_system_prompts, build_editable_mode_prompts
        os_name = os_name or ('windows' if sys.platform == 'win32' else 'mac' if sys.platform == 'darwin' else 'linux')
        if os_name not in {'windows', 'mac', 'linux'}:
            return {'ok': False, 'error': 'Unsupported operating system'}
        templates = build_mode_system_prompts(os_name=os_name, browser_only=browser_only)
        return {"ok": True, "backend": "native",
                "os_name": os_name, "prompt": templates['dom'], "prompts": templates,
                "editable_prompts": build_editable_mode_prompts(os_name=os_name, browser_only=browser_only)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/experiments/memory-strategies")
async def get_memory_strategies():
    return {"strategies": []}


@app.get("/experiments/vision-strategies")
async def get_vision_strategies():
    return {"strategies": []}


@app.get("/experiments/prompts")
async def get_prompt_presets():
    return {"presets": []}


@app.get("/experiments/prompts/{name}")
async def get_prompt_preset(name: str):
    return Response(status_code=404)


@app.post("/experiments/prompts")
async def post_prompt_preset(request: Request):
    return {"ok": False, "error": "新框架未接入提示词预设"}


@app.get("/experiments/token-count")
async def get_token_count():
    return {"ok": False, "error": "新框架未接入 tokenizer 服务"}


@app.get("/debug/snapshots")
async def list_debug_snapshots(call_type: str = ""):
    return {"snapshots": []}


@app.get("/debug/snapshots/{snapshot_id}")
async def get_debug_snapshot(snapshot_id: str):
    return Response(status_code=404)


@app.get("/debug/images/{filename}")
async def get_debug_image(filename: str):
    return Response(status_code=404)


@app.post("/debug/replay")
async def replay_debug_snapshot(request: Request):
    return {"ok": False, "error": "新框架未接入模型调试重放"}


def _extract_grounding_coords(content):
    """解析 grounding 响应，返回 (coords, kind) 或 None。

    coords 为 [0,1000] 归一化坐标；kind 为 'point' 或 'bbox'。
    支持 [x,y]、[x1,y1,x2,y2]、{"bbox_2d":[...]}、{"point":[...]}、{"x":..,"y":..}。
    """
    if not content:
        return None
    text = str(content).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            if "bbox_2d" in obj and isinstance(obj["bbox_2d"], (list, tuple)) and len(obj["bbox_2d"]) >= 4:
                b = obj["bbox_2d"]
                return [float(b[0]), float(b[1]), float(b[2]), float(b[3])], "bbox"
            if "point" in obj and isinstance(obj["point"], (list, tuple)) and len(obj["point"]) >= 2:
                return [float(obj["point"][0]), float(obj["point"][1])], "point"
            if "x" in obj and "y" in obj:
                return [float(obj["x"]), float(obj["y"])], "point"
    except Exception:
        pass
    import re
    if re.search(r"[\[\(]\s*-1\s*,\s*-1\s*[\]\)]", text):
        return None
    m4 = re.search(r"[\[\(]\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*[\]\)]", text)
    if m4:
        x1, y1, x2, y2 = map(int, m4.groups())
        return [x1, y1, x2, y2], "bbox"
    m2 = re.search(r"[\[\(]\s*(-?\d+)\s*,\s*(-?\d+)\s*[\]\)]", text)
    if m2:
        x, y = map(int, m2.groups())
        if x < 0 or y < 0:
            return None
        return [x, y], "point"
    return None


def _first_image_url(messages):
    """从 messages 里找第一张图片的 data URL（base64）。"""
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "image_url":
                    u = (c.get("image_url") or {}).get("url", "")
                    if isinstance(u, str) and u.startswith("data:image"):
                        return u
        elif isinstance(content, str) and content.startswith("data:image"):
            return content
    return None


def _make_annotation_image(messages, coords, kind, is_pixel=False):
    """在原图上画标注（point=绿圈，bbox=红框+中心点），返回 data URL。

    is_pixel: True 表示 coords 已是像素坐标（如 UI-TARS），False 表示 [0,1000] 归一化（如 UI-Venus-2）。
    """
    url = _first_image_url(messages)
    if not url:
        return None
    try:
        import base64
        from io import BytesIO
        from PIL import Image, ImageDraw
        b64 = url.split(",", 1)[1]
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)

        def to_px(cx, cy):
            if is_pixel:
                return cx, cy
            return cx / 1000 * w, cy / 1000 * h

        if kind == "bbox":
            x1, y1 = to_px(coords[0], coords[1])
            x2, y2 = to_px(coords[2], coords[3])
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], outline=(0, 255, 0), width=4)
        else:
            cx, cy = to_px(coords[0], coords[1])
            r = max(6, min(w, h) // 60)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 255, 0), width=4)
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(0, 255, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


@app.post("/debug/chat")
async def debug_chat(request: Request):
    """会话调试：代理前端请求到任意 OpenAI 兼容模型端点（决策 / grounding 都走这里）。

    前端 payload: {engine:{engine_type,model,base_url}, messages, generation, api_key}
    返回: {"ok": True, "response": ...} 或 {"ok": False, "error": ...}
    """
    try:
        payload = await request.json()
    except Exception:
        return {"ok": False, "error": "请求体不是合法 JSON"}

    started = time.monotonic()
    result = await asyncio.to_thread(_debug_chat_request, payload)
    result["latency_ms"] = round((time.monotonic() - started) * 1000, 2)
    return result


def _debug_chat_request(payload):
    """Run blocking model transport outside the UI event loop."""
    if not isinstance(payload, dict):
        return {"ok": False, "error": "请求体必须是 JSON 对象"}
    engine = payload.get("engine") or {}
    engine_type = str(engine.get("engine_type") or "vllm").lower()
    model = str(engine.get("model") or "").strip()
    base_url = str(engine.get("base_url") or "").strip()
    messages = payload.get("messages") or []
    generation = payload.get("generation") or {}
    from osworld_agent.config import resolve_secret
    try:
        api_key = resolve_secret(payload.get("api_key") or "EMPTY")
    except (RuntimeError, TypeError) as exc:
        return {"ok": False, "error": str(exc)}

    if not model:
        return {"ok": False, "error": "缺少模型名"}
    if not base_url:
        return {"ok": False, "error": "缺少模型 URL"}

    # 规范化 messages：content 保持原样（字符串或 OpenAI vision 多部分列表都兼容）
    normalized = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        normalized.append({"role": role, "content": m.get("content", "")})
    if not normalized:
        return {"ok": False, "error": "messages 为空"}

    temperature = float(generation.get("temperature", 0.0))
    max_tokens = int(generation.get("max_new_tokens", 4096))

    if engine_type in ("vllm", "openai", "open_router"):
        import requests as req
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        body = {
            "model": model,
            "messages": normalized,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # 关闭 Qwen 思考模式，让 grounding 直接输出 [x,y] 坐标而非冗长推理
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last_err = None
        for attempt in range(2):
            try:
                # Empty strings explicitly disable inherited ALL_PROXY as well.
                # Public DashScope can reuse this deployment's configured egress;
                # private model endpoints remain direct unless explicitly configured.
                from urllib.parse import urlsplit
                proxy = os.getenv("AGENT_DEBUG_HTTP_PROXY", "")
                host = urlsplit(base_url).hostname or ""
                if not proxy and (host == "dashscope.aliyuncs.com" or host.endswith(".dashscope.aliyuncs.com")):
                    proxy = os.getenv("AGENT_KNOWLEDGE_HTTP_PROXY", "")
                r = req.post(url, json=body, headers=headers, timeout=300,
                             proxies={"http": proxy, "https": proxy, "all": ""})
                if r.status_code != 200:
                    return {"ok": False, "error": f"模型返回 HTTP {r.status_code}: {r.text[:300]}"}
                data = r.json()
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                # grounding 响应：解析坐标并在原图上画标注，返回可视化图
                parsed = _extract_grounding_coords(content)
                annotation = None
                if parsed and payload.get("call_type") in (None, "grounding"):
                    # UI-TARS 返回像素坐标，UI-Venus-2 返回 [0,1000] 归一化
                    is_pixel = "tars" in model.lower() or "uitars" in model.lower()
                    annotation = _make_annotation_image(normalized, parsed[0], parsed[1], is_pixel=is_pixel)
                return {"ok": True, "response": content, "annotation_image": annotation,
                        "usage": data.get("usage") or {}, "model": data.get("model") or model}
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                # 若端点不支持 chat_template_kwargs，去掉后重试一次
                if "chat_template_kwargs" in body:
                    body.pop("chat_template_kwargs", None)
                    continue
                break
        return {"ok": False, "error": f"调用失败: {last_err}"}

    return {"ok": False, "error": f"暂不支持的 provider: {engine_type}（当前支持 vllm / openai / open_router）"}


def _keep_model_warm():
    """可选的自托管模型保活；API 决策模型仅在显式给出 URL 时才保活。"""
    import requests as _req
    from osworld_agent.prompts import build_system_prompt
    sys_prompt = build_system_prompt(browser_only=True)
    dec_url = os.getenv("OSWORLD_DECISION_URL", "")
    dec_model = os.getenv("OSWORLD_DECISION_MODEL", "Qwen3.6-27B")
    grd_url = os.getenv("OSWORLD_GROUNDING_URL", "http://7.246.80.189:8001/v1/chat/completions")
    while True:
        if dec_url:
            try:
                _req.post(dec_url, json={"model": dec_model,
                                         "messages": [{"role": "system", "content": sys_prompt},
                                                      {"role": "user", "content": "keepalive"}],
                                         "max_tokens": 1}, timeout=120)
            except Exception:  # noqa: BLE001
                pass
        try:
            _req.post(grd_url, json={"model": "UI-Venus-2-9B",
                                     "messages": [{"role": "user", "content": "keepalive"}],
                                     "max_tokens": 1}, timeout=60)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(60)


if __name__ == "__main__":
    port = int(os.getenv("AGENTS_VIZ_PORT", "8088"))
    if os.getenv("OSWORLD_KEEP_MODEL_WARM") == "1":
        threading.Thread(target=_keep_model_warm, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
