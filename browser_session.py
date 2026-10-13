"""Chromium AX/DOM 观察、任务会话、浏览器事件和确定性动作。

持久 Playwright async 连接由独立线程拥有，事件在浏览器调用期间派发；下载作为
同一事件循环的限时后台任务运行，不阻塞下一次观察或主线程评测器。

策略输出语义目标 target_ref（快照节点引用），执行器按 backend_id 精确定位。
DOM 坐标是 CSS viewport 坐标，不是桌面截图像素。
"""
from __future__ import annotations

import os
import re
import time
import uuid
import math
import copy
import hashlib
import json
import threading
import asyncio
import inspect
import queue
from concurrent.futures import Future
from functools import wraps
from dataclasses import dataclass


class BrowserActionError(RuntimeError):
    def __init__(self, message, dispatched=False, code="blocked"):
        super().__init__(message)
        self.dispatched = dispatched
        self.code = code


class _BrowserWorker:
    """Serialize operations while also pumping browser events between model calls."""
    def __init__(self, owner):
        self.owner = owner
        self.queue = queue.Queue()
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="browser-session", daemon=True)
        self.thread.start()

    def submit(self, fn):
        if self.closed:
            raise RuntimeError("Browser worker is closed")
        future = Future()
        self.queue.put((fn, future))
        return future

    def _run(self):
        while True:
            try:
                work = self.queue.get(timeout=0.025)
            except queue.Empty:
                loop = self.owner._loop
                if loop is not None and not loop.is_closed():
                    try:
                        loop.run_until_complete(asyncio.sleep(0))
                    except Exception:
                        pass  # The next explicit browser operation reports transport errors.
                continue
            if work is None:
                return
            fn, future = work
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn())
                except BaseException as exc:
                    future.set_exception(exc)

    def shutdown(self, wait=True):
        self.closed = True
        self.queue.put(None)
        if wait and threading.get_ident() != self.thread.ident:
            self.thread.join()


def browser_thread(method):
    """Keep Playwright's sync event loop on one owner thread, away from evaluators."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if self._worker_thread == threading.get_ident():
            return method(self, *args, **kwargs)
        if self._worker is None:
            self._worker = _BrowserWorker(self)
        def invoke():
            self._worker_thread = threading.get_ident()
            return method(self, *args, **kwargs)
        return self._worker.submit(invoke).result()
    return wrapped


class _AsyncBrowserObject:
    """Small public-API adapter; every await stays on the session owner thread."""
    def __init__(self, owner, raw):
        self.owner, self.raw = owner, raw

    def __getattr__(self, name):
        value = getattr(self.raw, name)
        if not callable(value):
            return self.owner._wrap_async(value)
        def call(*args, **kwargs):
            if name in {"on", "once"} and len(args) == 2:
                event, callback = args
                args = (event, lambda *items: callback(*(self.owner._wrap_async(x) for x in items)))
            args = tuple(self.owner._unwrap_async(x) for x in args)
            kwargs = {k: self.owner._unwrap_async(v) for k, v in kwargs.items()}
            # Even synchronous public methods such as on() schedule protocol
            # subscriptions and therefore require a running asyncio loop.
            if self.owner._loop is not None and self.owner._loop.is_running():
                result = value(*args, **kwargs)
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise RuntimeError("Async browser operation cannot be awaited from a synchronous event callback")
            else:
                async def invoke():
                    result = value(*args, **kwargs)
                    return await result if inspect.isawaitable(result) else result
                timeout_ms = kwargs.get("timeout") or self.owner.options.timeout_ms
                result = self.owner._await(invoke(), max(float(timeout_ms) / 1000, 1) + 1)
            return self.owner._wrap_async(result)
        return call


INTERACTIVE = {"button", "link", "textbox", "searchbox", "checkbox", "radio", "combobox",
               "menuitem", "menuitemcheckbox", "menuitemradio", "tab", "switch", "slider",
               "spinbutton", "listbox", "option", "treeitem"}
SKIP = {"none", "generic", "InlineTextBox", "LineBreak"}


@dataclass
class BrowserOptions:
    endpoint: str = "http://127.0.0.1:9222"
    timeout_ms: int = 4000
    max_nodes: int = 4000
    max_snapshot_age: float = 180.0
    require_focus: bool = True   # True=要求浏览器前台聚焦；Windows 本地模式设 False（虚拟鼠标，用户可同时用电脑）
    download_timeout_s: float = 300.0
    download_max_bytes: int = 50 * 1024 * 1024  # Resource cap, not paper identity; 0 disables.

    def __post_init__(self):
        if type(self.download_max_bytes) is not int or self.download_max_bytes < 0:
            raise ValueError('download_max_bytes must be a nonnegative integer')


class BrowserSession:
    """Task-owned CDP session with a dedicated async Playwright owner thread."""

    def __init__(self, options=None, foreground_check=None, downloads_dir=None, workspace_dir=None):
        self.options = options or BrowserOptions()
        self.foreground_check = foreground_check
        if downloads_dir is not None and workspace_dir is not None:
            raise ValueError('Specify workspace_dir or legacy downloads_dir, not both')
        workspace_root = workspace_dir or downloads_dir or os.getenv('OSWORLD_WORKSPACE_DIR') or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "workspace")
        self.downloads_dir = os.path.abspath(os.path.expanduser(os.fspath(workspace_root)))
        self.snapshot = None
        self.nodes = {}
        self._downloaded = {}  # url -> path，本次会话已下载的文件（幂等去重）
        self._downloaded_sizes = {}
        self._task_page_id = None
        self._known_page_ids = set()
        self._observation_metadata = {}
        self.last_resolution_error = None
        self._worker = None
        self._worker_thread = None
        self._playwright = None
        self._browser = None
        self._owned_context = None
        self._watched_pages = set()
        self._watched_contexts = set()
        self._dialogs = {}
        self._pending_downloads = []
        self._artifacts = []
        self._download_events = []
        self._parents = {}
        self._base_downloads_dir = self.downloads_dir
        self._workspace_prepared = False
        self._workspace_state = {}
        self.initial_state = {}
        self._last_screenshot = None
        self._last_screenshot_metadata = {}
        self._temporary_cdp = []
        self._loop = None
        self._async_objects = {}
        self._background_downloads = set()
        self._active_downloads = {}
        self._download_tasks = {}
        self._download_objects = {}
        self._reserved_artifact_paths = set()
        self._main_responses = {}
        self._page_target_ids = {}
        self._artifact_events = []
        self._download_sizes = {}
        self._download_guids = {}
        self._download_monitors = []

    # ---- 临时连接的 attach 帮助 ----
    def _resolve_cdp_endpoint(self):
        """Resolve local/LAN CDP directly; process proxy settings belong to webpage traffic."""
        from urllib.parse import urlsplit, urlunsplit
        endpoint = self.options.endpoint.strip()
        parsed = urlsplit(endpoint)
        if parsed.scheme in {"ws", "wss"}:
            return endpoint
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("CDP endpoint must be an HTTP(S) or WS(S) URL")
        import httpx
        discovery = urlunsplit((parsed.scheme, parsed.netloc,
            parsed.path.rstrip("/") + "/json/version/", parsed.query, ""))
        with httpx.Client(trust_env=False, timeout=max(0.1, self.options.timeout_ms / 1000)) as client:
            response = client.get(discovery)
            response.raise_for_status()
            websocket = response.json().get("webSocketDebuggerUrl", "")
        if not isinstance(websocket, str) or urlsplit(websocket).scheme not in {"ws", "wss"}:
            raise ValueError("CDP discovery did not provide a WebSocket debugger URL")
        return websocket

    def _connect(self):
        if self._browser is not None:
            return self._playwright, self._browser
        endpoint = self._resolve_cdp_endpoint()
        from playwright.async_api import async_playwright
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        p = self._wrap_async(self._await(async_playwright().start()))
        browser = None
        try:
            browser = p.chromium.connect_over_cdp(endpoint, timeout=self.options.timeout_ms)
            self._watch_contexts(browser)
        except Exception:
            if browser is not None:
                try:
                    browser.close()
                except Exception:
                    pass
            try:
                p.stop()
            except Exception:
                pass
            self._async_objects.clear()
            raise
        self._playwright, self._browser = p, browser
        return p, browser

    def _await(self, coroutine, timeout=None):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        if self._loop.is_running():
            raise RuntimeError("Awaiting browser operations inside synchronous event callbacks is unsupported")
        return self._loop.run_until_complete(asyncio.wait_for(
            coroutine, timeout if timeout is not None else max(1, self.options.timeout_ms / 1000) + 1))

    def _wrap_async(self, value):
        if isinstance(value, list):
            return [self._wrap_async(x) for x in value]
        if isinstance(value, tuple):
            return tuple(self._wrap_async(x) for x in value)
        if type(value).__module__.startswith("playwright.async_api"):
            if id(value) not in self._async_objects:
                self._async_objects[id(value)] = _AsyncBrowserObject(self, value)
            return self._async_objects[id(value)]
        return value

    def _unwrap_async(self, value):
        if isinstance(value, _AsyncBrowserObject):
            return value.raw
        if isinstance(value, (list, tuple)):
            return [self._unwrap_async(x) for x in value]
        if isinstance(value, dict):
            return {k: self._unwrap_async(v) for k, v in value.items()}
        return value

    def _contexts(self, browser):
        return [self._owned_context] if self._owned_context is not None else browser.contexts

    def _cdp(self, page):
        session = page.context.new_cdp_session(page)
        self._temporary_cdp.append(session)
        return session

    def _watch_contexts(self, browser):
        for context in self._contexts(browser):
            if id(context) not in self._watched_contexts and hasattr(context, "on"):
                context.on("page", self._watch_page)
                context.on("dialog", lambda dialog: self._record_dialog(dialog.page, dialog)
                           if getattr(dialog, "page", None) is not None else None)
                self._watched_contexts.add(id(context))
            for page in context.pages:
                self._watch_page(page)

    def _watch_page(self, page):
        if id(page) in self._watched_pages or not hasattr(page, "on"):
            return
        self._watched_pages.add(id(page))
        page.on("dialog", lambda dialog: self._record_dialog(page, dialog))
        page.on("download", lambda download: self._queue_download(page, download))
        page.on("response", lambda response: self._record_response(page, response))
        self._install_download_monitor(page)

    def _install_download_monitor(self, page):
        """Observe Chromium headers/progress without changing download destinations."""
        if self._loop is None:
            return
        async def install():
            try:
                raw = self._unwrap_async(page)
                cdp = await raw.context.new_cdp_session(raw)
                self._download_monitors.append(cdp)
                cdp.on('Network.responseReceived', lambda e: self._record_download_headers(e.get('response', {})))
                cdp.on('Page.downloadWillBegin', lambda e: self._download_guids.update({e['guid']: e.get('url', '')}))
                cdp.on('Page.downloadProgress', self._record_download_progress)
                await cdp.send('Network.enable')
                await cdp.send('Page.enable')
            except Exception:
                pass  # Final saved-size validation and operation deadline remain enforced.
        task = self._loop.create_task(install())
        self._background_downloads.add(task)
        task.add_done_callback(self._background_downloads.discard)

    def _record_download_headers(self, response):
        headers = {str(k).lower(): str(v) for k, v in response.get('headers', {}).items()}
        size = headers.get('content-length', '')
        if size.isdigit():
            self._track_download_size(response.get('url', ''), int(size))

    def _record_download_progress(self, event):
        url = self._download_guids.get(event.get('guid'))
        if url:
            sizes = [x for x in (event.get('totalBytes'), event.get('receivedBytes')) if type(x) in (int, float)]
            if sizes:
                self._track_download_size(url, max(sizes))
        if event.get('state') in {'completed', 'canceled'}:
            self._download_guids.pop(event.get('guid'), None)

    def _track_download_size(self, url, size):
        if not url:
            return
        key = self._url_key(url)
        self._download_sizes[key] = max(size, self._download_sizes.get(key, 0))
        if len(self._download_sizes) > 2048:
            self._download_sizes.pop(next(iter(self._download_sizes)))
        limit = self.options.download_max_bytes
        if limit and size > limit:
            for ident, item in list(self._active_downloads.items()):
                if self._url_key(item['url']) == key and ident in self._download_objects and not item.get('error_code'):
                    item['error_code'] = 'download_size_limit'
                    self._cancel_observed_download(self._download_objects[ident])

    def _record_response(self, page, response):
        """Keep headers of the current main resource, never its body."""
        try:
            if response.request.is_navigation_request() and response.frame is page.main_frame:
                headers = response.headers
                self._main_responses[id(page)] = {"url": response.url,
                    "content_type": headers.get("content-type", "").split(";", 1)[0].strip().lower(),
                    "status": response.status}
        except Exception:
            pass  # Some browser-internal responses do not expose a frame.

    def _record_dialog(self, page, dialog):
        if self._owned_context is not None and page.context is not self._owned_context:
            return
        if any(item["object"] is dialog for item in self._dialogs.values()):
            return
        dialog_id = uuid.uuid4().hex
        self._dialogs[dialog_id] = {"object": dialog, "page": page,
            "id": dialog_id, "type": dialog.type, "message": dialog.message,
            "default_value": dialog.default_value}

    def _queue_download(self, page, download):
        if self._owned_context is not None and page.context is not self._owned_context:
            return
        # Browser-generated signed URLs are kept intact. Only exact URL aliases
        # actually observed in a completed transfer are considered equivalent.
        self._observe_artifact_renames()
        duplicate = self._find_pending_download(download.url) or self._cached_artifact(download.url)
        if duplicate is not None:
            self._cancel_observed_download(download)
            self._download_events.append({'status': 'deduplicated', 'completed': bool(duplicate.get('completed')),
                'url': download.url, 'download_id': duplicate.get('download_id'),
                'artifact_id': duplicate.get('artifact_id'), 'path': duplicate.get('path'),
                'reason': 'same URL already pending or downloaded', 'finished_at': time.time()})
            return
        if (self.options.download_max_bytes and
                self._download_sizes.get(self._url_key(download.url), 0) > self.options.download_max_bytes):
            self._cancel_observed_download(download)
            self._download_events.append({'download_id': uuid.uuid4().hex, 'url': download.url,
                'page_url': page.url, 'status': 'failed', 'completed': False, 'finished_at': time.time(),
                'error_code': 'download_size_limit', 'error': '下载超过配置的字节上限；请核对链接或调整资源上限。'})
            return
        download_id = uuid.uuid4().hex
        source_page_url = page.url
        started_at = time.time()
        self._active_downloads[download_id] = {"download_id": download_id, "artifact_id": download_id,
            "url": download.url, "page_url": source_page_url, "page_id": self._page_target_ids.get(id(page)),
            "status": "in_progress", "completed": False, "started_at": started_at,
            "deadline_at": started_at + self.options.download_timeout_s, "transport": "browser"}
        if self._loop is None:  # Deterministic in-memory transports used by offline tests.
            self._pending_downloads.append((page, download, download_id, source_page_url))
            return
        try:
            path = self._artifact_path(download.suggested_filename, download_id)
        except OSError as exc:
            self._active_downloads.pop(download_id, None)
            self._download_events.append({"download_id": download_id, "artifact_id": download_id,
                "status": "failed", "completed": False, "url": download.url,
                "finished_at": time.time(), "error": str(exc)[:300]})
            return
        self._download_objects[download_id] = download
        raw = self._unwrap_async(download)
        deadline = time.monotonic() + self.options.download_timeout_s
        async def save():
            try:
                await asyncio.wait_for(raw.save_as(path), max(0, deadline - time.monotonic()))
                self._record_artifact(path=path, url=download.url, final_url=download.url,
                                      download_id=download_id, page_url=source_page_url)
            except BaseException as exc:
                try:
                    await asyncio.wait_for(raw.cancel(), 5)
                except BaseException:
                    pass
                self._download_events.append({"download_id": download_id, "artifact_id": download_id,
                    "status": "failed" if self._active_downloads.get(download_id, {}).get('error_code') else "canceled" if isinstance(exc, asyncio.CancelledError) else "failed",
                    "error_code": self._active_downloads.get(download_id, {}).get('error_code') or getattr(exc, 'code', ''),
                    "completed": False, "url": download.url,
                    "finished_at": time.time(),
                    "error": ('下载超过配置的字节上限；请核对链接或调整资源上限。'
                              if self._active_downloads.get(download_id, {}).get('error_code') == 'download_size_limit'
                              else "download deadline exceeded" if isinstance(exc, asyncio.TimeoutError) else str(exc)[:200] or type(exc).__name__)})
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except OSError:
                    pass
            finally:
                self._active_downloads.pop(download_id, None)
                self._download_tasks.pop(download_id, None)
                self._download_objects.pop(download_id, None)
                self._reserved_artifact_paths.discard(path)
        task = self._loop.create_task(save())
        self._download_tasks[download_id] = task
        self._background_downloads.add(task)
        task.add_done_callback(self._background_downloads.discard)

    def _cancel_observed_download(self, download):
        if self._loop is None:
            try:
                download.cancel()
            except Exception:
                pass
            return
        async def cancel():
            try:
                await asyncio.wait_for(self._unwrap_async(download).cancel(), 5)
            except Exception:
                pass
        task = self._loop.create_task(cancel())
        self._background_downloads.add(task)
        task.add_done_callback(self._background_downloads.discard)

    @browser_thread
    def prevent_duplicate_download(self, action, observation, route=None):
        from .browser_feedback import download_action_url
        url = download_action_url(action, observation, route)
        if not url:
            return None
        self._observe_artifact_renames()
        artifact = self._cached_artifact(url)
        if artifact:
            return {'status': 'rejected', 'dispatched': False, 'error_code': 'already_downloaded',
                    'reason': '此 URL 的已登记文件仍有效，已拦截重复下载；请处理改名或后续验收。',
                    'artifact_id': artifact['artifact_id'], 'path': artifact['path'], 'url': url}
        return None

    @staticmethod
    def _url_key(url):
        from urllib.parse import urldefrag
        return urldefrag(str(url))[0]  # Fragments never participate in HTTP transfer.

    def _find_pending_download(self, url):
        key = self._url_key(url)
        return next((d for d in self._active_downloads.values()
                     if self._url_key(d["url"]) == key), None)

    def _cached_artifact(self, url):
        key = self._url_key(url)
        for artifact in reversed(self._artifacts):
            if not any(self._url_key(artifact.get(k, "")) == key for k in ("url", "final_url")):
                continue
            try:
                stat = os.stat(artifact['path'], follow_symlinks=False)
                if (os.path.isfile(artifact['path']) and not os.path.islink(artifact['path'])
                        and [stat.st_dev, stat.st_ino] == artifact.get('file_identity')
                        and stat.st_size == artifact.get('bytes') and self._file_hash(artifact['path']) == artifact['sha256']):
                    return artifact
            except OSError:
                pass
        return None

    @browser_thread
    def cancel_download(self, download_id):
        """Cancel one observed transfer; never restart its initiating action."""
        task = self._download_tasks.get(download_id)
        if task is None:
            return self._failure("No active download with this ID", "not_found", action="cancel_download")
        async def cancel():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._await(cancel(), 6)
        return {"status": "executed", "dispatched": True, "action": "cancel_download",
                "download_id": download_id, "artifact_id": download_id, **self.event_metadata()}

    @staticmethod
    def _filename(value):
        from urllib.parse import unquote
        value = unquote(value or "download").replace("\\", "/").rsplit("/", 1)[-1]
        value = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", value).strip(" .")[:180] or "download"
        return "_" + value if re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", value, re.I) else value

    def _artifact_path(self, filename, download_id):
        os.makedirs(self.downloads_dir, exist_ok=True)
        name = self._filename(filename)
        path = os.path.join(os.path.abspath(self.downloads_dir), name)
        if os.path.lexists(path) or path in self._reserved_artifact_paths:
            stem, ext = os.path.splitext(name)
            path = os.path.join(os.path.abspath(self.downloads_dir), f"{stem}_{download_id[:12]}{ext}")
        while os.path.lexists(path) or path in self._reserved_artifact_paths:
            stem, ext = os.path.splitext(name)
            path = os.path.join(os.path.abspath(self.downloads_dir), f"{stem}_{uuid.uuid4().hex[:12]}{ext}")
        self._reserved_artifact_paths.add(path)
        return path

    @staticmethod
    def _file_hash(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _record_artifact(self, *, path, url, final_url, download_id, page_url="", content_type="", transport="browser"):
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            raise ValueError("download did not produce a nonempty regular file")
        if self.options.download_max_bytes and os.path.getsize(path) > self.options.download_max_bytes:
            raise BrowserActionError('下载超过配置的字节上限；请核对链接或调整资源上限。', code='download_size_limit')
        artifact = {"source_url": url, "url": url, "final_url": final_url,
            "path": os.path.abspath(path), "status": "completed", "completed": True,
            "download_id": download_id, "artifact_id": download_id, "version": 1, "rename_history": [],
            "page_url": page_url, "content_type": content_type,
            "bytes": os.path.getsize(path), "transport": transport, "completed_at": time.time()}
        artifact["sha256"] = self._file_hash(path)
        artifact["initial_sha256"] = artifact["sha256"]
        stat = os.stat(path, follow_symlinks=False)
        artifact["file_identity"] = [stat.st_dev, stat.st_ino]
        self._artifacts.append(artifact)
        self._downloaded[url] = artifact["path"]
        self._downloaded_sizes[url] = artifact["bytes"]
        return copy.deepcopy(artifact)

    @classmethod
    def _explicit_filename(cls, value):
        if not isinstance(value, str) or not value or cls._filename(value) != value:
            raise ValueError("filename must be a safe basename (no paths or reserved characters)")
        return value

    @browser_thread
    def rename_artifact(self, artifact_id, filename):
        """Compatibility rejection: filenames must be changed through the interface."""
        return self._failure("Use Save As or the file manager UI to rename files", "unsupported",
                             action="rename_artifact", artifact_id=artifact_id)

    def _observe_artifact_renames(self):
        """Track UI renames read-only, within this task's directory, without rebaselining content."""
        missing = [a for a in self._artifacts if not os.path.lexists(a["path"]) and a.get("file_identity")]
        if not missing:
            return
        try:
            with os.scandir(self.downloads_dir) as entries:
                candidates = [(os.path.abspath(e.path), e.stat(follow_symlinks=False))
                              for e in entries if e.is_file(follow_symlinks=False)]
        except OSError:
            return
        for artifact in missing:
            identity = artifact["file_identity"]
            matches = [(path, stat) for path, stat in candidates if [stat.st_dev, stat.st_ino] == identity
                       and stat.st_size == artifact["bytes"] and stat.st_nlink == 1]
            if len(matches) != 1 or not identity[1]:
                continue
            destination, _ = matches[0]
            try:
                digest = self._file_hash(destination)
                after = os.stat(destination, follow_symlinks=False)
                if digest != artifact["initial_sha256"] or [after.st_dev, after.st_ino] != identity:
                    continue
            except OSError:
                continue
            source, version = artifact["path"], artifact.get("version", 1)
            migration = {"from_path": source, "to_path": destination, "from_version": version,
                         "to_version": version + 1, "sha256": digest, "renamed_at": time.time(), "source": "observed_ui_rename"}
            artifact.setdefault("rename_history", []).append(migration)
            artifact.update(path=destination, version=version + 1)
            for url, path in list(self._downloaded.items()):
                if path == source:
                    self._downloaded[url] = destination
            self._artifact_events.append({"event": "artifact_renamed", "artifact_id": artifact["artifact_id"],
                                          "download_id": artifact["download_id"], **migration})

    def _drain_downloads(self):
        if self._loop is not None:
            self._await(asyncio.sleep(0))
            return
        pending, self._pending_downloads = self._pending_downloads, []
        for page, download, download_id, source_page_url in pending:
            path = None
            try:
                path = self._artifact_path(download.suggested_filename, download_id)
                download.save_as(path)
                self._record_artifact(path=path, url=download.url, final_url=download.url,
                                      download_id=download_id, page_url=source_page_url)
            except Exception as exc:
                self._cancel_observed_download(download)
                self._download_events.append({"download_id": download_id, "status": "failed",
                    "completed": False, "url": download.url, "error": str(exc)[:300],
                    "error_code": getattr(exc, 'code', '')})
                if path:
                    try:
                        if os.path.isfile(path):
                            os.remove(path)
                    except OSError:
                        pass
            finally:
                self._active_downloads.pop(download_id, None)
                self._reserved_artifact_paths.discard(path)

    @browser_thread
    def event_metadata(self):
        self._observe_artifact_renames()
        # Frequent lightweight polling can keep the worker queue nonempty, so
        # do not rely solely on its 25 ms idle timeout to pump browser events.
        if self._loop is not None and not self._loop.is_running():
            self._await(asyncio.sleep(0))
        return {"artifacts": copy.deepcopy(self._artifacts),
            "download_events": copy.deepcopy(self._download_events + list(self._active_downloads.values())),
            "pending_downloads": copy.deepcopy(list(self._active_downloads.values())),
            "artifact_events": copy.deepcopy(self._artifact_events),
            "dialogs": [{k: v for k, v in d.items() if k not in {"object", "page"}}
                        for d in self._dialogs.values()], "initial_state": copy.deepcopy(self.initial_state)}

    def _cancel_background_downloads(self):
        if self._loop is None or not self._background_downloads:
            return
        tasks = list(self._background_downloads)
        async def cancel_all():
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            self._await(cancel_all(), 6)
        finally:
            self._background_downloads.clear()
            self._active_downloads.clear()
            self._download_tasks.clear()

    @browser_thread
    def prepare_workspace(self, task_id=None):
        """Allocate once before planning; begin_episode adopts the same directory."""
        if not self._workspace_prepared:
            episode_id = str(task_id or uuid.uuid4().hex)
            directory_id = self._filename(episode_id) + '_' + uuid.uuid4().hex[:12]
            self.downloads_dir = os.path.join(self._base_downloads_dir, directory_id)
            os.makedirs(self.downloads_dir, exist_ok=False)
            self._workspace_state = {'episode_id': episode_id, 'workspace_dir': self.downloads_dir,
                'downloads_dir': self.downloads_dir, 'workspace_root': self._base_downloads_dir,
                'workspace_scope': 'episode', 'workspace_location': 'controller_host'}
            self._workspace_prepared = True
        return copy.deepcopy(self._workspace_state)

    @browser_thread
    def begin_episode(self, task_id=None, isolated=True, viewport=None):
        """Create a fresh context and output directory; never silently reuse on failure."""
        if not isolated:
            raise ValueError("begin_episode requires isolated=True")
        self.initial_state = {}
        _, browser = self._connect()
        self._cancel_background_downloads()
        if self._owned_context is not None:
            self._owned_context.close()
            self._owned_context = None
        self._async_objects = {id(item.raw): item for item in (self._browser, self._playwright)
                               if isinstance(item, _AsyncBrowserObject)}
        self.invalidate()
        self._task_page_id = None
        self._known_page_ids.clear()
        self._watched_pages.clear()
        self._watched_contexts.clear()
        self._parents.clear()
        self._dialogs.clear()
        self._pending_downloads.clear()
        self._artifacts.clear()
        self._download_events.clear()
        self._downloaded.clear()
        self._downloaded_sizes.clear()
        self._reserved_artifact_paths.clear()
        self._main_responses.clear()
        self._page_target_ids.clear()
        self._artifact_events.clear()
        self._download_objects.clear()
        self._download_sizes.clear()
        self._download_guids.clear()
        self._download_monitors.clear()
        self._last_screenshot = None
        self._last_screenshot_metadata = {}
        size = viewport or {"width": 1280, "height": 720}
        workspace = self.prepare_workspace(task_id)
        episode_id = workspace['episode_id']
        context = browser.new_context(accept_downloads=True, viewport=size, locale="en-US",
            timezone_id="UTC", color_scheme="light", reduced_motion="reduce", service_workers="block",
            ignore_https_errors=True)
        self._owned_context = context
        self._watch_contexts(browser)
        page = context.new_page()
        self._watch_page(page)
        # Other contexts (even about:blank pages) are not ours. A CDP attachment
        # does not authorize closing them; only _owned_context is cleaned above.
        self._task_page_id = self._target_info(page)["targetId"]
        self._page_target_ids[id(page)] = self._task_page_id
        profile = {"version": "blank-browser-v1", "viewport": size,
                   "browser_version": str(getattr(browser, "version", "unknown")),
                   "locale": "en-US", "timezone": "UTC", "color_scheme": "light",
                   "reduced_motion": "reduce", "service_workers": "block", "storage": "empty"}
        fingerprint = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:20]
        self.initial_state = {"isolated": True, "isolation_scope": "browser_context",
            "initial_state_id": "blank-browser-v1:" + fingerprint,
            "episode_id": episode_id, "context_id": uuid.uuid4().hex,
            **workspace, "profile": profile}
        self._workspace_prepared = False
        return copy.deepcopy(self.initial_state)

    def _tab_records(self, browser):
        records = []
        for context in self._contexts(browser):
            for page in context.pages:
                try:
                    item = self._target_info(page)
                    self._page_target_ids[id(page)] = item["targetId"]
                    self._parents[item["targetId"]] = item.get("openerId")
                    records.append({"tab_id": item["targetId"], "url": page.url,
                        "title": item.get("title", ""),
                        "opener_id": item.get("openerId"), "active": item["targetId"] == self._task_page_id})
                except Exception:
                    continue
        return records

    @browser_thread
    def list_tabs(self):
        _, browser = self._connect()
        self._pick_page(browser)
        return {"status": "executed", "dispatched": False, "action": "list_tabs", "tabs": self._tab_records(browser)}

    @browser_thread
    def switch_tab(self, tab_id):
        _, browser = self._connect()
        if tab_id not in {r["tab_id"] for r in self._tab_records(browser)}:
            return self._failure("Unknown task tab", "not_found")
        self._task_page_id = tab_id
        if self.options.require_focus:
            for context in self._contexts(browser):
                for page in context.pages:
                    if self._target_info(page)["targetId"] == tab_id:
                        page.bring_to_front()
        self.invalidate()
        return {"status": "executed", "dispatched": False, "action": "switch_tab", "tab_id": tab_id}

    @browser_thread
    def close_tab(self, tab_id=None):
        _, browser = self._connect()
        tab_id = tab_id or self._task_page_id
        for context in self._contexts(browser):
            for page in list(context.pages):
                if self._target_info(page)["targetId"] == tab_id:
                    parent = self._parents.get(tab_id)
                    page.close()
                    if self._task_page_id == tab_id:
                        self._task_page_id = parent
                    self.invalidate()
                    return {"status": "executed", "dispatched": True, "action": "close_tab", "tab_id": tab_id}
        return self._failure("Unknown task tab", "not_found")

    @browser_thread
    def handle_dialog(self, accept, prompt_text=None, dialog_id=None):
        if type(accept) is not bool:
            return self._failure("accept must be boolean", "unsupported")
        if dialog_id is None:
            if len(self._dialogs) != 1:
                return self._failure("Provide dialog_id when dialog is absent or ambiguous", "ambiguous")
            dialog_id = next(iter(self._dialogs))
        item = self._dialogs.get(dialog_id)
        if item is None:
            return self._failure("Unknown dialog", "not_found")
        try:
            if accept:
                item["object"].accept(prompt_text=prompt_text) if prompt_text is not None else item["object"].accept()
            else:
                item["object"].dismiss()
            del self._dialogs[dialog_id]
            self.invalidate()
            return {"status": "executed", "dispatched": True, "action": "handle_dialog", "dialog_id": dialog_id}
        except Exception as exc:
            return self._failure(exc, dispatched=True)

    @staticmethod
    def _focused_page(browser):
        """找聚焦的 http 页面；不唯一则返回 None。"""
        candidates = []
        for ctx in browser.contexts:
            for page in ctx.pages:
                if not page.url.startswith(("http://", "https://", "file://")):
                    continue
                try:
                    if page.evaluate("document.hasFocus() && document.visibilityState === 'visible'"):
                        candidates.append(page)
                except Exception:
                    continue
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _any_page(browser):
        """随便挑一个页面：优先 http 网页，其次任意页面（含 about:blank，可导航）。"""
        for ctx in browser.contexts:
            for page in ctx.pages:
                if page.url.startswith(("http://", "https://", "file://")):
                    return page
            if ctx.pages:
                return ctx.pages[0]
        return None

    def _pick_page(self, browser):
        """Keep the selected task page; new popups require an explicit tab switch."""
        pages = []
        self._watch_contexts(browser)
        for context in self._contexts(browser):
            for page in context.pages:
                try:
                    info = self._target_info(page)
                    self._page_target_ids[id(page)] = info["targetId"]
                    pages.append((page, info))
                    self._parents[info["targetId"]] = info.get("openerId")
                except Exception:
                    continue
        if self._task_page_id is None:
            chosen = (pages[0][0] if self._owned_context is not None and pages else
                      self._focused_page(browser) if self.options.require_focus else self._any_page(browser))
            match = next(((p, i) for p, i in pages if p is chosen), None)
            if match is None:
                return None
            self._task_page_id = match[1]["targetId"]
        self._known_page_ids.update(i["targetId"] for _, i in pages)
        page = next((p for p, i in pages if i["targetId"] == self._task_page_id), None)
        if page is None and self._task_page_id:
            parent = self._parents.get(self._task_page_id)
            page = next((p for p, i in pages if i["targetId"] == parent), None)
            if page is not None:
                self._task_page_id = parent
        if page is not None and self.options.require_focus and not any(d["page"] is page for d in self._dialogs.values()):
            try:
                if not page.evaluate("document.hasFocus() && document.visibilityState === 'visible'"):
                    return None
            except Exception:
                return None
        return page

    @staticmethod
    def _target_info(page, cdp=None):
        own = cdp is None
        cdp = cdp or page.context.new_cdp_session(page)
        try:
            return cdp.send("Target.getTargetInfo")["targetInfo"]
        finally:
            if own:
                try:
                    cdp.detach()
                except Exception:
                    pass

    def _identity(self, page, cdp):
        page_id = self._target_info(page, cdp)["targetId"]
        self._page_target_ids[id(page)] = page_id
        return {"page_id": page_id,
                "document_id": cdp.send("DOM.getDocument", {"depth": 0})["root"]["backendNodeId"]}

    @staticmethod
    def _failure(message, code="blocked", dispatched=False, started=None, **extra):
        result = {"status": "uncertain" if dispatched else "rejected", "dispatched": dispatched,
                  "channel": "browser_dom", "error": str(message)[:500],
                  "error_code": "uncertain" if dispatched else code, **extra}
        if started is not None:
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    def _disconnect(self, p, browser):
        sessions, self._temporary_cdp = self._temporary_cdp, []
        for session in sessions:
            try:
                session.detach()
            except Exception:
                pass
        if browser is self._browser:
            return
        for owner, method in ((browser, "close"), (p, "stop")):
            try:
                getattr(owner, method)()
            except Exception:
                pass

    # ---- collect ----
    def collect(self):
        return self._collect(False)[1]

    def collect_observation(self):
        """同一连接、页面和文档的 (CSS 截图, AX 快照, 观察元数据)。"""
        return self._collect(True)

    _RESOURCE_METADATA = """() => {
      const links = Array.from(document.querySelectorAll('a[href]')).filter(a =>
        a.hasAttribute('download') || /pdf|download|下载/i.test(a.textContent || '') ||
        /\\.pdf(?:$|[?#])|\\/pdf\\//i.test(a.href)).slice(0, 100).map(a => ({
          url:a.href, source:'anchor', label:(a.textContent || '').trim().slice(0,300),
          download:a.hasAttribute('download'), type:a.getAttribute('type') || ''}));
      const embeds = Array.from(document.querySelectorAll('embed,object')).slice(0,20).map(e => ({
        type:e.getAttribute('type') || '', src:e.getAttribute('src') || e.getAttribute('data') || '',
        original_url:e.getAttribute('original-url') || ''}));
      const citation = {};
      for (const e of document.querySelectorAll('meta[name]')) {
        const key = e.getAttribute('name').toLowerCase();
        if (['citation_title','citation_author','citation_pdf_url','citation_arxiv_id',
             'citation_doi','citation_date','citation_publication_date'].includes(key)) {
          (citation[key] ||= []).push(e.getAttribute('content') || '');
        }
      }
      return {content_type:document.contentType || '', base_url:document.baseURI,
              links, embeds, citation, pdf_viewer:!!document.querySelector('pdf-viewer')};
    }"""

    def _resource_metadata(self, page, identity):
        """Small resource descriptors derived from observed headers/DOM, not PDF text."""
        from urllib.parse import urljoin, urlsplit
        try:
            value = page.evaluate(self._RESOURCE_METADATA)
        except Exception:
            value = None
        value = value if isinstance(value, dict) else {}
        response = self._main_responses.get(id(page), {})
        if response.get("url") != page.url:
            response = {}
        content_type = str(response.get("content_type") or value.get("content_type") or "").split(";", 1)[0].lower()
        candidates, seen = [], set()
        def add(url, source, **details):
            if not isinstance(url, str) or not url:
                return
            resolved = urljoin(value.get("base_url") or page.url, url)
            try:
                parsed = urlsplit(resolved)
            except ValueError:
                return
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or (resolved, source) in seen:
                return
            seen.add((resolved, source))
            candidates.append({"url": resolved, "source": source, **identity, **details})
        pdf_embed = False
        for embed in value.get("embeds", []):
            if str(embed.get("type", "")).lower() in {"application/pdf", "application/x-google-chrome-pdf"}:
                pdf_embed = True
                add(embed.get("original_url"), "embed_original_url", resource_kind="pdf")
                add(embed.get("src"), "embed", resource_kind="pdf")
        viewer = pdf_embed or value.get("pdf_viewer") is True
        metadata = {"download_candidates": candidates}
        if content_type:
            metadata["content_type"] = content_type
        if viewer:
            metadata["document_kind"] = "pdf_viewer"
        elif content_type == "application/pdf":
            metadata["document_kind"] = "pdf"
        elif content_type in {"text/html", "application/xhtml+xml"}:
            metadata["document_kind"] = "html"
        # A .pdf URL is evidence of a candidate, never proof of the returned MIME.
        # An HTML viewer/embedding page is not itself the embedded PDF resource.
        if content_type == "application/pdf" or re.search(r"\.pdf(?:$|[?#])", page.url, re.I):
            add(page.url, "current_url", resource_kind="pdf", mime_confirmed=content_type == "application/pdf")
        for link in value.get("links", []):
            add(link.get("url"), "anchor", label=str(link.get("label") or "")[:300],
                download_attribute=link.get("download") is True)
        citation = value.get("citation") or {}
        for url in citation.get("citation_pdf_url", []):
            add(url, "citation_pdf_url", resource_kind="pdf")
        parsed = urlsplit(page.url)
        arxiv = parsed.hostname in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
        if arxiv and parsed.path.startswith("/pdf/"):
            add(page.url, "current_url", resource_kind="pdf", mime_confirmed=content_type == "application/pdf")
        if arxiv and citation:
            first = lambda key: next((str(x) for x in citation.get(key, []) if x), "")
            canonical_id = first("citation_arxiv_id")
            if not canonical_id:
                match = re.match(r"/(?:abs|pdf)/(.+?)(?:\.pdf)?$", parsed.path)
                canonical_id = match.group(1) if match else ""
            metadata["paper_metadata"] = {"source_url": page.url, "canonical_id": canonical_id,
                "title": first("citation_title"), "authors": citation.get("citation_author", []),
                "pdf_url": urljoin(value.get("base_url") or page.url, first("citation_pdf_url"))
                           if first("citation_pdf_url") else "",
                "doi": first("citation_doi"), "source": "observed_arxiv_meta"}
        return metadata

    @browser_thread
    def _collect(self, with_screenshot):
        start = time.perf_counter()
        self.invalidate()
        if self.foreground_check and not self.foreground_check():
            return None, None, {"error_code": "blocked", "error": "Browser lost focus"}
        try:
            p, browser = self._connect()
        except Exception as exc:
            return None, None, {"error_code": "not_found", "error": str(exc)[:200]}
        try:
            page = self._pick_page(browser)
            if page is None:
                return None, None, {"error_code": "not_found", "error": "No task page"}
            page.wait_for_timeout(1)  # dispatch pending browser events before observing
            dialogs = [d for d in self._dialogs.values() if d["page"] is page]
            if dialogs:
                metadata = {**self._last_screenshot_metadata, **self.event_metadata(),
                            "dialog": {k: v for k, v in dialogs[-1].items() if k not in {"object", "page"}},
                            "screenshot_stale": True, "observation_errors": ["JavaScript dialog blocks page observation"]}
                return self._last_screenshot, None, metadata
            self._drain_downloads()
            cdp = self._cdp(page)
            identity = self._identity(page, cdp)
            url = page.url
            snapshot, shot, errors = None, None, []
            try:
                snapshot = self._collect_page(page, cdp, identity, start)
            except Exception as exc:
                errors.append(str(exc)[:200])
            resources = self._resource_metadata(page, identity)
            viewport_script = "({width:innerWidth,height:innerHeight,x:scrollX,y:scrollY,dpr:devicePixelRatio})"
            capture_viewport = page.evaluate(viewport_script)
            if with_screenshot:
                try:
                    import io
                    from PIL import Image
                    shot = Image.open(io.BytesIO(page.screenshot(full_page=False, scale="css")))
                    shot.load()
                except Exception as exc:
                    errors.append(str(exc)[:200])
            if identity != self._identity(page, cdp) or page.url != url:
                raise BrowserActionError("Page changed during observation", code="stale")
            final_viewport = page.evaluate(viewport_script)
            if any(capture_viewport[k] != final_viewport[k] for k in ("width", "height", "x", "y")):
                raise BrowserActionError("Viewport changed during observation", code="stale")
            metadata = {**identity, "snapshot_id": snapshot["snapshot_id"] if snapshot else uuid.uuid4().hex[:12],
                        "captured_at": time.time(), "url": page.url,
                        "scroll_offset": [final_viewport["x"], final_viewport["y"]],
                        **resources,
                        **self.event_metadata(), "tabs": self._tab_records(browser)}
            for candidate in metadata.get("download_candidates", []):
                candidate["snapshot_id"] = metadata["snapshot_id"]
            if shot is not None:
                metadata.update(screenshot_space="css_viewport", screenshot_size=list(shot.size))
            if errors:
                metadata["observation_errors"] = errors
            if snapshot is not None:
                snapshot.update(copy.deepcopy(resources))
                metadata["ax_node_count"] = len(snapshot["nodes"])
                metadata["ax_direct_count"] = sum(bool(n.get("direct")) for n in snapshot["nodes"])
                self.snapshot = snapshot
                self.nodes = {n["ref"]: n for n in snapshot["nodes"]}
            self._observation_metadata = metadata
            if shot is not None:
                self._last_screenshot = shot
                self._last_screenshot_metadata = copy.deepcopy(metadata)
            return shot, snapshot, metadata
        except Exception as exc:
            self.invalidate()
            return None, None, {"error_code": getattr(exc, "code", "stale"), "error": str(exc)[:200], **self.event_metadata()}
        finally:
            self._disconnect(p, browser)

    def _collect_page(self, page, cdp, identity, start):
        cdp.send("Accessibility.enable")
        url = page.url
        viewport = page.evaluate("({width:innerWidth,height:innerHeight,x:scrollX,y:scrollY,dpr:devicePixelRatio})")
        dom = cdp.send("DOMSnapshot.captureSnapshot", {"computedStyles": []})
        strings, layout = dom["strings"], {}
        for doc in dom["documents"]:
            dn = doc["nodes"]
            li = doc.get("layout", {})
            bounds = dict(zip(li.get("nodeIndex", []), li.get("bounds", [])))
            for i, backend in enumerate(dn.get("backendNodeId", [])):
                attrs = dn.get("attributes", [])[i]
                at = {strings[attrs[k]]: strings[attrs[k + 1]] for k in range(0, len(attrs), 2)}
                layout[backend] = {"bounds": bounds.get(i), "tag": strings[dn["nodeName"][i]].lower(),
                                   "attrs": at, "frame_id": strings[doc["frameId"]]}
        main_frame = cdp.send("Page.getFrameTree")["frameTree"]["frame"]["id"]
        raw = cdp.send("Accessibility.getFullAXTree")["nodes"]
        snapshot_id = uuid.uuid4().hex[:12]
        retained = [n for n in raw if not n.get("ignored") and n.get("role", {}).get("value") not in SKIP]
        raw_map = {n["nodeId"]: n for n in raw}
        modal_ids = {n["nodeId"] for n in retained if n.get("role", {}).get("value") in {"dialog", "alertdialog"}}
        modal_tree = set(modal_ids)
        changed = True
        while changed:
            previous = len(modal_tree)
            modal_tree.update(n["nodeId"] for n in raw if n.get("parentId") in modal_tree)
            changed = previous != len(modal_tree)
        for nid in list(modal_tree):
            parent = raw_map.get(nid, {}).get("parentId")
            seen = set()
            while parent and parent not in seen:
                seen.add(parent)
                modal_tree.add(parent)
                parent = raw_map.get(parent, {}).get("parentId")
        selected_ids = {n["nodeId"] for n in retained[:self.options.max_nodes]} | modal_tree
        refs = {n["nodeId"]: str(i) for i, n in enumerate(retained)}  # 短数字 ref（ComponentBench 风格，模型易抄）
        nodes = []
        for n in retained:
            if n["nodeId"] not in selected_ids:
                continue
            backend = n.get("backendDOMNodeId")
            detail = layout.get(backend, {})
            attrs = detail.get("attrs", {})
            role = n.get("role", {}).get("value", "")
            states = {p["name"]: p.get("value", {}).get("value") for p in n.get("properties", [])
                      if p["name"] in {"checked", "selected", "expanded", "disabled", "focused", "required", "readonly", "level"}}
            parent = n.get("parentId")
            seen = set()
            while parent and parent not in refs and parent not in seen:
                seen.add(parent)
                parent = raw_map.get(parent, {}).get("parentId")
            bounds = detail.get("bounds")
            visible = bool(bounds and bounds[2] > 0 and bounds[3] > 0 and
                bounds[0] < viewport["x"] + viewport["width"] and bounds[0] + bounds[2] > viewport["x"] and
                bounds[1] < viewport["y"] + viewport["height"] and bounds[1] + bounds[3] > viewport["y"])
            node = {"ref": refs[n["nodeId"]], "parent": refs.get(parent), "role": role,
                    "name": str(n.get("name", {}).get("value", ""))[:400], "states": states,
                    "backend_id": backend, "frame_id": detail.get("frame_id"),
                    "tag": detail.get("tag", ""), "bounds": bounds, "in_viewport": visible,
                    "direct": bool(backend and role in INTERACTIVE and detail.get("frame_id") == main_frame)}
            if n["nodeId"] in modal_tree:
                node["modal_context"] = True
            if "value" in n:
                node["value"] = "[redacted]" if attrs.get("type") == "password" else str(n["value"].get("value", ""))[:400]
            if attrs.get("href"):
                node["href"] = attrs["href"][:500]
            if attrs.get("id"):
                node["id"] = attrs["id"][:200]
            nodes.append(node)
        warnings = []
        if len(retained) > self.options.max_nodes:
            warnings.append("node_limit: additional nodes omitted")
        if len(page.frames) > 1:
            warnings.append("iframe content may be incomplete; iframe actions use visual fallback")
        return {"kind": "browser_ax", "snapshot_id": snapshot_id, "url": url, "frame_id": main_frame, **identity,
            "title": page.title(), "viewport": viewport, "nodes": nodes, "warnings": warnings,
            "active_modals": [refs[n] for n in modal_ids if n in refs],
            "captured_at": time.time(), "capture_ms": round((time.perf_counter() - start) * 1000, 2)}

    # ---- route ----
    def _ref_hint(self, observation, limit=20):
        """生成"当前可用元素编号"提示，帮模型从编造/过期的 ref 里纠正过来。"""
        refs = observation.info.get("exposed_refs")
        if refs is None:
            refs = list(self.nodes.keys())
        parts = []
        for ref in refs:
            n = self.nodes.get(ref)
            if not n:
                continue
            name = (n.get("name") or "").strip()
            role = n.get("role") or ""
            if name:
                parts.append(f"[{ref}] {role} '{name}'")
            if len(parts) >= limit:
                break
        return "；".join(parts) if parts else "（无）"

    def route(self, action, observation):
        ref = action.args.get("target_ref")
        if ref is not None:
            ref = str(ref)
        if ref is None:
            # 模型输出 HTML id → 解析成 ref（比 12 位 hex ref 更短、更易抄）
            eid = action.args.get("element_id")
            if eid:
                ref = self.resolve_id(str(eid))
        if ref is None:
            return None
        info = observation.info or {}
        observation_id = info.get("snapshot_id") or info.get("scene", {}).get("snapshot_id")
        if observation_id and (not self.snapshot or observation_id != self.snapshot.get("snapshot_id")):
            return {"channel": "reobserve", "reason": "Observation snapshot changed", "error_code": "stale"}
        if ref not in observation.info.get("exposed_refs", self.nodes):
            hint = self._ref_hint(observation)
            return {"channel": "reobserve", "error_code": "stale",
                    "reason": f"target_ref '{ref}' 不在当前页面里，可能是编造或过期的编号。"
                              f"当前页面可用的元素编号（部分）：{hint}。"
                              f"请从中选一个正确的 ref，或改用 element_id / 英文 target。"}
        if not self.snapshot or ref not in self.nodes:
            return {"channel": "reobserve", "reason": "Unknown or stale target_ref", "error_code": "stale"}
        if action.action not in {"click", "double_click", "right_click", "type", "select", "scroll"}:
            return None
        node = self.nodes[ref]
        if action.action == "scroll" and node.get("frame_id") != self.snapshot.get("frame_id"):
            return {"channel": "reobserve", "reason": "Cross-frame container scrolling is unsupported",
                    "error_code": "unsupported"}
        if not node["direct"] and action.action != "scroll":
            return None
        return {"channel": "browser_dom", "snapshot_id": self.snapshot["snapshot_id"],
                "page_id": self.snapshot.get("page_id"), "document_id": self.snapshot.get("document_id"),
                "target_ref": ref, "coordinate_space": "css_viewport"}

    def resolve_id(self, element_id):
        """按 HTML id 属性找节点，返回 ref 或 None。"""
        self.last_resolution_error = None
        if not self.nodes or not element_id:
            return None
        matches = [ref for ref, n in self.nodes.items() if n.get("id") == element_id]
        self.last_resolution_error = "ambiguous" if len(matches) > 1 else None
        return matches[0] if len(matches) == 1 else None

    # ---- 语义回退：target 描述 → 节点 ref ----
    _STOPWORDS = {
        "the", "a", "an", "in", "of", "on", "at", "to", "and", "or", "with", "by",
        "for", "from", "is", "are", "this", "that", "it", "its", "left", "right",
        "top", "bottom", "center", "side", "menu", "dropdown", "option", "button",
        "icon", "tab", "link", "page", "section", "panel", "bar", "area", "field",
        "box", "item", "row", "list", "dialog", "window", "popup", "select",
    }

    def resolve_target(self, target, allowed=None):
        """把英文 target 描述确定性匹配到快照里的可交互节点，返回 best ref 或 None。

        供 base 模型兜底：它不抄 target_ref 时，用 target 描述在 AX 快照里找 ref，
        命中即走 DOM 定位（跳过视觉定位）；匹配不置信返回 None → 交给视觉定位。
        allowed：可选 set，只在这些 ref（= 已暴露给模型的节点）里匹配。
        """
        self.last_resolution_error = None
        if not self.snapshot or not self.nodes:
            return None
        t = (target or "").lower()
        if not t:
            return None
        # 模型有时把 ref 直接写进 target 描述里（如 "[87]"）→ 直接提取用
        m = re.search(r"\[(\d+)\]", t)
        if m and m.group(1) in self.nodes and self.nodes[m.group(1)].get("direct"):
            if allowed is None or m.group(1) in allowed:
                return m.group(1)
        # 引号里的精确短语（模型通常写成 "The 'X' option in ..."）
        quoted = re.findall(
            r"[\"'`\u2018\u2019\u201c\u201d]([^\"'`\u2018\u2019\u201c\u201d]{2,80})"
            r"[\"'`\u2018\u2019\u201c\u201d]", t)
        # 方括号里的短语（模型写成 "The [PDF] xxx link" 这种）也当精确短语提取
        quoted += re.findall(r"\[([^\[\]]{2,80})\]", t)
        tokens = set(re.findall(r"[a-z0-9]+", t)) - self._STOPWORDS
        allowed_set = set(allowed) if allowed is not None else None
        candidates = []
        requested_role = next((role for word, role in (
            ('textbox', 'textbox'), ('search box', 'textbox'), ('input', 'textbox'),
            ('button', 'button'), ('link', 'link')) if re.search(r'\b' + re.escape(word) + r'\b', t)), None)
        for ref, n in self.nodes.items():
            if allowed_set is not None and ref not in allowed_set:
                continue
            if not n.get("direct"):
                continue
            name = (n.get("name") or "").lower()
            value = (n.get("value") or "").lower()
            role = (n.get("role") or "").lower()
            hay = f"{role} {name} {value}"
            if requested_role and role != requested_role and not (requested_role == 'textbox' and role in {'searchbox', 'combobox'}):
                continue
            # Whole accessible-name match handles "Search button" without
            # lowering the threshold for weak token/substring guesses.
            score = 40 if name and re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', t) else 0
            for q in quoted:
                q = q.strip()
                if not q:
                    continue
                if name and q in name:
                    score += 40
                elif q in hay:
                    score += 24
            if tokens:
                score += 6 * sum(1 for tk in tokens if tk and (tk in name or tk in value))
            if role and role in t:
                score += 4
            if score >= 24:
                candidates.append((score, ref))
        # 只信「引号精确命中」这一档（≥24）或足够强的 token 命中；宁可不匹配走视觉定位，
        # 也不要在 DOM 里点错节点。
        candidates.sort(reverse=True)
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 12:
            self.last_resolution_error = "ambiguous"
            return None
        if candidates:
            return candidates[0][1]
        self.last_resolution_error = "not_found"
        return None

    def _validate_binding(self, binding, page=None, cdp=None):
        if not binding or not all(binding.get(k) is not None for k in ("page_id", "document_id", "snapshot_id", "captured_at")):
            raise BrowserActionError("Observation has no page/document binding", code="stale")
        age = time.time() - binding["captured_at"]
        if age < 0 or age > self.options.max_snapshot_age:
            raise BrowserActionError("Observation expired", code="stale")
        if page is not None:
            actual = self._identity(page, cdp)
            if any(actual[k] != binding[k] for k in ("page_id", "document_id")):
                raise BrowserActionError("Page or document changed", code="stale")
            if binding.get("url") and page.url != binding["url"]:
                raise BrowserActionError("Page navigated", code="stale")

    _INSPECT_PROPERTIES = r"""function() {
      const e = this.nodeType === 1 ? this : this.parentElement;
      if (!e || !e.isConnected) return {detached:true};
      const tag = e.tagName.toLowerCase(), type = (e.getAttribute('type') || '').toLowerCase();
      const autocomplete = (e.getAttribute('autocomplete') || '').toLowerCase();
      const labels = Array.from(e.labels || []).map(label => label.textContent || '').join(' ');
      const semantic = [e.getAttribute('name'), e.id, e.getAttribute('aria-label'),
                        e.getAttribute('placeholder'), labels].filter(Boolean).join(' ');
      const reasons = [];
      if (type === 'password') reasons.push('password');
      if (/(^|\s)(current-password|new-password|one-time-code|cc-[^\s]+)(\s|$)/.test(autocomplete))
        reasons.push('sensitive_autocomplete');
      const control = tag === 'input' || tag === 'textarea' || e.isContentEditable;
      if (control && /password|passcode|one.?time|(?:^|[\W_])otp(?:[\W_]|$)|otp(?:code|input|digit|field)|(?:verification|authentication|security).?(?:code|pin)|(?:credit|debit|card).?(?:number|security|verification|expiry|expiration)|\bcc[-_ ]?(?:number|num|exp|csc|cvv|cvc)\b|\b(?:cvv|cvc|ccnum)\b|密码|验证码|信用卡|银行卡|卡号/i.test(semantic))
        reasons.push('sensitive_label');
      const disabled = !!e.disabled || e.getAttribute('aria-disabled') === 'true' || !!e.closest('[inert]');
      const editable = !disabled && !e.readOnly && (e.isContentEditable || tag === 'textarea' ||
        (tag === 'input' && !['button','submit','reset','checkbox','radio','file','hidden','image','range','color'].includes(type)));
      let editable_has_text = null;
      if (editable) {
        const content = e.isContentEditable ? e.textContent : e.value;
        editable_has_text = typeof content === 'string' ? content.length > 0 : null;
      }
      const rect = e.getBoundingClientRect();
      return {tag, input_type:type, editable:!!editable, editable_has_text, sensitive:reasons.length > 0, sensitive_reasons:reasons,
              disabled, bounds:[rect.x,rect.y,rect.width,rect.height]};
    }"""

    _INSPECT_ANCESTOR = r"""function(skipSelf) {
      const parent = e => e.assignedSlot || e.parentElement || (e.getRootNode && e.getRootNode().host);
      let e = this.nodeType === 1 ? this : this.parentElement;
      if (skipSelf && e) e = parent(e);
      const roles = new Set(['button','link','textbox','searchbox','checkbox','radio','combobox',
        'menuitem','menuitemcheckbox','menuitemradio','tab','switch','slider','spinbutton',
        'listbox','option','treeitem']);
      for (let depth=0; e && depth<64; depth++, e=parent(e)) {
        if (e.matches('button,a[href],input,textarea,select,option,summary,[contenteditable]:not([contenteditable="false"])') ||
            roles.has(e.getAttribute('role'))) return e;
      }
      return null;
    }"""

    _INSPECT_SCROLL_ANCESTOR = r"""function() {
      const parent = e => e.assignedSlot || e.parentElement || (e.getRootNode && e.getRootNode().host);
      let e = this.nodeType === 1 ? this : this.parentElement;
      for (let depth=0; e && depth<64; depth++, e=parent(e)) {
        if (e === document.scrollingElement) return e;
        const overflow = getComputedStyle(e).overflowY;
        if (['auto','scroll','overlay'].includes(overflow) && e.scrollHeight > e.clientHeight) return e;
      }
      return document.scrollingElement;
    }"""

    @browser_thread
    def inspect_point(self, x, y, metadata):
        """Inspect a frozen CSS screenshot point without input or a new snapshot.

        A target_ref is returned only after a live backend-node match. Bounding
        boxes are diagnostics, never evidence for choosing a semantic target.
        """
        return self._inspect_target(metadata, point=(x, y))

    @browser_thread
    def inspect_focused(self, metadata):
        """Inspect focus, including open shadow roots; return only text presence, never content."""
        return self._inspect_target(metadata)

    @browser_thread
    def inspect_scroll_target(self, x, y, metadata):
        """Resolve the actual scroll container without dispatch or new AX refs."""
        return self._inspect_target(metadata, point=(x, y), scroll=True)

    @browser_thread
    def insert_focused_text(self, text, metadata, overwrite=False, expected_backend_id=None):
        """Insert human text at the existing caret, without a second click.

        Focus and sensitive-field checks happen before any keyboard dispatch.
        Once dispatch starts, an exception is uncertain and must not be retried.
        """
        started, dispatched = time.perf_counter(), False
        if not isinstance(text, str) or not isinstance(overwrite, bool):
            return self._failure("Text must be a string and overwrite a boolean", "unsupported")
        inspected = self.inspect_focused(metadata)
        if inspected.get("status") != "inspected":
            return inspected
        if inspected.get("sensitive") or inspected.get("sensitive_unknown"):
            return self._failure("Sensitive fields cannot be recorded as ordinary text input", "blocked")
        if not inspected.get("editable"):
            return self._failure("Focused element is not editable", "blocked")
        backend_id = inspected.get("focused_backend_id")
        if backend_id is None or (expected_backend_id is not None and expected_backend_id != backend_id):
            return self._failure("Focused element changed", "stale")
        p = browser = cdp = None
        group = "human-insert-" + uuid.uuid4().hex
        try:
            p, browser = self._connect()
            page = self._pick_page(browser)
            if page is None:
                raise BrowserActionError("Task page unavailable", code="stale")
            cdp = self._cdp(page)
            self._validate_binding(metadata, page, cdp)
            obj = cdp.send("DOM.resolveNode", {"backendNodeId": backend_id,
                                               "objectGroup": group})["object"].get("objectId")
            if not obj:
                raise BrowserActionError("Focused element detached", code="stale")
            inspection = cdp.send("Runtime.callFunctionOn", {"objectId": obj,
                "functionDeclaration": """function() {
                  let active=document.activeElement;
                  while(active && active.shadowRoot && active.shadowRoot.activeElement)
                    active=active.shadowRoot.activeElement;
                  return {same_focus:active === this, fields:(""" + self._INSPECT_PROPERTIES + """).call(this)};
                }""", "returnByValue": True, "objectGroup": group})
            focus = inspection.get("result", {}).get("value") or {}
            if inspection.get("exceptionDetails") or not focus.get("same_focus"):
                raise BrowserActionError("Focus changed before text input", code="stale")
            fields = focus.get("fields") or {}
            if fields.get("detached"):
                raise BrowserActionError("Focused element detached", code="stale")
            if fields.get("sensitive") or not fields.get("editable"):
                raise BrowserActionError("Focused element is sensitive or not editable", code="blocked")
            dispatched = True
            if overwrite:
                page.keyboard.press("ControlOrMeta+A")
                page.keyboard.press("Backspace")
            page.keyboard.insert_text(text)
            return {"status": "executed", "dispatched": True, "channel": "browser_dom",
                    "action": "type", "resolved": "focused_input", "backend_id": backend_id,
                    "page_id": metadata["page_id"], "document_id": metadata["document_id"],
                    "snapshot_id": metadata["snapshot_id"],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        except Exception as exc:
            return self._failure(exc, getattr(exc, "code", "blocked"), dispatched, started,
                                 action="type", resolved="focused_input")
        finally:
            if cdp is not None:
                try:
                    cdp.send("Runtime.releaseObjectGroup", {"objectGroup": group})
                except Exception:
                    pass
            if browser is not None:
                self._disconnect(p, browser)

    def _inspect_target(self, metadata, point=None, scroll=False):
        started = time.perf_counter()
        p = browser = cdp = None
        group = "human-inspect-" + uuid.uuid4().hex
        try:
            self._validate_binding(metadata)
            if metadata.get("snapshot_id") != self._observation_metadata.get("snapshot_id"):
                raise BrowserActionError("Observation superseded", code="stale")
            if metadata.get("screenshot_stale"):
                raise BrowserActionError("Screenshot is stale", code="stale")
            if point is not None and metadata.get("screenshot_space") != "css_viewport":
                raise BrowserActionError("Expected CSS viewport coordinates", code="unsupported")
            p, browser = self._connect()
            page = self._pick_page(browser)
            if page is None:
                raise BrowserActionError("Task page unavailable", code="stale")
            if any(d["page"] is page for d in self._dialogs.values()):
                raise BrowserActionError("JavaScript dialog blocks inspection", code="blocked")
            cdp = self._cdp(page)
            self._validate_binding(metadata, page, cdp)
            viewport_script = "({w: innerWidth, h: innerHeight, x: scrollX, y: scrollY})"
            viewport = page.evaluate(viewport_script)
            if metadata.get("screenshot_size") and metadata["screenshot_size"] != [viewport["w"], viewport["h"]]:
                raise BrowserActionError("Viewport resized", code="stale")
            if metadata.get("scroll_offset") != [viewport["x"], viewport["y"]]:
                raise BrowserActionError("Viewport scrolled", code="stale")
            identity = {key: metadata[key] for key in ("page_id", "document_id", "snapshot_id")}
            result = {"status": "inspected", "dispatched": False, "matched": False,
                      "target_label": "", "editable": False, "sensitive": False,
                      "sensitive_reasons": [], "coordinate_space": "css_viewport", **identity}
            main_frame = cdp.send("Page.getFrameTree")["frameTree"]["frame"]["id"]
            if point is not None:
                x, y = point
                if not (isinstance(x, (int, float)) and not isinstance(x, bool)
                        and isinstance(y, (int, float)) and not isinstance(y, bool)
                        and math.isfinite(x) and math.isfinite(y)
                        and 0 <= x < viewport["w"] and 0 <= y < viewport["h"]):
                    raise BrowserActionError("Point is outside the screenshot", code="not_found")
                result["point"] = [x, y]
                hit = cdp.send("DOM.getNodeForLocation", {"x": int(x), "y": int(y),
                    "includeUserAgentShadowDOM": True, "ignorePointerEventsNone": False})
                result["hit_backend_id"] = hit.get("backendNodeId")
                if hit.get("frameId") != main_frame:
                    result.update(reason="iframe_target", sensitive=True, sensitive_unknown=True,
                                  sensitive_reasons=["uninspectable_frame"])
                    obj = None
                else:
                    obj = cdp.send("DOM.resolveNode", {"backendNodeId": hit["backendNodeId"],
                                                       "objectGroup": group})["object"].get("objectId")
            else:
                focused = cdp.send("Runtime.evaluate", {"expression": """(() => {
                  let e=document.activeElement;
                  while(e && e.shadowRoot && e.shadowRoot.activeElement) e=e.shadowRoot.activeElement;
                  return e;
                })()""", "objectGroup": group, "returnByValue": False})
                if focused.get("exceptionDetails"):
                    raise BrowserActionError("Focus inspection failed", code="blocked")
                obj = focused.get("result", {}).get("objectId")

            def call(object_id, function, *, by_value=True, arguments=None):
                response = cdp.send("Runtime.callFunctionOn", {"objectId": object_id,
                    "functionDeclaration": function, "returnByValue": by_value,
                    "objectGroup": group, "arguments": [{"value": arg} for arg in (arguments or [])]})
                if response.get("exceptionDetails"):
                    raise BrowserActionError("Element inspection failed", code="stale")
                return response.get("result", {}).get("value" if by_value else "objectId")

            if scroll:
                result.update(root_scroll=False, semantic_unavailable=True)
                if obj:
                    container = call(obj, self._INSPECT_SCROLL_ANCESTOR, by_value=False)
                    if container:
                        properties = call(container, self._INSPECT_PROPERTIES)
                        if not properties or properties.get("detached"):
                            raise BrowserActionError("Scroll container detached", code="stale")
                        result.update(properties)
                        result["root_scroll"] = bool(call(container, "function(){return this === document.scrollingElement;}"))
                        result["reason"] = "root_scroll" if result["root_scroll"] else "nested_scroll"
                        if result["root_scroll"]:
                            result["semantic_unavailable"] = False
                        else:
                            backend = cdp.send("DOM.describeNode", {"objectId": container, "depth": 0})["node"]["backendNodeId"]
                            matches = [(ref, node) for ref, node in self.nodes.items()
                                       if node.get("backend_id") == backend and node.get("frame_id") == main_frame]
                            if len(matches) == 1 and self.snapshot and self.snapshot.get("snapshot_id") == metadata["snapshot_id"]:
                                ref, node = matches[0]
                                result.update(matched=True, backend_id=backend, role=node.get("role", ""),
                                              target_label=" ".join(filter(None, (node.get("role", ""), node.get("name", "")))))
                                if ref in (metadata.get("exposed_refs") or []):
                                    result.update(target_ref=ref, semantic_unavailable=False)
                                    if node.get("id"):
                                        result["element_id"] = node["id"]
                                else:
                                    result["unexposed"] = True
                    else:
                        result["reason"] = "no_scroll_target"
                obj = None

            if obj:
                if point is None:
                    result["focused_backend_id"] = cdp.send("DOM.describeNode", {"objectId": obj, "depth": 0})["node"]["backendNodeId"]
                properties = call(obj, self._INSPECT_PROPERTIES)
                if not properties or properties.get("detached"):
                    raise BrowserActionError("Element detached", code="stale")
                result.update(properties)
                if properties.get("tag") in {"iframe", "frame"}:
                    result.update(reason="iframe_target", sensitive=True, sensitive_unknown=True,
                                  sensitive_reasons=["uninspectable_frame"])
                elif properties.get("tag") == "canvas":
                    result["reason"] = "canvas_target"
                else:
                    candidate = call(obj, self._INSPECT_ANCESTOR, by_value=False, arguments=[False])
                    # Walk interactive ancestors, including shadow hosts. The
                    # first exact AX backend match is the most specific target.
                    for _ in range(16):
                        if not candidate:
                            break
                        backend = cdp.send("DOM.describeNode", {"objectId": candidate, "depth": 0})["node"]["backendNodeId"]
                        matches = [(ref, node) for ref, node in self.nodes.items()
                                   if node.get("backend_id") == backend and node.get("direct")
                                   and node.get("frame_id") == main_frame]
                        if len(matches) == 1 and self.snapshot and self.snapshot.get("snapshot_id") == metadata["snapshot_id"]:
                            ref, node = matches[0]
                            properties = call(candidate, self._INSPECT_PROPERTIES)
                            if not properties or properties.get("detached"):
                                raise BrowserActionError("Target detached", code="stale")
                            result.update(properties)
                            result.update(matched=True, backend_id=backend, role=node.get("role", ""),
                                          target_label=" ".join(filter(None, (node.get("role", ""), node.get("name", "")))))
                            exposed = metadata.get("exposed_refs")
                            if exposed is None or ref in exposed:
                                result["target_ref"] = ref
                                if node.get("id"):
                                    result["element_id"] = node["id"]
                            else:
                                result["unexposed"] = True
                            break
                        candidate = call(candidate, self._INSPECT_ANCESTOR, by_value=False, arguments=[True])
                    if not result["matched"]:
                        result["reason"] = "no_matching_ax_node"
            elif "reason" not in result:
                result["reason"] = "no_focused_element" if point is None else "no_dom_element"
            self._validate_binding(metadata, page, cdp)
            if page.evaluate(viewport_script) != viewport:
                raise BrowserActionError("Viewport changed during inspection", code="stale")
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return result
        except Exception as exc:
            return self._failure(exc, getattr(exc, "code", "blocked"), started=started)
        finally:
            if cdp is not None:
                try:
                    cdp.send("Runtime.releaseObjectGroup", {"objectGroup": group})
                except Exception:
                    pass
            if browser is not None:
                self._disconnect(p, browser)

    # ---- execute ----
    @browser_thread
    def execute(self, action, route):
        if action.action == "download":
            return self.download(str(action.args.get("url") or ""), filename=action.args.get("filename"))
        if action.action not in {"click", "double_click", "right_click", "type", "select", "scroll"}:
            return self._failure(f"Unsupported DOM action: {action.action}", "unsupported")
        try:
            self._validate_binding(self.snapshot)
            if route.get("snapshot_id") != self.snapshot["snapshot_id"]:
                raise BrowserActionError("Route snapshot changed", code="stale")
            for key in ("page_id", "document_id"):
                if route.get(key) is not None and route[key] != self.snapshot[key]:
                    raise BrowserActionError("Route page/document changed", code="stale")
        except BrowserActionError as exc:
            return self._failure(exc, exc.code)
        ref = str(route.get("target_ref"))
        node = self.nodes.get(ref)
        if node is None:
            return self._failure("Unknown target_ref", "stale")
        if node.get("frame_id") != self.snapshot.get("frame_id"):
            return self._failure("Cross-frame DOM coordinates are unsupported", "unsupported")
        if (self.foreground_check and not self.foreground_check()):
            return self._failure("Browser lost focus", "blocked")
        started, dispatched = time.perf_counter(), False
        try:
            p, browser = self._connect()
        except Exception as exc:
            return self._failure(exc, "not_found")
        try:
            page = self._pick_page(browser)
            if page is None:
                return self._failure("Task page unavailable", "stale")
            if any(d["page"] is page for d in self._dialogs.values()):
                return self._failure("JavaScript dialog must be handled first", "blocked", **self.event_metadata())
            cdp = self._cdp(page)
            self._validate_binding(self.snapshot, page, cdp)
            backend_id = node["backend_id"]
            try:
                obj = cdp.send("DOM.resolveNode", {"backendNodeId": backend_id})["object"].get("objectId")
            except Exception as exc:
                raise BrowserActionError("Target detached", code="stale") from exc
            if not obj:
                return self._failure("Target detached", "stale")
            cdp.send("DOM.scrollIntoViewIfNeeded", {"backendNodeId": backend_id})
            inspect = """function() {
              if (!this.isConnected || !(this instanceof Element)) return {error:'detached'};
              if (this.matches(':disabled') || this.getAttribute('aria-disabled') === 'true' || this.closest('[inert]')) return {error:'disabled'};
              const r=this.getBoundingClientRect(), s=getComputedStyle(this);
              if (r.width<=0 || r.height<=0 || s.visibility!=='visible' || s.display==='none') return {error:'hidden'};
              const x=(Math.max(0,r.left)+Math.min(innerWidth,r.right))/2;
              const y=(Math.max(0,r.top)+Math.min(innerHeight,r.bottom))/2;
              let hit=document.elementFromPoint(x,y);
              while(hit && hit.shadowRoot && hit.shadowRoot.elementFromPoint) {
                const next=hit.shadowRoot.elementFromPoint(x,y); if (!next || next===hit) break; hit=next;
              }
              if (!hit || !(this===hit || this.contains(hit))) return {error:'occluded',point:{x,y},
                blocker:hit ? {id:hit.id,tag:hit.tagName.toLowerCase(),role:hit.getAttribute('role'),
                  name:(hit.getAttribute('aria-label') || hit.innerText || '').slice(0,300)} : null};
              return {x,y,w:r.width,h:r.height,tag:this.tagName.toLowerCase(),readOnly:!!this.readOnly,
                      editable:this.isContentEditable || this.tagName==='TEXTAREA' || (this.tagName==='INPUT' && !['button','submit','checkbox','radio','file'].includes(this.type))};
            }"""
            def call(obj, fn, args=None):
                r = cdp.send("Runtime.callFunctionOn", {"objectId": obj, "functionDeclaration": fn,
                    "arguments": [{"value": a} for a in (args or [])], "returnByValue": True})
                if r.get("exceptionDetails"):
                    raise BrowserActionError("DOM inspection failed")
                return r.get("result", {}).get("value")
            first = call(obj, inspect)
            page.wait_for_timeout(35)
            point = call(obj, inspect)
            if not point or point.get("error"):
                reason = (point or {}).get("error", "Target not actionable")
                if reason == "occluded":
                    blocker = point.get("blocker") or {}
                    blocker_ref = self.resolve_id(blocker.get("id")) if blocker.get("id") else None
                    if blocker_ref is None and point.get("point"):
                        try:
                            at = point["point"]
                            backend = cdp.send("DOM.getNodeForLocation", {"x": int(at["x"]),
                                "y": int(at["y"]), "includeUserAgentShadowDOM": True}).get("backendNodeId")
                            blocker_ref = next((ref for ref, n in self.nodes.items() if n.get("backend_id") == backend), None)
                        except Exception:
                            pass
                    return self._failure(reason, "blocked", started=started, blocker=blocker,
                        blocker_ref=blocker_ref,
                        active_modals=copy.deepcopy(self.snapshot.get("active_modals", [])))
                raise BrowserActionError(reason, code="stale" if reason == "detached" else "blocked")
            if first != point:
                raise BrowserActionError("Target moved during actionability check")
            name, args = action.action, action.args
            if name == "type" and (not point["editable"] or point["readOnly"]):
                raise BrowserActionError("Target is not editable")
            if name == "select" and point["tag"] != "select":
                raise BrowserActionError("Custom dropdown: use click and then click its option", code="unsupported")
            if name == "select":
                matches = call(obj, "function(label){return Array.from(this.options).filter(o=>o.label===label && !o.disabled).map(o=>o.index)}", [str(args["option"])])
                if len(matches) != 1:
                    raise BrowserActionError("Select option is absent or ambiguous", code="ambiguous" if matches else "not_found")
                dispatched = True
                call(obj, """function(i){ this.selectedIndex=i; this.dispatchEvent(new Event('input',{bubbles:true})); this.dispatchEvent(new Event('change',{bubbles:true})); }""", [matches[0]])
            elif name == "scroll":
                dispatched = True
                page.mouse.move(point["x"], point["y"])
                page.mouse.wheel(0, int(args["amount"]) * 120 * (-1 if args["direction"] == "up" else 1))
            else:
                dispatched = True
                page.mouse.click(point["x"], point["y"], button="right" if name == "right_click" else "left",
                                  click_count=2 if name == "double_click" else 1)
                if name == "type":
                    if not call(obj, "function(){let a=document.activeElement;while(a&&a.shadowRoot&&a.shadowRoot.activeElement)a=a.shadowRoot.activeElement;return a===this || this.contains(a)}"):
                        raise BrowserActionError("Focus changed after click", dispatched=True)
                    if args.get("overwrite"):
                        page.keyboard.press("ControlOrMeta+A")
                        page.keyboard.press("Backspace")
                    page.keyboard.insert_text(str(args["text"]))
                else:
                    self._settle(page)
            return {"status": "executed", "dispatched": True, "channel": "browser_dom",
                    "target_ref": ref, "coord": [point["x"], point["y"]],
                    "coordinate_space": "css_viewport", "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        except Exception as exc:
            return self._failure(exc, getattr(exc, "code", "blocked"),
                                 dispatched or getattr(exc, "dispatched", False), started)
        finally:
            self._disconnect(p, browser)

    # ---- 纯 CDP 浏览器动作（不动真实鼠标键盘）----
    _KEY_MAP = {
        "ctrl": "ControlOrMeta", "control": "ControlOrMeta", "cmd": "Meta", "win": "Meta",
        "esc": "Escape", "escape": "Escape", "return": "Enter", "enter": "Enter",
        "del": "Delete", "delete": "Delete", "backspace": "Backspace",
        "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
        "tab": "Tab", "space": " ", "pageup": "PageUp", "pagedown": "PageDown",
        "home": "Home", "end": "End", "shift": "Shift", "alt": "Alt",
    }

    @classmethod
    def _norm_key(cls, key):
        k = (key or "").strip().lower()
        return cls._KEY_MAP.get(k, key)

    @browser_thread
    def goto(self, url):
        """CDP 导航（虚拟，不动真实鼠标键盘）。"""
        url = (url or "").strip()
        if not url:
            return {"status": "rejected", "dispatched": False, "channel": "browser_dom",
                    "error": "empty url"}
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        started, dispatched = time.perf_counter(), False
        try:
            p, browser = self._connect()
        except Exception as exc:
            return {"status": "rejected", "dispatched": False, "channel": "browser_dom",
                    "error": str(exc)[:200]}
        try:
            page = self._pick_page(browser)
            if page is None:
                return self._failure("Task page unavailable", "stale")
            dispatched = True
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            self._settle(page)
            return {"status": "executed", "dispatched": True, "channel": "browser_dom",
                    "action": "goto", "url": page.url[:300],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        except Exception as exc:
            return self._failure(exc, "blocked", dispatched, started)
        finally:
            self._disconnect(p, browser)

    @staticmethod
    def _settle(page, timeout_ms=4000):
        """等待点击/导航后页面稳定；非导航点击会立即返回，失败不阻塞。

        PDF 查看器（arxiv.org/pdf/...）的 load 事件只代表 HTML 就绪，PDF 本体是
        之后异步 fetch + 渲染的；谷歌学术等站点的搜索结果也是 JS 异步渲染。所以：
        1) networkidle 等主资源 fetch 完成（上限 4s）；
        2) 再等屏幕连续两次一致（异步渲染收尾），避免截图仍是空白/加载态。
        """
        try:
            page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception:  # noqa: BLE001  页面持续有连接时按超时放行，不阻塞主流程
            pass
        BrowserSession._wait_screen_stable(page)

    @staticmethod
    def _wait_screen_stable(page, max_wait_ms=2000, interval_ms=350):
        """连续两次 viewport 截图一致才返回；页面在异步渲染时会一直等到稳定。"""
        import hashlib
        deadline = time.monotonic() + max_wait_ms / 1000.0
        prev = None
        while time.monotonic() < deadline:
            try:
                buf = page.screenshot(full_page=False, scale="css")
                digest = hashlib.sha1(bytes(buf)).hexdigest()
            except Exception:  # noqa: BLE001
                return
            if prev == digest:
                from .visual_readiness import content_thumbnail
                url = str(getattr(page, 'url', '')).lower().split('?', 1)[0]
                if not ('/pdf/' in url or url.endswith('.pdf')) or content_thumbnail(bytes(buf)) is not None:
                    return
            prev = digest
            time.sleep(interval_ms / 1000.0)

    def screenshot(self):
        """CSS 像素 viewport 截图，同时建立其页面/文档绑定。"""
        return self.collect_observation()[0]

    @browser_thread
    def get_page_source(self, target_ref=None, max_chars=20000):
        """Read a sanitized bounded DOM subtree from the bound task document."""
        try:
            p, browser = self._connect()
        except Exception as exc:
            return f"页面源码不可用：{str(exc)[:200]}"
        try:
            page = self._pick_page(browser)
            if page is None:
                return "任务页面不可用，请重新观察"
            cdp = self._cdp(page)
            before = self._identity(page, cdp)
            url = page.url
            clean = """function(){const root=this.cloneNode(true);
              root.querySelectorAll('script,style,noscript,template,input[type=hidden]').forEach(n=>n.remove());
              [root,...root.querySelectorAll('*')].forEach(n=>{
                for(const a of [...n.attributes || []]) if(a.name.startsWith('on') ||
                  ['style','nonce','integrity','srcdoc'].includes(a.name)) n.removeAttribute(a.name);
                if(n.matches && n.matches('input[type=password]')) n.removeAttribute('value');
              }); return root.outerHTML;}"""
            if target_ref is not None:
                self._validate_binding(self.snapshot, page, cdp)
                node = self.nodes.get(str(target_ref))
                if node is None or not node.get("backend_id"):
                    return "Unknown or stale target_ref; reobserve first"
                obj = cdp.send("DOM.resolveNode", {"backendNodeId": node["backend_id"]})["object"].get("objectId")
                result = cdp.send("Runtime.callFunctionOn", {"objectId": obj,
                    "functionDeclaration": clean, "returnByValue": True})
                html = result.get("result", {}).get("value", "")
            else:
                html = page.evaluate("() => (" + clean + ").call(document.documentElement)")
            if not isinstance(html, str):
                raise ValueError("DOM source unavailable")
            if before != self._identity(page, cdp) or page.url != url:
                self.invalidate()
                return "读取源码期间页面发生导航，请重新观察"
            limit = min(50000, max(100, int(max_chars)))
            return f"URL: {url}\n{html[:limit]}" + ("\n[truncated]" if len(html) > limit else "")
        except Exception as exc:
            return f"页面源码不可用：{str(exc)[:200]}"
        finally:
            self._disconnect(p, browser)

    @browser_thread
    def viewport_size(self):
        """返回当前页面 viewport 尺寸 (width, height)。"""
        try:
            p, browser = self._connect()
        except Exception:
            return None
        try:
            page = self._pick_page(browser)
            if page is None:
                return None
            vp = page.evaluate("({w: innerWidth, h: innerHeight})")
            return int(vp["w"]), int(vp["h"])
        except Exception:
            return None
        finally:
            self._disconnect(p, browser)

    def click_at_viewport(self, action, vx, vy):
        """兼容旧调用；新环境用 execute_viewport_action 传入对应观察的 metadata。"""
        from copy import copy
        action = copy(action)
        action.args = {**action.args, "x": vx, "y": vy}
        return self.execute_viewport_action(action)

    @browser_thread
    def execute_viewport_action(self, action, metadata=None):
        """仅在截图绑定仍有效时执行 CSS viewport 输入。"""
        started, dispatched = time.perf_counter(), False
        name, args = action.action, action.args
        point_scroll = name == "scroll" and ("x" in args or "y" in args)
        without_coordinates = name in {"press", "hotkey"} or (name == "scroll" and not point_scroll)
        if name not in {"click", "double_click", "right_click", "type", "drag_and_drop", "scroll", "press", "hotkey"}:
            return self._failure(f"Unsupported visual action: {name}", "unsupported", resolved="visual")
        binding = metadata if metadata is not None else self._observation_metadata
        try:
            self._validate_binding(binding)
            if binding.get("snapshot_id") != self._observation_metadata.get("snapshot_id"):
                raise BrowserActionError("Visual observation superseded", code="stale")
            if not without_coordinates and binding.get("screenshot_space") != "css_viewport":
                raise BrowserActionError("Expected a CSS viewport screenshot", code="unsupported")
            width, height = binding.get("screenshot_size", (0, 0))
            if without_coordinates:
                points = []
            elif name == "drag_and_drop":
                points = [(args.get("x1"), args.get("y1")), (args.get("x2"), args.get("y2"))]
            else:
                points = [(args.get("x"), args.get("y"))]
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                       and isinstance(y, (int, float)) and not isinstance(y, bool)
                       and math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height
                       for x, y in points):
                raise BrowserActionError("Visual coordinates outside screenshot", code="not_found")
        except (BrowserActionError, KeyError, TypeError, ValueError) as exc:
            return self._failure(exc, getattr(exc, "code", "stale"), resolved="visual")
        try:
            p, browser = self._connect()
        except Exception as exc:
            return self._failure(exc, "not_found", resolved="visual")
        try:
            page = self._pick_page(browser)
            if page is None:
                return self._failure("Task page unavailable", "stale", resolved="visual")
            if any(d["page"] is page for d in self._dialogs.values()):
                return self._failure("JavaScript dialog must be handled first", "blocked", **self.event_metadata())
            cdp = self._cdp(page)
            self._validate_binding(binding, page, cdp)
            viewport = page.evaluate("({w: innerWidth, h: innerHeight, x: scrollX, y: scrollY})")
            if not without_coordinates and [viewport["w"], viewport["h"]] != [width, height]:
                raise BrowserActionError("Viewport resized after observation", code="stale")
            if not without_coordinates and binding.get("scroll_offset") != [viewport.get("x", 0), viewport.get("y", 0)]:
                raise BrowserActionError("Viewport scrolled after observation", code="stale")
            vx, vy = points[0] if points else (None, None)
            if name in ("click", "double_click", "right_click"):
                dispatched = True
                page.mouse.click(vx, vy, button="right" if name == "right_click" else "left",
                                 click_count=2 if name == "double_click" else 1)
                self._settle(page)
            elif name == "type":
                dispatched = True
                page.mouse.click(vx, vy)
                if action.args.get("overwrite"):
                    page.keyboard.press("ControlOrMeta+A")
                    page.keyboard.press("Backspace")
                page.keyboard.insert_text(str(action.args.get("text", "")))
            elif name == "drag_and_drop":
                page.mouse.move(vx, vy)
                dispatched = True
                try:
                    page.mouse.down()
                    page.mouse.move(*points[1], steps=12)
                finally:
                    page.mouse.up()
            elif name == "scroll":
                delta = int(args.get("amount", 1)) * 300 * (-1 if args.get("direction", "down") == "up" else 1)
                dispatched = True
                if point_scroll:
                    page.mouse.move(vx, vy)
                    page.mouse.wheel(0, delta)
                else:
                    page.evaluate("delta => window.scrollBy(0, delta)", delta)
            elif name in {"press", "hotkey"}:
                key = (self._norm_key(str(args.get("key", ""))) if name == "press" else
                       "+".join(self._norm_key(k) for k in args.get("keys", [])))
                if not key:
                    raise BrowserActionError("Missing key", code="unsupported")
                dispatched = True
                page.keyboard.press(key)
            return {"status": "executed" if dispatched else "rejected", "dispatched": dispatched,
                    "channel": "browser_dom", "resolved": "visual", "action": name,
                    "coord": [vx, vy], "coordinate_space": "css_viewport",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        except Exception as exc:
            return self._failure(exc, getattr(exc, "code", "blocked"), dispatched, started, resolved="visual")
        finally:
            self._disconnect(p, browser)

    @browser_thread
    def download(self, url, filename=None):
        """Legacy read-only status query; never initiates HTTP transfers or changes files."""
        pending = self._find_pending_download(url)
        if pending is not None:
            return {"status": "in_progress", "dispatched": False, "action": "download", "url": url,
                    "download_id": pending["download_id"], "artifact_id": pending["artifact_id"],
                    "pending_downloads": [copy.deepcopy(pending)], "download_events": [copy.deepcopy(pending)]}
        return self._failure("Direct download is disabled; click a DOM element or use the browser UI",
                             "unsupported", action="download", url=url, verified=False)

    @staticmethod
    def _response_filename(disposition, url, content_type):
        from email.message import Message
        from email.utils import collapse_rfc2231_value
        from urllib.parse import urlsplit, unquote
        message = Message()
        message["Content-Disposition"] = disposition
        params = message.get_params(header="content-disposition", unquote=True) or []
        extended = [v for k, v in params if k.lower() == "filename" and isinstance(v, tuple)]
        regular = [v for k, v in params if k.lower() == "filename" and isinstance(v, str)]
        name = collapse_rfc2231_value(extended[0]) if extended else regular[0] if regular else ""
        name = name or unquote(urlsplit(url).path.rsplit("/", 1)[-1]) or "download"
        mime = content_type.split(";", 1)[0].strip().lower()
        if mime == "application/pdf" and not name.lower().endswith(".pdf"):
            name += ".pdf"
        elif mime == "text/html" and "." not in name:
            name += ".html"
        return BrowserSession._filename(name)

    @browser_thread
    def execute_target(self, action, target, allowed=None):
        """纯 target 描述执行：先 resolve ref → DOM 点击；否则 Playwright locator 匹配。

        全程 CDP，不动真实鼠标。allowed = 已暴露给模型的 ref 集合。
        """
        if action.action == "download":
            return self.download(str(action.args.get("url") or ""), filename=action.args.get("filename"))
        if action.action in {"scroll", "press", "hotkey"}:
            return self.execute_viewport_action(action)
        ref = self.resolve_target(target, allowed=allowed)
        if ref:
            action.args["target_ref"] = ref
            route = {"channel": "browser_dom", "snapshot_id": self.snapshot["snapshot_id"],
                     "target_ref": ref, "coordinate_space": "css_viewport"}
            return self.execute(action, route)
        if self.last_resolution_error == "ambiguous":
            return self._failure("Target matches multiple nodes", "ambiguous")
        if allowed is not None:
            return self._failure("Target not found in exposed nodes", "not_found")
        return self._execute_locator(action, target)

    def _locator_for(self, page, target):
        """按 target 描述里的引号短语，构造 Playwright locator（role/placeholder/text 依次试）。"""
        if not target:
            return None
        quoted = re.findall(
            r"[\"'`\u2018\u2019\u201c\u201d]([^\"'`\u2018\u2019\u201c\u201d]{2,80})"
            r"[\"'`\u2018\u2019\u201c\u201d]", target)
        roles = ("button", "link", "textbox", "searchbox", "combobox", "menuitem", "tab",
                 "checkbox", "radio", "option", "listbox", "switch", "slider")
        for q in quoted:
            q = q.strip()
            if not q:
                continue
            try:
                ph = page.get_by_placeholder(q)
                if ph.count() > 0:
                    return ph
            except Exception:
                pass
            for role in roles:
                try:
                    r = page.get_by_role(role, name=q, exact=True)
                    if r.count() > 0:
                        return r
                except Exception:
                    pass
            try:
                t = page.get_by_text(q, exact=True)
                if t.count() > 0:
                    return t
            except Exception:
                pass
        return None

    @browser_thread
    def _execute_locator(self, action, target):
        """CDP locator 执行浏览器动作（虚拟输入）。"""
        started, dispatched = time.perf_counter(), False
        try:
            self._validate_binding(self._observation_metadata)
        except BrowserActionError as exc:
            return self._failure(exc, exc.code, resolved="locator")
        try:
            p, browser = self._connect()
        except Exception as exc:
            return self._failure(exc, "not_found", resolved="locator")
        try:
            page = self._pick_page(browser)
            if page is None:
                return self._failure("Task page unavailable", "stale", resolved="locator")
            self._validate_binding(self._observation_metadata, page, self._cdp(page))
            name = action.action
            if name in ("press", "hotkey"):
                # 键盘动作不需要 locator
                dispatched = True
                if name == "press":
                    page.keyboard.press(self._norm_key(str(action.args.get("key", ""))))
                else:
                    keys = [self._norm_key(k) for k in (action.args.get("keys") or [])]
                    if keys:
                        page.keyboard.press("+".join(keys))
                dispatched = True
                return {"status": "executed", "dispatched": True, "channel": "browser_dom",
                        "resolved": "locator", "action": name,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
            loc = self._locator_for(page, target)
            if loc is None:
                return self._failure("Target not found in DOM", "not_found", resolved="locator")
            if loc.count() != 1:
                return self._failure("Target matches multiple elements", "ambiguous", resolved="locator")
            if name in ("click", "double_click", "right_click"):
                dispatched = True
                loc.first.click(button="right" if name == "right_click" else "left",
                                click_count=2 if name == "double_click" else 1, timeout=5000)
            elif name == "type":
                l = loc.first
                dispatched = True
                l.click(timeout=5000)
                if action.args.get("overwrite"):
                    page.keyboard.press("ControlOrMeta+A")
                    page.keyboard.press("Backspace")
                page.keyboard.insert_text(str(action.args.get("text", "")))
                dispatched = True
            elif name == "select":
                dispatched = True
                loc.first.select_option(str(action.args.get("option", "")), timeout=5000)
                dispatched = True
            else:
                return self._failure(f"Unsupported action {name}", "unsupported", resolved="locator")
            return {"status": "executed", "dispatched": True, "channel": "browser_dom",
                    "resolved": "locator", "action": name,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        except Exception as exc:
            return self._failure(exc, getattr(exc, "code", "blocked"), dispatched, started, resolved="locator")
        finally:
            self._disconnect(p, browser)

    @browser_thread
    def _scroll(self, action):
        """页面级滚动（window.scrollBy），无需 target。"""
        amount = int(action.args.get("amount", 1))
        delta = amount * 300 * (-1 if action.args.get("direction", "up") == "up" else 1)
        started = time.perf_counter()
        try:
            p, browser = self._connect()
        except Exception as exc:
            return {"status": "rejected", "dispatched": False, "channel": "browser_dom",
                    "error": str(exc)[:200]}
        try:
            page = self._pick_page(browser)
            if page is None:
                return {"status": "rejected", "dispatched": False, "channel": "browser_dom",
                        "error": "No page"}
            page.evaluate(f"window.scrollBy(0, {delta})")
            return {"status": "executed", "dispatched": True, "channel": "browser_dom",
                    "action": "scroll", "delta": delta,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        except Exception as exc:
            return {"status": "rejected", "dispatched": False, "channel": "browser_dom",
                    "error": str(exc)[:200]}
        finally:
            self._disconnect(p, browser)

    # ---- 视觉回退：grounding 屏幕坐标 → CDP 虚拟坐标点击（不动真实鼠标）----
    @staticmethod
    def _screen_to_viewport(page, sx, sy):
        """把 grounding 的屏幕像素坐标（设备像素，匹配 pyautogui 截图）转成页面 viewport 坐标。"""
        try:
            geo = page.evaluate("({sx: window.screenX, sy: window.screenY, "
                                "iw: window.innerWidth, ih: window.innerHeight, "
                                "oh: window.outerHeight, dpr: window.devicePixelRatio})")
        except Exception:
            return None
        dpr = geo.get("dpr") or 1
        chrome_css = (geo.get("oh", 0) or 0) - (geo.get("ih", 0) or 0)  # 标题栏+标签+地址栏
        vx = sx / dpr - (geo.get("sx", 0) or 0)
        vy = sy / dpr - (geo.get("sy", 0) or 0) - max(0, chrome_css)
        return int(round(vx)), int(round(vy))

    @browser_thread
    def click_at_screen(self, action, sx, sy):
        """视觉定位回退：在屏幕坐标 (sx, sy) 处用 CDP 虚拟点击/输入，不动真实鼠标。"""
        started, dispatched = time.perf_counter(), False
        try:
            p, browser = self._connect()
        except Exception as exc:
            return {"status": "rejected", "dispatched": False, "channel": "browser_dom",
                    "resolved": "visual", "error": str(exc)[:200]}
        try:
            page = self._pick_page(browser)
            if page is None:
                return {"status": "rejected", "dispatched": False, "channel": "browser_dom",
                        "resolved": "visual", "error": "No page"}
            v = self._screen_to_viewport(page, sx, sy)
            if v is None:
                return {"status": "rejected", "dispatched": False, "channel": "browser_dom",
                        "resolved": "visual", "error": "viewport geometry unavailable"}
            vx, vy = v
            name = action.action
            if name in ("click", "double_click", "right_click"):
                page.mouse.click(vx, vy, button="right" if name == "right_click" else "left",
                                 click_count=2 if name == "double_click" else 1)
                dispatched = True
            elif name == "type":
                page.mouse.click(vx, vy)
                if action.args.get("overwrite"):
                    page.keyboard.press("ControlOrMeta+A")
                    page.keyboard.press("Backspace")
                page.keyboard.insert_text(str(action.args.get("text", "")))
                dispatched = True
            else:
                return {"status": "rejected", "dispatched": False, "channel": "browser_dom",
                        "resolved": "visual", "error": f"unsupported visual action {name}"}
            return {"status": "executed", "dispatched": True, "channel": "browser_dom",
                    "resolved": "visual", "action": name, "coord": [vx, vy],
                    "coordinate_space": "css_viewport",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
        except Exception as exc:
            return {"status": "uncertain" if dispatched else "rejected", "dispatched": dispatched,
                    "channel": "browser_dom", "resolved": "visual", "error": str(exc)[:300]}
        finally:
            self._disconnect(p, browser)

    def invalidate(self):
        self.snapshot = None
        self.nodes = {}
        self._observation_metadata = {}

    @browser_thread
    def _close_connection(self):
        self._cancel_background_downloads()
        if self._owned_context is not None:
            try:
                self._owned_context.close()
            except Exception:
                pass
            self._owned_context = None
        browser, playwright = self._browser, self._playwright
        self._browser = self._playwright = None
        if browser is not None:
            self._disconnect(playwright, browser)
        self.invalidate()
        self._task_page_id = None
        self._known_page_ids.clear()
        self._async_objects.clear()
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    def close(self):
        if self._worker is None and self._browser is None:
            self.invalidate()
            return
        try:
            self._close_connection()
        finally:
            worker, self._worker = self._worker, None
            self._worker_thread = None
            if worker is not None:
                worker.shutdown(wait=True)
