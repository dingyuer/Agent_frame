"""History scans must leave the ASGI loop free for live task updates."""
import asyncio
import json
import threading

import pytest

from osworld_agent.viz import server


@pytest.fixture(autouse=True)
def clean_summary_cache():
    with server._local_summary_cache_lock:
        server._local_summary_cache.clear()
    yield
    with server._local_summary_cache_lock:
        server._local_summary_cache.clear()


def write_run(tmp_path, monkeypatch, *, task="搜索论文", outcome=None):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({"task": {"instruction": task},
                                "steps": [{"step": 1}],
                                "outcome": outcome or {"success": True, "score": 1}}),
                    encoding="utf-8")
    monkeypatch.setattr(server, "_local_run_files",
                        lambda: iter([("run-1", path, tmp_path, "")]))
    return path


@pytest.mark.parametrize("endpoint", ["runs", "traj"])
def test_large_history_parse_runs_outside_event_loop(tmp_path, monkeypatch, endpoint):
    write_run(tmp_path, monkeypatch)
    original_load = server.json.load
    release = threading.Event()

    async def exercise():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        loop_thread = threading.get_ident()
        worker_threads = []

        def slow_load(file):
            worker_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(entered.set)
            release.wait(timeout=3)
            return original_load(file)

        monkeypatch.setattr(server.json, "load", slow_load)
        request = asyncio.create_task(server.local_runs() if endpoint == "runs"
                                      else server.local_traj("run-1"))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            # A websocket/log coroutine can run while a JSON read is still pending.
            await asyncio.sleep(0)
            assert worker_threads and loop_thread not in worker_threads
            assert not request.done()
        finally:
            release.set()
        result = await request
        assert result["runs"][0]["instruction"] == "搜索论文" if endpoint == "runs" else result["ok"]

    asyncio.run(exercise())


def test_summary_cache_reuses_unchanged_json_and_invalidates_file_changes(tmp_path, monkeypatch):
    path = write_run(tmp_path, monkeypatch)
    original_load = server.json.load
    reads = []

    def counted_load(file):
        reads.append(file.name)
        return original_load(file)

    monkeypatch.setattr(server.json, "load", counted_load)
    assert asyncio.run(server.local_runs())["runs"][0]["shots"] == 0
    (tmp_path / "step_1.png").write_bytes(b"png")
    # Screenshot counts stay fresh even when the trajectory summary is cached.
    assert asyncio.run(server.local_runs())["runs"][0]["shots"] == 1
    assert len(reads) == 1
    path.write_text(json.dumps({"task": {"instruction": "下载另一篇论文"},
                                "steps": [{"step": 1}, {"step": 2}]}), encoding="utf-8")
    row = asyncio.run(server.local_runs())["runs"][0]
    assert row["instruction"] == "下载另一篇论文" and row["steps"] == 2
    assert len(reads) == 2


def test_invalid_partial_json_is_not_cached_and_recovers(tmp_path, monkeypatch):
    path = write_run(tmp_path, monkeypatch)
    path.write_text('{"task":', encoding="utf-8")
    assert asyncio.run(server.local_runs())["runs"][0]["instruction"] == ""
    assert not server._local_summary_cache
    path.write_text(json.dumps({"task": {"instruction": "恢复后"}}), encoding="utf-8")
    assert asyncio.run(server.local_runs())["runs"][0]["instruction"] == "恢复后"


def test_summary_cache_is_bounded_and_callers_cannot_change_it(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_LOCAL_SUMMARY_CACHE_LIMIT", 2)
    paths = []
    for index in range(3):
        path = tmp_path / f"run_{index}.json"
        path.write_text(json.dumps({"task": {"instruction": str(index)}}), encoding="utf-8")
        paths.append(path)
        summary = server._local_run_summary(path)
        summary["instruction"] = "tampered"
    assert len(server._local_summary_cache) == 2
    assert str(paths[0].resolve()) not in server._local_summary_cache
    assert server._local_run_summary(paths[-1])["instruction"] == "2"


def test_planning_failure_summary_and_missing_trajectory_contract(tmp_path, monkeypatch):
    path = write_run(tmp_path, monkeypatch, outcome={"termination_reason": "planning_failed"})
    data = json.loads(path.read_text(encoding="utf-8"))
    data["planning_calls"] = [{"validation_error": "first"}, {"validation_error": "last"}]
    path.write_text(json.dumps(data), encoding="utf-8")
    row = asyncio.run(server.local_runs())["runs"][0]
    assert row["termination_reason"] == "planning_failed"
    assert row["failure_reason"] == "last"
    assert asyncio.run(server.local_traj("missing")) == {"ok": False, "error": "轨迹不存在"}
    assert asyncio.run(server.local_traj("../run-1")) == {"ok": False, "error": "轨迹不存在"}
