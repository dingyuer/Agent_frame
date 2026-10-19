import sys
import importlib.util
from pathlib import Path
import pytest

# Bind tests to this checkout regardless of the clone directory name.
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "osworld_agent", root / "__init__.py", submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules["osworld_agent"] = module
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def isolated_run_history(tmp_path, monkeypatch):
    """Mock launchers and CLI tests must never populate Studio's real history."""
    from osworld_agent.viz import server
    logs = str(tmp_path / 'local_runs')
    monkeypatch.setenv('OSWORLD_LOCAL_RUNS_DIR', logs)
    # server reads its default at import time; isolate that path as well.
    monkeypatch.setattr(server, 'LOCAL_RUNS_DIR', logs)


def pytest_addoption(parser):
    parser.addoption("--run-browser-integration", action="store_true", default=False,
                     help="Run local Chromium fixtures; never calls a model endpoint")
