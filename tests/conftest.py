import sys
import importlib.util
from pathlib import Path

# Bind tests to this checkout regardless of the clone directory name.
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "osworld_agent", root / "__init__.py", submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules["osworld_agent"] = module
spec.loader.exec_module(module)


def pytest_addoption(parser):
    parser.addoption("--run-browser-integration", action="store_true", default=False,
                     help="Run local Chromium fixtures; never calls a model endpoint")
