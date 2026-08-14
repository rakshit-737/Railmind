import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "agents"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    # Tests must never hit the real Claude API just because the developer has
    # a key exported; the LLM path is exercised via monkeypatched stubs only.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TWIN_BASE_URL", raising=False)
