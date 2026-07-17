"""pytest 全局 fixture：暴露 golden 轨迹数据给所有测试。"""
import sys
from pathlib import Path

import pytest

# 让 tests 能 import agentic_rl 包
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agentic_rl.utils.fixtures import fixture_names, load_fixture  # noqa: E402


@pytest.fixture(scope="session")
def all_fixtures():
    return {name: load_fixture(name) for name in fixture_names()}


@pytest.fixture(scope="session")
def retail_success():
    return load_fixture("retail_success")


@pytest.fixture(scope="session")
def telecom_success():
    return load_fixture("telecom_success_dualctrl")


@pytest.fixture(scope="session")
def telecom_fail():
    return load_fixture("telecom_fail_dualctrl")
