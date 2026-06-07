import sys
from unittest.mock import patch

import pytest


def test_ensure_gui_deps_noop_on_windows():
    from src.app import platform

    with patch.object(sys, "platform", "win32"):
        platform.ensure_gui_deps()


def test_u2net_setup_imports_without_fcntl():
    pytest.importorskip("src.roi.u2net_setup")
    import src.roi.u2net_setup as mod

    assert "fcntl" not in dir(mod)


def test_file_lock_context_manager(tmp_path):
    from src.utils.file_lock import exclusive_file_lock

    lock = tmp_path / "test.lock"
    with exclusive_file_lock(lock):
        assert lock.exists()
