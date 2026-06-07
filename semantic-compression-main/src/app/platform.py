"""Перевірка системних залежностей GUI перед запуском Qt."""

from __future__ import annotations

import ctypes.util
import os
import sys


def has_xcb_cursor() -> bool:
    if ctypes.util.find_library("xcb-cursor"):
        return True
    try:
        ctypes.CDLL("libxcb-cursor.so.0")
        return True
    except OSError:
        return False


def _linux_gui_install_hint() -> str:
    if os.path.exists("/etc/debian_version"):
        return "sudo apt install libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 libgl1"
    if os.path.exists("/etc/fedora-release"):
        return "sudo dnf install xcb-util-cursor libxkbcommon-x11 mesa-libGL"
    if os.path.exists("/etc/arch-release"):
        return "sudo pacman -S libxcb libxkbcommon libgl"
    return "встановіть libxcb-cursor (пакет libxcb-cursor0 на Debian/Ubuntu)"


def prepare_qt_platform() -> None:
    if sys.platform != "linux":
        return
    if os.environ.get("QT_QPA_PLATFORM"):
        return
    if os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "wayland")


def ensure_gui_deps() -> None:
    """Перевірки перед QApplication. На Windows/macOS додаткових кроків немає."""
    if sys.platform == "linux":
        _ensure_linux_gui_deps()


def _ensure_linux_gui_deps() -> None:
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return
    if os.environ.get("WAYLAND_DISPLAY"):
        prepare_qt_platform()
        return
    if has_xcb_cursor():
        return

    hint = _linux_gui_install_hint()
    print(
        "\nПомилка: для PyQt6 на Linux/X11 потрібна системна бібліотека libxcb-cursor.\n"
        f"  {hint}\n\n"
        "Після встановлення перезапустіть: make app\n"
        "Або (сесія Wayland): QT_QPA_PLATFORM=wayland make app\n",
        file=sys.stderr,
    )
    sys.exit(1)


ensure_linux_gui_deps = ensure_gui_deps
