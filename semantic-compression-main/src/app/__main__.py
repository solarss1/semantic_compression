"""Запуск: python -m src.app"""

from src.app.platform import ensure_gui_deps, prepare_qt_platform

prepare_qt_platform()
ensure_gui_deps()

from src.app.main_window import run_app

if __name__ == "__main__":
    run_app()
