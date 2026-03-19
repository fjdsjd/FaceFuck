import os

os.environ.setdefault("DEEPFACE_HOME", os.path.dirname(os.path.abspath(__file__)))

from src.qt_ui import run_qt_app


if __name__ == "__main__":
    raise SystemExit(run_qt_app())
