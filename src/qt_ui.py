import os
import time
import multiprocessing
import queue
import collections
from pathlib import Path
import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from src.recognizer import FaceRecognizer
from src.filter import StateFilter
from src.state_machine import FaceStateMachine
from src.bf_worker_process import run_bf_process


class ScaledPixmapLabel(QtWidgets.QLabel):
    def __init__(self, max_side: int = 40, parent=None):
        super().__init__(parent)
        self._src = None
        self._max_side = int(max_side)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def set_source_pixmap(self, pix: QtGui.QPixmap | None):
        self._src = pix
        self._refresh()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()

    def showEvent(self, e):
        super().showEvent(e)
        self._refresh()

    def _refresh(self):
        if self._src is None or self._src.isNull():
            self.clear()
            return
        r = self.contentsRect()
        if r.width() <= 0 or r.height() <= 0:
            return
        target_w = min(r.width(), self._max_side)
        target_h = min(r.height(), self._max_side)
        scaled = self._src.scaled(target_w, target_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(scaled)


class CameraWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    status_ready = QtCore.pyqtSignal(dict)
    char_ready = QtCore.pyqtSignal(str)
    control_ready = QtCore.pyqtSignal(str)

    def __init__(
        self,
        camera_index=0,
        analysis_interval=0.15,
        filter_window=7,
        hold_duration=1.5,
        vertical_hold_duration=0.8,
        vertical_filter_window=7,
        vertical_deadzone=0.08,
        parent=None,
    ):
        super().__init__(parent)
        self.camera_index = int(camera_index)
        self.analysis_interval = float(analysis_interval)
        self.filter_window = int(filter_window)
        self.hold_duration = float(hold_duration)
        self.vertical_hold_duration = float(vertical_hold_duration)
        self.vertical_filter_window = int(vertical_filter_window)
        self.vertical_deadzone = float(vertical_deadzone)
        self._stop = False

        self.recognizer = FaceRecognizer(enforce_detection=False)
        self.state_filter = StateFilter(window_size=self.filter_window)
        self.state_machine = FaceStateMachine(hold_duration=self.hold_duration)
        self.vertical_hist = collections.deque(maxlen=self.vertical_filter_window)
        self.vertical_state = None
        self.vertical_state_start = time.time()
        self.vertical_fired = False

    def stop(self):
        self._stop = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.status_ready.emit({"error": "Cannot open camera"})
            return

        last_analysis_time = 0.0
        cached_face = None
        tracked_box = None
        grab_failures = 0

        while not self._stop:
            ok, frame = cap.read()
            if not ok:
                grab_failures += 1
                if grab_failures == 30:
                    self.status_ready.emit({"error": "Failed to read camera frames (device busy or permission denied)"})
                self.msleep(10)
                continue
            grab_failures = 0

            frame = cv2.flip(frame, 1)
            now = time.time()

            if now - last_analysis_time >= self.analysis_interval:
                results = self.recognizer.analyze_frame(frame)
                cached_face = self._select_primary_face(results, tracked_box)
                tracked_box = cached_face.get("box") if cached_face else None
                last_analysis_time = now

            raw_emotion = None
            raw_position = None
            dominant_score = None
            box = None
            raw_vertical = None

            if cached_face:
                raw_emotion = cached_face.get("dominant_emotion")
                raw_position = cached_face.get("position")
                dominant_score = cached_face.get("dominant_score")
                box = cached_face.get("box")
                raw_vertical = cached_face.get("vertical_position")
            else:
                self.state_machine.reset()

            self.vertical_hist.append(raw_vertical)
            smoothed_vertical = None
            if self.vertical_hist:
                counts = collections.Counter(self.vertical_hist)
                smoothed_vertical = counts.most_common(1)[0][0]

            if smoothed_vertical != self.vertical_state:
                self.vertical_state = smoothed_vertical
                self.vertical_state_start = now
                self.vertical_fired = False

            if (
                (not self.vertical_fired)
                and self.vertical_state in ("up", "down")
                and (now - self.vertical_state_start) >= self.vertical_hold_duration
            ):
                self.vertical_fired = True
                self.control_ready.emit("RUN" if self.vertical_state == "up" else "STOP")

            smoothed_emotion, smoothed_position = self.state_filter.update(raw_emotion, raw_position, emotion_score=dominant_score)

            allow_code_input = smoothed_vertical == "middle"
            if not allow_code_input:
                self.state_machine.reset(current_time=now)
                triggered = None
            else:
                should_process = False
                if smoothed_position in ("left", "right"):
                    should_process = True
                elif smoothed_position == "center" and smoothed_emotion is not None:
                    should_process = True
                else:
                    self.state_machine.reset()

                triggered = None
                if should_process:
                    triggered = self.state_machine.process(smoothed_position, smoothed_emotion, current_time=now)
                    if triggered:
                        self.char_ready.emit(triggered)

            draw = frame.copy()
            green = (0, 255, 0)
            sky = (255, 200, 80)

            frame_h, frame_w = draw.shape[:2]
            x1 = frame_w // 3
            x2 = (frame_w * 2) // 3
            y1 = frame_h // 3
            y2 = (frame_h * 2) // 3
            line_color = (0, 180, 0)
            cv2.line(draw, (x1, 0), (x1, frame_h - 1), line_color, 1)
            cv2.line(draw, (x2, 0), (x2, frame_h - 1), line_color, 1)
            cv2.line(draw, (0, y1), (frame_w - 1, y1), line_color, 1)
            cv2.line(draw, (0, y2), (frame_w - 1, y2), line_color, 1)

            holding = self.state_machine.current_state
            holding_progress = (
                max(
                    0.0,
                    min(1.0, (now - self.state_machine.state_start_time) / self.state_machine.hold_duration),
                )
                if self.state_machine.current_state and (not self.state_machine.fired_for_state)
                else 0.0
            )

            if box:
                ibx = int(box.get("x", 0) or 0)
                iby = int(box.get("y", 0) or 0)
                ibw = int(box.get("w", 0) or 0)
                ibh = int(box.get("h", 0) or 0)
                if ibw > 0 and ibh > 0:
                    cv2.rectangle(draw, (ibx, iby), (ibx + ibw, iby + ibh), green, 2)
                    label_pos = smoothed_position if smoothed_position is not None else raw_position
                    label_emo = smoothed_emotion if smoothed_emotion is not None else raw_emotion
                    lp = label_pos if label_pos is not None else "?"
                    le = label_emo if label_emo is not None else "?"
                    cv2.putText(
                        draw,
                        f"{lp}/{le}",
                        (ibx, max(0, iby - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        green,
                        2,
                    )

            vertical_progress = (
                max(0.0, min(1.0, (now - self.vertical_state_start) / self.vertical_hold_duration))
                if (self.vertical_state in ("up", "down") and (not self.vertical_fired))
                else 0.0
            )
            vertical_action = "RUN" if self.vertical_state == "up" else ("STOP" if self.vertical_state == "down" else None)

            bar_x = 10
            bar_w = 220
            bar_h = 16
            hold_y = 20
            cv2.rectangle(draw, (bar_x, hold_y), (bar_x + bar_w, hold_y + bar_h), (80, 80, 80), -1)

            active_label = None
            active_progress = 0.0
            if vertical_action and (not self.vertical_fired):
                active_label = vertical_action
                active_progress = vertical_progress
            elif holding and (not self.state_machine.fired_for_state):
                active_label = holding
                active_progress = holding_progress

            bar_color = sky if (vertical_action and (not self.vertical_fired)) else green
            if active_label and active_progress > 0.0:
                cv2.rectangle(
                    draw,
                    (bar_x, hold_y),
                    (bar_x + int(bar_w * active_progress), hold_y + bar_h),
                    bar_color,
                    -1,
                )
                cv2.putText(
                    draw,
                    f"HOLD {active_label}",
                    (bar_x, max(0, hold_y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    bar_color,
                    2,
                )

            rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)

            status = {
                "raw_emotion": raw_emotion,
                "raw_position": raw_position,
                "dominant_score": dominant_score,
                "smoothed_emotion": smoothed_emotion,
                "smoothed_position": smoothed_position,
                "raw_vertical": raw_vertical,
                "smoothed_vertical": smoothed_vertical,
                "vertical_action": vertical_action,
                "vertical_progress": vertical_progress,
                "allow_code_input": allow_code_input,
                "holding": holding,
                "holding_progress": holding_progress,
                "triggered": triggered,
            }
            self.status_ready.emit(status)

            self.msleep(15)

        cap.release()

    def _select_primary_face(self, faces: list, tracked_box: dict | None):
        if not faces:
            return None
        valid = []
        for f in faces:
            b = f.get("box") or {}
            w = int(b.get("w", 0) or 0)
            h = int(b.get("h", 0) or 0)
            if w <= 0 or h <= 0:
                continue
            valid.append(f)
        if not valid:
            return faces[0]
        if tracked_box:
            best = None
            best_iou = -1.0
            for f in valid:
                iou = self._box_iou(tracked_box, f.get("box") or {})
                if iou > best_iou:
                    best_iou = iou
                    best = f
            if best is not None and best_iou >= 0.1:
                return best
        return max(valid, key=lambda f: int((f.get("box") or {}).get("w", 0) or 0) * int((f.get("box") or {}).get("h", 0) or 0))

    def _box_iou(self, a: dict, b: dict) -> float:
        ax1 = float(a.get("x", 0) or 0)
        ay1 = float(a.get("y", 0) or 0)
        ax2 = ax1 + float(a.get("w", 0) or 0)
        ay2 = ay1 + float(a.get("h", 0) or 0)

        bx1 = float(b.get("x", 0) or 0)
        by1 = float(b.get("y", 0) or 0)
        bx2 = bx1 + float(b.get("w", 0) or 0)
        by2 = by1 + float(b.get("h", 0) or 0)

        iw = min(ax2, bx2) - max(ax1, bx1)
        ih = min(ay2, by2) - max(ay1, by1)
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = float(iw * ih)
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union


class BFQueueReader(QtCore.QThread):
    output_ready = QtCore.pyqtSignal(str)
    status_ready = QtCore.pyqtSignal(dict)
    tape_ready = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal()

    def __init__(self, q, parent=None):
        super().__init__(parent)
        self._stop = False
        self.q = q

    def stop(self):
        self._stop = True

    def run(self):
        while not self._stop:
            try:
                msg = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            except Exception as e:
                self.status_ready.emit({"error": str(e)})
                self.finished.emit()
                return

            t = msg.get("type")
            if t == "output":
                self.output_ready.emit(msg.get("output", ""))
            elif t == "status":
                self.status_ready.emit(msg)
            elif t == "tape":
                self.tape_ready.emit(msg)
            elif t == "error":
                self.status_ready.emit(msg)
            elif t == "finished":
                self.finished.emit()
                return


class BFProcessController(QtCore.QObject):
    output_ready = QtCore.pyqtSignal(str)
    status_ready = QtCore.pyqtSignal(dict)
    tape_ready = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctx = multiprocessing.get_context("spawn")
        self._proc = None
        self._q = None
        self._ctrl = None
        self._reader = None

    def is_running(self):
        return self._proc is not None and self._proc.is_alive()

    def start(self, code: str):
        self.stop()
        self._q = self._ctx.Queue()
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        self._ctrl = parent_conn
        self._proc = self._ctx.Process(target=run_bf_process, args=(code, self._q, child_conn), daemon=True)
        self._proc.start()

        self._reader = BFQueueReader(self._q)
        self._reader.output_ready.connect(self.output_ready)
        self._reader.status_ready.connect(self.status_ready)
        self._reader.tape_ready.connect(self.tape_ready)
        self._reader.finished.connect(self._on_finished)
        self._reader.start()

    def stop(self):
        if self._reader:
            self._reader.stop()
            self._reader.wait(500)
            self._reader = None

        if self._ctrl:
            try:
                self._ctrl.send("STOP")
            except Exception:
                pass
            try:
                self._ctrl.close()
            except Exception:
                pass
            self._ctrl = None

        if self._proc:
            self._proc.join(timeout=0.5)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=0.5)
            self._proc = None

        self._q = None

    def _on_finished(self):
        self.finished.emit()
        self.stop()


class FaceFuckMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceFuck")
        self.setObjectName("FaceFuckMainWindow")

        self.code_buffer = ""
        self.camera_worker = None
        self.bf_controller = BFProcessController(self)
        self._example_image_dir = Path(__file__).resolve().parent.parent / "icon"

        central = QtWidgets.QWidget(self)
        central.setObjectName("central")
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        self.tape_label = QtWidgets.QLabel("Brainfuck Tape")
        self.tape_label.setObjectName("tapeTitle")
        root.addWidget(self.tape_label, stretch=0)

        self.tape_table = QtWidgets.QTableWidget()
        self.tape_table.setObjectName("tapeTable")
        self.tape_table.setRowCount(1)
        self.tape_table.setColumnCount(31)
        self.tape_table.verticalHeader().setVisible(False)
        self.tape_table.horizontalHeader().setVisible(False)
        self.tape_table.setShowGrid(False)
        self.tape_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tape_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.tape_table.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tape_table.setFixedHeight(70)
        for c in range(self.tape_table.columnCount()):
            self.tape_table.setColumnWidth(c, 38)
        self.tape_table.setRowHeight(0, 36)
        root.addWidget(self.tape_table, stretch=0)

        self.bf_status_label = QtWidgets.QLabel()
        self.bf_status_label.setObjectName("bfStatus")
        root.addWidget(self.bf_status_label, stretch=0)

        self.code_view = QtWidgets.QPlainTextEdit()
        self.code_view.setObjectName("codeView")
        self.code_view.setReadOnly(True)
        self.code_view.setMaximumBlockCount(1000)
        self.code_view.setPlaceholderText("Brainfuck code appears here")
        self.code_view.setFixedHeight(120)
        root.addWidget(self.code_view, stretch=0)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, stretch=1)

        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setSpacing(10)
        body.addLayout(left_panel, stretch=3)

        self.reference_table = QtWidgets.QTableWidget()
        self.reference_table.setObjectName("referenceTable")
        self.reference_table.setColumnCount(5)
        self.reference_table.setHorizontalHeaderLabels(["Sample", "Output", "Name", "Trigger", "Effect"])
        self.reference_table.verticalHeader().setVisible(False)
        self.reference_table.setShowGrid(False)
        self.reference_table.setAlternatingRowColors(True)
        self.reference_table.setWordWrap(True)
        self.reference_table.setTextElideMode(QtCore.Qt.ElideNone)
        self.reference_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.reference_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.reference_table.setFocusPolicy(QtCore.Qt.NoFocus)
        self.reference_table.horizontalHeader().setStretchLastSection(True)
        self.reference_table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.reference_table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.reference_table.setMinimumWidth(560)
        header = self.reference_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        left_panel.addWidget(self.reference_table, stretch=1)

        middle = QtWidgets.QVBoxLayout()
        middle.setSpacing(10)
        body.addLayout(middle, stretch=7)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setObjectName("video")
        self.video_label.setMinimumSize(720, 420)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setText("Camera")
        middle.addWidget(self.video_label, stretch=1)

        self.control_hint = QtWidgets.QLabel("Control: Face up = Run; Face down = Stop; Middle = Input")
        self.control_hint.setObjectName("hint")
        middle.addWidget(self.control_hint, stretch=0)

        self.hold_text = QtWidgets.QLabel("HOLD: -")
        self.hold_text.setObjectName("holdText")
        middle.addWidget(self.hold_text, stretch=0)

        self.hold_bar = QtWidgets.QProgressBar()
        self.hold_bar.setObjectName("holdBar")
        self.hold_bar.setRange(0, 100)
        self.hold_bar.setValue(0)
        self.hold_bar.setTextVisible(False)
        self.hold_bar.setFixedHeight(12)
        middle.addWidget(self.hold_bar, stretch=0)

        self.info_label = QtWidgets.QLabel()
        self.info_label.setObjectName("debug")
        self.info_label.setWordWrap(True)
        middle.addWidget(self.info_label, stretch=0)

        self.bf_controller.output_ready.connect(self.on_bf_output)
        self.bf_controller.status_ready.connect(self.on_bf_status)
        self.bf_controller.tape_ready.connect(self.on_bf_tape)
        self.bf_controller.finished.connect(self.on_bf_finished)

        self._init_reference_table()
        self._apply_style()
        self.start_camera()

    def _apply_style(self):
        self.setStyleSheet(
            """
            QWidget#central { background: #0b0f17; color: #e6e9ef; }
            QLabel#tapeTitle { font-size: 13px; font-weight: 600; color: #e6e9ef; padding: 2px 0; }
            QLabel#bfStatus { color: #c8d0dd; padding: 6px 10px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; }
            QPlainTextEdit#codeView { background: #0f1522; border: 1px solid rgba(255,255,255,0.10); border-radius: 12px; padding: 12px; color: #e6e9ef; selection-background-color: rgba(80,200,255,0.35); font-family: Consolas, 'Cascadia Mono', monospace; font-size: 16px; line-height: 1.40; }
            QLabel#video { background: #0f1522; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; }
            QLabel#hint { color: #a8b3c7; padding: 0 2px; }
            QLabel#holdText { font-weight: 600; color: #e6e9ef; padding: 0 2px; }
            QLabel#debug { color: #93a3bc; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 8px 10px; }

            QTableWidget#referenceTable { background: #0f1522; alternate-background-color: #111a2b; border: 1px solid rgba(255,255,255,0.10); border-radius: 12px; gridline-color: rgba(255,255,255,0.06); color: #e6e9ef; }
            QTableWidget#referenceTable::item { padding: 10px 10px; border-bottom: 1px solid rgba(255,255,255,0.06); color: #e6e9ef; }
            QTableWidget#referenceTable QHeaderView::section { background: #111a2b; color: #e6e9ef; border: none; padding: 10px 10px; font-weight: 700; }
            QTableWidget#referenceTable QTableCornerButton::section { background: #111a2b; border: none; }

            QTableWidget#tapeTable { background: #0f1522; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; }
            QTableWidget#tapeTable::item { background: rgba(255,255,255,0.02); color: #e6e9ef; border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; margin: 2px; font-size: 12px; }

            QProgressBar#holdBar { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; }
            QProgressBar#holdBar::chunk { background-color: rgb(0,255,0); border-radius: 6px; }

            QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
            QScrollBar::handle:vertical { background: rgba(255,255,255,0.15); border-radius: 5px; min-height: 28px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
            QScrollBar::handle:horizontal { background: rgba(255,255,255,0.15); border-radius: 5px; min-width: 28px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
            """
        )

    def _sample_pixmap(self, key: str) -> QtGui.QPixmap | None:
        if not key:
            return None
        base = self._example_image_dir
        keys = [key]
        if key == "middle":
            keys.append("center")
        elif key == "center":
            keys.append("middle")
        candidates = []
        for k in keys:
            candidates.extend(
                [
                    base / f"{k}.jpg",
                    base / f"{k}.png",
                ]
            )
        for p in candidates:
            if p.exists():
                pix = QtGui.QPixmap(str(p))
                if not pix.isNull():
                    return pix
        return None

    def _init_reference_table(self):
        rows = [
            {"sample": "up", "out": "RUN", "name": "RUN", "trigger": "Face box ≥50% in top third", "effect": "Start Brainfuck execution"},
            {"sample": "down", "out": "STOP", "name": "STOP", "trigger": "Face box ≥50% in bottom third", "effect": "Stop Brainfuck execution"},
            {"sample": "center", "out": "INPUT", "name": "INPUT_GATE", "trigger": "Face box ≥50% in middle third", "effect": "Only then allow emotion/left/right inputs"},
            {"sample": "left", "out": "[", "name": "LOOP_START", "trigger": "left", "effect": "Loop start"},
            {"sample": "right", "out": "]", "name": "LOOP_END", "trigger": "right", "effect": "Loop end"},
            {"sample": "happy", "out": "+", "name": "INC", "trigger": "happy", "effect": "Increment cell"},
            {"sample": "sad", "out": "-", "name": "DEC", "trigger": "sad", "effect": "Decrement cell"},
            {"sample": "angry", "out": "<", "name": "PTR_LEFT", "trigger": "angry", "effect": "Move pointer left"},
            {"sample": "surprise", "out": ">", "name": "PTR_RIGHT", "trigger": "surprise", "effect": "Move pointer right"},
            {"sample": "fear", "out": ",", "name": "IN", "trigger": "fear", "effect": "Read 1 byte (may block)"},
            {"sample": "disgust", "out": ".", "name": "OUT", "trigger": "disgust", "effect": "Write 1 byte"},
            {"sample": "neutral", "out": "BACKSPACE", "name": "BACKSPACE", "trigger": "neutral", "effect": "Delete last symbol"},
        ]

        self.reference_table.clearContents()
        self.reference_table.setRowCount(len(rows))
        self.reference_table.setIconSize(QtCore.QSize(56, 56))

        for r, item in enumerate(rows):
            sample_key = item.get("sample", "")
            pix = self._sample_pixmap(sample_key)
            if pix is not None:
                lbl = ScaledPixmapLabel(max_side=60)
                lbl.set_source_pixmap(pix)
                self.reference_table.setCellWidget(r, 0, lbl)
            else:
                self.reference_table.setItem(r, 0, QtWidgets.QTableWidgetItem(""))

            out_item = QtWidgets.QTableWidgetItem(item["out"])
            out_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.reference_table.setItem(r, 1, out_item)

            name_item = QtWidgets.QTableWidgetItem(item["name"])
            name_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.reference_table.setItem(r, 2, name_item)

            trig = item["trigger"] if item["trigger"] else ""
            self.reference_table.setItem(r, 3, QtWidgets.QTableWidgetItem(trig))
            self.reference_table.setItem(r, 4, QtWidgets.QTableWidgetItem(item["effect"]))
            self.reference_table.setRowHeight(r, 64)

        self.reference_table.resizeColumnsToContents()
        self.reference_table.setColumnWidth(0, max(68, self.reference_table.columnWidth(0)))
        self.reference_table.setColumnWidth(3, max(200, self.reference_table.columnWidth(3)))
        self.reference_table.setColumnWidth(4, max(260, self.reference_table.columnWidth(4)))
        self.reference_table.resizeRowsToContents()
        total_w = sum(self.reference_table.columnWidth(i) for i in range(self.reference_table.columnCount()))
        self.reference_table.setMinimumWidth(total_w + 40)

    def closeEvent(self, event):
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker.wait(1000)
        if self.bf_controller:
            self.bf_controller.stop()
        super().closeEvent(event)

    def start_camera(self):
        if self.camera_worker:
            return
        self.camera_worker = CameraWorker()
        self.camera_worker.frame_ready.connect(self.on_frame_ready)
        self.camera_worker.status_ready.connect(self.on_status_ready)
        self.camera_worker.char_ready.connect(self.on_char_ready)
        self.camera_worker.control_ready.connect(self.on_control_ready)
        self.camera_worker.start()

    @QtCore.pyqtSlot(QtGui.QImage)
    def on_frame_ready(self, img: QtGui.QImage):
        pix = QtGui.QPixmap.fromImage(img)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    @QtCore.pyqtSlot(dict)
    def on_status_ready(self, status: dict):
        if "error" in status:
            self.info_label.setText(status["error"])
            return

        raw_emotion = status.get("raw_emotion")
        raw_position = status.get("raw_position")
        dominant_score = status.get("dominant_score")
        smoothed_emotion = status.get("smoothed_emotion")
        smoothed_position = status.get("smoothed_position")
        raw_vertical = status.get("raw_vertical")
        smoothed_vertical = status.get("smoothed_vertical")
        vertical_action = status.get("vertical_action")
        vertical_progress = status.get("vertical_progress", 0.0)
        allow_code_input = status.get("allow_code_input")
        holding = status.get("holding")
        progress = status.get("holding_progress", 0.0)
        triggered = status.get("triggered")

        score_text = ""
        if dominant_score is not None:
            score_text = f"{dominant_score:.1f}%"

        self.info_label.setText(
            "Recognition\n"
            f"- Raw      : emotion={raw_emotion}({score_text}) position={raw_position}\n"
            f"- Smoothed : emotion={smoothed_emotion} position={smoothed_position}\n"
            f"- Vertical : raw={raw_vertical} smoothed={smoothed_vertical} allow_input={allow_code_input}\n"
            f"- Control  : {vertical_action}\n"
            f"- Holding  : {holding}\n"
        )

        active_label = None
        active_progress = 0.0
        if vertical_action and vertical_progress > 0.0:
            active_label = vertical_action
            active_progress = vertical_progress
        elif holding and progress > 0.0:
            active_label = holding
            active_progress = progress
        elif triggered:
            active_label = triggered
            active_progress = 0.0

        self.hold_text.setText(f"HOLD: {active_label}" if active_label else "HOLD: -")
        self.hold_bar.setValue(int(active_progress * 100))
        if vertical_action and vertical_progress > 0.0:
            self.hold_bar.setStyleSheet("QProgressBar::chunk { background-color: rgb(80,200,255); }")
        else:
            self.hold_bar.setStyleSheet("QProgressBar::chunk { background-color: rgb(0,255,0); }")

    @QtCore.pyqtSlot(str)
    def on_char_ready(self, ch: str):
        if ch == "BACKSPACE":
            self.code_buffer = self.code_buffer[:-1]
        else:
            self.code_buffer += ch
        self.code_view.setPlainText(self.code_buffer)

    @QtCore.pyqtSlot(str)
    def on_control_ready(self, cmd: str):
        if cmd == "RUN":
            if not self.bf_controller.is_running():
                self.bf_status_label.setText("BF: starting")
                self.bf_controller.start(self.code_buffer)
        elif cmd == "STOP":
            if self.bf_controller.is_running():
                self.bf_status_label.setText("BF: stopped")
                self.bf_controller.stop()

    @QtCore.pyqtSlot(str)
    def on_bf_output(self, out: str):
        return

    @QtCore.pyqtSlot(dict)
    def on_bf_status(self, st: dict):
        if st.get("type") == "error" or "error" in st:
            err = st.get("error", "")
            err_type = st.get("error_type", "")
            kind = st.get("kind", "")
            pc = st.get("pc")
            near = st.get("near", "")
            msg = "Runtime error"
            if err_type:
                msg += f"({err_type})"
            if kind:
                msg += f"[{kind}]"
            if pc is not None:
                msg += f": pc={pc}"
            if err:
                msg += f" {err}"
            if near:
                msg += f" near='{near}'"
            self.bf_status_label.setText(msg)
            return
        pc = st.get("pc")
        reason = st.get("blocked_reason")
        running = bool(st.get("hit_step_limit"))
        status = "running" if running else reason
        self.bf_status_label.setText(f"BF: pc={pc} status={status}")

    @QtCore.pyqtSlot(dict)
    def on_bf_tape(self, msg: dict):
        dp = int(msg.get("dp", 0))
        offset = int(msg.get("offset", 0))
        cells = msg.get("cells", []) or []
        cols = self.tape_table.columnCount()
        if len(cells) != cols:
            if len(cells) < cols:
                cells = list(cells) + [0] * (cols - len(cells))
            else:
                cells = list(cells[:cols])

        self.tape_label.setText(f"Brainfuck Tape (offset={offset}, dp={dp})")

        dp_col = dp - offset
        for c in range(cols):
            val = int(cells[c])
            item = self.tape_table.item(0, c)
            if item is None:
                item = QtWidgets.QTableWidgetItem()
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.tape_table.setItem(0, c, item)
            item.setText(str(val))
            if c == dp_col:
                item.setBackground(QtGui.QColor(255, 240, 150))
            else:
                item.setBackground(QtGui.QColor(0, 0, 0, 0))

    @QtCore.pyqtSlot()
    def on_bf_finished(self):
        if self.bf_controller.is_running():
            return
        if "running" in (self.bf_status_label.text() or ""):
            self.bf_status_label.setText("BF: finished")


def run_qt_app():
    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")
    app.setFont(QtGui.QFont("Segoe UI", 10))
    win = FaceFuckMainWindow()
    win.resize(1200, 800)
    win.show()
    return app.exec_()
