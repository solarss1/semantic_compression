"""Головне вікно desktop-додатку."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.app.models import AppSettings, CompressionOutput, SessionState
from src.app.service import (
    compress_classical,
    compress_semantic,
    compression_diff,
    extract_roi_mask,
    mask_to_rgb,
    quality_map_rgb,
    roi_overlay,
    sam_prompts_overlay,
)
from src.app.widgets import ImageViewer, StatsPanel
from src.app.worker import TaskWorker
from src.utils.config import load_config, project_root
from src.evaluation.metrics import raw_image_bytes
from src.utils.io import load_image, save_compression_result


def _samples_dir(config: dict) -> Path:
    raw = config.get("app", {}).get("samples_dir", "data/samples")
    p = Path(raw)
    return p if p.is_absolute() else project_root() / p


IMAGE_FILTER = "Зображення (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Семантичне стиснення зображень — ROI")
        self.resize(1280, 820)

        self._base_config = load_config(project_root() / "configs" / "default.yaml")
        self._samples_dir = _samples_dir(self._base_config)
        self._session = SessionState(settings=AppSettings.from_config(self._base_config))
        self._worker: TaskWorker | None = None
        self._active_result: CompressionOutput | None = None
        self._view_base: np.ndarray | None = None
        self._sam_live_box: tuple[int, int, int, int] | None = None

        self._build_ui()
        self._build_menu()
        self._sync_settings_from_ui()
        self._update_actions()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        left.setMaximumWidth(340)
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(self._build_file_group())
        left_layout.addWidget(self._build_roi_group())
        left_layout.addWidget(self._build_semantic_group())
        left_layout.addWidget(self._build_classical_group())
        left_layout.addWidget(self._build_actions_group())
        left_layout.addStretch()

        center = QWidget()
        center_layout = QVBoxLayout(center)

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Перегляд:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(
            [
                "Оригінал",
                "Стиснене",
                "Маска ROI",
                "ROI на зображенні",
                "Різниця (×10)",
                "Карта якості",
            ]
        )
        self._view_combo.currentIndexChanged.connect(self._refresh_view)
        view_row.addWidget(self._view_combo, stretch=1)

        view_row.addWidget(QLabel("Результат:"))
        self._result_combo = QComboBox()
        self._result_combo.addItems(["Класичне", "Семантичне (ROI)"])
        self._result_combo.currentIndexChanged.connect(self._on_result_changed)
        view_row.addWidget(self._result_combo)

        center_layout.addLayout(view_row)
        self._viewer = ImageViewer()
        self._viewer.image_clicked.connect(self._on_sam_image_click)
        self._viewer.box_drawn.connect(self._on_sam_box_drawn)
        self._viewer.box_dragging.connect(self._on_sam_box_dragging)
        self._viewer.box_drag_ended.connect(self._on_sam_box_drag_ended)
        center_layout.addWidget(self._viewer, stretch=1)

        self._path_label = QLabel("Файл: —")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet("color: #666; font-size: 11px;")
        center_layout.addWidget(self._path_label)

        right = QWidget()
        right.setMaximumWidth(320)
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("<b>Метрики стиснення</b>"))
        self._stats = StatsPanel()
        right_layout.addWidget(self._stats)
        right_layout.addStretch()

        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Готово")

    def _build_file_group(self) -> QGroupBox:
        box = QGroupBox("Зображення")
        layout = QVBoxLayout(box)
        self._btn_open = QPushButton("Відкрити файл…")
        self._btn_open.clicked.connect(self._open_image)
        self._btn_sample = QPushButton("Зразки (PNG, без стиснення)")
        self._btn_sample.clicked.connect(self._open_sample)
        layout.addWidget(self._btn_open)
        layout.addWidget(self._btn_sample)
        layout.addWidget(
            QLabel(
                "<small>Для коректного порівняння використовуйте "
                "нестиснені PNG/TIFF.</small>"
            )
        )
        return box

    def _build_roi_group(self) -> QGroupBox:
        box = QGroupBox("ROI (deep learning)")
        layout = QVBoxLayout(box)
        form = QFormLayout()
        self._roi_method = QComboBox()
        for value, label in [
            ("saliency_u2net", "U²-Net (салієнтність)"),
            ("saliency_resnet", "ResNet (baseline)"),
            ("segmentation_weighted", "DeepLab + ваги класів"),
            ("segmentation_binary", "DeepLab (бінарна)"),
            ("combined", "Combined (U²-Net + сегментація)"),
            ("ultralytics_sam", "Ultralytics SAM (клік / bbox)"),
        ]:
            self._roi_method.addItem(label, value)
        self._roi_method.currentIndexChanged.connect(self._on_roi_method_changed)
        self._device = QComboBox()
        self._device.addItems(["cpu", "cuda"])
        self._mask_sigma = QSpinBox()
        self._mask_sigma.setRange(0, 30)
        self._mask_sigma.setValue(5)
        form.addRow("Метод ROI:", self._roi_method)
        form.addRow("Пристрій:", self._device)
        form.addRow("Згладжування σ:", self._mask_sigma)
        layout.addLayout(form)

        self._sam_box = QGroupBox("Ultralytics SAM")
        sam_form = QFormLayout(self._sam_box)
        self._sam_model = QComboBox()
        for name in ("mobile_sam.pt", "sam_b.pt", "sam_l.pt"):
            self._sam_model.addItem(name, name)
        self._sam_prompt_mode = QComboBox()
        self._sam_prompt_mode.addItem("Клік (точка)", "point")
        self._sam_prompt_mode.addItem("Прямокутник (drag)", "box")
        self._sam_prompt_mode.currentIndexChanged.connect(self._on_roi_method_changed)
        self._sam_click_mode = QComboBox()
        self._sam_click_mode.addItem("Позитивний (+)", 1)
        self._sam_click_mode.addItem("Негативний (−)", 0)
        self._btn_clear_sam_prompts = QPushButton("Очистити промпти")
        self._btn_clear_sam_prompts.clicked.connect(self._clear_sam_prompts)
        self._sam_show_prompts = QCheckBox("Показувати кліки та bbox")
        self._sam_show_prompts.setChecked(True)
        self._sam_show_prompts.toggled.connect(self._on_sam_show_prompts_toggled)
        self._sam_prompts_label = QLabel("Промпти: немає")
        self._sam_prompts_label.setWordWrap(True)
        self._sam_prompts_label.setStyleSheet("color: #666; font-size: 11px;")
        sam_form.addRow("Модель:", self._sam_model)
        sam_form.addRow("Режим:", self._sam_prompt_mode)
        sam_form.addRow("Тип кліку:", self._sam_click_mode)
        sam_form.addRow("", self._btn_clear_sam_prompts)
        sam_form.addRow("", self._sam_show_prompts)
        sam_form.addRow("", self._sam_prompts_label)
        sam_form.addRow(
            "",
            QLabel(
                "<small>У режимі «Оригінал»: кліки уточнюють одну маску "
                "(+ / −). Bbox — окремий промпт. Після змін натисніть "
                "«Виділити ROI» знову.</small>"
            ),
        )
        layout.addWidget(self._sam_box)
        self._sam_box.setVisible(False)
        return box

    def _build_semantic_group(self) -> QGroupBox:
        box = QGroupBox("Семантичне стиснення")
        form = QFormLayout(box)
        self._q_roi = QSpinBox()
        self._q_roi.setRange(1, 100)
        self._q_roi.setValue(85)
        self._q_bg = QSpinBox()
        self._q_bg.setRange(1, 100)
        self._q_bg.setValue(35)
        self._tile = QSpinBox()
        self._tile.setRange(8, 128)
        self._tile.setSingleStep(8)
        self._tile.setValue(32)
        form.addRow("Якість ROI:", self._q_roi)
        form.addRow("Якість фону:", self._q_bg)
        form.addRow("Розмір тайла:", self._tile)
        return box

    def _build_classical_group(self) -> QGroupBox:
        box = QGroupBox("Класичне стиснення")
        form = QFormLayout(box)
        self._classical_method = QComboBox()
        self._classical_method.addItems(["jpeg", "webp", "png"])
        self._classical_q = QSpinBox()
        self._classical_q.setRange(1, 100)
        self._classical_q.setValue(50)
        form.addRow("Алгоритм:", self._classical_method)
        form.addRow("Якість:", self._classical_q)
        return box

    def _build_actions_group(self) -> QGroupBox:
        box = QGroupBox("Дії")
        layout = QVBoxLayout(box)
        self._btn_roi = QPushButton("1. Виділити ROI")
        self._btn_classical = QPushButton("2. Стиснути (класичне)")
        self._btn_semantic = QPushButton("3. Стиснути (семантичне)")
        self._btn_save = QPushButton("Зберегти стиснене…")
        self._btn_reset = QPushButton("Скинути ROI і результати")
        self._btn_roi.clicked.connect(self._run_roi)
        self._btn_classical.clicked.connect(self._run_classical)
        self._btn_semantic.clicked.connect(self._run_semantic)
        self._btn_save.clicked.connect(self._save_result)
        self._btn_reset.clicked.connect(self._reset_workflow)
        for btn in (self._btn_roi, self._btn_classical, self._btn_semantic, self._btn_save):
            layout.addWidget(btn)
        layout.addWidget(self._btn_reset)
        layout.addWidget(
            QLabel(
                "<small>Скидає маску ROI, SAM-промпти та результати "
                "стиснення. Зображення лишається — можна обрати інший метод ROI.</small>"
            )
        )
        return box

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("Файл")
        open_act = QAction("Відкрити…", self)
        open_act.triggered.connect(self._open_image)
        save_act = QAction("Зберегти стиснене…", self)
        save_act.triggered.connect(self._save_result)
        reset_act = QAction("Скинути ROI і результати", self)
        reset_act.triggered.connect(self._reset_workflow)
        quit_act = QAction("Вихід", self)
        quit_act.triggered.connect(self.close)
        menu.addAction(open_act)
        menu.addAction(save_act)
        menu.addSeparator()
        menu.addAction(reset_act)
        menu.addSeparator()
        menu.addAction(quit_act)

    def _sync_settings_from_ui(self) -> None:
        s = self._session.settings
        s.roi_method = self._roi_method.currentData() or self._roi_method.currentText()
        s.device = self._device.currentText()
        s.mask_smooth_sigma = float(self._mask_sigma.value())
        s.sam_model = self._sam_model.currentData() or self._sam_model.currentText()
        s.quality_roi = self._q_roi.value()
        s.quality_background = self._q_bg.value()
        s.tile_size = self._tile.value()
        s.classical_method = self._classical_method.currentText()
        s.classical_quality = self._classical_q.value()

    def _load_settings_to_ui(self, s: AppSettings) -> None:
        idx = self._roi_method.findData(s.roi_method)
        if idx < 0:
            idx = self._roi_method.findText(s.roi_method)
        if idx >= 0:
            self._roi_method.setCurrentIndex(idx)
        di = self._device.findText(s.device)
        if di >= 0:
            self._device.setCurrentIndex(di)
        self._mask_sigma.setValue(int(s.mask_smooth_sigma))
        idx = self._sam_model.findData(s.sam_model)
        if idx < 0:
            idx = self._sam_model.findText(s.sam_model)
        if idx >= 0:
            self._sam_model.setCurrentIndex(idx)
        self._q_roi.setValue(s.quality_roi)
        self._q_bg.setValue(s.quality_background)
        self._tile.setValue(s.tile_size)
        ci = self._classical_method.findText(s.classical_method)
        if ci >= 0:
            self._classical_method.setCurrentIndex(ci)
        self._classical_q.setValue(s.classical_quality)
        self._update_sam_prompts_label()
        self._on_roi_method_changed()

    def _is_sam_method(self) -> bool:
        return (self._roi_method.currentData() or "") == "ultralytics_sam"

    def _on_roi_method_changed(self) -> None:
        is_sam = self._is_sam_method()
        self._sam_box.setVisible(is_sam)
        self._sam_click_mode.setEnabled(
            is_sam and self._sam_prompt_mode.currentData() == "point"
        )
        self._update_sam_interaction()
        if self._session.has_image:
            self._refresh_view()

    def _sam_prompts_for_service(
        self,
    ) -> tuple[list[tuple[int, int, int]] | None, list[tuple[int, int, int, int]] | None]:
        if not self._is_sam_method():
            return None, None
        return list(self._session.sam_points), list(self._session.sam_boxes)

    def _update_sam_prompts_label(self) -> None:
        pts = self._session.sam_points
        boxes = self._session.sam_boxes
        if not pts and not boxes:
            self._sam_prompts_label.setText("Промпти: немає")
            return
        parts: list[str] = []
        for x, y, label in pts:
            sign = "+" if label else "−"
            parts.append(f"({x},{y}) {sign}")
        for x1, y1, x2, y2 in boxes:
            parts.append(f"box [{x1},{y1}–{x2},{y2}]")
        self._sam_prompts_label.setText(f"Промпти ({len(parts)}): " + ", ".join(parts))

    def _clear_sam_prompts(self) -> None:
        self._session.sam_points.clear()
        self._session.sam_boxes.clear()
        self._sam_live_box = None
        self._update_sam_prompts_label()
        self._refresh_view()
        self._status.showMessage("SAM промпти очищено")

    def _maybe_sam_overlay(self, image: np.ndarray | None) -> np.ndarray | None:
        if image is None or not self._is_sam_method():
            return image
        show = self._sam_show_prompts.isChecked()
        pts = list(self._session.sam_points) if show else []
        boxes = list(self._session.sam_boxes) if show else []
        draft = self._sam_live_box
        if not pts and not boxes and draft is None:
            return image
        if self._session.original is None:
            return image
        oh, ow = self._session.original.shape[:2]
        if image.shape[:2] != (oh, ow):
            return image
        return sam_prompts_overlay(image, pts, boxes, draft_box=draft)

    def _show_view_image(
        self, image: np.ndarray | None, placeholder: str = ""
    ) -> None:
        self._view_base = image
        self._viewer.show_image(self._maybe_sam_overlay(image), placeholder)

    def _repaint_view_base(self) -> None:
        if self._view_base is None:
            return
        self._viewer.show_image(self._maybe_sam_overlay(self._view_base))

    def _on_sam_show_prompts_toggled(self, _checked: bool) -> None:
        self._repaint_view_base()

    def _on_sam_box_dragging(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if not self._is_sam_method() or self._view_combo.currentIndex() != 0:
            return
        self._sam_live_box = (x1, y1, x2, y2)
        self._repaint_view_base()

    def _on_sam_box_drag_ended(self) -> None:
        self._sam_live_box = None
        if self._session.sam_boxes or self._session.sam_points:
            self._refresh_view()
        else:
            self._repaint_view_base()

    def _reset_workflow(self) -> None:
        if not self._session.has_image or self._worker is not None:
            return
        self._session.roi_mask = None
        self._session.sam_points.clear()
        self._session.sam_boxes.clear()
        self._sam_live_box = None
        self._session.classical = None
        self._session.semantic = None
        self._active_result = None
        self._update_sam_prompts_label()
        self._stats.clear()
        self._view_combo.setCurrentIndex(0)
        self._result_combo.setCurrentIndex(0)
        self._refresh_view()
        self._update_actions()
        self._status.showMessage(
            "Скинуто: маска ROI, SAM-промпти та результати стиснення"
        )

    def _on_sam_image_click(self, x: int, y: int) -> None:
        if not self._is_sam_method() or not self._session.has_image:
            return
        if self._view_combo.currentIndex() != 0:
            return
        if self._sam_prompt_mode.currentData() != "point":
            return
        label = int(self._sam_click_mode.currentData() or 1)
        self._session.sam_points.append((x, y, label))
        self._update_sam_prompts_label()
        self._refresh_view()
        sign = "+" if label else "−"
        self._status.showMessage(f"SAM: клік {sign} на ({x}, {y})")

    def _on_sam_box_drawn(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if not self._is_sam_method() or not self._session.has_image:
            return
        if self._view_combo.currentIndex() != 0:
            return
        if self._sam_prompt_mode.currentData() != "box":
            return
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        self._session.sam_boxes.append((x1, y1, x2, y2))
        self._update_sam_prompts_label()
        self._refresh_view()
        self._status.showMessage(f"SAM: bbox [{x1},{y1}]–[{x2},{y2}]")

    def _update_sam_interaction(self) -> None:
        active = (
            self._is_sam_method()
            and self._view_combo.currentIndex() == 0
            and self._session.has_image
            and self._worker is None
        )
        box_mode = active and self._sam_prompt_mode.currentData() == "box"
        self._viewer.set_interaction_enabled(active, box_mode=box_mode)

    def _set_busy(self, busy: bool, message: str = "") -> None:
        for w in (
            self._btn_open,
            self._btn_sample,
            self._btn_roi,
            self._btn_classical,
            self._btn_semantic,
            self._btn_save,
            self._btn_reset,
        ):
            w.setEnabled(not busy)
        self._status.showMessage(message or ("Обробка…" if busy else "Готово"))

    def _update_actions(self) -> None:
        has = self._session.has_image
        self._btn_roi.setEnabled(has and self._worker is None)
        self._btn_classical.setEnabled(has and self._worker is None)
        self._btn_semantic.setEnabled(has and self._worker is None)
        has_result = self._active_result is not None
        self._btn_save.setEnabled(has_result and self._worker is None)
        self._btn_reset.setEnabled(has and self._worker is None)

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Відкрити зображення", str(self._samples_dir), IMAGE_FILTER
        )
        if path:
            self._load_path(Path(path))

    def _open_sample(self) -> None:
        if not self._samples_dir.exists():
            QMessageBox.information(
                self,
                "Зразки",
                f"Каталог зразків відсутній.\n\n"
                f"Виконайте: make data\nабо покладіть PNG у:\n{self._samples_dir}",
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Обрати зразок", str(self._samples_dir), IMAGE_FILTER
        )
        if path:
            self._load_path(Path(path))

    def _load_path(self, path: Path) -> None:
        try:
            image = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Помилка", f"Не вдалося відкрити файл:\n{exc}")
            return
        file_size = path.stat().st_size
        self._session = SessionState(
            original=image,
            source_path=path,
            source_file_bytes=file_size,
            settings=AppSettings.from_config(self._base_config),
        )
        self._load_settings_to_ui(self._session.settings)
        self._session.classical = None
        self._session.semantic = None
        self._session.roi_mask = None
        self._session.sam_points.clear()
        self._session.sam_boxes.clear()
        self._sam_live_box = None
        self._active_result = None
        h, w = image.shape[:2]
        self._path_label.setText(
            f"Файл: {path.name} ({w}×{h}, на диску {file_size / 1024:.1f} KB, "
            f"RGB {raw_image_bytes(image) / 1024:.1f} KB)"
        )
        self._view_combo.setCurrentIndex(0)
        self._stats.clear()
        self._update_sam_prompts_label()
        self._refresh_view()
        self._update_actions()
        self._status.showMessage(f"Завантажено: {path.name}")

    def _run_task(self, message: str, task, on_success) -> None:
        if self._worker is not None:
            return
        self._sync_settings_from_ui()
        self._set_busy(True, message)
        self._worker = TaskWorker(task)
        self._worker.status.connect(lambda m: self._status.showMessage(m))
        self._worker.finished_ok.connect(on_success)
        self._worker.failed.connect(self._on_task_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_finished(self) -> None:
        self._worker = None
        self._set_busy(False)
        self._update_sam_interaction()
        self._update_actions()

    def _on_task_failed(self, msg: str) -> None:
        QMessageBox.critical(self, "Помилка", msg)

    def _run_roi(self) -> None:
        if not self._session.has_image:
            return

        def task():
            pts, boxes = self._sam_prompts_for_service()
            return extract_roi_mask(
                self._session.original,
                self._base_config,
                self._session.settings,
                self._session.source_path,
                sam_points=pts,
                sam_boxes=boxes,
            )

        def done(mask: np.ndarray) -> None:
            self._session.roi_mask = mask
            self._view_combo.setCurrentIndex(2)
            self._refresh_view()
            self._status.showMessage("ROI маску виділено")

        self._run_task("Виділення ROI…", task, done)

    def _run_classical(self) -> None:
        if not self._session.has_image:
            return

        def task():
            return compress_classical(self._session.original, self._session.settings)

        def done(out: CompressionOutput) -> None:
            self._session.classical = out
            self._result_combo.setCurrentIndex(0)
            self._set_active_result(out)
            self._status.showMessage(f"Класичне стиснення: {out.method_label}")

        self._run_task("Класичне стиснення…", task, done)

    def _run_semantic(self) -> None:
        if not self._session.has_image:
            return

        if self._session.roi_mask is None:
            self._status.showMessage("Спочатку виділяю ROI…")

            def roi_then_compress():
                pts, boxes = self._sam_prompts_for_service()
                mask = extract_roi_mask(
                    self._session.original,
                    self._base_config,
                    self._session.settings,
                    self._session.source_path,
                    sam_points=pts,
                    sam_boxes=boxes,
                )
                self._session.roi_mask = mask
                return compress_semantic(
                    self._session.original,
                    mask,
                    self._session.settings,
                )

            def done(out: CompressionOutput) -> None:
                self._session.semantic = out
                self._result_combo.setCurrentIndex(1)
                self._set_active_result(out)
                self._status.showMessage("ROI + семантичне стиснення завершено")

            self._run_task("ROI + семантичне стиснення…", roi_then_compress, done)
            return

        def task():
            return compress_semantic(
                self._session.original,
                self._session.roi_mask,
                self._session.settings,
            )

        def done(out: CompressionOutput) -> None:
            self._session.semantic = out
            self._result_combo.setCurrentIndex(1)
            self._set_active_result(out)
            self._status.showMessage("Семантичне стиснення завершено")

        self._run_task("Семантичне стиснення…", task, done)

    def _set_active_result(self, out: CompressionOutput) -> None:
        self._active_result = out
        m = out.metrics
        self._stats.show_stats(
            out.method_label,
            psnr=m.psnr,
            ssim=m.ssim,
            bpp=m.bpp,
            file_bytes=m.file_size_bytes,
            ratio=out.compression_ratio,
            encode_ms=out.encode_ms,
            decode_ms=out.decode_ms,
            original_bytes=out.original_bytes,
            source_file_bytes=self._session.source_file_bytes,
            roi_psnr=m.roi_psnr,
            bg_psnr=m.background_psnr,
        )
        self._view_combo.setCurrentIndex(1)
        self._refresh_view()
        self._update_actions()

    def _on_result_changed(self) -> None:
        idx = self._result_combo.currentIndex()
        out = self._session.semantic if idx == 1 else self._session.classical
        if out is not None:
            self._set_active_result(out)
        else:
            self._active_result = None
            self._stats.clear()
            self._refresh_view()
            self._update_actions()

    def _refresh_view(self) -> None:
        try:
            if not self._session.has_image:
                self._viewer.show_image(None)
                return

            mode = self._view_combo.currentIndex()
            img = self._session.original

            if mode == 0:
                self._show_view_image(img)
                return

            if mode == 2:
                if self._session.roi_mask is not None:
                    self._show_view_image(mask_to_rgb(self._session.roi_mask))
                else:
                    self._viewer.show_image(None, "Спочатку виділіть ROI")
                return

            if mode == 3:
                if self._session.roi_mask is not None:
                    self._show_view_image(roi_overlay(img, self._session.roi_mask))
                else:
                    self._viewer.show_image(None, "Спочатку виділіть ROI")
                return

            if mode == 5:
                qm = self._active_result.quality_map if self._active_result else None
                if qm is not None:
                    self._show_view_image(quality_map_rgb(qm))
                else:
                    self._viewer.show_image(
                        None, "Карта якості — лише для семантичного стиснення"
                    )
                return

            recon = self._active_result.reconstructed if self._active_result else None
            if mode == 1:
                self._show_view_image(recon, "Спочатку виконайте стиснення")
            elif mode == 4:
                if recon is not None:
                    self._show_view_image(compression_diff(img, recon))
                else:
                    self._viewer.show_image(None, "Спочатку виконайте стиснення")
        finally:
            self._update_sam_interaction()

    def _save_result(self) -> None:
        if self._active_result is None:
            return
        out = self._active_result
        ext = out.file_extension
        default_name = f"compressed.{ext}"
        if self._session.source_path:
            default_name = f"{self._session.source_path.stem}_{ext}.{ext}"

        filters = (
            f"JPEG (*.jpg *.jpeg);;"
            f"PNG (*.png);;"
            f"WebP (*.webp);;"
            f"Усі файли (*.*)"
        )
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Зберегти стиснене зображення",
            default_name,
            filters,
        )
        if not path:
            return

        save_path = Path(path)
        try:
            save_path = self._write_compression_output(out, save_path, selected_filter)
            self._status.showMessage(f"Збережено: {save_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Помилка", f"Не вдалося зберегти:\n{exc}")

    @staticmethod
    def _extension_from_filter(selected_filter: str, fallback: str) -> str:
        f = selected_filter.lower()
        if "webp" in f:
            return "webp"
        if "png" in f:
            return "png"
        if "jpeg" in f or "jpg" in f:
            return "jpg"
        return fallback

    def _write_compression_output(
        self,
        out: CompressionOutput,
        save_path: Path,
        selected_filter: str,
    ) -> Path:
        target_ext = self._extension_from_filter(selected_filter, out.file_extension)
        return save_compression_result(
            save_path,
            bitstream=out.bitstream or None,
            reconstructed=out.reconstructed,
            source_extension=out.file_extension,
            target_extension=target_ext,
            jpeg_quality=min(self._session.settings.quality_roi, 95),
        )


def run_app() -> None:
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
