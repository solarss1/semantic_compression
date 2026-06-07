"""Допоміжні віджети UI."""

from __future__ import annotations

from typing import Callable

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget


def numpy_rgb_to_qpixmap(image: np.ndarray) -> QPixmap:
    """RGB uint8 (H,W,3) → QPixmap."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    arr = np.ascontiguousarray(image.astype(np.uint8))
    h, w, ch = arr.shape
    qimg = QImage(arr.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class InteractiveImageLabel(QLabel):
    """Кліки та drag-bbox у координатах зображення."""

    image_clicked = pyqtSignal(int, int)
    box_drawn = pyqtSignal(int, int, int, int)
    box_dragging = pyqtSignal(int, int, int, int)
    box_drag_ended = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._map_point: Callable[[int, int], tuple[int, int] | None] | None = None
        self._interaction_enabled = False
        self._box_mode = False
        self._drag_start: tuple[int, int] | None = None

    def set_interaction(
        self,
        enabled: bool,
        *,
        box_mode: bool = False,
        mapper: Callable[[int, int], tuple[int, int] | None] | None = None,
    ) -> None:
        prev_enabled = self._interaction_enabled
        prev_box = self._box_mode
        self._interaction_enabled = enabled and mapper is not None
        self._box_mode = box_mode and self._interaction_enabled
        self._map_point = mapper if self._interaction_enabled else None
        if not self._interaction_enabled:
            self._drag_start = None
        elif prev_enabled != self._interaction_enabled or prev_box != self._box_mode:
            self._drag_start = None
        self.setCursor(
            Qt.CursorShape.CrossCursor
            if self._interaction_enabled
            else Qt.CursorShape.ArrowCursor
        )

    def mousePressEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if (
            event
            and self._interaction_enabled
            and self._map_point is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            pos = event.position()
            mapped = self._map_point(int(pos.x()), int(pos.y()))
            if mapped is not None:
                if self._box_mode:
                    self._drag_start = mapped
                else:
                    self.image_clicked.emit(mapped[0], mapped[1])
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if (
            event
            and self._box_mode
            and self._drag_start is not None
            and self._map_point is not None
        ):
            pos = event.position()
            mapped = self._map_point(int(pos.x()), int(pos.y()))
            if mapped is not None:
                x1, y1 = self._drag_start
                x2, y2 = mapped
                self.box_dragging.emit(x1, y1, x2, y2)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        was_dragging = self._drag_start is not None
        if (
            event
            and self._box_mode
            and self._drag_start is not None
            and self._map_point is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            pos = event.position()
            mapped = self._map_point(int(pos.x()), int(pos.y()))
            if mapped is not None:
                x1, y1 = self._drag_start
                x2, y2 = mapped
                if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3:
                    self.box_drawn.emit(x1, y1, x2, y2)
            self._drag_start = None
        if was_dragging and self._box_mode:
            self.box_drag_ended.emit()
        super().mouseReleaseEvent(event)


class ImageViewer(QScrollArea):
    """Прокручуваний перегляд зображення з масштабуванням під вікно."""

    image_clicked = pyqtSignal(int, int)
    box_drawn = pyqtSignal(int, int, int, int)
    box_dragging = pyqtSignal(int, int, int, int)
    box_drag_ended = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #2b2b2b;")

        self._label = InteractiveImageLabel("Завантажте зображення (PNG, JPEG, …)")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._label.setStyleSheet("color: #aaa; font-size: 14px; padding: 40px;")
        self._label.image_clicked.connect(self.image_clicked.emit)
        self._label.box_drawn.connect(self.box_drawn.emit)
        self._label.box_dragging.connect(self.box_dragging.emit)
        self._label.box_drag_ended.connect(self.box_drag_ended.emit)
        self.setWidget(self._label)
        self._pixmap: QPixmap | None = None
        self._image_shape: tuple[int, int] | None = None
        self._interaction_enabled = False
        self._box_mode = False

    def set_interaction_enabled(self, enabled: bool, *, box_mode: bool = False) -> None:
        self._interaction_enabled = enabled
        self._box_mode = box_mode
        self._apply_interaction()

    def show_image(self, image: np.ndarray | None, placeholder: str = "") -> None:
        if image is None:
            self._pixmap = None
            self._image_shape = None
            self._label.setPixmap(QPixmap())
            self._label.setText(placeholder or "Немає зображення для відображення")
            self._apply_interaction()
            return
        self._image_shape = (image.shape[0], image.shape[1])
        self._pixmap = numpy_rgb_to_qpixmap(image)
        self._fit_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_pixmap()

    def _apply_interaction(self) -> None:
        mapper = self._map_widget_to_image if self._interaction_enabled else None
        self._label.set_interaction(
            self._interaction_enabled,
            box_mode=self._box_mode,
            mapper=mapper,
        )

    def _fit_pixmap(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        viewport = self.viewport()
        if viewport is None:
            return
        scaled = self._pixmap.scaled(
            viewport.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.setText("")
        self._apply_interaction()

    def _map_widget_to_image(self, x: int, y: int) -> tuple[int, int] | None:
        if self._image_shape is None:
            return None
        pixmap = self._label.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        pw, ph = pixmap.width(), pixmap.height()
        lw, lh = self._label.width(), self._label.height()
        if pw <= 0 or ph <= 0:
            return None
        x0, y0 = (lw - pw) // 2, (lh - ph) // 2
        lx, ly = x - x0, y - y0
        if lx < 0 or ly < 0 or lx >= pw or ly >= ph:
            return None
        h, w = self._image_shape
        ix = int(round(lx * w / pw))
        iy = int(round(ly * h / ph))
        return max(0, min(w - 1, ix)), max(0, min(h - 1, iy))


class StatsPanel(QWidget):
    """Таблиця метрик стиснення."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = QLabel("Метрики з’являться після стиснення")
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._label.setStyleSheet(
            "font-family: monospace; font-size: 16px; padding: 12px; "
            "background: #1e1e1e; color: #eee; border-radius: 4px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def clear(self) -> None:
        self._label.setText("Метрики з’являться після стиснення")

    def show_stats(
        self,
        title: str,
        *,
        psnr: float,
        ssim: float,
        bpp: float,
        file_bytes: int,
        ratio: float,
        encode_ms: float,
        decode_ms: float,
        original_bytes: int,
        source_file_bytes: int | None = None,
        roi_psnr: float | None = None,
        bg_psnr: float | None = None,
    ) -> None:
        lines = [
            f"<b>{title}</b>",
            f"PSNR: {psnr:.2f} dB",
            f"SSIM: {ssim:.4f}",
            f"Розмір результату: {file_bytes / 1024:.1f} KB ({file_bytes} B)",
            f"bpp: {bpp:.4f}",
            f"Коеф. стиснення: {ratio:.2f}× (від RGB-бази)",
        ]
        if source_file_bytes is not None:
            lines.append(
                f"Файл на диску: {source_file_bytes / 1024:.1f} KB "
                f"({source_file_bytes} B)"
            )
        lines.extend(
            [
                f"Нестиснений RGB: {original_bytes / 1024:.1f} KB "
                f"({original_bytes} B)",
                f"Час кодування: {encode_ms:.1f} ms",
                f"Час декодування: {decode_ms:.1f} ms",
            ]
        )
        if roi_psnr is not None:
            lines.append(f"PSNR у ROI: {roi_psnr:.2f} dB")
        if bg_psnr is not None:
            lines.append(f"PSNR у фоні: {bg_psnr:.2f} dB")
        self._label.setText("<br>".join(lines))
