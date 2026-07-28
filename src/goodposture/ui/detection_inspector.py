"""Local developer-only visibility into live posture scoring."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from goodposture.adapters.runtime import FramePreview
from goodposture.app import PostureAssessment, SessionUpdate
from goodposture.core.alert_policy import AlertPolicyConfig
from goodposture.core.calibration import CalibrationBaseline
from goodposture.core.metrics import extract_posture_metrics
from goodposture.core.models import LandmarkName, PoseObservation
from goodposture.core.scoring import ScoringConfig


@dataclass(frozen=True, slots=True)
class MetricInspection:
    name: str
    live: float | None
    baseline: float | None
    standardized_deviation: float | None


@dataclass(frozen=True, slots=True)
class DetectionSnapshot:
    assessment: PostureAssessment
    raw_score: float | None
    smoothed_score: float | None
    confidence: float
    head_hunch_score: float | None
    forward_component_score: float
    downward_component_score: float
    threshold: float
    recovery_threshold: float
    continuous_ms: int
    continuous_limit_ms: int
    posture_debt_ms: float
    debt_limit_ms: int
    alert_state: str
    trigger: str | None
    metrics: tuple[MetricInspection, ...]


def _directional_component(
    *,
    live: float | None,
    baseline: float | None,
    scale: float | None,
    direction: int,
    maximum: float,
) -> float:
    if live is None or baseline is None or scale is None:
        return 0.0
    directional = (live - baseline) * direction
    return min(max(directional / scale, 0.0), maximum) / maximum * 100.0


def build_detection_snapshot(
    *,
    observation: PoseObservation,
    update: SessionUpdate,
    baseline: CalibrationBaseline,
    scoring_config: ScoringConfig,
    alert_config: AlertPolicyConfig,
) -> DetectionSnapshot:
    """Build a derived-only inspection snapshot without retaining camera pixels."""

    metrics = extract_posture_metrics(observation)
    live_by_name = metrics.as_dict()
    baseline_by_name = {item.name: item for item in baseline.features}
    rule_by_name = {item.name: item for item in scoring_config.rules}
    deviation_by_name = {
        item.name: item.standardized_deviation
        for item in (() if update.score is None else update.score.feature_deviations)
    }

    def scale(name: str) -> float | None:
        calibrated = baseline_by_name.get(name)
        rule = rule_by_name.get(name)
        if calibrated is None or rule is None:
            return None
        return max(
            rule.minimum_scale,
            calibrated.dispersion * rule.dispersion_multiplier,
        )

    depth_name = "head_depth_ratio"
    height_name = "head_vertical_offset_ratio"
    forward = _directional_component(
        live=live_by_name.get(depth_name),
        baseline=(
            None
            if depth_name not in baseline_by_name
            else baseline_by_name[depth_name].median
        ),
        scale=scale(depth_name),
        direction=1,
        maximum=scoring_config.maximum_standardized_deviation,
    )
    downward = _directional_component(
        live=live_by_name.get(height_name),
        baseline=(
            None
            if height_name not in baseline_by_name
            else baseline_by_name[height_name].median
        ),
        scale=scale(height_name),
        direction=-1,
        maximum=scoring_config.maximum_standardized_deviation,
    )
    names = tuple(rule.name for rule in scoring_config.rules)
    inspected_metrics = tuple(
        MetricInspection(
            name=name,
            live=live_by_name.get(name),
            baseline=(
                None if name not in baseline_by_name else baseline_by_name[name].median
            ),
            standardized_deviation=deviation_by_name.get(name),
        )
        for name in names
        if name in baseline_by_name
    )
    score = update.score
    alert = update.alert
    return DetectionSnapshot(
        assessment=update.posture_assessment,
        raw_score=None if score is None else score.raw_score,
        smoothed_score=None if score is None else score.smoothed_score,
        confidence=0.0 if score is None else score.confidence,
        head_hunch_score=None if score is None else score.head_hunch_score,
        forward_component_score=forward,
        downward_component_score=downward,
        threshold=alert_config.deviation_threshold,
        recovery_threshold=alert_config.recovery_threshold,
        continuous_ms=0 if alert is None else alert.continuous_deviation_ms,
        continuous_limit_ms=alert_config.continuous_deviation_duration_ms,
        posture_debt_ms=0.0 if alert is None else alert.posture_debt_ms,
        debt_limit_ms=alert_config.posture_debt_limit_ms,
        alert_state="unavailable" if alert is None else alert.state.value,
        trigger=None if alert is None or alert.trigger is None else alert.trigger.value,
        metrics=inspected_metrics,
    )


def _number(value: float | None, *, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


class DetectionInspectorWindow(QMainWindow):
    """Inspectable local preview; never records frames, landmarks, or samples."""

    def __init__(self, *, on_closed: Callable[[], None]) -> None:
        super().__init__()
        self._on_closed = on_closed
        self._observation: PoseObservation | None = None
        self.setWindowTitle("GoodPosture detection inspector — local developer tool")
        self.setMinimumSize(760, 760)
        self.resize(900, 860)

        root = QWidget()
        layout = QVBoxLayout(root)
        banner = QLabel(
            "Developer diagnostic only • local transient preview • nothing is "
            "recorded, labelled, uploaded, or used as training data"
        )
        banner.setWordWrap(True)
        banner.setObjectName("banner")
        layout.addWidget(banner)

        self.preview_label = QLabel("Open monitoring to begin the local preview.")
        self.preview_label.setAccessibleName("Local diagnostic camera preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(300)
        self.preview_label.setObjectName("preview")
        layout.addWidget(self.preview_label, stretch=1)

        status_grid = QGridLayout()
        self.assessment_label = QLabel("Assessment: Tracking uncertain")
        self.score_label = QLabel("Score: —")
        self.components_label = QLabel("Forward: — • Downward: —")
        self.timing_label = QLabel("Continuous: 0/0 s • Debt: 0/0 s")
        for label in (
            self.assessment_label,
            self.score_label,
            self.components_label,
            self.timing_label,
        ):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        status_grid.addWidget(self.assessment_label, 0, 0)
        status_grid.addWidget(self.score_label, 0, 1)
        status_grid.addWidget(self.components_label, 1, 0)
        status_grid.addWidget(self.timing_label, 1, 1)
        layout.addLayout(status_grid)

        self.metric_table = QTableWidget(0, 4)
        self.metric_table.setHorizontalHeaderLabels(
            ("Derived metric", "Live", "Baseline", "Deviation (scales)")
        )
        self.metric_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.metric_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metric_table.setAccessibleName("Live posture metrics and calibration baseline")
        layout.addWidget(self.metric_table)

        self.close_button = QPushButton("Close inspector")
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)
        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #1b201d; color: #e8eee9; }
            QLabel#banner { color: #b9c8be; padding: 6px; }
            QLabel#preview { background: #080b09; border: 1px solid #718078; }
            QTableWidget { background: #f7faf8; color: #172019; gridline-color: #bac6be; }
            QHeaderView::section { background: #dfe8e2; color: #172019; padding: 5px; }
            QPushButton { background: #f7faf8; color: #172019; padding: 8px; }
            """
        )

    def update_analysis(
        self,
        *,
        observation: PoseObservation,
        update: SessionUpdate,
        baseline: CalibrationBaseline,
        scoring_config: ScoringConfig,
        alert_config: AlertPolicyConfig,
    ) -> None:
        self._observation = observation
        snapshot = build_detection_snapshot(
            observation=observation,
            update=update,
            baseline=baseline,
            scoring_config=scoring_config,
            alert_config=alert_config,
        )
        assessment = {
            PostureAssessment.GOOD: "Good",
            PostureAssessment.NEEDS_ADJUSTMENT: "Needs adjustment",
            PostureAssessment.TRACKING_UNCERTAIN: "Tracking uncertain",
        }[snapshot.assessment]
        self.assessment_label.setText(f"Assessment: {assessment}")
        self.score_label.setText(
            f"Score: {_number(snapshot.smoothed_score, digits=1)} "
            f"(raw {_number(snapshot.raw_score, digits=1)}, "
            f"hunch {_number(snapshot.head_hunch_score, digits=1)}) • "
            f"enter {snapshot.threshold:.0f} / recover {snapshot.recovery_threshold:.0f}"
        )
        self.components_label.setText(
            f"Forward: {snapshot.forward_component_score:.1f} • "
            f"Downward: {snapshot.downward_component_score:.1f} • "
            f"confidence {snapshot.confidence:.2f}"
        )
        trigger = "" if snapshot.trigger is None else f" • trigger {snapshot.trigger}"
        self.timing_label.setText(
            f"Continuous: {snapshot.continuous_ms / 1_000:.1f}/"
            f"{snapshot.continuous_limit_ms / 1_000:.1f} s • "
            f"Debt: {snapshot.posture_debt_ms / 1_000:.1f}/"
            f"{snapshot.debt_limit_ms / 1_000:.1f} s • "
            f"alert {snapshot.alert_state}{trigger}"
        )
        self.metric_table.setRowCount(len(snapshot.metrics))
        for row, metric in enumerate(snapshot.metrics):
            for column, text in enumerate(
                (
                    metric.name,
                    _number(metric.live),
                    _number(metric.baseline),
                    _number(metric.standardized_deviation),
                )
            ):
                self.metric_table.setItem(row, column, QTableWidgetItem(text))

    def update_preview(self, preview: FramePreview) -> None:
        image = QImage(
            preview.rgb_bytes,
            preview.width,
            preview.height,
            preview.bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        self._paint_landmarks(image)
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setText("")
        self.preview_label.setPixmap(pixmap)

    def _paint_landmarks(self, image: QImage) -> None:
        if self._observation is None:
            return
        landmarks = self._observation.landmarks

        def point(name: LandmarkName) -> tuple[int, int] | None:
            landmark = landmarks.get(name)
            if landmark is None or landmark.confidence < 0.5:
                return None
            return (
                round((1.0 - landmark.x) * image.width()),
                round(landmark.y * image.height()),
            )

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        shoulder_left = point(LandmarkName.LEFT_SHOULDER)
        shoulder_right = point(LandmarkName.RIGHT_SHOULDER)
        hip_left = point(LandmarkName.LEFT_HIP)
        hip_right = point(LandmarkName.RIGHT_HIP)
        nose = point(LandmarkName.NOSE)
        left_eye = point(LandmarkName.LEFT_EYE)
        right_eye = point(LandmarkName.RIGHT_EYE)
        painter.setPen(QPen(QColor("#f1c75b"), 3))
        if shoulder_left is not None and shoulder_right is not None:
            painter.drawLine(*shoulder_left, *shoulder_right)
            shoulder_midpoint = (
                (shoulder_left[0] + shoulder_right[0]) // 2,
                (shoulder_left[1] + shoulder_right[1]) // 2,
            )
            if nose is not None:
                painter.drawLine(*shoulder_midpoint, *nose)
            if hip_left is not None and hip_right is not None:
                hip_midpoint = (
                    (hip_left[0] + hip_right[0]) // 2,
                    (hip_left[1] + hip_right[1]) // 2,
                )
                painter.setPen(QPen(QColor("#84d07b"), 3))
                painter.drawLine(*hip_left, *hip_right)
                painter.drawLine(*shoulder_midpoint, *hip_midpoint)
        painter.setPen(QPen(QColor("#61d6d0"), 7))
        for item in (nose, left_eye, right_eye):
            if item is not None:
                painter.drawPoint(*item)
        painter.end()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.preview_label.clear()
        self._observation = None
        self._on_closed()
        event.accept()
