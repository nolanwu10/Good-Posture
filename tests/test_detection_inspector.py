from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton

from goodposture.adapters.runtime import FramePreview
from goodposture.app import AnalysisSession, PostureAssessment
from goodposture.core.calibration import (
    CALIBRATION_CONFIG_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    CalibrationBaseline,
    CalibrationFeature,
)
from goodposture.core.models import Landmark, LandmarkName, PoseObservation
from goodposture.ui.detection_inspector import (
    DetectionInspectorWindow,
    build_detection_snapshot,
)


def application() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def baseline() -> CalibrationBaseline:
    return CalibrationBaseline(
        calibration_schema_version=CALIBRATION_SCHEMA_VERSION,
        calibration_config_version=CALIBRATION_CONFIG_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_id="pose-landmarker-lite@sha256:test",
        started_at_ms=0,
        completed_at_ms=100,
        accepted_sample_count=60,
        rejected_sample_count=0,
        features=(
            CalibrationFeature("shoulder_tilt_degrees", 0.0, 1.0),
            CalibrationFeature("head_lateral_offset_ratio", 0.0, 0.02),
            CalibrationFeature("head_vertical_offset_ratio", 0.73, 0.02),
            CalibrationFeature("head_depth_ratio", 0.60, 0.02),
        ),
    )


def head_down_observation(timestamp_ms: int) -> PoseObservation:
    return PoseObservation(
        timestamp_ms=timestamp_ms,
        landmarks={
            LandmarkName.NOSE: Landmark(0.50, 0.41, -0.18, 0.95, 0.95),
            LandmarkName.LEFT_EYE: Landmark(0.47, 0.39, -0.18, 0.95, 0.95),
            LandmarkName.RIGHT_EYE: Landmark(0.53, 0.39, -0.18, 0.95, 0.95),
            LandmarkName.LEFT_SHOULDER: Landmark(0.35, 0.50, 0.0, 0.95, 0.95),
            LandmarkName.RIGHT_SHOULDER: Landmark(0.65, 0.50, 0.0, 0.95, 0.95),
        },
    )


def test_snapshot_explains_downward_and_forward_hunch_components() -> None:
    active_baseline = baseline()
    session = AnalysisSession(
        model_id=active_baseline.model_id,
        baseline=active_baseline,
    )
    session.start(timestamp_ms=0)
    observation = head_down_observation(100)
    update = session.process_observation(observation)

    snapshot = build_detection_snapshot(
        observation=observation,
        update=update,
        baseline=active_baseline,
        scoring_config=session.scoring_config,
        alert_config=session.alert_config,
    )

    assert snapshot.assessment is PostureAssessment.NEEDS_ADJUSTMENT
    assert snapshot.downward_component_score >= 60.0
    assert snapshot.forward_component_score == 0.0
    assert snapshot.threshold == 75.0
    assert snapshot.continuous_limit_ms == 7_000
    assert snapshot.debt_limit_ms == 12_000


def test_developer_window_shows_preview_metrics_and_timing_without_capture_controls() -> None:
    application()
    active_baseline = baseline()
    session = AnalysisSession(
        model_id=active_baseline.model_id,
        baseline=active_baseline,
    )
    session.start(timestamp_ms=0)
    observation = head_down_observation(100)
    update = session.process_observation(observation)
    window = DetectionInspectorWindow(on_closed=lambda: None)

    window.update_analysis(
        observation=observation,
        update=update,
        baseline=active_baseline,
        scoring_config=session.scoring_config,
        alert_config=session.alert_config,
    )
    window.update_preview(
        FramePreview(
            width=4,
            height=3,
            bytes_per_line=12,
            rgb_bytes=b"\x00" * 36,
        )
    )

    assert window.preview_label.accessibleName() == "Local diagnostic camera preview"
    assert window.preview_label.pixmap().isNull() is False
    assert "Needs adjustment" in window.assessment_label.text()
    assert "Downward" in window.components_label.text()
    assert "Continuous" in window.timing_label.text()
    button_texts = {button.text().lower() for button in window.findChildren(QPushButton)}
    assert not any("capture" in text or "mark" in text for text in button_texts)
