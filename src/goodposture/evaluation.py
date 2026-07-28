"""Repeatable, privacy-bounded evaluation of the headless analysis session."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from goodposture.app.session import AnalysisSession, SessionEventType
from goodposture.core.alert_policy import (
    DEFAULT_ALERT_POLICY_CONFIG,
    AlertPolicyConfig,
)
from goodposture.core.calibration import CalibrationBaseline
from goodposture.core.models import Landmark, LandmarkName, PoseObservation
from goodposture.core.scoring import DEFAULT_SCORING_CONFIG, ScoreState, ScoringConfig

EVALUATION_SCHEMA_VERSION: Final = 1
DEFAULT_EVALUATION_CORPUS_PATH: Final = (
    Path(__file__).resolve().parents[2] / "evaluation" / "corpus-v1.json"
)
PREVIOUS_ALERT_POLICY_CONFIG: Final = AlertPolicyConfig(
    version=3,
    deviation_threshold=60.0,
    recovery_threshold=40.0,
    continuous_deviation_duration_ms=5_000,
    posture_debt_limit_ms=8_000,
    recovery_debt_decay_rate=0.5,
    cooldown_duration_ms=600_000,
    maximum_observation_gap_ms=2_000,
)
_ALLOWED_PROVENANCE: Final = frozenset(("synthetic", "consented_derived"))
_FORBIDDEN_MEDIA_KEYS: Final = frozenset(
    (
        "frame",
        "frames",
        "raw_frame",
        "raw_frames",
        "image",
        "images",
        "pixels",
        "screenshot",
        "screenshots",
        "video",
        "videos",
        "video_path",
        "image_path",
        "media_path",
        "recording",
        "recordings",
    )
)


@dataclass(frozen=True, slots=True)
class EvaluationSegment:
    """One synthetic pose template held for a bounded duration."""

    duration_ms: int
    template: str


@dataclass(frozen=True, slots=True)
class EvaluationScenario:
    """A labeled sequence and its expected reminder behavior."""

    id: str
    label: str
    expected_prompt: bool
    frame_interval_ms: int
    segments: tuple[EvaluationSegment, ...]
    deviation_start_ms: int | None = None
    robustness_group: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationCorpus:
    """Versioned synthetic or explicitly consented derived evaluation inputs."""

    schema_version: int
    corpus_version: str
    description: str
    provenance: str
    deidentified: bool
    consent_reference: str | None
    model_id: str
    baseline: CalibrationBaseline
    templates: Mapping[str, Mapping[LandmarkName, Landmark]]
    scenarios: tuple[EvaluationScenario, ...]
    known_limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScenarioEvaluation:
    """Aggregate result for one scenario; no observation history is retained."""

    id: str
    label: str
    expected_prompt: bool
    prompt_count: int
    first_prompt_timestamp_ms: int | None
    time_to_prompt_ms: int | None
    unknown_coverage: float
    frame_interval_ms: int
    robustness_group: str | None


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Corpus-level reminder quality and confidence-coverage metrics."""

    prompt_precision: float
    false_prompt_count: int
    missed_sustained_deviations: int
    mean_time_to_prompt_ms: float | None
    unknown_coverage: float
    frame_rate_robustness: float
    maximum_prompt_latency_spread_ms: int


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Deterministic evidence tied to exact corpus and policy versions."""

    report_schema_version: int
    corpus_version: str
    profile_name: str
    scoring_config_version: int
    alert_config_version: int
    metrics: EvaluationMetrics
    scenarios: tuple[ScenarioEvaluation, ...]
    known_limitations: tuple[str, ...]

    def to_json(self) -> str:
        """Serialize only report aggregates and version metadata."""

        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class EvaluationGate:
    """Agreed tolerances for accepting a candidate policy."""

    maximum_added_false_prompts: int = 0
    maximum_added_misses: int = 0
    maximum_prompt_latency_increase_ms: int = 2_500
    maximum_unknown_coverage_increase: float = 0.0
    minimum_frame_rate_robustness: float = 1.0

    def __post_init__(self) -> None:
        if self.maximum_added_false_prompts < 0:
            raise ValueError("maximum_added_false_prompts cannot be negative")
        if self.maximum_added_misses < 0:
            raise ValueError("maximum_added_misses cannot be negative")
        if self.maximum_prompt_latency_increase_ms < 0:
            raise ValueError("maximum_prompt_latency_increase_ms cannot be negative")
        if self.maximum_unknown_coverage_increase < 0.0:
            raise ValueError("maximum_unknown_coverage_increase cannot be negative")
        if not 0.0 <= self.minimum_frame_rate_robustness <= 1.0:
            raise ValueError("minimum_frame_rate_robustness must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class EvaluationComparison:
    """Result of applying a regression gate to before/after evidence."""

    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationProtocolResult:
    """Before/after reports and the explicit acceptance decision."""

    before: EvaluationReport
    after: EvaluationReport
    comparison: EvaluationComparison


DEFAULT_EVALUATION_GATE: Final = EvaluationGate()


def _required_dict(data: object, name: str) -> dict[str, object]:
    if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
        raise ValueError(f"{name} must be an object with string keys")
    return data


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_positive_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _reject_raw_media(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_MEDIA_KEYS:
                raise ValueError(f"evaluation corpus cannot contain raw media field {key!r}")
            _reject_raw_media(item)
    elif isinstance(value, list):
        for item in value:
            _reject_raw_media(item)


def _load_templates(
    data: object,
) -> dict[str, dict[LandmarkName, Landmark]]:
    items = _required_dict(data, "templates")
    templates: dict[str, dict[LandmarkName, Landmark]] = {}
    for template_name, raw_template in items.items():
        template_data = _required_dict(raw_template, f"template {template_name!r}")
        landmarks: dict[LandmarkName, Landmark] = {}
        for raw_name, raw_coordinates in template_data.items():
            try:
                name = LandmarkName(raw_name)
            except ValueError as error:
                raise ValueError(f"unknown landmark name: {raw_name!r}") from error
            if (
                not isinstance(raw_coordinates, list)
                or len(raw_coordinates) != 4
                or any(
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                    for value in raw_coordinates
                )
            ):
                raise ValueError(
                    f"landmark {raw_name!r} must contain x, y, z, and confidence"
                )
            x, y, z, confidence = (float(value) for value in raw_coordinates)
            if not all(math.isfinite(value) for value in (x, y, z, confidence)):
                raise ValueError(f"landmark {raw_name!r} values must be finite")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"landmark {raw_name!r} confidence must be 0 to 1")
            landmarks[name] = Landmark(
                x=x,
                y=y,
                z=z,
                visibility=confidence,
                presence=confidence,
            )
        templates[template_name] = landmarks
    if not templates:
        raise ValueError("templates cannot be empty")
    return templates


def _load_scenarios(
    data: object,
    templates: Mapping[str, Mapping[LandmarkName, Landmark]],
) -> tuple[EvaluationScenario, ...]:
    if not isinstance(data, list) or not data:
        raise ValueError("scenarios must be a non-empty list")
    scenarios: list[EvaluationScenario] = []
    seen_ids: set[str] = set()
    for raw_scenario in data:
        item = _required_dict(raw_scenario, "scenario")
        scenario_id = _required_string(item, "id")
        if scenario_id in seen_ids:
            raise ValueError(f"duplicate scenario id: {scenario_id!r}")
        seen_ids.add(scenario_id)
        expected_prompt = item.get("expected_prompt")
        if not isinstance(expected_prompt, bool):
            raise ValueError("expected_prompt must be a boolean")
        raw_segments = item.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError("scenario segments must be a non-empty list")
        segments: list[EvaluationSegment] = []
        for raw_segment in raw_segments:
            segment_data = _required_dict(raw_segment, "scenario segment")
            template = _required_string(segment_data, "template")
            if template not in templates:
                raise ValueError(f"scenario references unknown template: {template!r}")
            segments.append(
                EvaluationSegment(
                    duration_ms=_required_positive_int(segment_data, "duration_ms"),
                    template=template,
                )
            )
        deviation_start_ms = item.get("deviation_start_ms")
        if deviation_start_ms is not None and (
            not isinstance(deviation_start_ms, int)
            or isinstance(deviation_start_ms, bool)
            or deviation_start_ms < 0
        ):
            raise ValueError("deviation_start_ms must be a nonnegative integer")
        if expected_prompt and deviation_start_ms is None:
            raise ValueError("prompt-expected scenario requires deviation_start_ms")
        robustness_group = item.get("robustness_group")
        if robustness_group is not None and (
            not isinstance(robustness_group, str) or not robustness_group
        ):
            raise ValueError("robustness_group must be a non-empty string")
        scenarios.append(
            EvaluationScenario(
                id=scenario_id,
                label=_required_string(item, "label"),
                expected_prompt=expected_prompt,
                frame_interval_ms=_required_positive_int(item, "frame_interval_ms"),
                segments=tuple(segments),
                deviation_start_ms=deviation_start_ms,
                robustness_group=robustness_group,
            )
        )
    return tuple(scenarios)


def load_evaluation_corpus(path: Path) -> EvaluationCorpus:
    """Load and validate a corpus without accepting raw-media fields."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load evaluation corpus: {error}") from error
    data = _required_dict(raw, "evaluation corpus")
    _reject_raw_media(data)
    if data.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported evaluation schema version: {data.get('schema_version')!r}")
    provenance = _required_string(data, "provenance")
    if provenance not in _ALLOWED_PROVENANCE:
        raise ValueError(f"unsupported evaluation provenance: {provenance!r}")
    deidentified = data.get("deidentified")
    if not isinstance(deidentified, bool):
        raise ValueError("deidentified must be a boolean")
    consent_reference = data.get("consent_reference")
    if consent_reference is not None and (
        not isinstance(consent_reference, str) or not consent_reference
    ):
        raise ValueError("consent_reference must be null or a non-empty string")
    if provenance == "consented_derived" and (
        not deidentified or consent_reference is None
    ):
        raise ValueError(
            "consented_derived corpora require deidentified=true and consent_reference"
        )
    if provenance == "synthetic" and consent_reference is not None:
        raise ValueError("synthetic corpora cannot claim a consent_reference")

    templates = _load_templates(data.get("templates"))
    raw_limitations = data.get("known_limitations")
    if (
        not isinstance(raw_limitations, list)
        or not raw_limitations
        or any(not isinstance(item, str) or not item for item in raw_limitations)
    ):
        raise ValueError("known_limitations must be a non-empty string list")
    baseline = CalibrationBaseline.from_dict(data.get("baseline"))
    model_id = _required_string(data, "model_id")
    if baseline.model_id != model_id:
        raise ValueError("baseline model_id must match corpus model_id")
    return EvaluationCorpus(
        schema_version=EVALUATION_SCHEMA_VERSION,
        corpus_version=_required_string(data, "corpus_version"),
        description=_required_string(data, "description"),
        provenance=provenance,
        deidentified=deidentified,
        consent_reference=consent_reference,
        model_id=model_id,
        baseline=baseline,
        templates=templates,
        scenarios=_load_scenarios(data.get("scenarios"), templates),
        known_limitations=tuple(raw_limitations),
    )


def _observations(
    corpus: EvaluationCorpus,
    scenario: EvaluationScenario,
) -> tuple[PoseObservation, ...]:
    observations: list[PoseObservation] = []
    timestamp_ms = 0
    for segment in scenario.segments:
        end_ms = timestamp_ms + segment.duration_ms
        while timestamp_ms < end_ms:
            timestamp_ms = min(timestamp_ms + scenario.frame_interval_ms, end_ms)
            observations.append(
                PoseObservation(
                    timestamp_ms=timestamp_ms,
                    landmarks=corpus.templates[segment.template],
                )
            )
    return tuple(observations)


def _evaluate_scenario(
    corpus: EvaluationCorpus,
    scenario: EvaluationScenario,
    *,
    scoring_config: ScoringConfig,
    alert_config: AlertPolicyConfig,
) -> tuple[ScenarioEvaluation, int, int]:
    session = AnalysisSession(
        model_id=corpus.model_id,
        baseline=corpus.baseline,
        scoring_config=scoring_config,
        alert_config=alert_config,
    )
    session.start(timestamp_ms=0)
    prompt_timestamps: list[int] = []
    unknown_ms = 0
    evaluated_ms = 0
    previous_timestamp_ms: int | None = None
    previous_unknown = False
    for observation in _observations(corpus, scenario):
        update = session.process_observation(observation)
        if previous_timestamp_ms is not None:
            interval_ms = observation.timestamp_ms - previous_timestamp_ms
            evaluated_ms += interval_ms
            if previous_unknown:
                unknown_ms += interval_ms
        previous_timestamp_ms = observation.timestamp_ms
        previous_unknown = (
            update.score is not None and update.score.state is ScoreState.UNKNOWN
        )
        if any(event.type is SessionEventType.PROMPT for event in update.events):
            prompt_timestamps.append(observation.timestamp_ms)

    first_prompt = prompt_timestamps[0] if prompt_timestamps else None
    time_to_prompt = (
        first_prompt - scenario.deviation_start_ms
        if first_prompt is not None and scenario.deviation_start_ms is not None
        else None
    )
    return (
        ScenarioEvaluation(
            id=scenario.id,
            label=scenario.label,
            expected_prompt=scenario.expected_prompt,
            prompt_count=len(prompt_timestamps),
            first_prompt_timestamp_ms=first_prompt,
            time_to_prompt_ms=time_to_prompt,
            unknown_coverage=unknown_ms / evaluated_ms if evaluated_ms else 0.0,
            frame_interval_ms=scenario.frame_interval_ms,
            robustness_group=scenario.robustness_group,
        ),
        unknown_ms,
        evaluated_ms,
    )


def evaluate_corpus(
    corpus: EvaluationCorpus,
    *,
    profile_name: str,
    scoring_config: ScoringConfig = DEFAULT_SCORING_CONFIG,
    alert_config: AlertPolicyConfig = DEFAULT_ALERT_POLICY_CONFIG,
) -> EvaluationReport:
    """Replay every sequence through the production headless analysis path."""

    if not profile_name:
        raise ValueError("profile_name must be non-empty")
    scenario_results: list[ScenarioEvaluation] = []
    total_unknown_ms = 0
    total_evaluated_ms = 0
    for scenario in corpus.scenarios:
        result, unknown_ms, evaluated_ms = _evaluate_scenario(
            corpus,
            scenario,
            scoring_config=scoring_config,
            alert_config=alert_config,
        )
        scenario_results.append(result)
        total_unknown_ms += unknown_ms
        total_evaluated_ms += evaluated_ms

    true_prompts = sum(
        result.expected_prompt and result.prompt_count > 0
        for result in scenario_results
    )
    false_prompts = sum(
        result.prompt_count
        for result in scenario_results
        if not result.expected_prompt
    )
    missed = sum(
        result.expected_prompt and result.prompt_count == 0
        for result in scenario_results
    )
    prompt_latencies = [
        result.time_to_prompt_ms
        for result in scenario_results
        if result.expected_prompt and result.time_to_prompt_ms is not None
    ]
    grouped: dict[str, list[ScenarioEvaluation]] = defaultdict(list)
    for result in scenario_results:
        if result.robustness_group is not None:
            grouped[result.robustness_group].append(result)
    group_passes = 0
    latency_spreads: list[int] = []
    for results in grouped.values():
        prompted = [result.prompt_count > 0 for result in results]
        latencies = [
            result.time_to_prompt_ms
            for result in results
            if result.time_to_prompt_ms is not None
        ]
        spread = max(latencies) - min(latencies) if len(latencies) >= 2 else 0
        latency_spreads.append(spread)
        tolerance_ms = max(result.frame_interval_ms for result in results)
        if len(results) >= 2 and len(set(prompted)) == 1 and spread <= tolerance_ms:
            group_passes += 1
    precision_denominator = true_prompts + false_prompts
    return EvaluationReport(
        report_schema_version=EVALUATION_SCHEMA_VERSION,
        corpus_version=corpus.corpus_version,
        profile_name=profile_name,
        scoring_config_version=scoring_config.version,
        alert_config_version=alert_config.version,
        metrics=EvaluationMetrics(
            prompt_precision=(
                true_prompts / precision_denominator
                if precision_denominator
                else 1.0
            ),
            false_prompt_count=false_prompts,
            missed_sustained_deviations=missed,
            mean_time_to_prompt_ms=(
                sum(prompt_latencies) / len(prompt_latencies)
                if prompt_latencies
                else None
            ),
            unknown_coverage=(
                total_unknown_ms / total_evaluated_ms
                if total_evaluated_ms
                else 0.0
            ),
            frame_rate_robustness=(
                group_passes / len(grouped) if grouped else 1.0
            ),
            maximum_prompt_latency_spread_ms=max(latency_spreads, default=0),
        ),
        scenarios=tuple(scenario_results),
        known_limitations=corpus.known_limitations,
    )


def compare_evaluation_reports(
    before: EvaluationReport,
    after: EvaluationReport,
    *,
    gate: EvaluationGate = DEFAULT_EVALUATION_GATE,
) -> EvaluationComparison:
    """Require a candidate report to stay within explicit regression tolerances."""

    if before.corpus_version != after.corpus_version:
        raise ValueError("before and after reports must use the same corpus version")
    failures: list[str] = []
    if (
        after.metrics.false_prompt_count - before.metrics.false_prompt_count
        > gate.maximum_added_false_prompts
    ):
        failures.append("candidate added false prompts beyond tolerance")
    if (
        after.metrics.missed_sustained_deviations
        - before.metrics.missed_sustained_deviations
        > gate.maximum_added_misses
    ):
        failures.append("candidate added missed sustained deviations beyond tolerance")
    before_latency = before.metrics.mean_time_to_prompt_ms
    after_latency = after.metrics.mean_time_to_prompt_ms
    if before_latency is not None and after_latency is None:
        failures.append("candidate missed all sustained deviations")
    elif (
        before_latency is not None
        and after_latency is not None
        and after_latency - before_latency
        > gate.maximum_prompt_latency_increase_ms
    ):
        failures.append("candidate increased time-to-prompt beyond tolerance")
    if (
        after.metrics.unknown_coverage - before.metrics.unknown_coverage
        > gate.maximum_unknown_coverage_increase + 1e-12
    ):
        failures.append("candidate increased unknown coverage beyond tolerance")
    if (
        after.metrics.frame_rate_robustness
        < gate.minimum_frame_rate_robustness
    ):
        failures.append("candidate frame-rate robustness is below tolerance")
    return EvaluationComparison(passed=not failures, failures=tuple(failures))


def render_evaluation_markdown(report: EvaluationReport) -> str:
    """Render a concise, reviewable report without detailed observations."""

    metrics = report.metrics
    latency = (
        "n/a"
        if metrics.mean_time_to_prompt_ms is None
        else f"{metrics.mean_time_to_prompt_ms:.0f} ms"
    )
    lines = [
        f"# GoodPosture evaluation: {report.profile_name}",
        "",
        f"- Corpus: `{report.corpus_version}`",
        f"- Scoring config version: {report.scoring_config_version}",
        f"- Alert config version: {report.alert_config_version}",
        f"- Prompt precision: {metrics.prompt_precision:.3f}",
        f"- False prompts: {metrics.false_prompt_count}",
        f"- Missed sustained deviations: {metrics.missed_sustained_deviations}",
        f"- Mean time-to-prompt: {latency}",
        f"- Unknown coverage: {metrics.unknown_coverage:.3f}",
        f"- Frame-rate robustness: {metrics.frame_rate_robustness:.3f}",
        (
            "- Maximum prompt-latency spread across frame rates: "
            f"{metrics.maximum_prompt_latency_spread_ms} ms"
        ),
        "",
        "## Scenario outcomes",
        "",
        "| Scenario | Label | Expected prompt | Prompts | Time-to-prompt | Unknown |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scenario in report.scenarios:
        scenario_latency = (
            "n/a"
            if scenario.time_to_prompt_ms is None
            else f"{scenario.time_to_prompt_ms} ms"
        )
        lines.append(
            f"| {scenario.id} | {scenario.label} | "
            f"{'yes' if scenario.expected_prompt else 'no'} | "
            f"{scenario.prompt_count} | {scenario_latency} | "
            f"{scenario.unknown_coverage:.3f} |"
        )
    lines.extend(("", "## Known limitations", ""))
    lines.extend(f"- {limitation}" for limitation in report.known_limitations)
    lines.append("")
    return "\n".join(lines)


def run_evaluation_protocol(
    *,
    corpus_path: Path,
    output_directory: Path,
    gate: EvaluationGate = DEFAULT_EVALUATION_GATE,
) -> EvaluationProtocolResult:
    """Run the recorded Task 11/12 comparison and write aggregate evidence."""

    corpus = load_evaluation_corpus(corpus_path)
    before = evaluate_corpus(
        corpus,
        profile_name="task11-default-v3",
        alert_config=PREVIOUS_ALERT_POLICY_CONFIG,
    )
    after = evaluate_corpus(
        corpus,
        profile_name="task12-synthetic-v1",
        alert_config=DEFAULT_ALERT_POLICY_CONFIG,
    )
    comparison = compare_evaluation_reports(before, after, gate=gate)
    output_directory.mkdir(parents=True, exist_ok=True)
    report_items = (
        ("before-task11-default-v3", before),
        ("after-task12-synthetic-v1", after),
    )
    for stem, report in report_items:
        (output_directory / f"{stem}.json").write_text(
            report.to_json(),
            encoding="utf-8",
        )
        (output_directory / f"{stem}.md").write_text(
            render_evaluation_markdown(report),
            encoding="utf-8",
        )
    comparison_payload = {
        "report_schema_version": EVALUATION_SCHEMA_VERSION,
        "corpus_version": corpus.corpus_version,
        "before_profile": before.profile_name,
        "after_profile": after.profile_name,
        "gate": asdict(gate),
        "passed": comparison.passed,
        "failures": list(comparison.failures),
    }
    (output_directory / "comparison.json").write_text(
        json.dumps(comparison_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return EvaluationProtocolResult(
        before=before,
        after=after,
        comparison=comparison,
    )
