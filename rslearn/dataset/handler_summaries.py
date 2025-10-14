"""This module contains dataclasses for summarizing the results of dataset operations.

They can be used by callers to emit telemetry / logs, or discarded.
"""

from dataclasses import dataclass


@dataclass
class LayerPrepareSummary:
    """Results for preparing a single layer."""

    # Identity
    layer_name: str
    data_source_name: str

    # Timing
    duration_seconds: float

    # Counts
    windows_prepared: int
    windows_skipped: int
    get_items_attempts: int


@dataclass
class PrepareDatasetWindowsSummary:
    """Results from prepare_dataset_windows operation for telemetry purposes."""

    # Timing
    duration_seconds: float

    # Counts
    total_windows_requested: int

    # Per-layer summaries
    layer_summaries: list[LayerPrepareSummary]


@dataclass
class IngestCounts:
    """Known ingestion counts."""

    items_ingested: int
    geometries_ingested: int


@dataclass
class UnknownIngestCounts:
    """Indicates ingestion counts are unknown due to partial failure."""

    items_attempted: int
    geometries_attempted: int


@dataclass
class LayerIngestSummary:
    """Results for ingesting a single layer."""

    # Identity
    layer_name: str
    data_source_name: str

    # Timing
    duration_seconds: float

    # Counts - either known or unknown
    ingest_counts: IngestCounts | UnknownIngestCounts
    ingest_attempts: int


@dataclass
class IngestDatasetJobsSummary:
    """Results from ingesting a set of jobs; for telemetry purposes."""

    # Timing
    duration_seconds: float

    # Counts
    num_jobs: int

    # Per-layer summaries
    layer_summaries: list[LayerIngestSummary]


@dataclass
class MaterializeWindowLayerSummary:
    """Results for materializing a single window layer."""

    skipped: bool
    materialize_attempts: int


@dataclass
class MaterializeWindowLayersSummary:
    """Results for materialize a given layer for all windows in a materialize call."""

    # Identity
    layer_name: str
    data_source_name: str

    # Timing
    duration_seconds: float

    # Counts
    total_windows_requested: int
    num_windows_materialized: int
    materialize_attempts: int


@dataclass
class MaterializeDatasetWindowsSummary:
    """Results from materialize_dataset_windows operation for telemetry purposes."""

    # Timing
    duration_seconds: float

    # Counts
    total_windows_requested: int

    # Per-layer summaries
    layer_summaries: list[MaterializeWindowLayersSummary]


@dataclass
class ErrorOutcome:
    """TBD what goes in here, if anything."""

    # Timing
    duration_seconds: float
