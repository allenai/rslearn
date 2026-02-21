"""This module contains dataclasses for summarizing the results of dataset operations.

They can be used by callers to emit telemetry / logs, or discarded.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


def summarize_errors(
    error_messages: list[str], top_n: int = 5
) -> list[tuple[str, int]]:
    """Count occurrences of each unique error message, return top N by frequency."""
    return Counter(error_messages).most_common(top_n)


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
    windows_rejected: int
    get_items_attempts: int
    windows_failed: int = 0
    error_messages: list[str] = field(default_factory=list)

    def merge(self, other: LayerPrepareSummary) -> LayerPrepareSummary:
        """Combine two summaries for the same layer."""
        return LayerPrepareSummary(
            layer_name=self.layer_name,
            data_source_name=self.data_source_name,
            duration_seconds=self.duration_seconds + other.duration_seconds,
            windows_prepared=self.windows_prepared + other.windows_prepared,
            windows_skipped=self.windows_skipped + other.windows_skipped,
            windows_rejected=self.windows_rejected + other.windows_rejected,
            get_items_attempts=self.get_items_attempts + other.get_items_attempts,
            windows_failed=self.windows_failed + other.windows_failed,
            error_messages=self.error_messages + other.error_messages,
        )


@dataclass
class PrepareDatasetWindowsSummary:
    """Results from prepare_dataset_windows operation for telemetry purposes."""

    # Timing
    duration_seconds: float

    # Counts
    total_windows_requested: int

    # Per-layer summaries
    layer_summaries: dict[str, LayerPrepareSummary]

    def merge(
        self, other: PrepareDatasetWindowsSummary
    ) -> PrepareDatasetWindowsSummary:
        """Combine two summaries (e.g. from separate batches)."""
        merged: dict[str, LayerPrepareSummary] = dict(self.layer_summaries)
        for name, summary in other.layer_summaries.items():
            if name in merged:
                merged[name] = merged[name].merge(summary)
            else:
                merged[name] = summary
        return PrepareDatasetWindowsSummary(
            duration_seconds=self.duration_seconds + other.duration_seconds,
            total_windows_requested=self.total_windows_requested
            + other.total_windows_requested,
            layer_summaries=merged,
        )


@dataclass
class IngestCounts:
    """Ingestion counts for a layer."""

    items: int
    geometries: int

    def merge(self, other: IngestCounts) -> IngestCounts:
        """Combine two ingest counts."""
        return IngestCounts(
            items=self.items + other.items,
            geometries=self.geometries + other.geometries,
        )


@dataclass
class LayerIngestSummary:
    """Results for ingesting a single layer."""

    # Identity
    layer_name: str
    data_source_name: str

    # Timing
    duration_seconds: float

    # Counts
    ingest_counts: IngestCounts
    ingest_attempts: int
    error_messages: list[str] = field(default_factory=list)

    def merge(self, other: LayerIngestSummary) -> LayerIngestSummary:
        """Combine two summaries for the same layer."""
        return LayerIngestSummary(
            layer_name=self.layer_name,
            data_source_name=self.data_source_name,
            duration_seconds=self.duration_seconds + other.duration_seconds,
            ingest_counts=self.ingest_counts.merge(other.ingest_counts),
            ingest_attempts=self.ingest_attempts + other.ingest_attempts,
            error_messages=self.error_messages + other.error_messages,
        )


@dataclass
class IngestDatasetJobsSummary:
    """Results from ingesting a set of jobs; for telemetry purposes."""

    # Timing
    duration_seconds: float

    # Counts
    num_jobs: int

    # Per-layer summaries
    layer_summaries: dict[str, LayerIngestSummary]

    def merge(self, other: IngestDatasetJobsSummary) -> IngestDatasetJobsSummary:
        """Combine two summaries (e.g. from separate batches)."""
        merged: dict[str, LayerIngestSummary] = dict(self.layer_summaries)
        for name, summary in other.layer_summaries.items():
            if name in merged:
                merged[name] = merged[name].merge(summary)
            else:
                merged[name] = summary
        return IngestDatasetJobsSummary(
            duration_seconds=self.duration_seconds + other.duration_seconds,
            num_jobs=self.num_jobs + other.num_jobs,
            layer_summaries=merged,
        )


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
    windows_failed: int = 0
    error_messages: list[str] = field(default_factory=list)

    def merge(
        self, other: MaterializeWindowLayersSummary
    ) -> MaterializeWindowLayersSummary:
        """Combine two summaries for the same layer."""
        return MaterializeWindowLayersSummary(
            layer_name=self.layer_name,
            data_source_name=self.data_source_name,
            duration_seconds=self.duration_seconds + other.duration_seconds,
            total_windows_requested=self.total_windows_requested
            + other.total_windows_requested,
            num_windows_materialized=self.num_windows_materialized
            + other.num_windows_materialized,
            materialize_attempts=self.materialize_attempts + other.materialize_attempts,
            windows_failed=self.windows_failed + other.windows_failed,
            error_messages=self.error_messages + other.error_messages,
        )


@dataclass
class MaterializeDatasetWindowsSummary:
    """Results from materialize_dataset_windows operation for telemetry purposes."""

    # Timing
    duration_seconds: float

    # Counts
    total_windows_requested: int

    # Per-layer summaries
    layer_summaries: dict[str, MaterializeWindowLayersSummary]

    def merge(
        self, other: MaterializeDatasetWindowsSummary
    ) -> MaterializeDatasetWindowsSummary:
        """Combine two summaries (e.g. from separate batches)."""
        merged: dict[str, MaterializeWindowLayersSummary] = dict(self.layer_summaries)
        for name, summary in other.layer_summaries.items():
            if name in merged:
                merged[name] = merged[name].merge(summary)
            else:
                merged[name] = summary
        return MaterializeDatasetWindowsSummary(
            duration_seconds=self.duration_seconds + other.duration_seconds,
            total_windows_requested=self.total_windows_requested
            + other.total_windows_requested,
            layer_summaries=merged,
        )
