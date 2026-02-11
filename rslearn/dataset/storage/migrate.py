"""Utilities for migrating between WindowStorage backends."""

from typing import Any

import tqdm

from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.dataset.storage.storage import WindowStorage
from rslearn.dataset.window import get_window_layer_dir
from rslearn.log_utils import get_logger

logger = get_logger(__name__)


def migrate_window_storage(
    source: WindowStorage,
    target: WindowStorage,
    fail_if_target_nonempty: bool = True,
    source_get_windows_kwargs: dict[str, Any] | None = None,
) -> int:
    """Migrate all window metadata from source to target storage.

    Args:
        source: source storage to read windows from.
        target: target storage to write windows to.
        fail_if_target_nonempty: whether to fail if target already has windows.
        source_get_windows_kwargs: optional keyword args to pass to
            source.get_windows, e.g. {"workers": 8, "show_progress": True}
            for FileWindowStorage.

    Returns:
        number of migrated windows.
    """
    if fail_if_target_nonempty and len(target.get_windows()) > 0:
        raise ValueError(
            "target window storage is not empty; rerun with --no-fail-if-target-nonempty to bypass this check"
        )

    if source_get_windows_kwargs is None:
        source_get_windows_kwargs = {}
    windows = source.get_windows(**source_get_windows_kwargs)
    total = len(windows)
    logger.info(f"Found {total} windows in source storage")
    if total == 0:
        return 0

    for window in tqdm.tqdm(windows, total=total, desc="Migrating windows"):
        target.create_or_update_window(window)

        layer_datas = source.get_layer_datas(window.group, window.name)
        if layer_datas:
            target.save_layer_datas(window.group, window.name, layer_datas)

        for layer_name, group_idx in source.list_completed_layers(
            window.group, window.name
        ):
            if isinstance(target, FileWindowStorage):
                # FileWindowStorage expects the layer directory to exist before marking
                # completion, so ensure migration creates it.
                layer_dir = get_window_layer_dir(
                    target.get_window_root(window.group, window.name),
                    layer_name,
                    group_idx,
                )
                layer_dir.mkdir(parents=True, exist_ok=True)
            target.mark_layer_completed(
                window.group, window.name, layer_name, group_idx
            )

    return total
