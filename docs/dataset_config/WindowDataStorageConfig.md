[Back to DatasetConfig.md](../DatasetConfig.md)

## Window Data Storage

The window data storage controls the on-disk layout of materialized raster and
vector data inside each window. It is dataset-wide: all layers in a dataset
share the same window data storage.

### PerItemGroupStorage

The default window data storage is PerItemGroupStorage, which stores each item group of
each layer in a separate directory under `windows/{group_name}/{window_name}/layers/`.
Raster data is stored at `layers/{layer_name}.{group_idx}/{bandset_dir}/...` and vector
data at `layers/{layer_name}.{group_idx}/...`.

The configuration is like this (this is the default in case `window_data_storage` is
not set):

```jsonc
{
  "window_data_storage": {
    "class_path": "rslearn.dataset.window_data_storage.per_item_group.PerItemGroupStorageFactory"
  }
}
```

### PerLayerStorage

`PerLayerStorage` packs all item groups for a layer into a single combined
raster file per band set, with the groups concatenated along the time (T)
axis. A `window_storage_meta.json` sidecar records each group's number of
timesteps so individual groups can be sliced back out on read. It can be
configured like this:

```jsonc
{
  "window_data_storage": {
    "class_path": "rslearn.dataset.window_data_storage.per_layer.PerLayerStorageFactory"
  }
}
```

Raster on-disk layout: `layers/{layer_name}/{bandset_dir}/...`.

Vector data is not packed — vector layers fall through to the same
`layers/{layer_name}.{group_idx}/...` layout used by `PerItemGroupStorage`.
This means a dataset using `PerLayerStorage` can mix raster (packed
per-layer) and vector (per-item-group) layers without configuration.

Notes:

- All item groups for a (raster) layer are written together in a single
  end-of-context flush, so the materializer holds all of them in memory at
  once. Use `PerItemGroupStorage` when this is undesirable.
- Random access is not supported -- reads will always read all of the item groups.
- Use `PerLayerStorage` for sources where you commonly read all item groups
  at once during training, e.g. multi-temporal Sentinel-2 stacks.
- With the default window metadata storage, FileWindowStorage, a per-item-group
  completed sentinel file will be written, meaning the per-item-group directories of
  PerItemGroupStorage will still be created even when using PerLayerStorage. To reduce
  the number of files/directories, it is recommended to use
  [SQLiteWindowStorage](WindowStorageConfig.md#sqlitewindowstorage) or another similar
  window metadata storage when using PerLayerStorage.
