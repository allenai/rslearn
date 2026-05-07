[Back to DatasetConfig.md](../DatasetConfig.md)

## Window Data Storage

The window data storage controls the on-disk layout of materialized raster and
vector data inside each window. It is dataset-wide: all layers in a dataset
share the same window data storage. The default keeps every item group in its
own directory and is fully backward compatible with existing rslearn datasets.

The default window data storage writes one directory per item group:

```jsonc
{
  "window_data_storage": {
    "class_path": "rslearn.dataset.window_data_storage.per_item_group.PerItemGroupStorageFactory"
  }
}
```

`PerItemGroupStorage` lays out raster data at
`layers/{layer_name}.{group_idx}/{bandset_dir}/...` and vector data at
`layers/{layer_name}.{group_idx}/...`. Both raster and vector layers are
supported.

Below, we detail other window data storages.

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

Raster on-disk layout: `layers/{layer_name}/{bandset_dir}/...`. Per-group
completion markers (`layers/{layer_name}.{group_idx}/completed`) are still
written, so completion tracking is unchanged.

Vector data is not packed — vector layers fall through to the same
`layers/{layer_name}.{group_idx}/...` layout used by `PerItemGroupStorage`.
This means a dataset using `PerLayerStorageFactory` can mix raster (packed
per-layer) and vector (per-item-group) layers without configuration.

Notes:

- All item groups for a (raster) layer are written together in a single
  end-of-context flush, so the materializer holds all of them in memory at
  once. Use `PerItemGroupStorage` when this is undesirable.
- Use `PerLayerStorage` for sources where you commonly read all item groups
  at once during training, e.g. multi-temporal Sentinel-2 stacks.
