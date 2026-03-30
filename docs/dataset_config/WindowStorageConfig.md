[Back to DatasetConfig.md](../DatasetConfig.md)

## Window Storage

The window storage is responsible for keeping track of:

1. The windows in the rslearn dataset, including their name, group, projection, bounds,
   and time range.
2. The window layer datas, i.e., for each layer, the item groups produced by the matching process.
3. The completed layers for each window, i.e., which layers have been materialized
   successfully for each window.

The default window storage is file-based:

```jsonc
{
  "storage": {
    "class_path": "rslearn.dataset.storage.file.FileWindowStorageFactory"
  }
}
```

[DatasetFormat.md](./DatasetFormat.md) details the file structure. `FileWindowStorage` works
across all filesystem types (e.g. local filesystem but also object storage), and doesn't need
a database to be set up, but it is slow when there are a lot of windows because it stores the
information for each window in a separate file (so many files need to be read to list windows).

Below, we detail other window storages.

### SQLiteWindowStorage

SQLiteWindowStorage uses an sqlite database. It can be configured like this:

```jsonc
{
  "storage": {
    "class_path": "rslearn.dataset.storage.file.SQLiteWindowStorageFactory"
  }
}
```
