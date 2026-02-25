## rslearn.data_sources.worldpop.WorldPop

This data source is for world population data from worldpop.org.

Currently, this only supports the WorldPop Constrained 2020 100 m Resolution dataset.
See https://hub.worldpop.org/project/categories?id=3 for details.

The data is split by country. We implement with LocalFiles data source for simplicity,
but it means that all of the data must be downloaded first.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.worldpop.WorldPop",
  "init_args": {
    // Required local path to store the downloaded WorldPop data.
    "worldpop_dir": "cache/worldpop"
  }
}
```
