[Back to DatasetConfig.md](../DatasetConfig.md)

## Data Source Config

Layers populated automatically from data sources specify additional configuration
options beyond those detailed in [LayerConfig](./LayerConfig.md):

```jsonc
{
  "data_source": {
    // The class path of the data source.
    "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
    // Data sources may expose additional configuration options, passed via init_args.
    // class_path and init_args are handled by jsonargparse to instantiate the data
    // source class.
    "init_args": {
      // ...
    },
    // The query configuration specifies how items should be matched to windows. It is
    // optional, and the values below are defaults.,
    "query_config": {
      // The space mode must be "MOSAIC" (default), "CONTAINS", "INTERSECTS", or "SINGLE_COMPOSITE".
      "space_mode": "MOSAIC",
      // The max matches defaults to 1.
      "max_matches": 1,
      // For MOSAIC, the number of overlapping items wanted within each item group covering
      // the window (default 1). Set higher for compositing.
      "mosaic_compositing_overlaps": 1,
      // By default, the space mode controls how multiple item groups are created in case
      // max_matches > 1. If period_duration is set, the window time range is instead
      // divided into periods of this duration, and the space mode is applied within each
      // period to produce one item group per period.
      "period_duration": null,
      // When period_duration is set, whether to return item groups in reverse temporal
      // order (most recent first). Should always be set to false when setting period_duration.
      // The default is true for backwards compatibility (deprecated).
      "per_period_mosaic_reverse_time_order": false,
    },
    // The time offset is optional. It defaults to 0.
    "time_offset": "0d",
    // The duration is optional. It defaults to null.
    "duration": null,
    // The ingest flag is optional, and defaults to true.
    "ingest": true
  },
  // Re-sampling method to use during materialization. This only applies to raster
  // layers with a data source. It is used when there is a difference in CRS or
  // resolution between the item from the data source and the window's target.
  // It is one of "nearest", "bilinear" (default), "cubic", "cubic_spline".
  "resampling_method": "bilinear",
  // The compositing method to use, to handle raster item groups with more than one item.
  // It can be FIRST_VALID (default), MEAN, MEDIAN, or SPATIAL_MOSAIC_TEMPORAL_STACK.
  "compositing_method": "FIRST_VALID"
}
```

The `class_path` and `init_args` options configure the data source itself. See
[DataSources](DataSources.md) for details on all of the built-in data sources in rslearn.

rslearn retrieves data from data sources in three steps: prepare, ingest, and materialize.

### Prepare

In the prepare stage, we match items in the data source with each window in the dataset.
The output of prepare is a list of *item groups* for each window, where each group
specifies a different list of items that should be composited to form a different
vector or raster file for that window. The following options affect prepare:

- `space_mode`
- `max_matches`
- `mosaic_compositing_overlaps`
- `period_duration`
- `per_period_mosaic_reverse_time_order`
- `time_offset`
- `duration`

### Ingest

In the ingest stage, we download items from the data source that matched with at least
one window. The following options affect ingest:

- `ingest` (ingest flag)

### Materialize

In the materialize stage, we re-project, crop, and composite items within each item
group to align with the window. The following options affect materialize:

- `resampling_method`
- `compositing_method`

Below, we detail these options in order by stage.

## Prepare Stage Configuration

For each window, the prepare stage starts with a list of items provided by the data source
that intersect the window's spatial extent and time range. The output from matching is a
`list[list[Item]]` (list of item groups), where each item group corresponds to the items
that will be used to create one composite of raster or vector data.

### Time Offset and Duration

By default, the time range used for requesting items from the data source and applying
the matching strategy is the time range of the window. The request time range can be
adjusted by setting `time_offset` and/or `duration`. This is particularly useful when
the desired time range varies across layers.

`time_offset` specifies a positive or negative time delta. If set, the time delta is
added to the time range (both the start and the end time). It is parsed by
[pytimeparse](https://github.com/wroberts/pytimeparse). For example:

- "30d" means to adjust the time range 30 days into the future.
- "-30d" means to adjust the time range 30 days into the past.

`duration` specifies a positive time delta. If set, the end time of the request time
range is set to the start time plus `duration`.

Suppose the window time range is [2024-01-01, 2024-02-01].

- With time_offset=30d, the request time range is [2024-01-31, 2024-03-02].
- With duration=180d, the request time range is [2024-01-01, 2024-06-29].
- With time_offset=30d AND duration=180d, the request time range is [2024-01-31, 2024-07-29].

### Space Mode and Max Matches

The `space_mode` defines the matching strategy. It interacts with `max_matches`, which
specifies the maximum number of item groups to produce.

**CONTAINS.** Use items that fully contain the window bounds. The resulting item groups
will each consist of exactly one item. This strategy iterates over the items in the
order they are provided by the data source (some data sources provide sorting
options, e.g. sort by cloud cover), filtering ones that do not contain the window,
and creating single-item item groups for the rest, continuing until there are no more
items or `max_matches` item groups have been created.

**INTERSECTS.** Use items that intersect the window bounds. As with CONTAINS, the
resulting item groups will each consist of exactly one item.

**MOSAIC.** Create mosaics, where each item group combines multiple items from the data
source as needed to cover the entire window. In this case, each item group may
include multiple items. This strategy initializes a buffer of `max_matches` empty
item groups. It then iterates over the items, adding each item to the first group
that the item provides additional coverage for (skipping groups that already cover
all the portions of the window that the new item covers). Finally, the non-empty
groups are returned.

**SINGLE_COMPOSITE.** Put all items into one item group. This is most useful when computing
composites over all of the available data.

**Example.**
Consider a window covering a 10km x 10km region with a time range of January 1 to April
1. The data source returns four items in order:

- Item A: covers the full window (10km x 10km), from January 15
- Item B: covers the left half of the window (5km x 10km), from January 20
- Item C: covers the right half of the window (5km x 10km), from March 10
- Item D: covers the full window (10km x 10km), from March 20

With `max_matches=2`:

- **CONTAINS** returns `[[A], [D]]`. Both A and D fully contain the window. B and C are
  skipped because they only partially cover the window.
- **INTERSECTS** returns `[[A], [B]]`. All four items intersect the window, but we stop
  at 2 due to max_matches. Each item becomes its own single-item group.
- **MOSAIC** returns `[[A], [B, C]]`. Item A covers the full window, completing the
  first mosaic. Item B doesn't add coverage to the first mosaic (A already covers it),
  so B starts the second mosaic. Item C adds the right half to the second mosaic. Item D
  doesn't add new coverage to either mosaic.
- **SINGLE_COMPOSITE** returns `[[A, B, C, D]]`.

### Period Duration

By default, when `period_duration` is not set, the space mode determines how to handle
matching with `max_matches > 1`. For example, with MOSAIC, the resulting item groups
can arbitrarily combine items from across the request time range.

If `period_duration` is set, rslearn divides the request time range into periods of that
duration, and the space mode is applied within each period to obtain one item group per
period. It starts from the most recent period within the time range, finding all items
temporally intersecting that period and passing them to the space mode strategy. If no
items are found, the period is skipped. It continues until either there are no more
periods (i.e., it reaches the beginning of the time range) or it has created max_matches
item groups.

In the example above, when using MOSAIC with `period_duration="30d"`, rslearn returns
`[[A], [C, D]]`. The time range is split into January, February, and March periods. For
March, items C and D are combined into one mosaic. February is skipped since there are
no matching items. For January, item A covers the full window. In other words, we end
up with one monthly mosaic for each 30-day period in the request time range.

### Compositing Overlaps

For MOSAIC, the default behavior is to create item groups that cover the window's
spatial extent once. `mosaic_compositing_overlaps` can be set greater than 1 to have
each item group cover the window multiple times. This is useful when computing mean or
median composites for each item group.

In the example above, when using MOSAIC with `mosaic_compositing_overlaps = 2` and
`max_matches=2`, it returns `[[A, B, C], [D]]`. The first item group is completed
once it covers the window's spatial extent twice. The second item group only covers
the window once, but it is still returned.

## Ingest Stage Configuration

During the ingest stage, rslearn downloads all items that appear in an item group for at
least one window.

### Ingest Flag

The ingest flag specifies whether this data source should be ingested.

The default interface for data sources is represented as a collection of items, where
the items are matched to windows and then the items need to first be ingested before
they can be re-projected and cropped to align with individual windows. However, some
data sources support (or require) directly materializing data into the window.

For example, `XyzTiles` represents a slippy map tiles layer, i.e. a mosaic covering the
entire world that is broken up into tiles. Rather than representing each tile as a
separate item (which would be inefficient), it only supports directly materializing the
data into windows. Then, when using this data source, the ingest flag should be set to
false.

Other data sources like PlanetaryComputer (which uses COGs on Microsoft Planetary
Computer) support both approaches (download entire COGs and then align locally, or read
crops directly from the remote COGs). In this case, ingestion will be faster for dense
windows while direct materialization will be faster for sparse windows.

## Materialize Stage Configuration

During the materialize stage, items are re-projected and cropped to align with the window.
Each item group corresponds to a distinct vector or raster materialized output. For item
groups with more than one item, vector data is handled by concatenating vector features,
while raster data is handled via a configurable compositing method, e.g. computing a mean
composite across items in the group.

### Compositing Method

For vector data, non-singleton item groups are handled by concatenating the vector
features across items in the group.

Compositing raster data is more complex, and a `compositing_method` option is provided
to control the behavior. By default, `compositing_method = FIRST_VALID`; for each
pixel and band, the value is set based on the first item that is not NODATA at that
pixel and band.

The `compositing_method` can instead be set to MEAN or MEDIAN to compute the mean or
median across all items in the group that are not NODATA at that pixel and band.

#### SPATIAL_MOSAIC_TEMPORAL_STACK

The SPATIAL_MOSAIC_TEMPORAL_STACK compositing method is also available to handle
multi-temporal rasters (typically used with SINGLE_COMPOSITE).

Suppose the data source returns items spanning calendar months with hourly observations
(e.g., hourly precipitation). For a window spanning January 15 - February 15, we would
match with the January and February items. SPATIAL_MOSAIC_TEMPORAL_STACK will first
compute the union of time ranges across the items in the item group clipped to the
window time range; in this case, that is every hour from January 15 to February 15.
Then, it initializes a 3D THW grid where the T dimension corresponds to those time
ranges, while the H/W dimensions correspond to the window's spatial extent. It populates
the grid by iterating over items and copying them into output cells that are still
NODATA.

### Resampling Method

For raster data, this option configures how resampling should be performed when
re-projection is required (i.e., when the native projection of the data source item
does not match the window projection). The default is bilinear resampling, but it can
be set to  "nearest", "cubic", or "cubic_spline" instead.

For data sources with rasters consisting of categorical pixel values, you should use
"nearest".
