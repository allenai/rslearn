## Create Windows

The `rslearn dataset add_windows` command creates windows in an rslearn dataset from the
command-line interface. Windows can be specified either by a bounding box or by a vector
file (e.g. GeoJSON, Shapefile).

For more flexibility, see [ProgrammaticWindows](examples/ProgrammaticWindows.md) for
creating windows programmatically via the Python API.

## Basic Usage

### Create Windows from a Bounding Box

This creates one window under the EPSG:32610 projection at 10 m/pixel. The bounds of
the window in pixel coordinates will be set to match the provided WGS84 (EPSG:4326) box.

```
rslearn dataset add_windows --root ./dataset --group default \
    --src_crs EPSG:4326 --crs EPSG:32610 --resolution 10 \
    --box=-122.4,47.6,-122.3,47.7 \
    --start 2024-06-01T00:00:00+00:00 --end 2024-09-01T00:00:00+00:00
```

Pass `--grid_size N` to create windows at each cell along a NxN grid that
intersects the given box, or `--window_size N` to create an NxN window centered at the
center of the box.

### Create Windows from a Vector File

This will create one window for each feature in `regions.geojson`, under an appropriate
UTM CRS for each feature. The bounds of each window will match the bounds of the
corresponding feature.

```
rslearn dataset add_windows --root ./dataset --group default \
    --utm --resolution 10 \
    --fname regions.geojson \
    --start 2024-06-01T00:00:00+00:00 --end 2024-09-01T00:00:00+00:00
```

If some or all of the geometries are point geometries, make sure to set either
`--grid_size` or `--window_size`; otherwise, the corresponding windows will be 0x0.

## Arguments

See `rslearn dataset add_windows --help` for more details.

### Required

| Argument | Description |
|----------|-------------|
| `--root` | Path to the dataset root directory (containing `config.json`). |
| `--group` | The group to add the windows to. |

Exactly one of `--box` or `--fname` must be specified to define the spatial extent:

| Argument | Description |
|----------|-------------|
| `--box` | Bounding box as comma-separated coordinates `x1,y1,x2,y2`. The coordinates are in the source CRS (see `--src_crs`). |
| `--fname` | Path to a vector file (GeoJSON, Shapefile, etc.). One or more windows are created from the geometries in the file. The CRS is read from the file. |

### Output Projection

These arguments control the CRS and resolution of the created windows.

| Argument | Default | Description |
|----------|---------|-------------|
| `--crs` | EPSG:4326 | The CRS of the output windows, e.g. `EPSG:32610`. |
| `--resolution` | None | The resolution (projection units per pixel) of the output windows. Sets both X and Y resolution (Y is negated). If not set, `--x_res` and `--y_res` are used instead. |
| `--x_res` | 1 | The X resolution of the output windows. |
| `--y_res` | -1 | The Y resolution of the output windows. |
| `--utm` | false | Automatically select an appropriate UTM projection based on the centroid of each geometry (when set, `--crs` is ignored). |

### Source Projection (box only)

When using `--box`, these arguments specify the CRS and resolution of the input
coordinates. They are not used with `--fname` (the CRS is read from the file).

| Argument | Default | Description |
|----------|---------|-------------|
| `--src_crs` | Same as `--crs` | The CRS of the input box coordinates, e.g. `EPSG:4326` for longitude/latitude. |
| `--src_resolution` | None | The resolution of the input coordinates. If not set, `--src_x_res` and `--src_y_res` are used. |
| `--src_x_res` | 1 | The X resolution of the input coordinates. |
| `--src_y_res` | 1 | The Y resolution of the input coordinates. |

### Window Sizing

By default, one window is created per input geometry, sized to fit the geometry's
bounding box. These options change that behavior.

| Argument | Default | Description |
|----------|---------|-------------|
| `--grid_size` | None | Instead of one window per geometry, tile the geometry's bounding box into a grid of this cell size (in pixels) and create one window per grid cell that intersects the geometry. |
| `--window_size` | None | Instead of using the geometry's bounding box, create a fixed-size window (in pixels) centered at the centroid of each geometry. |

Only one of `--grid_size` and `--window_size` can be specified.

### Time Range

| Argument | Default | Description |
|----------|---------|-------------|
| `--start` | None | Start time for the windows (ISO 8601 format, e.g. `2024-06-01T00:00:00Z`). |
| `--end` | None | End time for the windows. |

If specified, data sources will only match items within this time range during the
prepare stage. If not specified, the windows will not have a time range.

### Naming

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | None | Name of the output window. If multiple windows are created (e.g. via `--grid_size`), this is used as a prefix. If not specified, windows are named based on their pixel coordinates and time range. |
