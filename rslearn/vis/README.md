# Dataset Visualizer

The rslearn **visualization server** allows for easily inspecting windows from the
browser, including viewing per-window rasters and vector labels.

The server (`python -m rslearn.vis.vis_server`) loads a dataset's `config.json`,
samples windows, and serves a small Flask UI with per-window imagery and labels.

## Prerequisites

- An rslearn dataset (directory containing `config.json` and windows).
- Layers you want to view must be materialized; incomplete windows will still appear in the table but may skip missing layers.

## Usage

```bash
python -m rslearn.vis.vis_server /path/to/dataset \
  [raster / vector options] \
  --max_samples 100 \
  --port 8000
```

You must enable **at least one** of:

- `--raster_groups` — raster layers as PNGs
- `--vector_text_groups` — vector layers shown as text (GeoJSON string or a single class label)
- `--vector_image_groups` — vector layers rendered as PNGs (detection overlay or segmentation mask)

Optional:

- `--groups` — restrict which window groups are loaded (e.g. `predict` only). If omitted, all groups under `windows/` are loaded.
- `--resampling` — JSON object mapping group name to a PIL resampling method used when resizing to the display size (512×512). Defaults to `nearest`. Valid methods: `nearest`, `bilinear`, `bicubic`, `lanczos`. Example: `'{"sentinel2": "bilinear", "label": "nearest"}'`.

### Item group names (`layer` vs `layer.N`)

Paths in a window's layer directory use an **item group** name: the layer name, or
`layer_name.<index>` for the *N*th temporal/group slot (e.g. `sentinel2`, `sentinel2.1`).
The CLI uses the same strings for `--raster_groups`, `--vector_text_groups`, and
`--vector_image_groups`.

## 1. Rasters only

Every raster group needs:

1. **`--bands`** — JSON object mapping that group name to a list of band names to visualize.
2. **`--raster_render`** — JSON object mapping that group name to a render spec: `{ "name": "<method>", "args": { ... } }` (optional `args`).

Supported `name` values (see `rslearn.vis.render_raster`):

| `name`           | Typical use |
|------------------|-------------|
| `sentinel2_rgb`  | Sentinel-2–style reflectance (divide by 10, clip to 0–255). |
| `percentile`     | Per-band 2–98% stretch to uint8. |
| `minmax`         | Per-band min/max stretch. |
| `linear`         | Fixed user-configured range, e.g. `args`: `vmin`, `vmax`. |
| `classes`        | Class index raster. |

Example — RGB + label raster:

```bash
python -m rslearn.vis.vis_server /path/to/dataset \
  --raster_groups sentinel2 landsat label_raster \
  --bands '{"sentinel2": ["B04", "B03", "B02"], "landsat": ["B4", "B3", "B2"], "label_raster": ["class"]}' \
  --raster_render '{"sentinel2": {"name": "sentinel2_rgb"}, "landsat": {"name": "linear", "args": {"vmin": 5000, "vmax": 17000}}, "label_raster": {"name": "classes"}}'
```

Multi-temporal rasters use separate group names per slot, e.g. `sentinel2`, `sentinel2.1`, each with their own `bands` and `raster_render` entries.

## 2. Vector as text

Use **`--vector_text_groups`** and **`--vector_text_render`**. Each vector group must have a render entry: `{ "name": "text" }` or `{ "name": "property" }`.

### `text` — full GeoJSON FeatureCollection

- Renders **all** features in the window as a pretty-printed **GeoJSON** string (geometries + properties).
- Use when you want to inspect raw attributes, multiple features, or geometry.

```bash
python -m rslearn.vis.vis_server /path/to/dataset \
  --vector_text_groups labels \
  --vector_text_render '{"labels": {"name": "text"}}' \
  --max_samples 100
```

### `property` — single string from `class_property_name`

- Reads **`layer_config.class_property_name`** from the dataset config and returns the **first non-null** value for that property among features as plain text (stringified).
- Matches **ClassificationTask** setups where one window has one primary class label stored on vector features.

Requirements:

- Vector layer in `config.json` must set **`class_property_name`** to the GeoJSON property holding the class (same as training).

```bash
python -m rslearn.vis.vis_server /path/to/dataset \
  --raster_groups sentinel2 \
  --vector_text_groups labels \
  --bands '{"sentinel2": ["B04", "B03", "B02"]}' \
  --raster_render '{"sentinel2": {"name": "sentinel2_rgb"}}' \
  --vector_text_render '{"labels": {"name": "property"}}' \
```

## 3. Vector as image

Use **`--vector_image_groups`** and **`--vector_image_render`**.
If `class_names` is set, it will be used to select colors from `DEFAULT_COLORS` in `rslearn.utils.colors`,
otherwise, labels fallback to a default red color.

Both modes require **`class_property_name`** on the vector layer so each feature’s class can be colored.

### `detection` — points on a reference image (or black background)

- Expects **point** geometries (see `flatten_shape` / point drawing in the renderer).
- If you also pass **`--raster_groups`** / **`--bands`** / **`--raster_render`**, the first raster group that appears in both `bands` and `raster_render` is used as the **background image**; vectors are drawn on top.
- If no reference raster is available, draws on a **black** canvas sized to the window bounds.

```bash
python -m rslearn.vis.vis_server /path/to/dataset \
  --raster_groups sentinel2 \
  --vector_image_groups boxes \
  --bands '{"sentinel2": ["B04", "B03", "B02"]}' \
  --raster_render '{"sentinel2": {"name": "sentinel2_rgb"}}' \
  --vector_image_render '{"boxes": {"name": "detection"}}' \
```

Here `boxes` is the vector layer name in `config.json` (with `class_property_name` set for detection labels).

### `segmentation` — polygon fill mask

- Renders **Polygon** / **MultiPolygon** features into a full-window **RGB mask** (window pixel size).
- Polygons are sorted by label so overlapping order is deterministic; holes in polygons are supported (interiors filled black).
- Does not composite over a raster by default (mask only).

```bash
python -m rslearn.vis.vis_server /path/to/dataset \
  --vector_image_groups land_polygons \
  --vector_image_render '{"land_polygons": {"name": "segmentation"}}' \
```

**Detection vs segmentation:** **`detection`** = points + optional satellite underlay; **`segmentation`** = filled polygons as a label map.

## 4. Combined example (RGB + label raster + vector text + detection overlay)

```bash
python -m rslearn.vis.vis_server /path/to/dataset \
  --raster_groups sentinel2 worldcover \
  --vector_text_groups meta \
  --vector_image_groups objects \
  --bands '{"sentinel2": ["B04", "B03", "B02"], "worldcover": ["B1"]}' \
  --raster_render '{"sentinel2": {"name": "sentinel2_rgb"}, "worldcover": {"name": "classes"}}' \
  --vector_text_render '{"meta": {"name": "text"}}' \
  --vector_image_render '{"objects": {"name": "detection"}}' \
  --resampling '{"sentinel2": "bilinear", "worldcover": "nearest"}' \
  --groups default \
  --max_samples 80 \
  --port 8000
```

## 5. Limitations and behavior notes

- **Sampling:** Up to **`--max_samples`** windows are chosen at random from those loaded (seed not fixed — order changes between runs).
- **Display size:** Served PNGs are **resized to 512×512** (`VISUALIZATION_IMAGE_SIZE` in `rslearn.vis.utils`) for the browser; source windows can be larger or smaller.

For training-time PNG dumps from the model, use **`rslearn model test`** with `visualize_dir` and task-specific `Task.visualize`; that path is separate from this dataset browser.
