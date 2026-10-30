## rslearn.data_sources.european_soil_properties.EuropeanSoilProperties

This data source provides the 10 m European clay, sand, and silt maps published by
BOKU University with the data descriptor [High-resolution European soil property
maps based on LUCAS and AlphaEarth Foundations satellite
embeddings](https://doi.org/10.1038/s41597-026-08273-1).

The maps represent the 0--20 cm soil layer and cover the EU-27 and the United
Kingdom. They were produced from 2018 covariates and LUCAS 2009 soil texture
samples. The dataset is distributed as 122 EPSG:4326 Cloud-Optimized GeoTIFF tiles
through the [version 2 dataset record](https://doi.org/10.5281/zenodo.21281853).

rslearn reads only the required COG byte ranges during materialization and converts
the stored byte fractions to percentages. Empty pixels are returned as `NaN`. The
source is licensed under CC BY 4.0; users should cite the paper and dataset record.

### Configuration

Data is read directly from the public COGs, so set `"ingest": false`:

```jsonc
{
  "soil_texture": {
    "type": "raster",
    "band_sets": [
      {
        "dtype": "float32",
        "bands": ["clay", "sand", "silt"]
      }
    ],
    "data_source": {
      "name": "rslearn.data_sources.european_soil_properties.EuropeanSoilProperties",
      "ingest": false,
      "cache_dir": "cache/european_soil_properties"
    }
  }
}
```

On first initialization, rslearn reads the headers of all 122 COGs to discover
their exact bounds and saves `tile_index.json` in `cache_dir`. Subsequent
initializations reuse that index without metadata requests. Relative cache paths
are resolved against the dataset directory. Delete the index to refresh it. The
tiles are approximately three degrees across, but their bounds do not form an
exact 3-degree grid.

### Available Bands

| Band | Unit | Description |
|---|---|---|
| `clay` | % | Predicted clay fraction |
| `sand` | % | Predicted sand fraction |
| `silt` | % | Predicted silt fraction |

The three bands are stored together and must be configured in the order shown. For
valid pixels, their sum is approximately 100%, subject to byte quantization.
