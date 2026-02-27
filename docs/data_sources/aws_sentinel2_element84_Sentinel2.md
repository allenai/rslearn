## rslearn.data_sources.aws_sentinel2_element84.Sentinel2

This data source is for Sentinel-2 L2A imagery from the Element 84 Earth Search STAC
API on AWS. It uses the `s3://sentinel-cogs` S3 bucket which provides Cloud-Optimized
GeoTIFFs, enabling direct materialization without ingestion.

See https://aws.amazon.com/marketplace/pp/prodview-ykj5gyumkzlme for details.

The bucket is public and free so no credentials are needed.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.aws_sentinel2_element84.Sentinel2",
  "init_args": {
    // Optional STAC query filter.
    // Example: {"eo:cloud_cover": {"lt": 20}}
    "query": null,
    // Sort by this STAC property, e.g. "eo:cloud_cover".
    "sort_by": null,
    // Whether to sort ascending or descending (default ascending).
    "sort_ascending": true,
    // Optional directory to cache discovered items.
    "cache_dir": null,
    // Flag (default false) to harmonize pixel values across different processing
    // baselines (recommended), see
    // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    "harmonize": false,
    // Timeout for requests.
    "timeout": "10s"
  }
}
```

### Available Bands

- B01 (uint16, from coastal asset)
- B02 (uint16, from blue asset)
- B03 (uint16, from green asset)
- B04 (uint16, from red asset)
- B05 (uint16, from rededge1 asset)
- B06 (uint16, from rededge2 asset)
- B07 (uint16, from rededge3 asset)
- B08 (uint16, from nir asset)
- B09 (uint16, from nir09 asset)
- B11 (uint16, from swir16 asset)
- B12 (uint16, from swir22 asset)
- B8A (uint16, from nir08 asset)
- R, G, B (uint8, from visual asset)

### Example

See [planetary_computer_Sentinel2](./planetary_computer_Sentinel2.md#example) for
example usage of a related data source.
