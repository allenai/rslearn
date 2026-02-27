## rslearn.data_sources.aws_open_data.Sentinel2

This data source is for Sentinel-2 L1C and L2A imagery on AWS. It uses the
sentinel-s2-l1c and sentinel-s2-l2a S3 buckets maintained by Sinergise. They state the
data is "added regularly, usually within few hours after they are available on
Copernicus OpenHub".

See https://aws.amazon.com/marketplace/pp/prodview-2ostsvrguftb2 for details about the
buckets. AWS credentials must be configured for use with boto3.

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.aws_open_data.Sentinel2",
  "init_args": {
    // Required modality, either "L1C" or "L2A".
    "modality": "L1C",
    // Required cache directory to cache product metadata files.
    "metadata_cache_dir": "cache/sentinel2",
    // Sort by this attribute, either null (default, meaning arbitrary ordering) or
    // "cloud_cover".
    "sort_by": null,
    // Flag (default false) to harmonize pixel values across different processing
    // baselines (recommended), see
    // https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
    "harmonize": false
  }
}
```

### Available Bands

These uint16 bands are available:

- B01
- B02
- B03
- B04
- B05
- B06
- B07
- B08
- B09
- B10 (L1C only)
- B11
- B12
- B8A

These uint8 bands are available:

- R (from TCI asset; derived from B04)
- G (from TCI asset; derived from B03)
- B (from TCI asset; derived from B02)

### Example

See [planetary_computer_Sentinel2](./planetary_computer_Sentinel2.md#example) for
example usage of a related data source.
