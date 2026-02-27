## rslearn.data_sources.aws_sentinel1.Sentinel1

This data source is for Sentinel-1 GRD imagery on AWS. It uses the sentinel-s1-l1c S3
bucket maintained by Element 84. See
https://aws.amazon.com/marketplace/pp/prodview-uxrsbvhd35ifw for details about the
bucket.

Although other Sentinel-1 scenes are available, the data source currently only supports
the GRD IW DV scenes (vv+vh bands). It uses the Copernicus API for metadata search
(prepare step).

### Configuration

```jsonc
{
  "class_path": "rslearn.data_sources.aws_sentinel1.Sentinel1",
  "init_args": {
    // Optional orbit direction to filter by, either "ASCENDING" or "DESCENDING". The
    // default is to not filter (so both types of scenes are included/mixed).
    "orbit_direction": null
  }
}
```

### Available Bands

- vv
- vh

### Example

See [planetary_computer_Sentinel1](./planetary_computer_Sentinel1.md) for example usage
of a related data source.
