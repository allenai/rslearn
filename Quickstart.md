Quickstart
----------

This is a quick example of building a remote sensing dataset, and then training a model
on that dataset, using rslearn. We assume you have already setup rslearn, so follow the
setup instructions if you haven't.

TODO: modify below to include training, inference, etc.

Here is an example of acquiring a NAIP and a Sentinel-2 image of the AI2 office.
We will get RGB version of both images.

You will need to set up AWS credentials since this example will acquire NAIP images from AWS.
You can remove the naip layer to avoid the requirement for setting up credentials.

First create a directory `/path/to/dataset` and corresponding configuration file at `/path/to/dataset/config.json` as follows:

    {
        "layers": {
            "naip": {
                "type": "raster",
                "band_sets": [{
                    "dtype": "uint8",
                    "bands": ["R", "G", "B"],
                    "format": {"name": "single_image", "format": "png"}
                }],
                "data_source": {
                    "name": "rslearn.data_sources.aws_open_data.Naip",
                    "index_cache_dir": "cache/naip/",
                    "use_rtree_index": true,
                    "query_config": {
                        "max_matches": 1
                    }
                }
            },
            "sentinel2": {
                "type": "raster",
                "band_sets": [{
                    "dtype": "uint8",
                    "bands": ["R", "G", "B"],
                    "format": {"name": "single_image", "format": "png"},
				    "zoom_offset": -3
                }],
                "data_source": {
                    "name": "rslearn.data_sources.gcp_public_data.Sentinel2",
                    "index_cache_dir": "cache/sentinel2/",
                    "max_time_delta": "1d",
                    "sort_by": "cloud_cover",
                    "use_rtree_index": false,
                    "query_config": {
                        "max_matches": 1
                    }
                }
            }
        },
        "tile_store": {
                "name": "file",
                "root_dir": "tiles"
        }
    }

You can replace the format `{"name": "single_image", "format": "png"}` with `{"name": "geotiff"}` to get a GeoTIFF instead of PNG.

Create a spatiotemporal window:

    python -m rslearn.main dataset add_windows --root /path/to/dataset --group default --utm --resolution 0.6 --window_size 4096 --src_crs EPSG:4326 --box=-122.33129,47.64796,-122.33129,47.64796 --start 2019-01-01T00:00:00+00:00 --end 2024-01-01T00:00:00+00:00 --name ai2

Then prepare, ingest, and materialize:

    mkdir -p /path/to/dataset/cache/{naip,sentinel2}
    python -m rslearn.main dataset prepare --root /path/to/dataset
    python -m rslearn.main dataset ingest --root /path/to/dataset --workers 8
    python -m rslearn.main dataset materialize --root /path/to/dataset --workers 8
