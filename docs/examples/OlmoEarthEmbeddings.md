## Computing Embeddings using OlmoEarth

This tutorial shows how to compute OlmoEarth embeddings on a target location and time
of interest. We will use rslearn to materialize satellite images that we will then pass
to the OlmoEarth encoder. For an introduction to rslearn, see
[the main README](../../README.md) and [CoreConcepts](../CoreConcepts.md).

We proceed in four steps:

1. Create windows in an rslearn dataset that define the spatiotemporal boxes for which
   we want to compute embeddings.

2. Materialize satellite images in the rslearn dataset.

3. Initialize the OlmoEarth pre-trained model and compute and save embeddings.

## Create Windows

First, create a new folder to contain the rslearn dataset, and copy the provided
dataset configuration file, which obtains Sentinel-2, Sentinel-1, and Landsat satellite
images identical in format to those used for pre-training:

```
cd /path/to/rslearn
export DATASET_PATH=/path/to/dataset
mkdir $DATASET_PATH
cp docs/examples/OlmoEarthEmbeddings/dataset_config.json $DATASET_PATH/config.json
```

Now, create a window corresponding to the spatiotemporal box of interest. We use a
10 m/pixel resolution and UTM projection since that matches what was used for
pre-training.

```
rslearn dataset add_windows --root $DATASET_PATH --group default --name default --utm --resolution 10 --src_crs EPSG:4326 --box=-122.4,47.6,-122.3,47.7 --start 2024-01-01T00:00:00+00:00 --end 2025-01-01T00:00:00+00:00
```

Above, the `--box` argument is in the form `lon1,lat1,lon2,lat2`.

The duration of the time range can be adjusted depending on the application -- where
possible, we recommend using a one-year time range, since that is the maximum time
range used during pre-training. For features that change more quickly, it may make
sense to use a shorter time range. If you want to compute embeddings on a specific
satellite image, you can narrow the time range to the minute around the timestamp of
that image. The `dataset_config.json` specifies to create one image mosaic per 30-day
period within the time range, which is recommended since it matches pre-training, but
you could try obtaining images more frequently if desired.

If the box exceeds 10 km x 10 km, we recommend passing `--grid_size` to create multiple
windows that are each limited to 1024x1024:

```
rslearn dataset add_windows --root $DATASET_PATH --group default --name default --utm --resolution 10 --src_crs EPSG:4326 --box=-122.6,47.4,-122.1,47.9 --start 2024-06-01T00:00:00+00:00 --end 2024-08-01T00:00:00+00:00 --grid_size 1024
```

## Materialize Satellite Images

Now, we can use rslearn to materialize the satellite images for the window(s):

```
rslearn dataset prepare --root $DATASET_PATH --workers 32 --disabled-layers landsat --retry-max-attempts 5 --retry-backoff-seconds 5
rslearn dataset materialize --root $DATASET_PATH --workers 32 --no-use-initial-job --disabled-layers landsat --retry-max-attempts 5 --retry-backoff-seconds 5
```

Here, we only obtain Sentinel-2 and Sentinel-1 images. To also obtain Landsat images,
you will need to setup AWS credentials (set the `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` environment variables) for access to the
[`usgs-landsat` requester pays bucket](https://registry.opendata.aws/usgs-landsat/),
however for most tasks we find that OlmoEarth produces high-quality embeddings from
Sentinel-2 and Sentinel-1 alone.

If you used a single window, then the first Sentinel-2 L2A GeoTIFF should appear here:

```
qgis $DATASET_PATH/windows/default/default/layers/sentinel2_l2a/B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12/geotiff.tif
```

With multiple timesteps, you should see folders like `layers/sentinel2_l2a.1`, `layers/sentinel2_l2a.2`, and so on.

## Compute and Save Embeddings

Finally, we can run the provided script to compute and save embeddings for each window.
This will apply the model on each 64x64 within the rslearn window.

```
python docs/examples/OlmoEarthEmbeddings/get_embeddings.py --ds_path $DATASET_PATH --patch_size 4 --model_id OlmoEarth-v1-Base --workers 4 --batch_size 8 --input_size 64 --modalities sentinel2_l2a,sentinel1
```

You can visualize the output embeddings in qgis:

```
qgis $DATASET_PATH/windows/default/default/olmoearth_embeddings.tif
```
