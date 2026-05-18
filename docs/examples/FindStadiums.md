Download the stadium CSV and create windows:

```
export DATASET_PATH=./dataset
mkdir $DATASET_PATH
cp docs/examples/FindStadiums/config.json $DATASET_PATH/config.json
wget https://raw.githubusercontent.com/gboeing/data-visualization/refs/heads/main/ncaa-football-stadiums/data/stadiums-geocoded.csv
python docs/examples/FindStadiums/create_windows.py --csv_path stadiums-geocoded.csv --ds_path $DATASET_PATH
```

Materialize the dataset:

```
rslearn dataset prepare --root $DATASET_PATH
rslearn dataset materialize --root $DATASET_PATH
```

Visualize some GeoTIFF:

```
qgis $DATASET_PATH/windows/default/Ball_State/layers/sentinel2.1/B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12/geotiff.tif $DATASET_PATH/windows/default/Ball_State/layers/label/label/geotiff.tif
```

Fit the model:

```
export MANAGEMENT_DIR=./project_data
rslearn model fit \
  --config docs/examples/FindStadiums/config.yaml \
  --project_name 2026_05_15_find_stadiums \
  --run_name olmoearth_tiny_00 \
  --management_dir $MANAGEMENT_DIR
```

Create a 1024x1024 prediction window centered on downtown Seattle (Space Needle). At
10 m/pixel this covers ~10 km x 10 km, which includes Lumen Field to the south:

```
rslearn dataset add_windows \
  --root $DATASET_PATH \
  --group predict \
  --utm \
  --resolution 10 \
  --src_crs EPSG:4326 \
  --box=-122.3493,47.6205,-122.3493,47.6205 \
  --window_size 1024 \
  --start 2025-06-01T00:00:00+00:00 \
  --end 2025-08-30T00:00:00+00:00 \
  --name seattle
```

Prepare and materialize the prediction window:

```
rslearn dataset prepare --root $DATASET_PATH --group predict
rslearn dataset materialize --root $DATASET_PATH --group predict
```

Run prediction using the latest checkpoint:

```
rslearn model predict \
  --config docs/examples/FindStadiums/config.yaml \
  --ckpt_path $MANAGEMENT_DIR/2026_05_15_find_stadiums/olmoearth_tiny_00/last.ckpt
```

Visualize the Sentinel-2 image and model output in QGIS:

```
qgis $DATASET_PATH/windows/predict/seattle/layers/sentinel2/*/geotiff.tif $DATASET_PATH/windows/predict/seattle/layers/output/*/geotiff.tif
```
