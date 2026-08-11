# Examples

The [Quickstart](examples/Quickstart.md) provides an end-to-end example showing how to
create an rslearn dataset with aligned Sentinel-2 images and ESA WorldCover land cover
labels, how to fine-tune a model on the dataset, and how to apply the model to map land
cover in new regions.

Below we list other examples, organized by their focus.

## Data-Focused Examples

- [WindowsFromGeojson](examples/WindowsFromGeojson.md): create windows based on a
  GeoJSON file of point features and acquire Sentinel-2 images. Then, train a model to
  predict the point positions.
- [ConvertEuroSAT](examples/ConvertEuroSAT.md): convert the multispectral EuroSAT
  dataset into rslearn windows with raster inputs and vector class labels, then
  fine-tune OlmoEarth for classification.
- [NaipSentinel2](examples/NaipSentinel2.md): create windows based on the timestamp
  that NAIP is available. Then, acquire NAIP images at each window, along with
  Sentinel-2 images captured within one month of the NAIP image. This dataset could be
  used e.g. for super-resolution training.
- [GetImageTimeSeries](examples/GetImageTimeSeries.md): a quick example to obtain time
  series of satellite images for several locations.
- [FindStadiums](examples/FindStadiums.md): a fine-tuning example that shows how to turn
  point labels into rasters for training a segmentation model, and how to compare
  several remote sensing foundation models.

## Training-Focused Examples

- [FinetuneOlmoEarth](examples/FinetuneOlmoEarth.md): fine-tune OlmoEarth to segment
  Sentinel-2 images for land cover and crop type mapping. In this example, the labels
  are taken from the USDA Cropland Data Layer.
- [BitemporalSentinel2](examples/BitemporalSentinel2.md): acquire Sentinel-2 images
  from 2016 and 2024, and train a model to predict which is earlier. This example shows
  how to specify more complex model architectures (it applies OlmoEarth independently
  on the two images and then concatenates the feature maps), and also how to add custom
  augmentations (to randomize the order of the images).

## Computing Embeddings

- [OlmoEarthEmbeddings](examples/OlmoEarthEmbeddings.md): compute OlmoEarth embeddings
  on a location and time range of interest.

## Other Examples

- [EarthDailySentinel2Biophysical](examples/EarthDailySentinel2Biophysical.md): align
  EarthDaily Sentinel-2 scenes with derived LAI/FAPAR/FCOVER products.
- [EarthDailyCloudMaskClearCover](examples/EarthDailyCloudMaskClearCover.md): rank
  EarthDaily Sentinel-2 EDA cloud masks by AOI clear cover, then retrieve the related
  Sentinel-2 L2A scene.
