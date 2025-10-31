Examples
--------

- [WindowsFromGeojson](examples/WindowsFromGeojson.md): create windows based on a
  GeoJSON file of point features and acquire Sentinel-2 images. Then, train a model to
  predict the point positions.
- [ProgrammaticWindows](examples/ProgrammaticWindows.md): a simple example of creating
  windows programmatically, in case the `dataset add_windows` command is insufficient
  for your use case. This example also shows how to programmatically add raster and
  vector data into your dataset.
- [NaipSentinel2](examples/NaipSentinel2.md): create windows based on the timestamp
  that NAIP is available. Then, acquire NAIP images at each window, along with
  Sentinel-2 images captured within one month of the NAIP image. This dataset could be
  used e.g. for super-resolution training.
- [BitemporalSentinel2](examples/BitemporalSentinel2.md): acquire Sentinel-2 images
  from 2016 and 2024, and train a model to predict which is earlier. This example shows
  how to specify more complex model architectures (it applies SatlasPretrain
  independently on the two images and then concatenates the feature maps), and also how
  to add custom augmentations (to randomize the order of the images).
- [OlmoEarthEmbeddings](examples/OlmoEarthEmbeddings.md): compute OlmoEarth embeddings
  on a location and time range of interest.
