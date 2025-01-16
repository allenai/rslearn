Examples
--------

- [WindowsFromGeojson](examples/WindowsFromGeojson.md): create windows based on a
  GeoJSON file of point features and acquire Sentinel-2 images. Then, train a model to
  predict the point positions.
- [ProgrammaticWindows](examples/ProgrammaticWindows.md): a simple example of creating
  windows programmatically, in case the `dataset add_windows` command is insufficient
  for your use case.
- [NaipSentinel2](examples/NaipSentinel2.md): create windows based on the timestamp
  that NAIP is available. Then, acquire NAIP images at each window, along with
  Sentinel-2 images captured within one month of the NAIP image. This dataset could be
  used e.g. for super-resolution training.
- [BitemporalSentinel2](examples/BitemporalSentinel2.md): acquire Sentinel-2 images
  before and after a certain timestamp.
