# rslearn

rslearn is a tool for developing remote sensing datasets and models.
rslearn helps with:

1. **Developing remote sensing datasets**, starting with defining spatiotemporal
   windows (3D boxes in height/width/time) that are roughly equivalent to training
   examples.
2. **Importing raster and vector data** from various online or local data sources into
   the dataset.
3. **Fine-tuning remote sensing foundation models** on these datasets.
4. **Applying models** on new locations and times.

## New to rslearn?

For a quick introduction to rslearn, walk through the [Quickstart](examples/Quickstart.md).
The quickstart builds a dataset with paired Sentinel-2 images and ESA WorldCover land cover
labels, and fine-tunes an [OlmoEarth](https://github.com/allenai/olmoearth_pretrain) model
for land cover segmentation.

For a deeper introduction:

- Read [Core Concepts](CoreConcepts.md), which summarizes key concepts in rslearn,
  including datasets, windows, layers, and data sources.
- Check out the [Usage Overview](Usage.md), which walk through the typical workflow
  when using rslearn, from defining windows to training models.
- Browse through the [Examples](Examples.md) and find one that's relevant to your project.

## Installation

rslearn requires Python 3.11+ (Python 3.12 is recommended).

```bash
git clone https://github.com/allenai/rslearn.git
cd rslearn
pip install .[extra]
```

## Contact

For questions and suggestions, please
[open an issue on GitHub](https://github.com/allenai/rslearn/issues).
