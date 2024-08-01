Overview
--------

rslearn is a library and tool for developing remote sensing datasets and models.

rslearn helps with:

1. Developing remote sensing datasets, starting with defining spatiotemporal windows
   (roughly equivalent to training examples) that should be annotated.
2. Importing raster and vector data from various online or local data sources into the
   dataset.
3. Annotating new categories of vector data (like points, polygons, and classification
   labels) using integrated web-based labeling apps.
4. Fine-tuning remote sensing foundation models on these datasets.
5. Applying models on new locations and times.


Quick links:
- [CoreConcepts](CoreConcepts.md) summarizes key concepts in rslearn, including
  datasets, windows, layers, and data sources.
- [Quickstart](Quickstart.md) goes through a quick example of developing a remote
  sensing model using rslearn.
- [Examples](Examples.md) contains more examples, including customizing different
  stages of rslearn with additional code.


Setup
-----

rslearn requires Python 3.10+.

    conda create -n rslearn python=3.12
    conda activate rslearn
    pip install -r requirements.txt
    pip install -r extra_requirements.txt

`extra_requirements.txt` contains requirements specific to individual data sources.
