Here we explore applying a model on global WebMercator image tiles, similar to
the workflow here:

https://github.com/allenai/remote-sensing-data-hub

In particular, we would like to parallelize retrieving images and applying the
model across different zoom 7 tiles (there are 2^7 = 128 tiles along each
access, so 16384 tiles/jobs total).

We can initialize the jobs with the dataset configuration, model configuration,
and model weights:

```
dataset_root/
    config.json
model_root/
    config.json
    runs/
        7d7795e1-df9c-48b6-b8a5-f3738bdd3097/
            best.pth
```

In the job, we can first create the corresponding window and retrieve the
needed images:

```
rslearn dataset add_windows --root dataset_root/ --crs epsg:3857 \ --zoom 7
    --tile-size 32768 --size 512 --time ... --group inference
    --box 32768,32768,65536,65536 --size 512
rslearn dataset materialize
```

Above, the example `--box` corresponds to tile `(1, 1)`.

Then apply the model:

TODO

We can then upload the outputs to object storage and collect them later.
