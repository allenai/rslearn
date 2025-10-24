"""Script to collect OlmoEarth embeddings for an rslearn dataset."""

import argparse

import torch
import tqdm
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.model_loader import ModelID
from upath import UPath

from rslearn.dataset.dataset import Dataset, Window
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.models.olmoearth_pretrain.norm import OlmoEarthNormalize
from rslearn.train.data_module import collate_fn
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.transforms.sentinel1 import Sentinel1ToDecibels
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_format import GeotiffRasterFormat


def initialize_dataset_for_olmoearth(
    dataset: Dataset, modalities: list[str], workers: int = 32
) -> ModelDataset:
    """Create a ModelDataset for predicting OlmoEarth embeddings.

    It will read satellite images only, from the rslearn dataset.

    Args:
        dataset: the rslearn dataset to use.
        modalities: the list of modalities to load from the dataset.
        workers: the number of workers to use for initializing the ModelDataset.

    Returns:
        the ModelDataset.
    """
    transforms = []
    if "sentinel1" in modalities:
        transforms.append(
            Sentinel1ToDecibels(
                selectors=["sentinel1"],
            ),
        )

    # Setup the OlmoEarthNormalize normalizer. It needs to know which modalities are
    # present.
    norm_band_names = {}
    if "sentinel2_l2a" in modalities:
        norm_band_names["sentinel2_l2a"] = Modality.SENTINEL2_L2A.band_order
    if "sentinel1" in modalities:
        norm_band_names["sentinel1"] = Modality.SENTINEL1.band_order
    if "landsat" in modalities:
        norm_band_names["landsat"] = Modality.LANDSAT.band_order
    transforms.append(OlmoEarthNormalize(band_names=norm_band_names))

    # Setup the data inputs to read. We read all timesteps for each enabled modality.
    # We will use batch size 1 for data loading so that it works even when there are
    # different numbers of timesteps across windows.
    data_inputs = dict(
        label=DataInput(
            data_type="vector",
            layers=["label"],
            is_target=True,
        ),
    )
    if "sentinel2_l2a" in modalities:
        data_inputs["sentinel2_l2a"] = DataInput(
            data_type="raster",
            layers=["sentinel2_l2a"],
            bands=Modality.SENTINEL2_L2A.band_order,
            passthrough=True,
            load_all_layers=True,
            load_all_item_groups=True,
        )
    if "sentinel1" in modalities:
        data_inputs["sentinel1"] = DataInput(
            data_type="raster",
            layers=["sentinel1"],
            bands=Modality.SENTINEL1.band_order,
            passthrough=True,
            load_all_layers=True,
            load_all_item_groups=True,
        )
    if "landsat" in modalities:
        data_inputs["landsat"] = DataInput(
            data_type="raster",
            layers=["landsat"],
            bands=Modality.LANDSAT.band_order,
            passthrough=True,
            load_all_layers=True,
            load_all_item_groups=True,
        )

    model_dataset = ModelDataset(
        dataset=dataset,
        split_config=SplitConfig(
            transforms=transforms,
            skip_targets=True,
        ),
        inputs=data_inputs,
        # We set a dummy task since it is required.
        task=ClassificationTask(property_name="placeholder", classes=["cls0"]),
        workers=workers,
    )
    return model_dataset


def get_embeddings(
    model: OlmoEarth,
    model_dataset: ModelDataset,
    batch_size: int,
    input_size: int,
    patch_size: int,
    modalities: list[str],
    workers: int = 16,
) -> None:
    """Get OlmoEarth embeddings for each window in the ModelDataset.

    Inference is done in a sliding window manner within the rslearn window, where the
    size of each input crop is controlled by input_size.

    Args:
        model: the OlmoEarth model to apply.
        model_dataset: the dataset to apply the model on.
        batch_size: batch size to use for applying the model within a window.
        input_size: the sliding crop size for inference.
        patch_size: the transformer patch size used in the model.
        modalities: the list of modalities to process.
        workers: number of data loader worker processes.
    """
    data_loader = torch.utils.data.DataLoader(
        dataset=model_dataset,
        num_workers=workers,
        collate_fn=collate_fn,
        # We load one window at a time here since different windows may have different
        # numbers of timesteps. However, we assume
        batch_size=1,
    )

    with torch.no_grad():
        for inputs, _, metadatas in tqdm.tqdm(data_loader):
            assert len(metadatas) == 1
            metadata = metadatas[0]

            gpu_inputs: list[dict] = [
                {
                    k: (v.to(device) if v is not None else None)
                    for k, v in inputs[0].items()
                }
            ]
            features = None
            # Use Sentinel-2 to get the shape.
            image = gpu_inputs[0]["sentinel2_l2a"]
            height = image.shape[1]
            width = image.shape[2]

            print(
                f"processing window {metadata['window_name']} with height={height}, width={width}"
            )
            for modality in modalities:
                print(modality, "shape", gpu_inputs[0][modality].shape)

            rows = list(range(0, height - input_size, input_size)) + [
                height - input_size
            ]
            cols = list(range(0, width - input_size, input_size)) + [width - input_size]
            all_positions = [(row, col) for row in rows for col in cols]

            for batch_idx in range(0, len(all_positions), batch_size):
                batch = all_positions[batch_idx : batch_idx + batch_size]
                print(f"row={batch[0][0]} col={batch[0][1]}")
                cropped_inputs = []
                for row, col in batch:
                    cropped_inputs.append(
                        {
                            k: (
                                v[:, row : row + input_size, col : col + input_size]
                                if v is not None
                                else None
                            )
                            for k, v in gpu_inputs[0].items()
                        }
                    )
                cur_features = model(cropped_inputs)[0]
                cur_features = cur_features.to("cpu")

                if features is None:
                    features = torch.zeros(
                        (
                            cur_features.shape[1],
                            height // patch_size,
                            width // patch_size,
                        ),
                        dtype=torch.float32,
                        device="cpu",
                    )
                for (row, col), feat in zip(batch, cur_features):
                    features[
                        :,
                        row // patch_size : (row + input_size) // patch_size,
                        (col // patch_size) : (col + input_size) // patch_size,
                    ] = feat

            assert features is not None

            # When writing the GeoTIFF, we need to update the Projection because of the patch size.
            projection = Projection(
                metadata["projection"].crs,
                metadata["projection"].x_resolution * patch_size,
                metadata["projection"].y_resolution * patch_size,
            )
            bounds = (
                metadata["bounds"][0] // patch_size,
                metadata["bounds"][1] // patch_size,
                metadata["bounds"][2] // patch_size,
                metadata["bounds"][3] // patch_size,
            )

            window_path = Window.get_window_root(
                dataset.path, metadata["group"], metadata["window_name"]
            )
            GeotiffRasterFormat().encode_raster(
                window_path,
                projection,
                bounds,
                features.numpy(),
                fname="olmoearth_embeddings.tif",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_path",
        type=str,
        help="The path to the rslearn dataset",
        required=True,
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        help="Patch size to use for OlmoEarth model",
        required=True,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="OlmoEarth model ID",
        default="OlmoEarth-v1-Base",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes for dataset initialization and data loading",
        default=4,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Model batch size.",
        default=8,
    )
    parser.add_argument(
        "--input_size",
        type=int,
        help="Model input size. We apply the model in a sliding window manner within each rslearn window.",
        default=64,
    )
    parser.add_argument(
        "--modalities",
        type=str,
        help="Comma-separated list of modalities to input, like 'sentinel2_l2a' or 'sentinel2_l2a,sentinel1'",
        default="sentinel2",
    )
    args = parser.parse_args()
    modalities = args.modalities.split(",")

    print("Initializing dataset")
    dataset = Dataset(UPath(args.ds_path))
    model_dataset = initialize_dataset_for_olmoearth(
        dataset,
        workers=args.workers,
        modalities=modalities,
    )

    print("Initializing OlmoEarth model")
    device = torch.device("cuda")
    model = OlmoEarth(
        model_id=ModelID(args.model_id),
        patch_size=args.patch_size,
    )
    model.to(device)
    model.eval()

    get_embeddings(
        model=model,
        model_dataset=model_dataset,
        batch_size=args.batch_size,
        input_size=args.input_size,
        patch_size=args.patch_size,
        modalities=modalities,
        workers=args.workers,
    )
