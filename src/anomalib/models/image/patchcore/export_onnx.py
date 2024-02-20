import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf, DictConfig

from anomalib.data import MVTec
from anomalib.deploy import ExportType
from anomalib.deploy.inferencers import ONNXInferencer
from anomalib.engine import Engine
from anomalib.models.image.patchcore import Patchcore


def load_metadata(path: str | Path) -> dict | DictConfig:
    """Load the meta data from the given path.

    Args:
        path (str | Path | dict | None, optional): Path to JSON file containing the metadata.
            If no path is provided, it returns an empty dict. Defaults to None.

    Returns:
        dict | DictConfig: Dictionary containing the metadata.
    """
    metadata: dict[str, float | np.ndarray | torch.Tensor] | DictConfig = {}
    if path is not None:
        config = OmegaConf.load(path)
        metadata = cast(DictConfig, config)
    return metadata


if __name__ == '__main__':
    ckpt_path = "./lightning_logs/version_0/checkpoints/epoch=0-step=7.ckpt"

    checkpoint = torch.load(ckpt_path)
    image_threshold = checkpoint["state_dict"]["image_threshold.value"]
    pixel_threshold = checkpoint["state_dict"]["pixel_threshold.value"]
    print(f"threshold for image score: {image_threshold.item()}")
    print(f"threshold for pixel map: {pixel_threshold.item()}")

    model = Patchcore(input_size=(256, 256), backbone="wide_resnet50_2", coreset_sampling_ratio=0.1, num_neighbors=None)
    datamodule = MVTec(root="/home/nico/Dataset/mvtec", category="bottle", image_size=(256, 256), center_crop=None)
    engine = Engine()
    engine.export(model=model, export_type=ExportType.ONNX, export_root="./",
                  datamodule=datamodule, input_size=(256, 256), ckpt_path=ckpt_path)

    # test for exported model
    inferencer = ONNXInferencer(
        path="weights/onnx/model.onnx",
        metadata="weights/onnx/metadata.json",
        device="GPU",
    )
    image = np.array(Image.open("/home/nico/Dataset/mvtec/bottle/test/broken_large/000.png"))
    result = inferencer.predict(image)

    plt.subplot(121)
    plt.imshow(result.image)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(result.anomaly_map.squeeze(), cmap="jet")
    plt.axis("off")
    plt.show()

    ##### deprecated codes about manually exporting and inferencing
    # model = Patchcore.load_from_checkpoint(checkpoint_path="./lightning_logs/version_0/checkpoints/epoch=0-step=7.ckpt")
    # example_input_array = torch.randn(2, 3, 224, 224)
    #
    # dynamic_axes = {
    #     "input": {0: "batch_size"},
    #     "anomaly_map": {0: "batch_size"},
    #     "pred_score": {0: "batch_size"},
    # }
    # model.to_onnx("patchcore.onnx",
    #               input_sample=example_input_array,
    #               input_names=["input"],
    #               output_names=["anomaly_map", "pred_score"],
    #               dynamic_axes=dynamic_axes,
    #               # do_constant_folding=False,
    #               # opset_version=17,
    #               )
    #
    # import onnxruntime as ort
    # import numpy as np
    #
    # input_array = np.random.randn(10, 3, 224, 224).astype(np.float32)
    # ort_session = ort.InferenceSession("patchcore.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # inputs = {"input": input_array}
    # outputs = ort_session.run(output_names=["anomaly_map", "pred_score"], input_feed=inputs)
    # print(outputs[0].shape, outputs[1].shape)
