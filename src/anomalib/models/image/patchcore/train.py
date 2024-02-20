import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models.image.patchcore import Patchcore

if __name__ == '__main__':
    model = Patchcore(input_size=(256, 256), backbone="wide_resnet50_2", coreset_sampling_ratio=0.1, num_neighbors=None)
    datamodule = MVTec(root="/home/nico/Dataset/mvtec", category="bottle", image_size=(256, 256), center_crop=None)
    engine = Engine(
        threshold="F1AdaptiveThreshold",
        image_metrics=["AUROC", "BinaryF1Score"],
        pixel_metrics=["AUROC", "BinaryF1Score"],
    )
    engine.fit(datamodule=datamodule, model=model)
