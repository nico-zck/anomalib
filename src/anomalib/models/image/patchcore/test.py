import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models.image.patchcore import Patchcore
from anomalib.utils.visualization import ImageVisualizer, MetricsVisualizer

if __name__ == '__main__':
    ckpt_path = "./lightning_logs/version_0/checkpoints/epoch=0-step=7.ckpt"

    model = Patchcore(input_size=(256, 256), backbone="wide_resnet50_2", coreset_sampling_ratio=0.1, num_neighbors=None)
    datamodule = MVTec(root="/home/nico/Dataset/mvtec", category="bottle", image_size=(256, 256), center_crop=None)

    visualizers = [ImageVisualizer(), MetricsVisualizer()]

    engine = Engine(
        threshold="F1AdaptiveThreshold",
        image_metrics=["AUROC", "BinaryF1Score", "OverKill", "MissKill"],
        pixel_metrics=["AUROC", "BinaryF1Score", "BinaryJaccardIndex"],  # JaccardIndex is IoU
        visualizers=visualizers,
        save_image=True,
    )
    engine.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
