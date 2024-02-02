import torch

from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models.image.patchcore import Patchcore

if __name__ == '__main__':
    # model = Patchcore()
    # datamodule = MVTec(root="D:/Data/mvtec", category="bottle")
    # engine = Engine()
    # engine.fit(datamodule=datamodule, model=model)

    model = Patchcore.load_from_checkpoint(checkpoint_path="./lightning_logs/version_0/checkpoints/epoch=0-step=7.ckpt")
    example_input_array = torch.randn(2, 3, 224, 224)
    model.to_onnx("patchcore.onnx", input_sample=example_input_array)
