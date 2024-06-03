import torch
import torchvision.models as models
import numpy
model = models.resnet50(pretrained=True)

model.eval()

dummy_input = torch.rand(1, 3, 224, 224)

input_names = ["best(3).tp"]
output_names = ["best.onnx"]

torch.onnx.export(model, dummy_input, "resnet50.onnx", verbose=False, input_names=input_names, output_names=output_names, export_params=True,)

