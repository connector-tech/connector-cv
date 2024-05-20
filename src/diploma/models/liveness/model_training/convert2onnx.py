import torch
import torch.nn as nn
import torchvision.models as models

model_file = (
    "E:/kbtu_courses/diploma_project/src/diploma/models/liveness/weights/liveness_v5.pt"
)

model = models.efficientnet_b3()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model.load_state_dict(torch.load(model_file))
model.eval()

dummy_input = torch.randn(1, 3, 300, 300)

try:
    torch.onnx.export(
        model,
        dummy_input,
        "model_liveness.onnx",
        export_params=True,
        opset_version=17,
    )
    print("Model converted successfully.")
except Exception as e:
    print("Error during conversion:", e)
