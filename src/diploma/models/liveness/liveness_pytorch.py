from typing import Any

import timm
import torch
import numpy as np
from torch import nn

from diploma.utils import HelpMeta

from .preprocess import preprocess

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class LivenessModel(torch.nn.Module):
    "This class defines a efficientnet_b3 neural network for image classification tasks."

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone_out_channels: int = 1000,
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels for the input image.
            out_channels (int): Number of output channels/classes for classification.
            backbone_out_channels (int, optional): Number of output channels from the
                backbone network's classifier layer. Defaults to 1000.
        """
        super(LivenessModel, self).__init__()
        self.input = nn.Conv2d(in_channels, 3, 1)
        self.backbone = timm.create_model("efficientnet_b3", False)
        self.backbone.classifier = nn.Linear(
            self.backbone.classifier.in_features, backbone_out_channels, bias=True
        )
        self.output = nn.Linear(backbone_out_channels, out_channels)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.
        """
        x = self.input(x)
        x = self.backbone(x)
        x = self.output(x)
        return x.view(-1)


class LivenessPytorch(torch.nn.Module, metaclass=HelpMeta):
    "A wrapper class for a Pytorch model used for prediction."

    def __init__(
        self,
        model_file: str,
        device: str = "cpu",
        fp16: bool = False,
        imsize: int = 300,
        normalize: bool = True,
    ) -> None:
        """
        Args:
            model_file (str): Path to the model checkpoint file.
            device (str, optional): Device to load the model onto("cpu", "cuda").
                Defaults to "cpu".
            fp16 (bool, optional): Whether to use half-precision floating-point format.
                Defaults to False.
            imsize (int, optional): Size to which input images are resized.
                Defaults to 300.
            normalize (bool, optional): Whether to normalize input images.
                Defaults to True.
        """
        super().__init__()
        self.model = LivenessModel(3, 1)

        ckp = torch.load(model_file, map_location="cpu")["state_dict"]

        self.load_state_dict(ckp)
        self.to(device)
        self.eval()
        if fp16:
            self.model.half()
        self.imsize = imsize
        self.device = device
        self.fp16 = fp16
        self.normalize = normalize

    @torch.inference_mode()
    def predict_single(self, img: np.ndarray, rgb: bool = False) -> Any:
        """
        Predict the output for a single input image.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            rgb (bool, optional): Whether the image is in RGB format. Defaults to False.

        Returns:
            Any: Predicted output for the input image.
        """
        work_img = preprocess(img, rgb, self.imsize, self.normalize)
        img = torch.tensor(work_img, dtype=torch.float, device=self.device)
        if self.fp16:
            img = img.half()
        with torch.cuda.amp.autocast(enabled=self.fp16):
            return torch.sigmoid(self.forward(img)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model's output tensor.
        """
        return self.model(x)

    def __call__(self, img: np.ndarray, rgb: bool = False) -> Any:
        """
        Call the model to predict the output for an input image.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            rgb (bool, optional): Whether the is in RGB format. Defaults to False.

        Returns:
            Any: Predicted output for the input image.
        """
        return self.predict_single(img, rgb)
