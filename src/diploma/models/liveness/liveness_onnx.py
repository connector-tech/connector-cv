from typing import Optional

import cv2
import numpy as np
import onnxruntime
import albumentations as A

from diploma.utils import HelpMeta, get_providers, sigmoid


class LivenessONNX(metaclass=HelpMeta):
    "A class representing an ONNX model for LivenessONNX inference."

    def __init__(
        self,
        model_file: Optional[str] = None,
        imsize: int = 300,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_file (Optional[str], optional): Path to the model weights.
            imsize (int, optional): Size to which input images are resized.
                Defaults to 300.
            device (str, optional): Device to run the inference('cpu', 'cuda', 'cuda:1').
                Defaults to "cpu".
        """
        super().__init__()

        self.imsize = imsize
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        providers = get_providers(device)
        self.session = onnxruntime.InferenceSession(model_file, providers=providers)

        self.transform = A.Compose(
            [
                A.Resize(self.imsize, self.imsize, p=1),
                A.Normalize(mean=self.mean, std=self.std, p=1),
            ]
        )

    def __call__(self, image: np.ndarray) -> float:
        """
        Perform inference using the LivenessONNX model.

        Args:
            image (np.ndarray): Input image as  a Numpy array.

        Returns:
            float: Inference result for the input image.
        """
        image = self.transform(image=image)["image"]

        image = image.transpose((2, 0, 1))

        output = self.session.run([], input_feed={"input.1": image[None, :, :, :]})[0]

        return sigmoid(output)[0][0]
