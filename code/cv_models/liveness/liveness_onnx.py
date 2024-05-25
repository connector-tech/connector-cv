from typing import Optional

import cv2
import numpy as np
import onnxruntime
import albumentations as A
from loguru import logger

from code.utils import get_providers, HelpMeta, sigmoid


class LivenessONNX(metaclass=HelpMeta):
    "A class representing an ONNX model for LivenessONNX inference."
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logger.info('Creating LivenessONNX instance')
            cls._instance = super(LivenessONNX, cls).__new__(cls)
        return cls._instance

    def __init__(
            self,
            model_file: Optional[bytes] = None,
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
        if not hasattr(self, 'initialized'):
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
            self.initialized = True

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
