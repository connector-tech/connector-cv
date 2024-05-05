from typing import Optional

import numpy as np
import onnxruntime

from diploma.utils import HelpMeta, get_providers, sigmoid

from .preprocess import preprocess


class LivenessONNX(metaclass=HelpMeta):
    "A class representing an ONNX model for LivenessONNX inference."

    def __init__(
        self,
        model_file: Optional[str] = None,
        imsize: int = 300,
        normalize: bool = True,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_file (Optional[str], optional): Path to the model weights.
            imsize (int, optional): Size to which input images are resized.
                Defaults to 300.
            normalize (bool, optional): Whether to normalize an input images.
                Defaults to True.
            device (str, optional): Device to run the inference('cpu', 'cuda', 'cuda:1').
                Defaults to "cpu".
        """

        self.imsize = imsize
        self.normalize = normalize
        providers = get_providers(device)
        self.session = onnxruntime.InferenceSession(model_file, providers=providers)

    def __call__(self, img: np.ndarray, rgb: bool = False) -> float:
        """
        Perform inference using the LivenessONNX model.

        Args:
            img (np.ndarray): Input image as  a Numpy array.
            rgb (bool, optional): Whether the image is in RGB format. Defaults to False.

        Returns:
            float: Inference result for the input image.
        """
        work_img = preprocess(img, rgb, self.imsize, self.normalize).astype(np.float32)
        outputs = self.session.run(None, {"input.1": work_img})[0][0]
        return sigmoid(outputs)
