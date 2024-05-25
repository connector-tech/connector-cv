import cv2
import onnxruntime
import numpy as np
import onnxruntime
from loguru import logger

from code.utils import get_providers


onnxruntime.set_default_logger_severity(3)


class FaceRecONNX:
    "A class representing an ONNX model for FaceRec inference."
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logger.info('Creating FaceRecONNX instance')
            cls._instance = super(FaceRecONNX, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_file: bytes | None = None, device: str = "cpu"):
        """
        This is the model for face recognition.

        Args:
            model_file (bytes): Path to onnx model.
            device (str, optional): [cpu, cuda, cuda:0 ...]. Defaults to "cpu".
        """
        if not hasattr(self, 'initialized'):
            logger.info('Initializing FaceRecONNX')
            super().__init__()

            providers = get_providers(device)
            self.onnx_session = onnxruntime.InferenceSession(
                model_file, providers=providers,
            )
            self.initialized = True

    def __preprocess(self, image):
        work_image = cv2.resize(image, (112, 112))[:, :, ::-1]
        work_image = np.transpose(work_image, (2, 0, 1))
        work_image = np.expand_dims(work_image, 0)
        work_image = np.float32(((work_image / 255.0) - 0.5) / 0.5)
        return work_image

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Calculates embeddings of images containing faces

        Args:
            image (np.ndarray): BGR face crop.

        Returns:
            np.ndarray: embedding with shape (1, 512)
        """
        work_image = self.__preprocess(image)
        embedding = self.onnx_session.run(
            None, {self.onnx_session.get_inputs()[0].name: work_image},
        )[0][0]
        return embedding[None, ...]
