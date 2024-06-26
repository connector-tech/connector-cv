import cv2
import onnxruntime
import numpy as np

from diploma.utils import get_providers

onnxruntime.set_default_logger_severity(3)


class FaceRecONNX:
    "A class representing an ONNX model for FaceRec inference."

    def __init__(self, model_file: str, device: str = "cpu"):
        """
        This is the model for face recognition.

        Args:
            model_file (str): Path to onnx model.
            device (str, optional): [cpu, cuda, cuda:0 ...]. Defaults to "cpu".
        """
        super().__init__()

        providers = get_providers(device)
        self.session = onnxruntime.InferenceSession(model_file, providers=providers)

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
        embedding = self.session.run(
            None, {self.session.get_inputs()[0].name: work_image}
        )[0][0]
        return embedding[None, ...]
