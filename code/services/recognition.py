import numpy as np
from loguru import logger

from code.consts import RECOGNITION_THRESHOLD
from code.cv_models import FaceDetectorONNX, FaceRecONNX
from code.utils import align_face, cosine_simularity
from code.utils.general import read_image


class RecognitionCheckService:
    def __init__(self, image_1: bytes, image_2: bytes):
        self.face_detector = FaceDetectorONNX()
        self.recognition_model = FaceRecONNX()
        self.image_1 = image_1
        self.image_2 = image_2

    def _align_face(self, image: np.ndarray) -> np.ndarray:
        boxes, landmarks = self.face_detector(image)
        if boxes.shape[0] <= 0:
            raise ValueError('No faces detected')
        aligned_face = align_face(image, boxes[0][:-1], landmarks[0])
        return aligned_face

    def _get_embedding(self, image: bytes) -> np.ndarray:
        image = read_image(image)

        aligned_face = self._align_face(image)

        return self.recognition_model(aligned_face)

    def process(self) -> bool:
        embedding_1 = self._get_embedding(self.image_1)
        embedding_2 = self._get_embedding(self.image_2)

        similarity = cosine_simularity(embedding_1, embedding_2)
        return similarity > RECOGNITION_THRESHOLD
