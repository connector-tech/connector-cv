import asyncio
from typing import Tuple
from uuid import uuid4

import numpy as np

from code.clients.boto3 import S3
from code.cv_models import LivenessONNX, FaceDetectorONNX, bestframe_score
from code.models import UserSessionPhoto
from code.utils import crop_by_coordinates, estimate_pitch, estimate_yaw, estimate_blur
from code.utils.general import read_image


class LivenessService:
    def __init__(self, file: bytes, user_id: str, session_id: str | None = None):
        self.face_detector = FaceDetectorONNX()
        self.liveness_model = LivenessONNX()
        self.file = file
        self.user_id = user_id
        self.session_id = uuid4() if not session_id else session_id
        self.crop = None
        self.landmarks = None

    def _crop_faces(self, image) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
        boxes, landmarks = self.face_detector(image)
        if boxes.shape[0] <= 0:
            raise ValueError('No faces detected')
        crop = crop_by_coordinates(image, boxes[0, :4])
        return crop, landmarks

    def _get_scores(self) -> Tuple[float, np.ndarray | float]:
        image = read_image(self.file)
        self.crop, self.landmarks = self._crop_faces(image)

        bestframe_response = bestframe_score(
            estimate_pitch(self.landmarks[0]),
            estimate_yaw(self.landmarks[0]),
            estimate_blur(self.crop)
        )

        liveness_response = self.liveness_model(self.crop)

        return bestframe_response, liveness_response

    async def process(self) -> str:
        bestframe_response, liveness_response = self._get_scores()

        photo_id = uuid4()
        kwargs = {
            'user_id': self.user_id,
            'photo_id': photo_id,
            'session_id': self.session_id,
            'bestframe_score': bestframe_response,
            'liveness_score': liveness_response,
        }

        await asyncio.gather(
            UserSessionPhoto.create(**kwargs),
            S3.upload_image(self.file, f'cv/{self.user_id}/{self.session_id}/{photo_id}.png')
        )
        return str(self.session_id)
