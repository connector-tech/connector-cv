import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tortoise import Tortoise

from code.api import setup_routers
from code.clients.boto3 import S3
from code.config import settings, TORTOISE_CONFIG
from code.cv_models import FaceDetectorONNX, FaceRecONNX, LivenessONNX


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await asyncio.gather(
            Tortoise.init(config=TORTOISE_CONFIG),
            S3.connect(
                access_key_id=settings.s3_access_key_id,
                secret_access_key=settings.s3_secret_access_key,
                bucket_name=settings.s3_bucket_name,
                region=settings.aws_region,
            ),
        )

        detector_model = '/app/cv_models/model_detector.onnx'
        liveness_model = '/app/cv_models/model_liveness.onnx'
        recognition_model = '/app/cv_models/model_recognition.onnx'

        FaceDetectorONNX(detector_model)
        FaceRecONNX(recognition_model)
        LivenessONNX(liveness_model)
        yield
    finally:
        await asyncio.gather(
            Tortoise.close_connections(),
            S3.disconnect(),
        )


app = FastAPI(
    lifespan=lifespan,
    docs_url='/cv/docs/',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

setup_routers(app)

if __name__ == '__main__':
    uvicorn.run('code.app:app', host='0.0.0.0', reload=settings.debug)
