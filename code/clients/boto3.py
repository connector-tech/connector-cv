import io

from aioboto3 import Session
from loguru import logger


class S3:
    _instance = None
    _bucket_name = None

    @classmethod
    async def connect(cls, access_key_id: str, secret_access_key: str, bucket_name: str, region: str):
        if cls._instance is None:
            session = Session(
                region_name=region,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
            )

            cls._instance = await session.client('s3').__aenter__()

            cls._bucket_name = bucket_name

        return cls._instance

    @classmethod
    async def get_model(cls, model_path: str) -> bytes:
        try:
            model = await cls._instance.get_object(Bucket=cls._bucket_name, Key=model_path)
            file = await model['Body'].read()
        except Exception as e:
            logger.info(f'Error occurred during getting model exc={e}')
            raise e

        return file

    @classmethod
    async def upload_image(cls, file: bytes, path: str):
        try:
            await cls._instance.put_object(
                Bucket=cls._bucket_name,
                Key=path,
                Body=file
            )
        except Exception as e:
            logger.info(f'Error occurred during uploading photo exc={e}')
            raise e

    @classmethod
    async def get_image(cls, path: str) -> bytes:
        try:
            photo = await cls._instance.get_object(
                Bucket=cls._bucket_name,
                Key=path,
            )
            file = await photo['Body'].read()
        except Exception as e:
            logger.info(f'Error occurred during getting photo exc={e}')
            raise e

        return file

    @classmethod
    async def disconnect(cls):
        if cls._instance is not None:
            await cls._instance.__aexit__(None, None, None)

            cls._instance = None
            cls._bucket_name = None
