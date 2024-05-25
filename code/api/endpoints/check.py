import asyncio

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from loguru import logger

from code.clients.boto3 import S3
from code.consts import LIVENESS_THRESHOLD
from code.dto import CheckBaseResponse, ErrorResponse, CheckBaseRequest
from code.models import UserSessionPhoto
from code.services.recognition import RecognitionCheckService

router = APIRouter(prefix='/check', tags=['check'])


@router.post('/liveness/', responses={
    200: {'model': CheckBaseResponse},
    400: {'model': ErrorResponse},
})
async def liveness(data: CheckBaseRequest):
    try:
        user_photos = (
            await UserSessionPhoto.filter(
                user_id=data.user_id,
                session_id=data.session_id
            ).order_by('-bestframe_score').only('liveness_score').limit(5)
        )

        unlive_photos = [photo for photo in user_photos if photo.liveness_score < LIVENESS_THRESHOLD]
        logger.info(f'Unlive photos count={len(unlive_photos)}')
        return {'has_access': len(unlive_photos) < 2}
    except Exception as e:
        logger.error(f'Error occurred during liveness check exc={e}')
        return JSONResponse(
            content={'message': 'Internal server error'},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post('/recognition/', responses={
    200: {'model': CheckBaseResponse},
    400: {'model': ErrorResponse},
})
async def recognition(data: CheckBaseRequest):
    try:
        first_session_id = (
            await UserSessionPhoto.filter(
                user_id=data.user_id,
            ).order_by('created_at').only('session_id').first()
        ).session_id

        first_session_photo, current_session_photo = await asyncio.gather(
            UserSessionPhoto.filter(
                user_id=data.user_id,
                session_id=first_session_id
            ).order_by('-bestframe_score').only('photo_id').first(),
            UserSessionPhoto.filter(
                user_id=data.user_id,
                session_id=data.session_id
            ).order_by('-bestframe_score').only('photo_id').first(),
            return_exceptions=True
        )

        first_session_photo, current_session_photo = await asyncio.gather(
            S3.get_image(f'cv/{data.user_id}/{first_session_id}/{first_session_photo.photo_id}.png'),
            S3.get_image(f'cv/{data.user_id}/{data.session_id}/{current_session_photo.photo_id}.png'),
            return_exceptions=True
        )

        return JSONResponse(
            content={
                'has_access': RecognitionCheckService(
                    image_1=first_session_photo,
                    image_2=current_session_photo
                ).process()
            },
            status_code=status.HTTP_200_OK
        )
    except ValueError as e:
        logger.error(f'{str(e)}')
        return JSONResponse(
            content={'message': str(e) if str(e) else 'Bad Request'},
            status_code=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f'Error occurred during recognition check exc={e}')
        return JSONResponse(
            content={'message': 'Internal server error'},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
