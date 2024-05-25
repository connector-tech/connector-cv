from fastapi import APIRouter, Form, UploadFile, File, status
from fastapi.responses import JSONResponse
from loguru import logger

from code.dto import SessionBaseResponse, ErrorResponse
from code.services.liveness import LivenessService

router = APIRouter(prefix='/session', tags=['session'])


@router.post('/', responses={
    200: {'model': SessionBaseResponse},
    400: {'model': ErrorResponse}
})
async def session(user_id: str = Form(...), session_id: str = Form(None), file: UploadFile = File(...)):
    try:
        session_id = await LivenessService(
            file=file.file.read(),
            user_id=user_id,
            session_id=session_id
        ).process()
        return JSONResponse(content={'session_id': session_id}, status_code=status.HTTP_200_OK)
    except ValueError as e:
        logger.error(f'{str(e)}')
        return JSONResponse(
            content={'message': str(e) if str(e) else 'Internal server error'},
            status_code=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f'Error occurred during liveness check exc={e}')
        return JSONResponse(
            content={'message': 'Internal server error'},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
