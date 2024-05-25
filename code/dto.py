from pydantic import BaseModel


class SessionBaseResponse(BaseModel):
    session_id: str


class ErrorResponse(BaseModel):
    message: str


class CheckBaseResponse(BaseModel):
    has_access: bool


class CheckBaseRequest(BaseModel):
    user_id: str
    session_id: str
