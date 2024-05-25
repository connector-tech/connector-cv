from fastapi import FastAPI

from code.api.endpoints import check, session


def setup_routers(app: FastAPI):
    app.include_router(check.router)
    app.include_router(session.router)
