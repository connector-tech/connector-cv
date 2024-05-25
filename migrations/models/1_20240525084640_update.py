from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "user_session_photo" ADD "liveness_score" DOUBLE PRECISION NOT NULL;
        ALTER TABLE "user_session_photo" ADD "bestframe_score" DOUBLE PRECISION NOT NULL;
        ALTER TABLE "user_session_photo" ADD "session_id" UUID NOT NULL;"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        ALTER TABLE "user_session_photo" DROP COLUMN "liveness_score";
        ALTER TABLE "user_session_photo" DROP COLUMN "bestframe_score";
        ALTER TABLE "user_session_photo" DROP COLUMN "session_id";"""
