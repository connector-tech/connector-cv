from pydantic import PostgresDsn
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    debug: bool = True
    database_url: PostgresDsn
    s3_access_key_id: str
    s3_secret_access_key: str
    s3_bucket_name: str
    aws_region: str


settings = Settings()

TORTOISE_CONFIG = {
    'connections': {'default': str(settings.database_url)},
    'apps': {
        'models': {
            'models': ['code.models', 'aerich.models'],
            'default_connection': 'default',
        },
    },
}
