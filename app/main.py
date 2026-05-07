from fastapi import FastAPI

from app.core.config import Settings, get_settings
from app.routes.status import router as status_router


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title=resolved_settings.service_name,
        version=resolved_settings.service_version,
    )
    app.include_router(status_router)
    return app


app = create_app()
