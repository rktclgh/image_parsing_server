from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.routes.parse import router as parse_router
from app.routes.status import router as status_router


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    app = FastAPI(
        title=resolved_settings.service_name,
        version=resolved_settings.service_version,
    )
    app.dependency_overrides[get_settings] = lambda: resolved_settings
    app.add_exception_handler(AppError, _app_error_handler)
    app.include_router(parse_router)
    app.include_router(status_router)
    return app


def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    request_id = request.headers.get("x-request-id")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_response(request_id=request_id).model_dump(mode="json"),
    )


app = create_app()
