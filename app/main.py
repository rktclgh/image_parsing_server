from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.model.gemma4_loader import Gemma4Loader, Gemma4LoaderConfig
from app.model.runtime import Loader, VLMRuntime
from app.routes.parse import router as parse_router
from app.routes.status import router as status_router
from app.services.runtime_mode import RuntimeModeController


def create_app(
    settings: Settings | None = None,
    *,
    model_loader: Loader | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    loader = model_loader or Gemma4Loader(
        Gemma4LoaderConfig(
            model_id=resolved_settings.model_id,
            quantization=resolved_settings.quantization,
        )
    )
    runtime = VLMRuntime(
        loader=loader,
        mode=resolved_settings.vlm_mode,
        load_on_startup=resolved_settings.model_load_on_startup,
        unload_after_request=resolved_settings.unload_after_request,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime.startup()
        try:
            yield
        finally:
            runtime.unload()

    app = FastAPI(
        title=resolved_settings.service_name,
        version=resolved_settings.service_version,
        lifespan=lifespan,
    )
    app.state.model_loader = loader
    app.state.vlm_runtime = runtime
    app.state.runtime_mode_controller = RuntimeModeController(
        resolved_settings,
        runtime=runtime,
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
