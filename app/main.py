from fastapi import FastAPI

from app.routes.status import router as status_router


def create_app() -> FastAPI:
    app = FastAPI(title="image-parsing-server", version="v1")
    app.include_router(status_router)
    return app


app = create_app()
