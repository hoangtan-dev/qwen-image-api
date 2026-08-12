import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dishka import AsyncContainer
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI
from loguru import logger

from app.core.config import settings
from app.core.logging import configure_logging
from app.core.providers.factory import make_container
from app.presentation.api.router import router
from app.services.image_generation import ImageGenerationService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings.log_level)
    logger.info("Starting application in {} environment", settings.app_env)
    if settings.load_model_on_startup:
        image_service = await app.state.dishka_container.get(ImageGenerationService)
        await asyncio.to_thread(image_service.load)
        await asyncio.to_thread(image_service.warmup)
    yield
    logger.info("Stopping application")
    await app.state.dishka_container.close()


def create_app(container: AsyncContainer | None = None) -> FastAPI:
    container = container or make_container(settings)
    application = FastAPI(title="Qwen Image Server", debug=settings.debug, lifespan=lifespan)
    application.include_router(router)
    setup_dishka(container, application)
    return application


app = create_app()
