from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter

from app.presentation.api.v1.endpoints.images import router as images_router

router = APIRouter(prefix="/api", route_class=DishkaRoute)
router.include_router(images_router)
