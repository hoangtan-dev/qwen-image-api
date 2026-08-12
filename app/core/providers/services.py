from dishka import Provider, Scope, provide

from app.core.config import Settings
from app.services.image_generation import ImageGenerationService


class ServicesProvider(Provider):
    @provide(scope=Scope.APP)
    def provide_image_generation_service(self, settings: Settings) -> ImageGenerationService:
        return ImageGenerationService(settings)
