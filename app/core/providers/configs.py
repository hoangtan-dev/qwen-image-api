from dishka import Provider, Scope, provide

from app.core.config import Settings


class ConfigsProvider(Provider):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings

    @provide(scope=Scope.APP)
    def provide_settings(self) -> Settings:
        return self._settings
