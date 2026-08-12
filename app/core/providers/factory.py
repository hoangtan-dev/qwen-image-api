from dishka import AsyncContainer, make_async_container

from app.core.config import Settings
from app.core.providers.configs import ConfigsProvider
from app.core.providers.connections import ConnectionsProvider
from app.core.providers.services import ServicesProvider


def make_container(settings: Settings) -> AsyncContainer:
    return make_async_container(
        ConfigsProvider(settings),
        ConnectionsProvider(),
        ServicesProvider(),
    )
