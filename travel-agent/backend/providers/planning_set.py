from dataclasses import dataclass

from backend.providers.interfaces import (
    HoursProvider,
    PlaceProvider,
    RouteProvider,
    TravelProductProvider,
    WeatherProvider,
)


@dataclass(frozen=True)
class PlanningProviderSet:
    places: PlaceProvider
    routes: RouteProvider
    hours: HoursProvider
    products: TravelProductProvider
    weather: WeatherProvider | None
