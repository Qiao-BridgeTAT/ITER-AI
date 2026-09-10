"""Load and resolve the nationwide city registry without city-specific branches."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from backend.contracts.city_content import (
    CityContentPackage,
    RegisteredCityContentPackage,
)
from backend.contracts.city_registry import (
    CityAdministrativeCodes,
    CityContentRegistration,
    CityProviderCapability,
    CityProviderCodes,
    CityRegistration,
    CityRegistryCatalog,
    MainlandCityCatalog,
    normalize_city_reference,
)
from backend.contracts.enums import CoordinateSystem, ProviderCode
from backend.providers.contracts import ProviderCityScope

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CITY_REGISTRY_PATH = PROJECT_ROOT / "config" / "cities.v1.json"
DEFAULT_MAINLAND_CITY_CATALOG_PATH = PROJECT_ROOT / "config" / "china-cities.v1.json"
SUPPORTED_CITY_REGISTRY_MAJOR = 1
SUPPORTED_MAINLAND_CITY_CATALOG_MAJOR = 1
MAINLAND_PROVIDER_CAPABILITIES = (
    CityProviderCapability.AMAP_PLACES,
    CityProviderCapability.AMAP_ROUTES,
    CityProviderCapability.AMAP_HOURS,
    CityProviderCapability.FLYAI_PRODUCTS,
    CityProviderCapability.WEATHER_FORECAST,
)


class CityRegistryError(ValueError):
    """Base error for safe, explicit city-registry failures."""


class CityRegistryLoadError(CityRegistryError):
    pass


class CityRegistryVersionError(CityRegistryError):
    pass


class UnknownCityError(CityRegistryError):
    pass


class AmbiguousCityError(CityRegistryError):
    pass


class CityProviderUnavailableError(CityRegistryError):
    pass


class CityContentPackageLoadError(CityRegistryError):
    pass


class CityRegistry:
    def __init__(self, catalog: CityRegistryCatalog, *, project_root: Path = PROJECT_ROOT) -> None:
        self._catalog = catalog
        self._project_root = project_root.resolve()

    @property
    def version(self) -> str:
        return self._catalog.registry_version

    @property
    def cities(self) -> tuple[CityRegistration, ...]:
        return self._catalog.cities

    def resolve(self, reference: str) -> CityRegistration:
        normalized = normalize_city_reference(reference)
        matches = [city for city in self.cities if normalized in self._references_for(city)]
        return _one_city(reference, matches)

    def mentioned_in(self, text: str) -> tuple[CityRegistration, ...]:
        """Return registered cities explicitly named in free text.

        This is deterministic entity resolution, not intent inference. It lets the
        model use a public city name while the server remains authoritative for
        the open-ended city_id used by planning and Providers.
        """

        normalized_text = normalize_city_reference(text)
        if not normalized_text:
            return ()
        matches = [
            (
                city,
                max(
                    (
                        reference
                        for reference in self._references_for(city)
                        if reference and reference in normalized_text
                    ),
                    key=len,
                    default="",
                ),
            )
            for city in self.cities
        ]
        matches = [(city, reference) for city, reference in matches if reference]
        # Prefer the most specific explicitly written place. For example,
        # "阿勒泰地区" must not also bind the shorter county-level "阿勒泰市".
        return tuple(
            city
            for city, reference in matches
            if not any(
                reference != other_reference and reference in other_reference
                for _, other_reference in matches
            )
        )

    def resolve_provider_code(self, provider: ProviderCode, code: str) -> CityRegistration:
        field = _provider_field(provider)
        normalized = normalize_city_reference(code)
        matches = [
            city
            for city in self.cities
            if (provider_code := getattr(city.provider_codes, field)) is not None
            and normalize_city_reference(provider_code) == normalized
        ]
        return _one_city(f"{provider.value}:{code}", matches)

    def provider_scope(self, reference: str, provider: ProviderCode) -> ProviderCityScope:
        city = self.resolve(reference)
        field = _provider_field(provider)
        code = getattr(city.provider_codes, field)
        if code is None:
            raise CityProviderUnavailableError(
                f"city {city.city_id} has no {provider.value} provider registration"
            )
        return ProviderCityScope(
            city_id=city.city_id,
            provider_city_code=code,
            provider_transit_city_code=(
                city.provider_codes.amap_transit if provider is ProviderCode.AMAP else None
            ),
            display_name=city.display_name,
        )

    def supports(self, reference: str, capability: CityProviderCapability) -> bool:
        return capability in self.resolve(reference).enabled_provider_capabilities

    def content_registration(self, reference: str) -> CityContentRegistration | None:
        return self.resolve(reference).content_package

    def load_content_package(self, reference: str) -> RegisteredCityContentPackage | None:
        city = self.resolve(reference)
        registration = city.content_package
        if registration is None:
            return None
        path = (self._project_root / registration.relative_path).resolve()
        if not path.is_relative_to(self._project_root):
            raise CityContentPackageLoadError("city content path leaves the project root")
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
            package = self._parse_content_package(city, payload)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise CityContentPackageLoadError(
                f"failed to load content package for {city.city_id}"
            ) from exc
        if package.content_version != registration.content_version:
            raise CityContentPackageLoadError(f"content version mismatch for {city.city_id}")
        return package

    @staticmethod
    def _parse_content_package(
        city: CityRegistration,
        payload: Any,
    ) -> RegisteredCityContentPackage:
        if isinstance(payload, dict) and "city_id" in payload:
            package = RegisteredCityContentPackage.model_validate(payload)
            if package.city_id != city.city_id:
                raise CityContentPackageLoadError(
                    f"content package city mismatch for {city.city_id}"
                )
            return package
        legacy = CityContentPackage.model_validate(payload)
        if city.legacy_city_code is None or legacy.city is not city.legacy_city_code:
            raise CityContentPackageLoadError(
                f"legacy content package city mismatch for {city.city_id}"
            )
        return RegisteredCityContentPackage.from_legacy(city.city_id, legacy)

    @staticmethod
    def _references_for(city: CityRegistration) -> set[str]:
        values = {
            city.city_id,
            city.display_name,
            city.administrative_codes.prefecture,
            *city.aliases,
        }
        if city.legacy_city_code is not None:
            values.add(city.legacy_city_code.value)
        return {normalize_city_reference(value) for value in values}


def load_city_registry(
    path: Path = DEFAULT_CITY_REGISTRY_PATH,
    *,
    mainland_catalog_path: Path | None = None,
) -> CityRegistry:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        catalog = CityRegistryCatalog.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise CityRegistryLoadError(f"failed to load city registry: {path.name}") from exc
    major = int(catalog.registry_version.split(".", maxsplit=1)[0])
    if major != SUPPORTED_CITY_REGISTRY_MAJOR:
        raise CityRegistryVersionError(
            f"unsupported city registry major version: {catalog.registry_version}"
        )
    if mainland_catalog_path is None and path.resolve() == DEFAULT_CITY_REGISTRY_PATH.resolve():
        mainland_catalog_path = DEFAULT_MAINLAND_CITY_CATALOG_PATH
    if mainland_catalog_path is not None:
        mainland = _load_mainland_catalog(mainland_catalog_path)
        catalog = _merge_mainland_catalog(catalog, mainland)
    return CityRegistry(catalog)


@lru_cache(maxsize=1)
def default_city_registry() -> CityRegistry:
    return load_city_registry()


def _load_mainland_catalog(path: Path) -> MainlandCityCatalog:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        catalog = MainlandCityCatalog.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise CityRegistryLoadError(f"failed to load mainland city catalog: {path.name}") from exc
    major = int(catalog.catalog_version.split(".", maxsplit=1)[0])
    if major != SUPPORTED_MAINLAND_CITY_CATALOG_MAJOR:
        raise CityRegistryVersionError(
            f"unsupported mainland city catalog major version: {catalog.catalog_version}"
        )
    return catalog


def _merge_mainland_catalog(
    curated: CityRegistryCatalog, mainland: MainlandCityCatalog
) -> CityRegistryCatalog:
    curated_by_adcode = {
        city.provider_codes.amap: city
        for city in curated.cities
        if city.provider_codes.amap is not None
    }
    cities: list[CityRegistration] = []
    for entry in mainland.cities:
        override = curated_by_adcode.pop(entry.adcode, None)
        if override is not None:
            provider_codes = {
                **override.provider_codes.model_dump(mode="python"),
                "amap": entry.adcode,
                "amap_transit": entry.citycode,
                "flyai": override.provider_codes.flyai or override.display_name,
                "weather": override.provider_codes.weather or override.display_name,
            }
            override_payload = override.model_dump(mode="python")
            override_payload.update(
                provider_codes=provider_codes,
                enabled_provider_capabilities=tuple(
                    dict.fromkeys(
                        (
                            *MAINLAND_PROVIDER_CAPABILITIES,
                            *override.enabled_provider_capabilities,
                        )
                    )
                ),
            )
            cities.append(CityRegistration.model_validate(override_payload))
            continue
        display_name = entry.official_name.removesuffix("市")
        cities.append(
            CityRegistration(
                city_id=f"cn-{entry.adcode}",
                display_name=display_name,
                aliases=(entry.official_name,),
                country_name="中国",
                province_name=entry.province_name,
                administrative_codes=CityAdministrativeCodes(
                    country="CN",
                    province=entry.province_adcode,
                    prefecture=entry.adcode,
                ),
                timezone="Asia/Shanghai",
                coordinate_system=CoordinateSystem.GCJ_02,
                provider_codes=CityProviderCodes(
                    amap=entry.adcode,
                    amap_transit=entry.citycode,
                    flyai=display_name,
                    weather=display_name,
                ),
                enabled_provider_capabilities=MAINLAND_PROVIDER_CAPABILITIES,
            )
        )
    if curated_by_adcode:
        missing = ", ".join(sorted(city.city_id for city in curated_by_adcode.values()))
        raise CityRegistryLoadError(
            "curated cities are missing from the mainland source catalog: " + missing
        )
    return CityRegistryCatalog(
        registry_version=curated.registry_version,
        cities=tuple(cities),
    )


def _one_city(reference: str, matches: list[CityRegistration]) -> CityRegistration:
    if not matches:
        raise UnknownCityError(f"unregistered city reference: {reference}")
    if len(matches) > 1:
        city_ids = ", ".join(sorted(city.city_id for city in matches))
        raise AmbiguousCityError(f"ambiguous city reference {reference}: {city_ids}")
    return matches[0]


def _provider_field(provider: ProviderCode) -> str:
    fields = {
        ProviderCode.AMAP: "amap",
        ProviderCode.BAIDU: "baidu",
        ProviderCode.FLYAI: "flyai",
        ProviderCode.WEATHER: "weather",
    }
    try:
        return fields[provider]
    except KeyError as exc:
        raise CityProviderUnavailableError(
            f"provider {provider.value} does not use a city registry code"
        ) from exc
