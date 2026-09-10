import cityRegistryData from "../../../../config/cities.v1.json";
import { validatePublicContract } from "../contracts/validation";
import type { CityRegistryCatalog } from "../generated/contracts";
import type { CityCode } from "../generated/enums";

if (
  !validatePublicContract("city_registry_catalog", cityRegistryData).success
) {
  throw new Error("Invalid city registry");
}
const CITY_REGISTRY = cityRegistryData as CityRegistryCatalog;

export function getSupportedCity(city: CityCode): { name: string } {
  const registration = CITY_REGISTRY.cities.find(
    (entry) => entry.legacy_city_code === city,
  );
  if (!registration) throw new Error("Missing city catalog entry");
  return { name: registration.display_name };
}

export function getRegisteredCityName(cityId: string): string | null {
  return (
    CITY_REGISTRY.cities.find((entry) => entry.city_id === cityId)
      ?.display_name ?? null
  );
}
