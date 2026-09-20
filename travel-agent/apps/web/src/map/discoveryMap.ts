import curated from "../../../../config/cities.v1.json";
import type { CardOption } from "../generated/v4/contracts";

export type MapPoint = [number, number];
export interface DiscoveryMapPlace {
  id: string;
  label: string;
  point?: MapPoint;
  amapId?: string;
}
export interface DiscoveryMapPreview {
  city: { code: string; name: string } | null;
  places: DiscoveryMapPlace[];
  focusedId: string | null;
  focusRequest?: number;
}
export function discoveryCity(id?: string | null, name?: string | null) {
  if (!id && !name) return null;
  const known = curated.cities.find((city) => city.city_id === id);
  return {
    code: known?.provider_codes.amap ?? id?.replace(/^cn-/, "") ?? name!,
    name: name || known?.display_name || "目的地"
  };
}
export function discoveryPlace(option: CardOption): DiscoveryMapPlace {
  const coordinates = option.coordinates;
  const source = option.entity_ref?.provider_entity_refs.find((ref) =>
    ref.startsWith("provider:amap:")
  );
  return {
    id: option.option_id,
    label: option.label,
    point: coordinates
      ? [coordinates.longitude, coordinates.latitude]
      : undefined,
    amapId: source?.slice("provider:amap:".length)
  };
}

type Location = {
  getLng?: () => number;
  getLat?: () => number;
  lng?: number;
  lat?: number;
};
export function mapPoint(
  value: Location | string | undefined
): MapPoint | undefined {
  const parts =
    typeof value === "string"
      ? value.split(",").map(Number)
      : [value?.getLng?.() ?? value?.lng, value?.getLat?.() ?? value?.lat];
  const [lng, lat] = parts;
  return typeof lng === "number" &&
    typeof lat === "number" &&
    Number.isFinite(lng) &&
    Number.isFinite(lat) &&
    Math.abs(lng) <= 180 &&
    Math.abs(lat) <= 90
    ? [lng, lat]
    : undefined;
}
export interface DistrictResult {
  boundaries?: unknown[];
  center?: Location;
}
export interface DiscoveryApi {
  DistrictSearch?: new (options: object) => {
    search: (
      keyword: string,
      callback: (
        status: string,
        result: { districtList?: DistrictResult[] }
      ) => void
    ) => void;
  };
  PlaceSearch?: new (options: object) => {
    getDetails: (
      id: string,
      callback: (
        status: string,
        result: { poiList?: { pois?: { location?: Location | string }[] } }
      ) => void
    ) => void;
  };
}
const districts = new Map<string, Promise<DistrictResult>>();
const locations = new Map<string, Promise<MapPoint>>();
function cached<T>(
  cache: Map<string, Promise<T>>,
  key: string,
  request: (resolve: (value: T) => void, reject: () => void) => void
): Promise<T> {
  const existing = cache.get(key);
  if (existing) return existing;
  const pending = new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("地图信息查询超时")), 8000);
    request(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      () => {
        clearTimeout(timer);
        reject(new Error("地图信息暂不可用"));
      }
    );
  });
  cache.set(key, pending);
  void pending.catch(() => {
    if (cache.get(key) === pending) cache.delete(key);
  });
  return pending;
}
export function loadDistrict(api: DiscoveryApi, code: string) {
  return cached(districts, code, (resolve, reject) => {
    if (!api.DistrictSearch) return reject();
    new api.DistrictSearch({ subdistrict: 0, extensions: "all" }).search(
      code,
      (status, result) => {
        const district = result.districtList?.[0];
        if (status === "complete" && district) resolve(district);
        else reject();
      }
    );
  });
}
export function loadPlacePoint(api: DiscoveryApi, id: string) {
  return cached(locations, id, (resolve, reject) => {
    if (!api.PlaceSearch) return reject();
    new api.PlaceSearch({ extensions: "base" }).getDetails(
      id,
      (status, result) => {
        const point = mapPoint(result.poiList?.pois?.[0]?.location);
        if (status === "complete" && point) resolve(point);
        else reject();
      }
    );
  });
}
