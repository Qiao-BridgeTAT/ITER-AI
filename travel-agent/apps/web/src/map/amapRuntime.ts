import AMapLoader from "@amap/amap-jsapi-loader";
import type { AmapConfig, AmapApi } from "./ItineraryMapCanvas";
export function readAmapConfig(): AmapConfig | null {
  const key = import.meta.env.VITE_AMAP_JS_API_KEY?.trim();
  const securityCode = import.meta.env.VITE_AMAP_JS_SECURITY_CODE?.trim();
  const serviceHost = import.meta.env.VITE_AMAP_SERVICE_HOST?.trim();
  if (
    !key ||
    (!securityCode && !serviceHost) ||
    /replace|example/i.test(`${key}${securityCode ?? ""}`)
  )
    return null;
  return { key, securityCode, serviceHost };
}
export async function loadAmapApi(config: AmapConfig): Promise<AmapApi> {
  window._AMapSecurityConfig = config.serviceHost
    ? { serviceHost: `${window.location.origin}${config.serviceHost}` }
    : { securityJsCode: config.securityCode };
  return AMapLoader.load({
    key: config.key,
    version: "2.0",
    plugins: ["AMap.DistrictSearch", "AMap.PlaceSearch", "AMap.Scale"]
  }) as Promise<AmapApi>;
}
