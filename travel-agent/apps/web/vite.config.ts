import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";
import { defineConfig, loadEnv, type ProxyOptions } from "vite";

const projectRoot = fileURLToPath(new URL("../..", import.meta.url));
const disableLocalMap =
  Boolean(process.env.VITEST) ||
  process.env.TRAVEL_AGENT_DISABLE_LOCAL_MAP_CONFIG === "1";
const localMapEnv = disableLocalMap ? {} : loadEnv("amap", projectRoot, "");
const apiPort = process.env.TRAVEL_AGENT_API_PORT ?? "8000";
const mapKey = disableLocalMap
  ? undefined
  : (process.env.VITE_AMAP_JS_API_KEY ?? localMapEnv.VITE_AMAP_JS_API_KEY);
const mapSecurity =
  process.env.AMAP_JS_SECURITY_CODE ?? localMapEnv.AMAP_JS_SECURITY_CODE;
const mapProxy = (target: string): ProxyOptions => ({
  target,
  changeOrigin: true,
  rewrite: (path) => {
    const rewritten = path.replace(/^\/_AMapService/u, "");
    return `${rewritten}${rewritten.includes("?") ? "&" : "?"}jscode=${encodeURIComponent(mapSecurity ?? "")}`;
  },
});

export default defineConfig({
  cacheDir: process.env.TRAVEL_AGENT_VITE_CACHE_DIR,
  plugins: [react()],
  define: {
    ...(mapKey
      ? { "import.meta.env.VITE_AMAP_JS_API_KEY": JSON.stringify(mapKey) }
      : {}),
    "import.meta.env.VITE_AMAP_SERVICE_HOST": JSON.stringify(
      mapKey && mapSecurity ? "/_AMapService" : "",
    ),
  },
  server: {
    proxy: {
      "/_AMapService/v4/map/styles": mapProxy("https://webapi.amap.com"),
      "/_AMapService": mapProxy("https://restapi.amap.com"),
      "/api": {
        target: `http://127.0.0.1:${apiPort}`,
        changeOrigin: true,
        ws: true,
      },
    },
  },
});
