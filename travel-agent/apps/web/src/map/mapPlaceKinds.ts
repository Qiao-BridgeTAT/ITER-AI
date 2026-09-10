import { Bed, ForkKnife, MapPin, Mountains } from "@phosphor-icons/react";

export const MAP_PLACE_LEGEND = [
  { kind: "attraction", label: "景点", Icon: Mountains },
  { kind: "restaurant", label: "餐饮", Icon: ForkKnife },
  { kind: "hotel", label: "住宿", Icon: Bed },
] as const;

export function mapPlaceCategory(kind: string) {
  return (
    MAP_PLACE_LEGEND.find((entry) => entry.kind === kind) ?? {
      kind: "other",
      label: kind === "transport" ? "交通" : "地点",
      Icon: MapPin,
    }
  );
}
