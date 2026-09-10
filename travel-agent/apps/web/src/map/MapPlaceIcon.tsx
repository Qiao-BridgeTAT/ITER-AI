import { mapPlaceCategory } from "./mapPlaceKinds";

export function MapPlaceIcon({ kind }: { kind: string }) {
  const { Icon, kind: category } = mapPlaceCategory(kind);
  return (
    <span
      className="map-place-icon"
      data-place-kind={category}
      aria-hidden="true"
    >
      <Icon size={20} weight="duotone" />
    </span>
  );
}
