import type { CardOption } from "../generated/v4/contracts";
import "../styles/lodging-preference.css";

const AREA_ICONS: Record<string, string> = {
  transit: "area-transit",
  attraction: "area-attractions",
  commercial: "area-shopping"
};

const QUALITY_ICONS = {
  economy: "tier-2",
  comfort: "tier-3",
  upscale: "tier-4",
  luxury: "tier-5"
};

function iconPath(name: string) {
  return `${import.meta.env.BASE_URL}images/lodging/${name}.png`;
}

export function LodgingAreaIllustration({ option }: { option: CardOption }) {
  const semantic = option.semantic_value;
  const name =
    semantic.kind === "direction"
      ? AREA_ICONS[semantic.direction_id]
      : undefined;
  return name ? (
    <img
      className="lodging-option-illustration"
      src={iconPath(name)}
      alt=""
      width={128}
      height={96}
      loading="lazy"
      decoding="async"
    />
  ) : null;
}

export function LodgingQualityOption({
  option,
  selected,
  disabled,
  onToggle
}: {
  option: CardOption;
  selected: boolean;
  disabled: boolean;
  onToggle: () => void;
}) {
  const semantic = option.semantic_value;
  const icon =
    semantic.kind !== "direction"
      ? undefined
      : semantic.hotel_quality_tier
        ? QUALITY_ICONS[semantic.hotel_quality_tier]
        : semantic.property_type === "酒店"
          ? "type-hotel"
          : semantic.property_type === "民宿"
            ? "type-homestay"
            : undefined;
  return (
    <button
      type="button"
      className="lodging-quality-choice"
      data-option-id={option.option_id}
      aria-label={`选择${option.label}`}
      aria-pressed={selected}
      disabled={disabled || option.selection_state === "unavailable"}
      onClick={onToggle}
    >
      {icon ? (
        <img
          className="lodging-option-illustration"
          src={iconPath(icon)}
          alt=""
          width={128}
          height={96}
          loading="lazy"
          decoding="async"
        />
      ) : null}
      <span className="lodging-quality-label">{option.label}</span>
      <span className="lodging-choice-indicator" aria-hidden="true">
        {selected ? "✓" : null}
      </span>
    </button>
  );
}
