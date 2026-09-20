import { ContinuousPreferenceSlider } from "../components/ContinuousPreferenceSlider";

type PreferenceSliderAttachmentProps = {
  label: string;
  startLabel: string;
  endLabel: string;
  value: number;
  valueText: string;
  flexibleSelected?: boolean;
  disabled?: boolean;
  onChange: (value: number) => void;
  onFlexible: () => void;
  onConfirm: () => void;
};

export function PreferenceSliderAttachment({
  label,
  startLabel,
  endLabel,
  value,
  valueText,
  flexibleSelected = false,
  disabled = false,
  onChange,
  onFlexible,
  onConfirm
}: PreferenceSliderAttachmentProps) {
  return (
    <div className="preference-slider-attachment">
      <ContinuousPreferenceSlider
        label={label}
        startLabel={startLabel}
        endLabel={endLabel}
        value={value}
        valueText={valueText}
        disabled={disabled}
        onChange={onChange}
      />
      <div className="preference-slider-actions">
        <button
          className="preference-slider-flexible"
          type="button"
          disabled={disabled}
          aria-pressed={flexibleSelected}
          onClick={onFlexible}
        >
          都可以
        </button>
        <button
          className="preference-slider-confirm"
          type="button"
          disabled={disabled || flexibleSelected}
          onClick={onConfirm}
        >
          就这样
        </button>
      </div>
    </div>
  );
}
