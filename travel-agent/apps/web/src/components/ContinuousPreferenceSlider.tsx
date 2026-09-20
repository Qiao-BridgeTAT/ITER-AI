import { CSSProperties, useId } from "react";

interface ContinuousPreferenceSliderProps {
  label: string;
  startLabel: string;
  endLabel: string;
  startDescription?: string;
  endDescription?: string;
  feedbackLabel?: string;
  value: number;
  valueText: string;
  disabled?: boolean;
  showHeading?: boolean;
  onChange: (value: number) => void;
}

export function ContinuousPreferenceSlider({
  label,
  startLabel,
  endLabel,
  startDescription,
  endDescription,
  feedbackLabel,
  value,
  valueText,
  disabled = false,
  showHeading = true,
  onChange
}: ContinuousPreferenceSliderProps) {
  const feedbackId = useId();
  const hasEndpointDescriptions = Boolean(startDescription || endDescription);
  const progressStyle = {
    "--slider-progress": `${value}%`,
    "--slider-feedback-shift": `${-value}%`,
    "--slider-thumb-offset": `${10 - value * 0.2}px`
  } as CSSProperties;

  return (
    <div className="continuous-preference-slider" style={progressStyle}>
      {showHeading ? (
        <div className="preference-slider-heading">
          <strong>{label}</strong>
        </div>
      ) : null}
      <div className="preference-slider-control">
        <output
          id={feedbackId}
          className="preference-slider-feedback"
          aria-live="polite"
        >
          {feedbackLabel ? <small>{feedbackLabel}</small> : null}
          <strong>{valueText}</strong>
        </output>
        <input
          type="range"
          min="0"
          max="100"
          step="1"
          value={value}
          disabled={disabled}
          aria-label={label}
          aria-describedby={feedbackId}
          aria-valuetext={valueText}
          onChange={(event) => onChange(Number(event.currentTarget.value))}
        />
        <div
          className={`preference-slider-endpoints${hasEndpointDescriptions ? " has-descriptions" : ""}`}
          aria-hidden="true"
        >
          <span className="preference-slider-endpoint">
            <strong>{startLabel}</strong>
            {startDescription ? <small>{startDescription}</small> : null}
          </span>
          <span className="preference-slider-endpoint">
            <strong>{endLabel}</strong>
            {endDescription ? <small>{endDescription}</small> : null}
          </span>
        </div>
      </div>
    </div>
  );
}
