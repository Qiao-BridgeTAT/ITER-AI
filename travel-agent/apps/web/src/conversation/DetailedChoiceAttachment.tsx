import { CompactChoiceOption } from "./CompactChoiceAttachment";

export type DetailedChoiceOption = CompactChoiceOption & {
  meta: string;
};

type DetailedChoiceAttachmentProps = {
  label: string;
  name: string;
  options: DetailedChoiceOption[];
  selectedId?: string;
  disabled?: boolean;
  onSelect: (option: DetailedChoiceOption) => void;
};

export function DetailedChoiceAttachment({
  label,
  name,
  options,
  selectedId,
  disabled = false,
  onSelect
}: DetailedChoiceAttachmentProps) {
  const visibleOptions = options.slice(0, 5);

  return (
    <fieldset
      className={`detailed-choice-attachment detailed-choice-count-${visibleOptions.length}`}
    >
      <legend className="visually-hidden">{label}</legend>
      {visibleOptions.map((option) => (
        <label className="detailed-choice-card" key={option.id}>
          <input
            type="radio"
            name={name}
            value={option.id}
            checked={selectedId === option.id}
            disabled={disabled}
            onChange={() => onSelect(option)}
          />
          <span className="detailed-choice-card-content">
            <strong>{option.label}</strong>
            {option.description ? <span>{option.description}</span> : null}
            <small>{option.meta}</small>
          </span>
        </label>
      ))}
    </fieldset>
  );
}
