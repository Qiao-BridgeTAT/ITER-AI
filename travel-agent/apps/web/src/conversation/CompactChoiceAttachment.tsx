export type CompactChoiceOption = {
  id: string;
  label: string;
  description?: string;
  submitText?: string;
};

type CompactChoiceAttachmentProps = {
  label: string;
  name: string;
  options: CompactChoiceOption[];
  selectedId?: string;
  disabled?: boolean;
  onSelect: (option: CompactChoiceOption) => void;
};

export function CompactChoiceAttachment({
  label,
  name,
  options,
  selectedId,
  disabled = false,
  onSelect
}: CompactChoiceAttachmentProps) {
  const visibleOptions = options.slice(0, 4);

  return (
    <fieldset
      className={`compact-choice-attachment compact-choice-count-${visibleOptions.length}`}
    >
      <legend className="visually-hidden">{label}</legend>
      {visibleOptions.map((option) => (
        <label className="compact-choice-card" key={option.id}>
          <input
            type="radio"
            name={name}
            value={option.id}
            checked={selectedId === option.id}
            disabled={disabled}
            onChange={() => onSelect(option)}
          />
          <span className="compact-choice-card-body">
            <strong>{option.label}</strong>
            {option.description ? <small>{option.description}</small> : null}
          </span>
          <span className="compact-choice-indicator" aria-hidden="true" />
        </label>
      ))}
    </fieldset>
  );
}

type CompactMultiChoiceAttachmentProps = {
  label: string;
  options: CompactChoiceOption[];
  selectedIds: string[];
  disabled?: boolean;
  onToggle: (option: CompactChoiceOption) => void;
  onConfirm: () => void;
};

export function CompactMultiChoiceAttachment({
  label,
  options,
  selectedIds,
  disabled = false,
  onToggle,
  onConfirm
}: CompactMultiChoiceAttachmentProps) {
  const visibleOptions = options.slice(0, 4);

  return (
    <div
      className={`compact-multi-choice-attachment compact-multi-choice-count-${visibleOptions.length}`}
    >
      <fieldset
        className={`compact-choice-attachment compact-choice-count-${visibleOptions.length}`}
      >
        <legend className="visually-hidden">{label}</legend>
        {visibleOptions.map((option) => {
          const selected = selectedIds.includes(option.id);
          return (
            <label
              className="compact-choice-card compact-choice-card-multiple"
              key={option.id}
            >
              <input
                type="checkbox"
                value={option.id}
                checked={selected}
                disabled={disabled}
                onChange={() => onToggle(option)}
              />
              <span className="compact-choice-card-body">
                <strong>{option.label}</strong>
                {option.description ? (
                  <small>{option.description}</small>
                ) : null}
              </span>
              <svg
                className="compact-choice-check"
                viewBox="0 0 35.6 35.6"
                aria-hidden="true"
              >
                <circle
                  className="compact-choice-check-background"
                  cx="17.8"
                  cy="17.8"
                  r="17.3"
                />
                <circle
                  className="compact-choice-check-stroke"
                  cx="17.8"
                  cy="17.8"
                  r="14.37"
                />
                <polyline
                  className="compact-choice-check-mark"
                  points="11.78 18.12 15.55 22.23 25.17 12.87"
                />
              </svg>
            </label>
          );
        })}
      </fieldset>
      <button
        className="compact-multi-choice-confirm"
        type="button"
        disabled={disabled || selectedIds.length === 0}
        onClick={onConfirm}
      >
        选好了
      </button>
    </div>
  );
}
