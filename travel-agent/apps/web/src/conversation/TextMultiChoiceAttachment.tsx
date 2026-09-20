import { CompactChoiceOption } from "./CompactChoiceAttachment";

export type TextChoiceOption = CompactChoiceOption & {
  meta?: string;
};

type TextMultiChoiceAttachmentProps = {
  label: string;
  options: TextChoiceOption[];
  selectedIds: string[];
  exclusiveOptionId?: string;
  disabled?: boolean;
  onToggle: (option: TextChoiceOption) => void;
  onConfirm: () => void;
};

export function TextMultiChoiceAttachment({
  label,
  options,
  selectedIds,
  exclusiveOptionId,
  disabled = false,
  onToggle,
  onConfirm
}: TextMultiChoiceAttachmentProps) {
  return (
    <div className="text-multi-choice-attachment">
      <fieldset className="text-multi-choice-selector">
        <legend className="visually-hidden">{label}</legend>
        {options.slice(0, 6).map((option) => (
          <label className="text-multi-choice-row" key={option.id}>
            <span className="text-multi-choice-control">
              <input
                type="checkbox"
                value={option.id}
                checked={selectedIds.includes(option.id)}
                disabled={disabled}
                onChange={() => onToggle(option)}
              />
              <span className="text-multi-choice-ball" aria-hidden="true" />
            </span>
            <span className="text-multi-choice-copy">
              <strong>{option.label}</strong>
              {option.description ? <span>{option.description}</span> : null}
              {option.meta ? <small>{option.meta}</small> : null}
              {option.id === exclusiveOptionId ? (
                <span className="visually-hidden">此选项会替代其他选择</span>
              ) : null}
            </span>
          </label>
        ))}
      </fieldset>
      <button
        className="compact-multi-choice-confirm text-multi-choice-confirm"
        type="button"
        disabled={disabled || selectedIds.length === 0}
        onClick={onConfirm}
      >
        选好了
      </button>
    </div>
  );
}
