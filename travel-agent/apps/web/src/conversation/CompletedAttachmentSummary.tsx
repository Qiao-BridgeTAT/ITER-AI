type CompletedAttachmentSummaryProps = {
  label: string;
  value: string;
  disabled?: boolean;
  onEdit: () => void;
};

export function CompletedAttachmentSummary({
  label,
  value,
  disabled = false,
  onEdit
}: CompletedAttachmentSummaryProps) {
  return (
    <div className="completed-attachment-summary">
      <span>
        <small>{label}</small>
        <strong>{value}</strong>
      </span>
      <button
        type="button"
        disabled={disabled}
        aria-label={`修改${label}`}
        onClick={onEdit}
      >
        修改
      </button>
    </div>
  );
}
