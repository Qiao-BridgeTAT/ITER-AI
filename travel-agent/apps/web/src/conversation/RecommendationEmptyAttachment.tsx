type RecommendationEmptyAttachmentProps = {
  label: string;
  message?: string;
};

export function RecommendationEmptyAttachment({
  label,
  message = "这次没有找到足够可靠的候选。可以换个方向，或直接告诉我你更想找什么。"
}: RecommendationEmptyAttachmentProps) {
  return (
    <section className="recommendation-empty-attachment" aria-label={label}>
      <strong>{label}</strong>
      <p>{message}</p>
    </section>
  );
}
