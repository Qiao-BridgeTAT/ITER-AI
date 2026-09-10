export function RecommendationImagePlaceholder() {
  return (
    <span className="recommendation-image-placeholder" aria-hidden="true">
      <svg viewBox="0 0 32 26">
        <circle cx="23" cy="8" r="2.5" />
        <path d="m5 21 7.4-8 4.6 4.7 3.5-3.4L27 21" />
      </svg>
      <span>暂无图片</span>
    </span>
  );
}
