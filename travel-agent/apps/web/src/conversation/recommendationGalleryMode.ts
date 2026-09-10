export type RecommendationGalleryMode = "accordion" | "depth";

export function recommendationGalleryMode(
  candidateCount: number,
): RecommendationGalleryMode {
  return candidateCount > 7 ? "depth" : "accordion";
}
