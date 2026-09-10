import type {
  CategoryCostSummary,
  CostEstimate,
  CostLineItem,
  TripCostEstimate,
} from "../generated/contracts";

export function buildPublishedPlanBudget(
  estimate: TripCostEstimate,
): CostEstimate | null {
  if (estimate.known_total_per_person == null) return null;
  const items = estimate.categories.map(categoryToLineItem);
  return {
    total_per_person: estimate.known_total_per_person,
    items,
    lodging_share_divisor: estimate.lodging_share_divisor ?? 2,
    excluded_costs: estimate.excluded_costs,
    fetched_at_note: `${estimate.pricing_note}（估算于 ${formatDateTime(estimate.generated_at)}）`,
  };
}

function categoryToLineItem(category: CategoryCostSummary): CostLineItem {
  const hasAmount = category.amount_per_person != null;
  return {
    category: category.category,
    availability:
      category.status === "available"
        ? "available"
        : category.status === "partial"
          ? "partial"
          : "missing",
    amount_per_person: category.amount_per_person,
    missing_reason: hasAmount
      ? category.note
      : (category.note ??
        (category.status === "not_applicable"
          ? "本次没有这一类费用"
          : "当前还没有可展示的估算")),
    source_fact_ids: category.source_reference_ids,
  };
}

function formatDateTime(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : new Intl.DateTimeFormat("zh-CN", {
        month: "numeric",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }).format(parsed);
}
