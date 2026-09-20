import { Star } from "@phosphor-icons/react";
import type { DiningDisplayFacts } from "../generated/v4/contracts";

export function DiningFacts({
  facts,
  showRating = true
}: {
  facts?: DiningDisplayFacts | null;
  showRating?: boolean;
}) {
  if (!facts) return null;
  const cost = facts.average_cost;
  const amount = (fen: number) => (fen / 100).toLocaleString("zh-CN");
  const price = cost
    ? `参考人均 ¥${amount(cost.minimum_fen)}${cost.maximum_fen === cost.minimum_fen ? "" : `–${amount(cost.maximum_fen)}`}`
    : null;
  const rating = showRating && facts.rating != null ? facts.rating : null;
  if (rating == null && !price) return null;
  return (
    <span className="dining-display-facts" aria-label="餐厅信息">
      {rating != null ? (
        <span aria-label={`${facts.source_name}评分 ${rating.toFixed(1)} 分`}>
          <Star size={13} weight="fill" aria-hidden="true" />
          {facts.source_name} {rating.toFixed(1)}分
        </span>
      ) : null}
      {price ? <span>{price}</span> : null}
    </span>
  );
}
