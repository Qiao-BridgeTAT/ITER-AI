import { useEffect, useId, useRef, useState, type ComponentType } from "react";
import { createPortal } from "react-dom";
import {
  Bed,
  Car,
  ForkKnife,
  Ticket,
  Info,
  X,
  type IconProps
} from "@phosphor-icons/react";

import type { CostCategory } from "../generated/enums";

export type PlanBudgetRange = {
  currency?: string;
  minimum_fen: number;
  maximum_fen: number;
};

export type PlanBudgetItem = {
  category: CostCategory;
  availability: "available" | "partial" | "missing";
  amount_per_person?: PlanBudgetRange | null;
  missing_reason?: string | null;
};

export type PlanBudgetEstimate = {
  items: PlanBudgetItem[];
  detail_items?: PlanBudgetDetail[];
  total_per_person?: PlanBudgetRange | null;
  lodging_share_divisor?: number;
  fetched_at_note: string;
};

export type PlanBudgetDetail = {
  id: string;
  label: string;
  service_date: string;
  category: CostCategory;
  amount_per_person?: PlanBudgetRange | null;
  missing_label?: string;
  reference_consumption?: PlanBudgetRange | null;
};

interface BudgetCategoryPresentation {
  category: CostCategory;
  className: "transport" | "tickets" | "dining" | "lodging";
  label: string;
  note: string;
  icon: ComponentType<IconProps>;
}

const BUDGET_CATEGORIES: readonly BudgetCategoryPresentation[] = [
  {
    category: "local_transport",
    className: "transport",
    label: "交通",
    note: "行程内移动",
    icon: Car
  },
  {
    category: "attraction_tickets",
    className: "tickets",
    label: "景点门票",
    note: "需购票项目",
    icon: Ticket
  },
  {
    category: "dining",
    className: "dining",
    label: "饮食",
    note: "正餐与途中补给",
    icon: ForkKnife
  },
  {
    category: "lodging",
    className: "lodging",
    label: "住宿",
    note: "按同行分摊口径",
    icon: Bed
  }
];

const CURRENCY_FORMATTER = new Intl.NumberFormat("zh-CN");

export function PlanBudgetSummary({
  estimate
}: {
  estimate: PlanBudgetEstimate | null | undefined;
}) {
  const headingId = useId();
  if (estimate == null) return null;

  const itemsByCategory = new Map(
    estimate.items.map((item) => [item.category, item])
  );
  const visibleItems = BUDGET_CATEGORIES.map((presentation) => ({
    presentation,
    item: itemsByCategory.get(presentation.category)
  }));
  const pricedItems = visibleItems.filter(
    (entry): entry is typeof entry & { item: PlanBudgetItem } =>
      entry.item?.amount_per_person != null
  );
  const hasKnownAmount = pricedItems.length > 0;
  const shownTotal =
    estimate.total_per_person ??
    sumBudgetRanges(pricedItems.map(({ item }) => item.amount_per_person));
  const distributionLabel = pricedItems
    .map(({ presentation, item }) =>
      item?.amount_per_person
        ? `${presentation.label}${formatBudgetRange(item.amount_per_person)}`
        : `${presentation.label}${item?.missing_reason ?? "暂未估算"}`
    )
    .join("，");

  return (
    <section className="inline-plan-budget" aria-labelledby={headingId}>
      <header>
        <div>
          <h3 id={headingId}>预算</h3>
          {estimate.detail_items?.length ? (
            <BudgetDetails items={estimate.detail_items} />
          ) : null}
        </div>
        <div className="inline-plan-budget-total">
          <small>
            {visibleItems.some(
              ({ item }) => item == null || item.availability !== "available"
            )
              ? "已知参考费用"
              : "参考合计"}
          </small>
          <strong>
            {shownTotal ? formatBudgetRange(shownTotal) : "暂无估算"}
          </strong>
          <span>/ 人</span>
        </div>
      </header>

      {hasKnownAmount ? (
        <div
          className="inline-plan-budget-distribution"
          role="img"
          aria-label={`人均预算结构：${distributionLabel}`}
        >
          {pricedItems.map(({ presentation, item }) => (
            <span
              className={`is-${presentation.className}`}
              key={presentation.category}
              style={{ flexGrow: midpoint(item?.amount_per_person) }}
            />
          ))}
        </div>
      ) : null}

      <ul className="inline-plan-budget-breakdown" aria-label="人均预算明细">
        {visibleItems.map(({ presentation, item }) => {
          const ItemIcon = presentation.icon;
          return (
            <li key={presentation.category}>
              <span
                className={`inline-plan-budget-icon is-${presentation.className}`}
              >
                <ItemIcon size={15} weight="bold" aria-hidden="true" />
              </span>
              <span className="inline-plan-budget-label">
                <strong>{presentation.label}</strong>
              </span>
              {item?.amount_per_person ? (
                <data value={item.amount_per_person.maximum_fen / 100}>
                  {formatBudgetRange(item.amount_per_person)}
                </data>
              ) : (
                <span className="inline-plan-budget-missing">
                  {item?.missing_reason ?? "本次未计"}
                </span>
              )}
            </li>
          );
        })}
      </ul>

      <p className="inline-plan-budget-note">
        {formatBudgetNote(estimate.fetched_at_note)}
      </p>
    </section>
  );
}

function BudgetDetails({ items }: { items: PlanBudgetDetail[] }) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const panelId = useId();
  useEffect(() => {
    if (!open) return;
    panel.current?.querySelector<HTMLButtonElement>("button")?.focus();
    const outside = (event: Event) => {
      if (
        event.target instanceof Node &&
        !panel.current?.contains(event.target) &&
        !trigger.current?.contains(event.target)
      )
        setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setOpen(false);
        trigger.current?.focus();
      }
    };
    const resize = () => setOpen(false);
    document.addEventListener("pointerdown", outside);
    document.addEventListener("focusin", outside);
    document.addEventListener("keydown", escape);
    window.addEventListener("resize", resize);
    window.addEventListener("scroll", outside, true);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("focusin", outside);
      document.removeEventListener("keydown", escape);
      window.removeEventListener("resize", resize);
      window.removeEventListener("scroll", outside, true);
    };
  }, [open]);
  return (
    <>
      <button
        type="button"
        className="inline-plan-budget-info"
        ref={trigger}
        aria-label="查看预算逐项明细"
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-controls={open ? panelId : undefined}
        onClick={() => {
          const rect = trigger.current?.getBoundingClientRect();
          if (rect)
            setPosition({
              left: Math.max(16, Math.min(rect.left, window.innerWidth - 376)),
              top: Math.max(
                16,
                Math.min(
                  rect.bottom + 8,
                  window.innerHeight - Math.min(460, window.innerHeight - 32)
                )
              )
            });
          setOpen((value) => !value);
        }}
      >
        <Info size={19} aria-hidden="true" />
      </button>
      {open
        ? createPortal(
            <div
              id={panelId}
              ref={panel}
              role="dialog"
              aria-label="预算逐项明细"
              className="inline-plan-budget-details"
              style={position}
            >
              <header>
                <h3>
                  费用明细 <span>/ 人</span>
                </h3>
                <button
                  type="button"
                  aria-label="关闭预算明细"
                  onClick={() => {
                    setOpen(false);
                    trigger.current?.focus();
                  }}
                >
                  <X size={18} aria-hidden="true" />
                </button>
              </header>
              <div className="inline-plan-budget-details-scroll">
                {BUDGET_CATEGORIES.map(({ category, label }) => {
                  const lines = items.filter(
                    (item) => item.category === category
                  );
                  if (!lines.length) return null;
                  return (
                    <section key={category} aria-label={`${label}逐项价格`}>
                      <h4>{label}</h4>
                      <ul>
                        {lines.map((item) => (
                          <li key={item.id}>
                            <div>
                              <span>{item.label}</span>
                              <small>
                                {item.service_date.slice(5).replace("-", "/")}
                                {item.reference_consumption
                                  ? ` · 参考消费 ${formatBudgetRange(item.reference_consumption)}`
                                  : ""}
                              </small>
                            </div>
                            <strong>
                              {item.amount_per_person
                                ? category === "attraction_tickets" &&
                                  item.amount_per_person.maximum_fen === 0
                                  ? "免费"
                                  : formatBudgetRange(item.amount_per_person)
                                : (item.missing_label ?? "—")}
                            </strong>
                          </li>
                        ))}
                      </ul>
                    </section>
                  );
                })}
                <p>— 表示价格暂缺，不计入合计。参考价以实际消费为准。</p>
              </div>
            </div>,
            document.body
          )
        : null}
    </>
  );
}

function formatBudgetRange(range: PlanBudgetRange): string {
  const minimum = Math.round(range.minimum_fen / 100);
  const maximum = Math.round(range.maximum_fen / 100);
  const symbol =
    range.currency == null || range.currency === "CNY"
      ? "¥"
      : `${range.currency} `;
  return minimum === maximum
    ? `${symbol}${CURRENCY_FORMATTER.format(minimum)}`
    : `${symbol}${CURRENCY_FORMATTER.format(minimum)}–${CURRENCY_FORMATTER.format(maximum)}`;
}

function midpoint(range: PlanBudgetRange | null | undefined): number {
  if (range == null) return 0;
  return Math.max(0, (range.minimum_fen + range.maximum_fen) / 2);
}

function sumBudgetRanges(
  ranges: Array<PlanBudgetRange | null | undefined>
): PlanBudgetRange | null {
  const known = ranges.filter(
    (range): range is PlanBudgetRange => range != null
  );
  if (known.length === 0) return null;
  const currency = known[0].currency ?? "CNY";
  if (known.some((range) => (range.currency ?? "CNY") !== currency))
    return null;
  return {
    currency,
    minimum_fen: known.reduce((total, range) => total + range.minimum_fen, 0),
    maximum_fen: known.reduce((total, range) => total + range.maximum_fen, 0)
  };
}

function formatBudgetNote(note: string): string {
  return /不含.*(?:飞机|高铁|跨城交通|大交通)/u.test(note)
    ? note
    : `${note.replace(/[。；;]+$/u, "")}；不含飞机、高铁和其他跨城交通。`;
}
