import { useId } from "react";

import type { PendingInteraction } from "../generated/v4/contracts";

export function V4CardRecovery({
  pending,
  disabled,
  conflictMessage,
  onRetry,
  onReload,
}: {
  pending: PendingInteraction | null;
  disabled: boolean;
  conflictMessage?: string;
  onRetry: (interactionId: string) => void;
  onReload: () => void;
}) {
  const descriptionId = useId();
  if (
    !pending?.recovery ||
    pending.kind !== "free_text_question" ||
    (pending.status ?? "active") !== "active"
  ) {
    return null;
  }

  return (
    <section
      className="v4-card-recovery"
      data-v4-card-recovery={pending.interaction_id}
      aria-label="卡片生成恢复"
    >
      <p id={descriptionId} role="status">
        这一步的卡片暂未生成，已保存的选择不会丢失。可以重试，也可以直接输入新的需求或已有酒店信息。
      </p>
      <button
        type="button"
        className="v4-card-primary-action"
        aria-describedby={descriptionId}
        disabled={disabled}
        onClick={() => onRetry(pending.interaction_id)}
      >
        重新生成卡片
      </button>
      {conflictMessage ? (
        <div className="attachment-submit-conflict" role="alert">
          <span>{conflictMessage}</span>
          <button type="button" onClick={onReload} disabled={disabled}>
            刷新最新状态
          </button>
        </div>
      ) : null}
    </section>
  );
}
