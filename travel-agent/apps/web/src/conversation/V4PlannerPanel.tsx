import { useId, useState } from "react";

import type { PlannerWorkspacePublicView } from "../generated/v4/contracts";
import "./v4-planner.css";

type Props = {
  workspace: PlannerWorkspacePublicView | null;
  generating: boolean;
  progressMessage: string | null;
  disabled: boolean;
  conflictMessage?: string;
  onResume: () => void;
  onAnswer: (optionId: string, text?: string) => void;
  onReload: () => void;
};

export function V4PlannerPanel({
  workspace,
  generating,
  progressMessage,
  disabled,
  conflictMessage,
  onResume,
  onAnswer,
  onReload,
}: Props) {
  const textId = useId();
  const [answerDraft, setAnswerDraft] = useState({
    interactionId: "",
    text: "",
  });
  if (!workspace && !progressMessage) return null;
  const status = workspace?.status ?? "planning";
  const interaction =
    status === "awaiting_user" ? workspace?.active_interaction : null;
  const text =
    answerDraft.interactionId === interaction?.interaction_id
      ? answerDraft.text
      : "";
  const needsNewBook =
    workspace?.last_interaction_action === "revise_task_book" ||
    workspace?.last_interaction_action === "supply_booking_detail";
  const canResume =
    workspace &&
    !generating &&
    !needsNewBook &&
    [
      "planning",
      "draft_ready",
      "ready_to_publish",
      "failed",
      "cancelled",
    ].includes(status);
  const title = interaction
    ? "有一项安排需要你决定"
    : generating ||
        ["planning", "draft_ready", "ready_to_publish"].includes(status)
      ? "正在生成正式行程"
      : status === "failed"
        ? "本次规划未完成"
        : status === "stale"
          ? "行程要求已更新"
          : "规划已暂停";
  const statusLabel = interaction
    ? "等待确认"
    : status === "failed"
      ? "需要重试"
      : status === "cancelled"
        ? "已暂停"
        : status === "stale"
          ? "需要重新规划"
          : ["draft_ready", "ready_to_publish"].includes(status)
            ? "即将完成"
            : "进行中";

  return (
    <section
      className="v4-planner-panel"
      aria-label="行程规划工作区"
      data-v4-planner-status={status}
      data-planner-revision={workspace?.workspace_revision}
    >
      <header>
        <h3>{title}</h3>
        <span>{statusLabel}</span>
      </header>
      {generating ? (
        <p role="status" aria-live="polite">
          {progressMessage ?? "正在根据已确认的要求编排行程并补充费用信息。"}
        </p>
      ) : null}
      {!generating && ["draft_ready", "ready_to_publish"].includes(status) ? (
        <p role="status" aria-live="polite">
          已完成逐日编排，正在整理正式行程。
        </p>
      ) : null}
      {interaction ? (
        <div
          className="v4-planner-decision"
          data-planner-interaction={interaction.interaction_id}
        >
          <p>
            这些选择会影响已确认的要求。修改任务书需要重新确认；选择保留原要求时，不会提交下方的修改说明。
          </p>
          <label htmlFor={textId}>
            补充说明
            {interaction.reason_code === "missing_user_owned_booking_detail"
              ? "（请填写预订信息）"
              : "（选填）"}
          </label>
          <textarea
            id={textId}
            value={text}
            onChange={(event) =>
              setAnswerDraft({
                interactionId: interaction.interaction_id,
                text: event.target.value,
              })
            }
            disabled={disabled}
            maxLength={2000}
            rows={3}
          />
          <div className="v4-planner-actions">
            {interaction.option_contracts.map((option) => (
              <button
                key={option.option_id}
                type="button"
                disabled={
                  disabled ||
                  (option.semantic_action === "supply_booking_detail" &&
                    !text.trim())
                }
                onClick={() =>
                  onAnswer(
                    option.option_id,
                    option.semantic_action === "keep_task_book"
                      ? undefined
                      : text.trim() || undefined,
                  )
                }
              >
                {option.verified_impact_summary}
              </button>
            ))}
          </div>
        </div>
      ) : null}
      {needsNewBook ? (
        <p>
          请在下方补充新的要求或预订信息，更新任务书并再次确认后再开始规划。
        </p>
      ) : null}
      {canResume ? (
        <div className="v4-planner-actions">
          <p>
            已保存工作状态，尚未生成正式行程。可以恢复，也可以直接输入新的需求。
          </p>
          <button
            type="button"
            className="v4-card-primary-action"
            disabled={disabled}
            onClick={onResume}
          >
            继续规划
          </button>
        </div>
      ) : null}
      {conflictMessage ? (
        <div className="attachment-submit-conflict" role="alert">
          <p>{conflictMessage}</p>
          <button type="button" onClick={onReload}>
            刷新最新状态
          </button>
        </div>
      ) : null}
    </section>
  );
}
