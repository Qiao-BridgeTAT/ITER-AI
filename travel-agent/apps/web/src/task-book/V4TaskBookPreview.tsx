import { createPortal } from "react-dom";
import { motion, useReducedMotion } from "motion/react";
import { useCallback, useId, useMemo, useState } from "react";

import { useModalFocus } from "../accessibility/useModalFocus";
import type { TaskBookV4, TripSemanticState } from "../generated/v4/contracts";
import { buildCompactTaskBookSummary } from "./v4TaskBookSummary";

type V4TaskBookPreviewProps = {
  triggerVariant?: "reference" | "conversation";
  taskBook: TaskBookV4;
  semanticState?: TripSemanticState | null;
  disabled?: boolean;
  conflictMessage?: string;
  onConfirm: () => boolean;
  onModify?: () => void;
  onReload?: () => void;
};

export function V4TaskBookPreview({
  taskBook,
  triggerVariant = "reference",
  semanticState = null,
  disabled = false,
  conflictMessage,
  onConfirm,
  onModify,
  onReload
}: V4TaskBookPreviewProps) {
  const [open, setOpen] = useState(false);
  const prefersReducedMotion = useReducedMotion();
  const generatedId = useId().replaceAll(":", "");
  const panelId = `v4-travel-task-book-${generatedId}`;
  const titleId = `${panelId}-title`;
  const isConfirmed = taskBook.status === "confirmed";
  const summary = useMemo(
    () => buildCompactTaskBookSummary(taskBook, semanticState),
    [semanticState, taskBook]
  );
  const close = useCallback(() => setOpen(false), []);
  const dialogRef = useModalFocus<HTMLElement>(open ? close : () => undefined);

  const modify = () => {
    setOpen(false);
    window.setTimeout(() => onModify?.(), 0);
  };

  const confirm = () => {
    if (disabled || isConfirmed) return;
    if (onConfirm()) setOpen(false);
  };

  const openTransition = prefersReducedMotion
    ? { duration: 0 }
    : { duration: 0.22, ease: [0.16, 1, 0.3, 1] as const };

  return (
    <>
      <button
        className={
          triggerVariant === "conversation"
            ? "v4-task-book-open v4-card-primary-action"
            : "task-book-reference-link"
        }
        type="button"
        aria-label={
          triggerVariant === "conversation"
            ? "打开本次旅行任务书"
            : isConfirmed
              ? "查看已确认的旅行任务书"
              : "查看旅行任务书"
        }
        aria-expanded={open}
        aria-controls={panelId}
        data-v4-attachment-kind="task_book"
        data-task-book-id={taskBook.task_book_id}
        data-task-book-status={taskBook.status}
        data-task-book-version={taskBook.version}
        data-task-book-state-version={taskBook.based_on_state_version}
        onClick={() => setOpen((current) => !current)}
      >
        {triggerVariant === "conversation" ? (
          "打开本次旅行任务书"
        ) : (
          <>
            <span>旅行任务书</span>
            <small>{isConfirmed ? "已确认" : "查看"}</small>
          </>
        )}
      </button>

      {open
        ? createPortal(
            <motion.div
              className="task-book-overlay"
              initial={prefersReducedMotion ? false : { opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: prefersReducedMotion ? 0 : 0.14 }}
              role="presentation"
              onMouseDown={close}
            >
              <motion.section
                id={panelId}
                ref={dialogRef}
                className="travel-task-book-panel v4-task-book-preview"
                role="dialog"
                aria-modal="true"
                aria-labelledby={titleId}
                tabIndex={-1}
                data-task-book-status={taskBook.status}
                initial={
                  prefersReducedMotion
                    ? false
                    : { opacity: 0, scale: 0.965, y: -8 }
                }
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={openTransition}
                onMouseDown={(event) => event.stopPropagation()}
              >
                <header className="travel-task-book-header">
                  <div className="travel-task-book-identity">
                    <img src="/brand/iter-mark-black-64.png" alt="" />
                    <h2 id={titleId}>旅行任务书</h2>
                  </div>
                  <div className="travel-task-book-header-actions">
                    <span
                      className={`travel-task-book-status${
                        isConfirmed ? " is-confirmed" : ""
                      }`}
                    >
                      {isConfirmed ? "已确认" : "待确认"}
                    </span>
                    <button
                      type="button"
                      onClick={close}
                      aria-label="关闭旅行任务书"
                    >
                      关闭
                    </button>
                  </div>
                </header>

                <div className="travel-task-book-content">
                  <dl className="travel-task-book-summary">
                    <TaskBookFact
                      label="目的地与时间"
                      value={summary.destinationAndDates}
                    />
                    <TaskBookFact
                      label="景点偏好"
                      value={summary.attractionPreference}
                    />
                    <TaskBookFact
                      label="饮食偏好"
                      value={summary.diningPreference}
                    />
                    <TaskBookFact
                      label="住宿偏好"
                      value={summary.lodgingPreference}
                    />
                  </dl>

                  {conflictMessage ? (
                    <div className="attachment-submit-conflict" role="alert">
                      <span>{conflictMessage}</span>
                      {onReload ? (
                        <button type="button" onClick={onReload}>
                          刷新最新任务书
                        </button>
                      ) : null}
                    </div>
                  ) : null}
                </div>

                <footer className="travel-task-book-actions is-compact">
                  <div>
                    {onModify ? (
                      <button
                        className="travel-task-book-modify"
                        type="button"
                        onClick={modify}
                      >
                        返回对话修改
                      </button>
                    ) : null}
                    {!isConfirmed ? (
                      <button
                        className="travel-task-book-confirm"
                        type="button"
                        disabled={disabled}
                        onClick={confirm}
                      >
                        确认任务书
                      </button>
                    ) : null}
                  </div>
                </footer>
              </motion.section>
            </motion.div>,
            document.body
          )
        : null}
    </>
  );
}

function TaskBookFact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
