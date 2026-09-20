import { motion, useReducedMotion } from "motion/react";
import { useCallback, useState } from "react";

import { useModalFocus } from "../accessibility/useModalFocus";

export type TravelTaskBookContent = {
  destinationAndDates: string;
  attractionPreference: string;
  diningPreference: string;
  lodgingPreference: string;
};

type FloatingTaskBookProps = {
  content: TravelTaskBookContent;
  onModify: () => void;
  onConfirm: () => void;
};

export function FloatingTaskBook({
  content,
  onModify,
  onConfirm
}: FloatingTaskBookProps) {
  const [open, setOpen] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const prefersReducedMotion = useReducedMotion();

  const close = useCallback(() => setOpen(false), []);
  const dialogRef = useModalFocus<HTMLElement>(open ? close : () => undefined);

  const modify = () => {
    setOpen(false);
    window.setTimeout(onModify, 0);
  };

  const confirm = () => {
    if (confirmed) {
      return;
    }
    setConfirmed(true);
    onConfirm();
    setOpen(false);
  };

  const openTransition = prefersReducedMotion
    ? { duration: 0 }
    : { duration: 0.22, ease: [0.16, 1, 0.3, 1] as const };

  return (
    <>
      <button
        className="task-book-reference-link"
        type="button"
        aria-label={confirmed ? "查看已确认的旅行任务书" : "查看旅行任务书"}
        aria-expanded={open}
        aria-controls="travel-task-book-panel"
        onClick={() => setOpen((current) => !current)}
      >
        <span>旅行任务书</span>
        <small>{confirmed ? "已确认" : "查看"}</small>
      </button>
      {open ? (
        <motion.div
          className="task-book-overlay"
          initial={prefersReducedMotion ? false : { opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: prefersReducedMotion ? 0 : 0.14 }}
          role="presentation"
          onMouseDown={close}
        >
          <motion.section
            id="travel-task-book-panel"
            ref={dialogRef}
            className="travel-task-book-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="travel-task-book-title"
            tabIndex={-1}
            initial={
              prefersReducedMotion ? false : { opacity: 0, scale: 0.965, y: -8 }
            }
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={openTransition}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header className="travel-task-book-header">
              <div className="travel-task-book-identity">
                <img src="/brand/iter-mark-black-64.png" alt="" />
                <h2 id="travel-task-book-title">旅行任务书</h2>
              </div>
              <div className="travel-task-book-header-actions">
                <span
                  className={`travel-task-book-status${
                    confirmed ? " is-confirmed" : ""
                  }`}
                >
                  {confirmed ? "已确认" : "待确认"}
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
                  value={content.destinationAndDates}
                />
                <TaskBookFact
                  label="景点偏好"
                  value={content.attractionPreference}
                />
                <TaskBookFact
                  label="饮食偏好"
                  value={content.diningPreference}
                />
                <TaskBookFact
                  label="住宿偏好"
                  value={content.lodgingPreference}
                />
              </dl>
            </div>

            <footer className="travel-task-book-actions is-compact">
              <div>
                <button
                  className="travel-task-book-modify"
                  type="button"
                  onClick={modify}
                >
                  返回对话修改
                </button>
                <button
                  className="travel-task-book-confirm"
                  type="button"
                  disabled={confirmed}
                  onClick={confirm}
                >
                  {confirmed ? "已确认" : "确认任务书"}
                </button>
              </div>
            </footer>
          </motion.section>
        </motion.div>
      ) : null}
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
