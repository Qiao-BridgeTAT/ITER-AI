import { useCallback } from "react";

import { useModalFocus } from "../accessibility/useModalFocus";
import type { TaskBook } from "../generated/contracts";

type ServerTaskBookPreviewProps = {
  taskBookId: string;
  label: string;
  taskBook: TaskBook | null;
  placeNames: ReadonlyMap<string, string>;
  onClose: () => void;
  onModify: () => void;
  onConfirm: () => void;
  confirmError?: string | null;
};

export function ServerTaskBookPreview({
  taskBookId,
  label,
  taskBook,
  placeNames,
  onClose,
  onModify,
  onConfirm,
  confirmError,
}: ServerTaskBookPreviewProps) {
  const close = useCallback(onClose, [onClose]);
  const dialogRef = useModalFocus<HTMLElement>(close);

  return (
    <div className="task-book-overlay" role="presentation" onMouseDown={close}>
      <section
        ref={dialogRef}
        className="travel-task-book-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="server-task-book-title"
        data-task-book-id={taskBookId}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="travel-task-book-header">
          <div className="travel-task-book-identity">
            <img src="/brand/iter-mark-black-64.png" alt="" />
            <div>
              <p>旅行任务书</p>
              <h2 id="server-task-book-title">{label}</h2>
            </div>
          </div>
          <div className="travel-task-book-header-actions">
            {taskBook ? (
              <span
                className={`travel-task-book-status${
                  taskBook.status === "confirmed" ? " is-confirmed" : ""
                }`}
              >
                {taskBook.status === "confirmed" ? "已确认" : "待确认"}
              </span>
            ) : null}
            <button type="button" onClick={close} aria-label="关闭旅行任务书">
              关闭
            </button>
          </div>
        </header>

        <div className="travel-task-book-content">
          {taskBook ? (
            <>
              <p className="travel-task-book-intro">
                这是服务端当前稳定快照中的旅行任务书。你可以关闭预览后继续通过对话修改。
              </p>
              <dl className="travel-task-book-summary">
                <TaskBookFact
                  label="目的地与时间"
                  value={`${taskBook.start_date} 至 ${taskBook.end_date}`}
                />
                <TaskBookFact
                  label="本次采用的偏好"
                  value={taskBook.preference_summary}
                />
                <TaskBookFact
                  label="必去与想去"
                  value={formatPlaces(
                    taskBook.strong_attraction_ids,
                    placeNames,
                    "暂无强意愿景点",
                  )}
                />
                <TaskBookFact
                  label="重要餐饮"
                  value={formatPlaces(
                    taskBook.important_restaurant_ids,
                    placeNames,
                    "没有需要专程前往的餐厅",
                  )}
                />
                <TaskBookFact
                  label="最终住宿"
                  value={
                    taskBook.selected_hotel_id
                      ? (placeNames.get(taskBook.selected_hotel_id) ??
                        "已确定酒店")
                      : "本次不安排住宿"
                  }
                />
              </dl>
              <details className="travel-task-book-details">
                <summary>查看取舍与当前假设</summary>
                <div>
                  <TaskBookList
                    title="关键限制"
                    items={taskBook.key_constraints ?? []}
                    empty="暂无额外关键限制"
                  />
                  <TaskBookList
                    title="未满足的强意愿"
                    items={taskBook.omitted_strong_desires ?? []}
                    empty="当前没有未满足的强意愿"
                  />
                  <TaskBookList
                    title="关键取舍"
                    items={taskBook.tradeoffs ?? []}
                    empty="暂无额外取舍"
                  />
                  <TaskBookList
                    title="当前假设"
                    items={(taskBook.assumptions ?? []).map(
                      (assumption) => assumption.description,
                    )}
                    empty="当前没有额外假设"
                  />
                </div>
              </details>
            </>
          ) : (
            <p className="travel-task-book-intro" role="status">
              任务书内容暂时不可用，请刷新后重试。
            </p>
          )}
        </div>

        <footer className="travel-task-book-actions">
          <p role={confirmError ? "alert" : undefined}>
            {confirmError ??
              (taskBook?.status === "confirmed"
                ? "这版任务书已经由服务端确认。"
                : "确认结果以服务端当前状态为准。")}
          </p>
          <div>
            <button
              className="travel-task-book-modify"
              type="button"
              onClick={() => {
                close();
                window.setTimeout(onModify, 0);
              }}
            >
              返回对话修改
            </button>
            <button
              className="travel-task-book-confirm"
              type="button"
              disabled={!taskBook || taskBook.status === "confirmed"}
              onClick={onConfirm}
            >
              {taskBook?.status === "confirmed" ? "已确认" : "确认任务书"}
            </button>
          </div>
        </footer>
      </section>
    </div>
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

function TaskBookList({
  title,
  items,
  empty,
}: {
  title: string;
  items: string[];
  empty: string;
}) {
  return (
    <section>
      <h3>{title}</h3>
      {items.length > 0 ? (
        <ul>
          {items.map((item, index) => (
            <li key={`${index}:${item}`}>{item}</li>
          ))}
        </ul>
      ) : (
        <p>{empty}</p>
      )}
    </section>
  );
}

function formatPlaces(
  ids: string[] | undefined,
  names: ReadonlyMap<string, string>,
  empty: string,
) {
  if (!ids?.length) return empty;
  return ids.map((id) => names.get(id) ?? "已选择地点").join("、");
}
