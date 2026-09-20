import { BootLoadingVisual } from "../app/BootLoadingVisual";

export function TripRestoreStatus({
  failed,
  errorCode,
  onRetry,
  onCancel
}: {
  failed: boolean;
  errorCode: string | null;
  onRetry: () => void;
  onCancel: () => void;
}) {
  const message =
    errorCode === "trip_restore_timeout"
      ? "打开行程超时，已保存的内容不受影响。"
      : errorCode?.includes("contract") || errorCode?.includes("incompatible")
        ? "这段行程的数据暂时无法恢复，并不是网络故障。"
        : errorCode?.includes("session") || errorCode?.includes("unauth")
          ? "登录或临时会话已失效，请重新登录后打开。"
          : errorCode === "trip_not_found"
            ? "找不到这段行程，或当前账号无权查看。"
            : "暂时无法连接行程服务，已保存的内容不受影响。";
  return (
    <section className="trip-restore-status" aria-label="恢复行程">
      {failed ? (
        <p role="alert">{message}</p>
      ) : (
        <div role="status" aria-live="polite">
          <BootLoadingVisual text="稍等一下，正在打开你的行程。" />
        </div>
      )}
      <div className="app-boot-actions">
        {failed ? (
          <button type="button" onClick={onRetry}>
            重试
          </button>
        ) : null}
        <button type="button" onClick={onCancel}>
          {failed ? "返回首页" : "取消，返回首页"}
        </button>
      </div>
    </section>
  );
}
