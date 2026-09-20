import {
  type FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useState
} from "react";

import { useModalFocus } from "../accessibility/useModalFocus";

type AccountSessionOverlayProps = {
  userId: string;
  maskedPhone: string;
  nickname?: string | null;
  onClose: () => void;
  onUpdateNickname: (nickname: string) => Promise<boolean>;
  onLogout: () => Promise<boolean>;
};

function defaultNickname(userId: string): string {
  const compact = userId.replaceAll("-", "");
  const suffix = Number.parseInt(compact.slice(-8), 16) % 10_000;
  return `用户${String(suffix).padStart(4, "0")}`;
}

export function AccountSessionOverlay({
  userId,
  maskedPhone,
  nickname,
  onClose,
  onUpdateNickname,
  onLogout
}: AccountSessionOverlayProps) {
  const displayNickname = useMemo(
    () => nickname?.trim() || defaultNickname(userId),
    [nickname, userId]
  );
  const [editing, setEditing] = useState(false);
  const [draftNickname, setDraftNickname] = useState(displayNickname);
  const [nicknameBusy, setNicknameBusy] = useState(false);
  const [logoutBusy, setLogoutBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const close = useCallback(onClose, [onClose]);
  const dialogRef = useModalFocus<HTMLElement>(close);

  useEffect(() => {
    if (!editing) setDraftNickname(displayNickname);
  }, [displayNickname, editing]);

  const saveNickname = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const normalized = draftNickname.trim();
    if (normalized.length < 1 || normalized.length > 20) {
      setError("昵称需要填写 1–20 个字符。");
      return;
    }
    setNicknameBusy(true);
    setError(null);
    const completed = await onUpdateNickname(normalized);
    setNicknameBusy(false);
    if (completed) {
      setEditing(false);
      return;
    }
    setError("昵称没有保存成功，请稍后重试。");
  };

  return (
    <div
      className="account-session-backdrop"
      role="presentation"
      onMouseDown={close}
    >
      <section
        ref={dialogRef}
        className="account-session-popover"
        role="dialog"
        aria-modal="true"
        aria-labelledby="account-session-title"
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="account-session-header">
          <h2 id="account-session-title">我的账号</h2>
          <button
            type="button"
            className="account-session-close"
            aria-label="关闭账号"
            onClick={close}
          >
            <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <path
                d="m5.5 5.5 9 9m0-9-9 9"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </header>

        <div className="account-session-details">
          <div className="account-session-row account-session-nickname-row">
            <span className="account-session-label">昵称</span>
            {editing ? (
              <form
                className="account-session-nickname-form"
                onSubmit={saveNickname}
              >
                <input
                  aria-label="昵称"
                  autoFocus
                  maxLength={20}
                  value={draftNickname}
                  disabled={nicknameBusy}
                  onChange={(event) => setDraftNickname(event.target.value)}
                />
                <button type="submit" disabled={nicknameBusy}>
                  {nicknameBusy ? "保存中…" : "确认"}
                </button>
              </form>
            ) : (
              <div className="account-session-value-line">
                <strong>{displayNickname}</strong>
                <button
                  type="button"
                  onClick={() => {
                    setDraftNickname(displayNickname);
                    setError(null);
                    setEditing(true);
                  }}
                >
                  修改昵称
                </button>
              </div>
            )}
          </div>

          <div className="account-session-row">
            <span className="account-session-label">手机号</span>
            <strong>{maskedPhone}</strong>
          </div>
        </div>

        {error ? (
          <p className="account-session-error" role="alert">
            {error}
          </p>
        ) : null}

        <footer className="account-session-footer">
          <button
            type="button"
            className="account-session-logout"
            disabled={logoutBusy || nicknameBusy}
            onClick={() => {
              setLogoutBusy(true);
              setError(null);
              void onLogout().then((completed) => {
                setLogoutBusy(false);
                if (completed) close();
                else setError("退出没有完成，请检查网络后重试。");
              });
            }}
          >
            {logoutBusy ? "正在退出…" : "退出账号"}
          </button>
        </footer>
      </section>
    </div>
  );
}
