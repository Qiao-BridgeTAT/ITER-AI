import { type FormEvent, useCallback, useState } from "react";

import { useModalFocus } from "../accessibility/useModalFocus";

interface LoginOverlayProps {
  onClose: () => void;
  onSendCode: (phone: string) => boolean | Promise<boolean>;
  onSubmit: (phone: string, code: string) => boolean | Promise<boolean>;
}

type AccountMode = "login" | "register" | "recover";

const MODE_COPY: Record<
  AccountMode,
  { title: string; action: string; codeNotice: string }
> = {
  login: {
    title: "欢迎回来",
    action: "登录并保存旅程",
    codeNotice: "验证码已发送，验证后会恢复你的行程和长期偏好。",
  },
  register: {
    title: "创建账号",
    action: "完成注册",
    codeNotice: "验证码已发送，验证后会创建账号并保存当前旅程。",
  },
  recover: {
    title: "找回账号",
    action: "恢复账号",
    codeNotice: "验证码已发送，验证原手机号即可恢复账号。",
  },
};

export function LoginOverlay({
  onClose,
  onSendCode,
  onSubmit,
}: LoginOverlayProps) {
  const [mode, setMode] = useState<AccountMode>("login");
  const [phone, setPhone] = useState("");
  const [verificationCode, setVerificationCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [sendingCode, setSendingCode] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [codeSent, setCodeSent] = useState(false);
  const close = useCallback(() => onClose(), [onClose]);
  const dialogRef = useModalFocus<HTMLElement>(close);
  const copy = MODE_COPY[mode];

  const changeMode = (nextMode: AccountMode) => {
    setMode(nextMode);
    setError(null);
    setNotice(null);
    setVerificationCode("");
    setCodeSent(false);
  };

  const sendCode = async () => {
    const normalizedPhone = normalizeChinesePhone(phone);
    if (normalizedPhone === null) {
      setError("请输入有效的中国大陆手机号。");
      return;
    }

    setSendingCode(true);
    setError(null);
    setNotice(null);
    try {
      const sent = await onSendCode(normalizedPhone);
      setCodeSent(sent);
      if (sent) {
        setNotice(copy.codeNotice);
      } else {
        setError("验证码暂时无法发送，请稍后再试。");
      }
    } catch {
      setCodeSent(false);
      setError("验证码暂时无法发送，请稍后再试。");
    } finally {
      setSendingCode(false);
    }
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const normalizedPhone = normalizeChinesePhone(phone);
    if (normalizedPhone === null) {
      setError("请输入有效的中国大陆手机号。");
      return;
    }
    if (!/^\d{4,8}$/.test(verificationCode)) {
      setError("请输入 4–8 位数字验证码。");
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const completed = await onSubmit(normalizedPhone, verificationCode);
      if (!completed) {
        setError("登录没有完成，请检查手机号和验证码后重试。");
      }
    } catch {
      setError("登录没有完成，请检查手机号和验证码后重试。");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      className="overlay-backdrop auth-backdrop"
      role="presentation"
      onMouseDown={onClose}
    >
      <section
        ref={dialogRef}
        className="auth-overlay"
        role="dialog"
        aria-modal="true"
        aria-labelledby="auth-title"
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <button
          type="button"
          className="auth-close"
          onClick={close}
          aria-label="关闭登录"
        >
          <span aria-hidden="true">×</span>
        </button>

        <div className="auth-brand" aria-label="ITER AI">
          <img src="/brand/iter-mark-black-64.png" alt="" />
          <span>ITER AI</span>
        </div>

        <div className="auth-heading" key={`heading-${mode}`}>
          <h2 id="auth-title">{copy.title}</h2>
        </div>

        <form className="auth-form" key={mode} onSubmit={submit} noValidate>
          <label className="auth-field">
            <span className="visually-hidden">手机号</span>
            <input
              type="tel"
              autoComplete="tel"
              inputMode="tel"
              placeholder="手机号"
              value={phone}
              onChange={(event) => {
                setPhone(event.target.value);
                setVerificationCode("");
                setCodeSent(false);
                setNotice(null);
                setError(null);
              }}
            />
          </label>

          <div className="auth-field auth-code-field">
            <label
              className="visually-hidden"
              htmlFor="account-verification-code"
            >
              验证码
            </label>
            <input
              id="account-verification-code"
              type="text"
              autoComplete="one-time-code"
              inputMode="numeric"
              placeholder="验证码"
              value={verificationCode}
              onChange={(event) => setVerificationCode(event.target.value)}
            />
            <button
              type="button"
              disabled={sendingCode}
              onClick={() => void sendCode()}
            >
              {sendingCode ? "发送中…" : codeSent ? "重新发送" : "获取验证码"}
            </button>
          </div>

          {mode === "login" ? (
            <button
              type="button"
              className="auth-text-action auth-forgot"
              onClick={() => changeMode("recover")}
            >
              找回账号
            </button>
          ) : null}

          {error ? (
            <p className="auth-feedback auth-error" role="alert">
              {error}
            </p>
          ) : null}
          {notice ? (
            <p className="auth-feedback auth-notice" role="status">
              {notice}
            </p>
          ) : null}

          <button
            type="submit"
            className="auth-primary-action"
            disabled={submitting}
          >
            {submitting ? "正在验证…" : copy.action}
          </button>

          {mode === "login" ? (
            <button
              type="button"
              className="auth-secondary-action"
              onClick={() => changeMode("register")}
            >
              注册账号
            </button>
          ) : (
            <button
              type="button"
              className="auth-text-action auth-back-action"
              onClick={() => changeMode("login")}
            >
              返回登录
            </button>
          )}
        </form>

        <p className="auth-agreement">继续即表示你同意用户协议与隐私政策</p>
      </section>
    </div>
  );
}

function normalizeChinesePhone(value: string): string | null {
  const digits = value.replace(/\D/g, "");
  const local =
    digits.startsWith("86") && digits.length === 13 ? digits.slice(2) : digits;
  return /^1\d{10}$/.test(local) ? `+86${local}` : null;
}
