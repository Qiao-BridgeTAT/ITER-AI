import { X } from "@phosphor-icons/react";
import { createPortal } from "react-dom";

import { useModalFocus } from "../accessibility/useModalFocus";
import "./help-modal.css";

export function HelpModal({ onClose }: { onClose: () => void }) {
  const dialogRef = useModalFocus<HTMLElement>(onClose);

  return createPortal(
    <div
      className="account-session-backdrop"
      role="presentation"
      onMouseDown={onClose}
    >
      <section
        ref={dialogRef}
        className="account-session-popover help-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="help-modal-title"
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="account-session-header">
          <h2 id="help-modal-title">帮助</h2>
          <button
            className="account-session-close"
            type="button"
            aria-label="关闭帮助"
            onClick={onClose}
          >
            <X size={20} aria-hidden="true" />
          </button>
        </header>
        <div className="help-modal-content">
          <p>我暂时还帮不了你，因为我自己也不是很懂。</p>
          <p>有问题欢迎联系我的邮箱。</p>
          <a href="mailto:Li.qiao02@outlook.com">Li.qiao02@outlook.com</a>
        </div>
      </section>
    </div>,
    document.body
  );
}
