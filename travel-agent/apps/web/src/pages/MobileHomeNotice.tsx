import { useEffect, useRef, type RefObject } from "react";
import { createPortal } from "react-dom";

import "./mobile-home-notice.css";

interface MobileHomeNoticeProps {
  onContinue: () => void;
  returnFocusRef: RefObject<HTMLElement>;
}

export function MobileHomeNotice({
  onContinue,
  returnFocusRef,
}: MobileHomeNoticeProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const continueRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    const homeMain = returnFocusRef.current;
    const previousFocus = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    dialog?.showModal();
    document.body.style.overflow = "hidden";
    continueRef.current?.focus({ preventScroll: true });

    return () => {
      dialog?.close();
      document.body.style.overflow = previousOverflow;
      const target = homeMain ?? previousFocus;
      if (target instanceof HTMLElement && target.isConnected) {
        target.focus({ preventScroll: true });
      }
    };
  }, [returnFocusRef]);

  return createPortal(
    <dialog
      ref={dialogRef}
      className="iter-mobile-notice"
      aria-label="移动端使用提示"
      aria-describedby="iter-mobile-notice-copy"
      aria-modal="true"
      onCancel={(event) => {
        event.preventDefault();
        onContinue();
      }}
    >
      <div className="iter-mobile-notice-brand" aria-hidden="true">
        <img src="/brand/iter-mark-black-64.png" alt="" />
        <span>ITER AI</span>
      </div>
      <p id="iter-mobile-notice-copy">
        由于时间关系，ITER AI
        暂时还未对移动端进行使用体验优化。更建议您使用电脑端进行旅行规划安排！
      </p>
      <button
        ref={continueRef}
        className="iter-mobile-notice-continue"
        type="button"
        onClick={onContinue}
      >
        继续在手机上使用
      </button>
    </dialog>,
    document.body,
  );
}
