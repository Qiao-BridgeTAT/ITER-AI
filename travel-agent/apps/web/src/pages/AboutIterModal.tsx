import { X } from "@phosphor-icons/react";
import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

import "./about-iter.css";

export function AboutIterModal({ onClose }: { onClose: () => void }) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    const previousFocus = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    dialog?.showModal();
    document.body.style.overflow = "hidden";
    // Start a long letter at its heading, not a link near the bottom.
    titleRef.current?.focus({ preventScroll: true });
    return () => {
      dialog?.close();
      document.body.style.overflow = previousOverflow;
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected) {
        previousFocus.focus();
      }
    };
  }, []);

  return createPortal(
    <dialog
      ref={dialogRef}
      className="iter-about-dialog"
      aria-labelledby="iter-about-title"
      aria-modal="true"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="iter-about-sheet">
        <button
          className="iter-about-close"
          type="button"
          aria-label="关闭关于 ITER AI"
          onClick={onClose}
        >
          <X size={22} weight="light" aria-hidden="true" />
        </button>
        <article className="iter-about-letter">
          <div
            className="iter-about-body"
            tabIndex={0}
            aria-label="关于 ITER AI 正文"
          >
            <header className="iter-about-heading">
              <h2 id="iter-about-title" ref={titleRef} tabIndex={-1}>
                关于 ITER AI
              </h2>
              <img src="/brand/iter-mark-black-64.png" alt="" />
            </header>

            <div className="iter-about-passage">
              <p>
                ITER AI 的名字来自拉丁语 <em>iter</em>，意为“道路”或“旅程”。
              </p>
              <p>
                旅行往往始于一个模糊的念头，而 ITER AI
                想做的，是把这个念头整理成一条可以真正出发的路线。
              </p>
              <p>
                在生成一个完整规划前，ITER
                将会询问你关于景点偏好、饮食偏好和住宿偏好三个方向的问题。它会根据目的地的特点，给你发散性的多角度选项，帮助你更好地表达自己的偏好。
              </p>
            </div>

            <div className="iter-about-passage">
              <p>
                ITER AI 是一个基于 LangGraph 的多 Agent 旅行规划类 AI
                产品。它基于 Qwen 3.7 Plus
                驱动，分为需求梳理与行程规划两个阶段：先在对话中理解你的想法，形成旅行任务书，再结合高德地图、FlyAI
                等外部服务提供的信息，安排具体行程。
              </p>
            </div>

            <div className="iter-about-passage">
              <p>
                做一个这样的
                Agent，不避讳地讲，更多是因为现实的原因：为了面试，我得做个拿得出手的项目。
              </p>
              <p>
                但是，也更多源自于自我乐趣吧，我享受看到一个想法从零开始，逐渐成为可以使用的功能产品；
                <br />
                一个由我定义，什么是对，什么是错的过程。
              </p>
            </div>

            <div className="iter-about-passage">
              <p>
                这是我第一次独立上线一个完整产品。它一定还不够完善，因为我总能发现各种各样的
                Bug
                和需要优化的产品功能，但我也一定会继续根据真实使用中的问题逐步迭代。
              </p>
              <p className="iter-about-thanks">
                感谢你使用 ITER AI。如果你发现问题或有任何建议，欢迎联系：
              </p>
              <a
                className="iter-about-contact"
                href="mailto:Li.qiao02@outlook.com"
              >
                Li.qiao02@outlook.com
              </a>
              <p>另外，这个产品已经开源在 GitHub，欢迎交流。</p>
              <a
                className="iter-about-contact iter-about-repository"
                href="https://github.com/Qiao-BridgeTAT/ITER-AI"
                target="_blank"
                rel="noopener noreferrer"
              >
                https://github.com/Qiao-BridgeTAT/ITER-AI
              </a>
            </div>
          </div>
        </article>
      </div>
    </dialog>,
    document.body
  );
}
