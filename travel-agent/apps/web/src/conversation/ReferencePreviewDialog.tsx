import { useCallback } from "react";

import { useModalFocus } from "../accessibility/useModalFocus";
import { FileAttachment, formatFileSize } from "./attachments";

type ReferencePreviewDialogProps = {
  attachment: FileAttachment;
  onClose: () => void;
};

export function ReferencePreviewDialog({
  attachment,
  onClose,
}: ReferencePreviewDialogProps) {
  const close = useCallback(() => onClose(), [onClose]);
  const dialogRef = useModalFocus<HTMLElement>(close);
  const isImage = attachment.mimeType.startsWith("image/");
  const isPdf = attachment.mimeType === "application/pdf";

  return (
    <div
      className="reference-preview-backdrop"
      role="presentation"
      onMouseDown={close}
    >
      <section
        ref={dialogRef}
        className="reference-preview-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="reference-preview-title"
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <div>
            <p>本次资料</p>
            <h2 id="reference-preview-title">{attachment.name}</h2>
          </div>
          <button type="button" onClick={close} aria-label="关闭资料预览">
            关闭
          </button>
        </header>

        <div className="reference-preview-content">
          {isImage && attachment.url ? (
            <img src={attachment.url} alt={attachment.name} />
          ) : null}
          {isPdf && attachment.url ? (
            <iframe src={attachment.url} title={attachment.name} />
          ) : null}
          {attachment.textPreview !== undefined ? (
            <pre>{attachment.textPreview || "这个文本文件是空的。"}</pre>
          ) : null}
          {!isImage && !isPdf && attachment.textPreview === undefined ? (
            <div className="reference-preview-unavailable">
              <p>这个文件已加入本次资料。</p>
              <p>当前格式将在接入解析服务后提供网页内预览。</p>
            </div>
          ) : null}
        </div>

        <footer>
          <span>{formatFileSize(attachment.size)}</span>
          {attachment.url ? (
            <a href={attachment.url} target="_blank" rel="noreferrer">
              打开原文件
            </a>
          ) : null}
        </footer>
      </section>
    </div>
  );
}
