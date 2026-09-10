export function AgentLoadingIndicator() {
  return (
    <article className="conversation-message-row conversation-message-row-agent">
      <div
        className="message-bubble message-bubble-agent message-bubble-loading"
        role="status"
        aria-label="Agent 正在生成回复"
        aria-live="polite"
        aria-atomic="true"
      >
        <span className="agent-loading-indicator" aria-hidden="true">
          <span className="agent-loading-bar" />
          <span className="agent-loading-bar" />
          <span className="agent-loading-bar" />
        </span>
      </div>
    </article>
  );
}
