import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./agent-markdown.css";

/** Untrusted Agent text is Markdown, never executable HTML. */
export function AgentMarkdown({ text }: { text: string }) {
  // Older persisted Planner replies joined this server prefix to a Markdown
  // heading. Repair only that known boundary; user text and HTML remain inert.
  const content = text.replace(/^(正式行程已生成。)(?=#{1,6}[ \t])/u, "$1\n\n");
  return (
    <div className="agent-markdown">
      <Markdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
          img: () => null,
          table: ({ children }) => (
            <div className="agent-markdown-table">
              <table>{children}</table>
            </div>
          )
        }}
      >
        {content}
      </Markdown>
    </div>
  );
}
