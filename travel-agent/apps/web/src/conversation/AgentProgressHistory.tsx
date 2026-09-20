import { CaretRight } from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import type { AgentProgressEntry } from "../generated/v4/contracts";
import {
  groupAgentProgress,
  visibleAgentProgress
} from "./agentProgressPresentation";
import "./agent-progress.css";

export function AgentProgressHistory({
  entries
}: {
  entries: AgentProgressEntry[];
}) {
  const groups = groupAgentProgress(entries);
  if (!groups.size) return null;
  return (
    <div className="agent-progress-history">
      {[...groups].map(([id, progress]) => (
        <details
          className="agent-progress-run"
          key={id}
          data-progress-generation={id}
        >
          <summary>
            查看规划过程
            <CaretRight
              className="agent-progress-chevron"
              size={14}
              aria-hidden="true"
            />
          </summary>
          <AgentProgressContent entries={progress} />
        </details>
      ))}
    </div>
  );
}

export function AgentProgressContent({
  entries,
  id,
  active = false
}: {
  entries: AgentProgressEntry[];
  id?: string;
  active?: boolean;
}) {
  const reducedMotion = useReducedMotion();
  return (
    <div
      id={id}
      className="agent-progress-content"
      role="region"
      aria-label="规划过程"
      tabIndex={0}
    >
      <AnimatePresence initial={false}>
        {visibleAgentProgress(entries).map((entry) => (
          <motion.p
            key={entry.event_id}
            data-progress-source={entry.source}
            initial={
              active && !reducedMotion && !document.hidden
                ? { opacity: 0 }
                : false
            }
            animate={{ opacity: 1 }}
            transition={{ duration: reducedMotion ? 0 : 0.18 }}
          >
            {entry.text}
          </motion.p>
        ))}
      </AnimatePresence>
    </div>
  );
}
