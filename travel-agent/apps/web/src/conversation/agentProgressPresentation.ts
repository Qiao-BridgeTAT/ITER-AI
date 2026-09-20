import type {
  AgentProgressEntry,
  ConversationMessageV4
} from "../generated/v4/contracts";

const GENERIC_TOOL_RECEIPTS = new Set([
  "这一步已完成",
  "这一步未完成，已记录原因"
]);

export function visibleAgentProgress(entries: AgentProgressEntry[]) {
  return entries.filter(
    (entry) =>
      !GENERIC_TOOL_RECEIPTS.has(entry.text.trim().replace(/[。.!！]+$/u, ""))
  );
}

export function groupAgentProgress(entries: AgentProgressEntry[]) {
  const groups = new Map<string, AgentProgressEntry[]>();
  for (const entry of visibleAgentProgress(entries)) {
    const group = groups.get(entry.generation_id) ?? [];
    group.push(entry);
    groups.set(entry.generation_id, group);
  }
  for (const group of groups.values()) {
    group.sort((a, b) => a.progress_index - b.progress_index);
  }
  return groups;
}

export function positionAgentProgress(
  entries: AgentProgressEntry[],
  messages: ConversationMessageV4[],
  activeGenerationId: string | null
) {
  const replies = new Map<string, ConversationMessageV4>();
  const priority = (message: ConversationMessageV4) =>
    message.message_type === "plan" ? 3 : message.role === "assistant" ? 2 : 1;
  for (const message of messages) {
    if (message.role === "user") continue;
    const previous = replies.get(message.generation_id);
    if (!previous || priority(message) > priority(previous)) {
      replies.set(message.generation_id, message);
    }
  }
  const beforeMessage = new Map<string, AgentProgressEntry[]>();
  const unmatched: AgentProgressEntry[] = [];
  let active: AgentProgressEntry[] = [];
  for (const [generationId, group] of groupAgentProgress(entries)) {
    if (generationId === activeGenerationId) {
      active = group;
    } else {
      const reply = replies.get(generationId);
      if (reply) beforeMessage.set(reply.message_id, group);
      else unmatched.push(...group);
    }
  }
  return { beforeMessage, active, unmatched };
}
