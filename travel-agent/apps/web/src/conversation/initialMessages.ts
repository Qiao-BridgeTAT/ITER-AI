import type { ConversationMessage } from "./messageTypes";

export const INITIAL_MESSAGES: ConversationMessage[] = [
  {
    id: "welcome",
    role: "agent",
    content: "先说说你想去哪里，或者想拥有一段怎样的旅程。",
    status: "sent",
  },
];
