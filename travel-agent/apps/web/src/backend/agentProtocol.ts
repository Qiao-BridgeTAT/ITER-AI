export type AgentProtocol = "v2" | "v4";

export function requestedAgentProtocol(search: string): AgentProtocol {
  return new URLSearchParams(search).get("protocol") === "v2" ? "v2" : "v4";
}

export function pathWithAgentProtocol(
  path: string,
  protocol: AgentProtocol
): string {
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}protocol=${protocol}`;
}
