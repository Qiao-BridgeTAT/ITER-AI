export type AgentProtocol = "v2" | "v4";

export function requestedAgentProtocol(_search: string): AgentProtocol {
  return "v4";
}

export function pathWithAgentProtocol(
  path: string,
  protocol: AgentProtocol,
): string {
  if (protocol !== "v4") return path;
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}protocol=v4`;
}
