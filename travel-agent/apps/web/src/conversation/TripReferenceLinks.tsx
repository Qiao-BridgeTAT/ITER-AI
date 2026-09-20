import { ArrowUpRight, LinkSimple } from "@phosphor-icons/react";
import type { TripReferenceLink } from "../generated/v4/contracts";
import "./trip-reference-links.css";

export function TripReferenceLinks({ links }: { links: TripReferenceLink[] }) {
  const visible = links.flatMap((link) => {
    try {
      const url = new URL(link.url);
      return ["https:", "http:"].includes(url.protocol) &&
        !url.username &&
        !url.password
        ? [{ ...link, host: url.hostname.replace(/^www\./, "") }]
        : [];
    } catch {
      return [];
    }
  });
  if (!visible.length) return null;
  return (
    <div className="trip-reference-links">
      <h3>参考链接</h3>
      <ul aria-label="本次旅行参考链接">
        {visible.map((link) => (
          <li key={link.url}>
            <a href={link.url} target="_blank" rel="noopener noreferrer">
              <LinkSimple size={16} aria-hidden="true" />
              <span>
                <strong>{link.title}</strong>
                <small>{link.host}</small>
              </span>
              <ArrowUpRight size={14} aria-hidden="true" />
              <span className="visually-hidden">（在新标签页打开）</span>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
