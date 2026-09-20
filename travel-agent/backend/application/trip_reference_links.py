"""Public links from persisted, attributed evidence; never from model prose."""

import json
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from backend.agent.planner.react_web_tools import public_web_url
from backend.contracts.v4.conversation import TripReferenceLink


def collect_reference_links(records: Iterable[dict[str, Any]]) -> tuple[TripReferenceLink, ...]:
    links: dict[str, TripReferenceLink] = {}

    def add(value: Any, title: Any, source: str) -> None:
        url = public_web_url(value)
        if not url:
            return
        label = " ".join(title.split())[:200] if isinstance(title, str) else ""
        previous = links.get(url)
        if previous and (not label or previous.title != urlsplit(url).hostname):
            return
        links[url] = TripReferenceLink(
            title=label or urlsplit(url).hostname, url=url, source=source
        )

    for record in records:
        for receipt in record.get("receipts") or []:
            if not isinstance(receipt, dict) or receipt.get("status") != "completed":
                continue
            name = ((receipt.get("call") or {}).get("function") or {}).get("name")
            if name not in {"tavily_search", "tavily_extract"}:
                continue
            try:
                result = json.loads(receipt.get("result") or "{}")
            except (ValueError, TypeError):
                continue
            if not isinstance(result, dict):
                continue
            rows = result.get("results")
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict):
                    add(row.get("url"), row.get("title"), "web_search")
        for fact in record.get("facts") or []:
            if isinstance(fact, dict):
                for source in fact.get("source_reference_ids") or []:
                    add(source, None, "provider_source")
    return tuple(links.values())
