"""Run the local MCP endpoint; proxy credentials stay in server environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn
from dotenv import dotenv_values

from backend.config.settings import AmapSearchProxySettings
from services.amap_search_mcp.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    env = dict(os.environ)
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("env file does not exist")
        env.update({k: v for k, v in dotenv_values(args.env_file).items() if v is not None})
    env.setdefault("AMAP_SEARCH_PROXY_KEY_PARAMETER", "key")
    settings = AmapSearchProxySettings.from_environment(env)
    if settings is None:
        parser.error("AMAP_SEARCH_PROXY_URL and AMAP_SEARCH_PROXY_KEY are required")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
