"""Validated configuration loading without exposing secret values."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "config" / "contract.json"
PROFILE_DIR = PROJECT_ROOT / "config" / "environments"
ALLOWED_ENVIRONMENTS = frozenset({"development", "test", "production"})
ALLOWED_SMS_PROVIDERS = frozenset({"", "unavailable", "tencent_cloud", "aliyun_phone_verification"})
SMS_PROVIDER_REQUIRED_CONFIGURATION = {
    "tencent_cloud": frozenset(
        {
            "SMS_API_KEY",
            "SMS_API_SECRET",
            "SMS_APP_ID",
            "SMS_SIGN_NAME",
            "SMS_TEMPLATE_ID",
            "SMS_REGION",
        }
    ),
    "aliyun_phone_verification": frozenset(
        {"SMS_API_KEY", "SMS_API_SECRET", "SMS_SIGN_NAME", "SMS_TEMPLATE_ID"}
    ),
}


class ConfigurationError(ValueError):
    """Raised with configuration names only, never with their values."""


@dataclass(frozen=True)
class AmapSearchProxySettings:
    """Optional server-only transport for explicitly allowed POI search endpoints."""

    base_url: str
    api_key: str = field(repr=False, compare=False)
    key_parameter: Literal["KEY", "key"] = "KEY"

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> AmapSearchProxySettings | None:
        proxy_url = environ.get("AMAP_SEARCH_PROXY_URL", "").strip()
        proxy_key = environ.get("AMAP_SEARCH_PROXY_KEY", "").strip()
        if bool(proxy_url) != bool(proxy_key):
            raise ConfigurationError(
                "AMAP_SEARCH_PROXY_URL and AMAP_SEARCH_PROXY_KEY must be configured together"
            )
        key_parameter = environ.get("AMAP_SEARCH_PROXY_KEY_PARAMETER", "KEY").strip()
        if key_parameter not in {"KEY", "key"}:
            raise ConfigurationError("AMAP_SEARCH_PROXY_KEY_PARAMETER must be KEY or key")
        return (
            cls(
                base_url=proxy_url,
                api_key=proxy_key,
                key_parameter=cast(Literal["KEY", "key"], key_parameter),
            )
            if proxy_url
            else None
        )

    def __post_init__(self) -> None:
        try:
            parsed = urlsplit(self.base_url)
            valid = (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and parsed.port in {None, 80 if parsed.scheme == "http" else 443}
                and parsed.username is None
                and parsed.password is None
                and parsed.path in {"", "/"}
                and not parsed.query
                and not parsed.fragment
            )
        except ValueError:
            valid = False
        if not valid:
            raise ConfigurationError(
                "AMAP_SEARCH_PROXY_URL requires an HTTP(S) origin without credentials/path/query"
            ) from None
        if not self.api_key.strip():
            raise ConfigurationError("AMAP_SEARCH_PROXY_KEY must not be empty")
        if self.key_parameter not in {"KEY", "key"}:
            raise ConfigurationError("AMAP_SEARCH_PROXY_KEY_PARAMETER must be KEY or key")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))


@dataclass(frozen=True)
class AmapPolygonProxySettings:
    """Separate user-configured transport and credential for v5 polygon only."""

    base_url: str
    api_key: str = field(repr=False, compare=False)
    key_parameter: Literal["KEY", "key"] = "key"

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> AmapPolygonProxySettings | None:
        proxy_url = environ.get("AMAP_POLYGON_PROXY_URL", "").strip()
        proxy_key = environ.get("AMAP_POLYGON_PROXY_KEY", "").strip()
        if bool(proxy_url) != bool(proxy_key):
            raise ConfigurationError(
                "AMAP_POLYGON_PROXY_URL and AMAP_POLYGON_PROXY_KEY must be configured together"
            )
        parameter = environ.get("AMAP_POLYGON_PROXY_KEY_PARAMETER", "key").strip()
        if parameter not in {"KEY", "key"}:
            raise ConfigurationError("AMAP_POLYGON_PROXY_KEY_PARAMETER must be KEY or key")
        return (
            cls(proxy_url, proxy_key, cast(Literal["KEY", "key"], parameter)) if proxy_url else None
        )

    def __post_init__(self) -> None:
        try:
            parsed = urlsplit(self.base_url)
            valid = (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and parsed.port in {None, 80 if parsed.scheme == "http" else 443}
                and parsed.username is None
                and parsed.password is None
                and parsed.path in {"", "/"}
                and not parsed.query
                and not parsed.fragment
            )
        except ValueError:
            valid = False
        if not valid:
            raise ConfigurationError(
                "AMAP_POLYGON_PROXY_URL requires an HTTP(S) origin without credentials/path/query"
            ) from None
        if not self.api_key.strip():
            raise ConfigurationError("AMAP_POLYGON_PROXY_KEY must not be empty")
        if self.key_parameter not in {"KEY", "key"}:
            raise ConfigurationError("AMAP_POLYGON_PROXY_KEY_PARAMETER must be KEY or key")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    base_url: str
    model_name: str
    temperature: float
    timeout_seconds: float
    prompt_version: str
    api_key: str = field(repr=False, compare=False)

    @property
    def enabled(self) -> bool:
        return self.provider == "qwen" and bool(self.api_key)


@dataclass(frozen=True)
class ModelAuditSettings:
    local_path: str
    ttl_days: int
    encryption_key: str = field(repr=False, compare=False)


@dataclass(frozen=True)
class TavilyMcpSettings:
    auth_mode: Literal["key", "keyless"]
    api_key: str = field(default="", repr=False, compare=False)


@dataclass(frozen=True)
class Settings:
    app_env: str
    provider_mode: str
    conversation_mode: str
    anonymous_session_ttl_seconds: int
    model: ModelSettings
    model_audit: ModelAuditSettings
    values: Mapping[str, str] = field(repr=False)
    amap_search_proxy: AmapSearchProxySettings | None = None
    amap_polygon_proxy: AmapPolygonProxySettings | None = None
    tavily_mcp: TavilyMcpSettings | None = None

    @property
    def v4_planner_enabled(self) -> bool:
        """Enable the confirmed-task-book Planner unless explicitly rolled back."""

        return self.values.get("V4_PLANNER_ENABLED", "true") == "true"

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> Settings:
        raw = dict(os.environ if environ is None else environ)
        app_env = raw.get("APP_ENV", "").strip()
        if app_env not in ALLOWED_ENVIRONMENTS:
            raise ConfigurationError("APP_ENV must be one of: development, test, production")

        contract = _load_json(CONTRACT_PATH)
        profile = _load_toml(PROFILE_DIR / f"{app_env}.toml")
        required = list(contract["server"]["always_required"])
        if app_env == "production" or profile["strict_secrets"]:
            required.extend(contract["server"]["production_required"])

        missing = sorted(name for name in required if not raw.get(name, "").strip())
        if missing:
            raise ConfigurationError("Missing server configuration: " + ", ".join(missing))

        public_app_url = raw["PUBLIC_APP_URL"].strip()
        parsed_public_url = urlsplit(public_app_url)
        if not parsed_public_url.scheme or not parsed_public_url.netloc:
            raise ConfigurationError("PUBLIC_APP_URL must be an absolute URL")
        if app_env == "production" and parsed_public_url.scheme.lower() != "https":
            raise ConfigurationError("PUBLIC_APP_URL must use HTTPS in production")

        sms_provider = raw.get("SMS_PROVIDER", "").strip()
        if sms_provider not in ALLOWED_SMS_PROVIDERS:
            raise ConfigurationError(
                "SMS_PROVIDER must be unavailable, tencent_cloud, or aliyun_phone_verification"
            )
        if app_env == "production" and sms_provider not in SMS_PROVIDER_REQUIRED_CONFIGURATION:
            raise ConfigurationError("SMS_PROVIDER must select a production SMS provider")
        sms_required = SMS_PROVIDER_REQUIRED_CONFIGURATION.get(sms_provider, frozenset())
        missing_sms = sorted(name for name in sms_required if not raw.get(name, "").strip())
        if missing_sms:
            raise ConfigurationError("Missing server configuration: " + ", ".join(missing_sms))

        ttl_text = raw.get(
            "ANONYMOUS_SESSION_TTL_SECONDS",
            str(profile["anonymous_session_ttl_seconds"]),
        )
        try:
            ttl = int(ttl_text)
        except ValueError as exc:
            raise ConfigurationError("ANONYMOUS_SESSION_TTL_SECONDS must be an integer") from exc
        if ttl <= 0:
            raise ConfigurationError("ANONYMOUS_SESSION_TTL_SECONDS must be greater than zero")

        provider_mode = raw.get("PROVIDER_MODE", str(profile["provider_mode"]))
        if provider_mode not in {"live", "replay"}:
            raise ConfigurationError("PROVIDER_MODE must be live or replay")

        conversation_mode = raw.get("CONVERSATION_MODE", str(profile["conversation_mode"]))
        if conversation_mode not in {"disabled", "replay"}:
            raise ConfigurationError("CONVERSATION_MODE must be disabled or replay")
        if app_env == "production" and conversation_mode == "replay":
            raise ConfigurationError("CONVERSATION_MODE cannot be replay in production")
        if raw.get("V4_PLANNER_ENABLED", "true") not in {"true", "false"}:
            raise ConfigurationError("V4_PLANNER_ENABLED must be true or false")

        if raw.get("V4_PLANNER_ENGINE", "legacy") not in {"legacy", "langgraph-react-2"}:
            raise ConfigurationError("V4_PLANNER_ENGINE must be legacy or langgraph-react-2")

        if raw.get("V4_PLANNER_TIME_LIMIT_ENABLED", "true") not in {"true", "false"}:
            raise ConfigurationError("V4_PLANNER_TIME_LIMIT_ENABLED must be true or false")
        try:
            planner_max_decisions = int(raw.get("V4_PLANNER_MAX_DECISIONS", "24"))
        except ValueError:
            raise ConfigurationError(
                "V4_PLANNER_MAX_DECISIONS must be an integer from 1 to 24"
            ) from None
        if not 1 <= planner_max_decisions <= 24:
            raise ConfigurationError("V4_PLANNER_MAX_DECISIONS must be an integer from 1 to 24")

        amap_search_proxy = AmapSearchProxySettings.from_environment(raw)
        amap_polygon_proxy = AmapPolygonProxySettings.from_environment(raw)

        model_provider = raw.get("MODEL_PROVIDER", str(profile["model_provider"])).strip()
        if model_provider not in {"disabled", "qwen"}:
            raise ConfigurationError("MODEL_PROVIDER must be disabled or qwen")
        model_base_url = raw.get("QWEN_BASE_URL", str(profile["qwen_base_url"])).strip()
        parsed_model_url = urlsplit(model_base_url)
        if parsed_model_url.scheme.lower() != "https" or not parsed_model_url.netloc:
            raise ConfigurationError("QWEN_BASE_URL must be an absolute HTTPS URL")
        model_name = raw.get("QWEN_MODEL", str(profile["qwen_model"])).strip()
        if not model_name:
            raise ConfigurationError("QWEN_MODEL must not be empty")
        try:
            model_temperature = float(
                raw.get("MODEL_TEMPERATURE", str(profile["model_temperature"]))
            )
            model_timeout = float(
                raw.get("MODEL_TIMEOUT_SECONDS", str(profile["model_timeout_seconds"]))
            )
        except ValueError as exc:
            raise ConfigurationError(
                "MODEL_TEMPERATURE and MODEL_TIMEOUT_SECONDS must be numbers"
            ) from exc
        if not 0 <= model_temperature <= 2:
            raise ConfigurationError("MODEL_TEMPERATURE must be between 0 and 2")
        if not 0 < model_timeout <= 300:
            raise ConfigurationError("MODEL_TIMEOUT_SECONDS must be between 0 and 300")
        prompt_version = raw.get(
            "MODEL_PROMPT_VERSION", str(profile["model_prompt_version"])
        ).strip()
        if not prompt_version:
            raise ConfigurationError("MODEL_PROMPT_VERSION must not be empty")
        model_settings = ModelSettings(
            provider=model_provider,
            base_url=model_base_url,
            model_name=model_name,
            temperature=model_temperature,
            timeout_seconds=model_timeout,
            prompt_version=prompt_version,
            api_key=raw.get("QWEN_API_KEY", "").strip(),
        )
        try:
            model_audit_ttl_days = int(raw.get("LLM_AUDIT_TTL_DAYS", "30"))
        except ValueError as exc:
            raise ConfigurationError("LLM_AUDIT_TTL_DAYS must be an integer") from exc
        if not 1 <= model_audit_ttl_days <= 365:
            raise ConfigurationError("LLM_AUDIT_TTL_DAYS must be between 1 and 365")
        model_audit_path = raw.get(
            "LLM_AUDIT_LOCAL_PATH",
            str(PROJECT_ROOT / ".local" / "private-model-audit"),
        ).strip()
        if not model_audit_path:
            raise ConfigurationError("LLM_AUDIT_LOCAL_PATH must not be empty")
        model_audit_key = (
            raw.get("LLM_AUDIT_ENCRYPTION_KEY", "").strip()
            or raw.get("PII_ENCRYPTION_KEY", "").strip()
            or "development-only-model-audit-key"
        )
        model_audit_settings = ModelAuditSettings(
            local_path=model_audit_path,
            ttl_days=model_audit_ttl_days,
            encryption_key=model_audit_key,
        )

        search_mcp_url = raw.get("AMAP_SEARCH_MCP_URL", "").strip()
        if search_mcp_url:
            from backend.providers.amap_mcp import validate_search_mcp_url

            try:
                validate_search_mcp_url(search_mcp_url)
            except ValueError:
                raise ConfigurationError(
                    "AMAP_SEARCH_MCP_URL requires a loopback HTTP /mcp URL"
                ) from None

        tavily_mcp = None
        tavily_enabled = raw.get("TAVILY_MCP_ENABLED", "false").strip()
        if tavily_enabled not in {"true", "false"}:
            raise ConfigurationError("TAVILY_MCP_ENABLED must be true or false")
        if tavily_enabled == "true":
            mode = raw.get("TAVILY_MCP_AUTH_MODE", "key").strip()
            key = raw.get("TAVILY_API_KEY", "").strip()
            if mode not in {"key", "keyless"} or (mode == "key" and not key):
                raise ConfigurationError(
                    "Tavily requires key mode with TAVILY_API_KEY or explicit keyless mode"
                )
            if mode == "keyless" and key:
                raise ConfigurationError("Tavily keyless mode cannot carry TAVILY_API_KEY")
            tavily_mcp = TavilyMcpSettings(cast(Literal["key", "keyless"], mode), key)
        known_names = set(contract["server"]["always_required"])
        known_names.update(contract["server"]["production_required"])
        known_names.update(contract["server"]["optional"])
        safe_values = {
            name: raw[name]
            for name in known_names
            if name in raw
            and name
            not in {
                "QWEN_API_KEY",
                "LLM_AUDIT_ENCRYPTION_KEY",
                "AMAP_SEARCH_PROXY_KEY",
                "AMAP_POLYGON_PROXY_KEY",
                "TAVILY_API_KEY",
            }
        }
        return cls(
            app_env=app_env,
            provider_mode=provider_mode,
            conversation_mode=conversation_mode,
            anonymous_session_ttl_seconds=ttl,
            model=model_settings,
            model_audit=model_audit_settings,
            values=MappingProxyType(safe_values),
            amap_search_proxy=amap_search_proxy,
            amap_polygon_proxy=amap_polygon_proxy,
            tavily_mcp=tavily_mcp,
        )


def public_configuration_names() -> tuple[str, ...]:
    contract = _load_json(CONTRACT_PATH)
    return tuple(contract["public"]["required"])


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Configuration contract must be an object: {path.name}")
    return cast(dict[str, Any], payload)


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)
