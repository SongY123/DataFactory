from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from utils.config_loader import get_config

_model_override_context: ContextVar[dict[str, Any] | None] = ContextVar("model_override_context", default=None)


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def _normalize_provider(config: dict[str, Any]) -> str:
    provider = _clean_string(config.get("provider"))
    if provider:
        return provider.lower()

    mode = _clean_string(config.get("mode"))
    if mode == "local":
        return "ollama"
    if mode == "api":
        return "openai"

    return str(get_config("model.provider", "ollama") or "ollama").strip().lower()


def _normalize_ollama_host(value: Any) -> str | None:
    host = _clean_string(value)
    if not host:
        return None
    normalized = host.rstrip("/")
    if normalized.lower().endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    return normalized or None


def _normalize_model_override(config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(config, dict):
        return None

    provider = _normalize_provider(config)
    normalized = {
        "provider": provider,
        "mode": _clean_string(config.get("mode")),
        "name": _clean_string(config.get("name")),
        "model_name": _clean_string(config.get("model_name") or config.get("modelName")),
        "host": _normalize_ollama_host(config.get("host")) if provider == "ollama" else _clean_string(config.get("host")),
        "api_key": _clean_string(config.get("api_key") or config.get("apiKey")),
        "base_url": _clean_string(config.get("base_url") or config.get("baseUrl")),
        "organization": _clean_string(config.get("organization")),
        "client_type": _clean_string(config.get("client_type") or config.get("clientType")),
        "enable_thinking": _clean_bool(config.get("enable_thinking") if "enable_thinking" in config else config.get("enableThinking")),
    }
    return {key: value for key, value in normalized.items() if value is not None}


def set_model_override(config: dict[str, Any] | None) -> Token:
    return _model_override_context.set(_normalize_model_override(config))


def reset_model_override(token: Token) -> None:
    _model_override_context.reset(token)


def get_model_override() -> dict[str, Any] | None:
    config = _model_override_context.get()
    return dict(config) if isinstance(config, dict) else None


def _merge_non_none(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if override:
        for key, value in override.items():
            if value is not None:
                merged[key] = value
    return merged


def _get_provider() -> str:
    override = get_model_override()
    if override and override.get("provider"):
        return str(override["provider"]).strip().lower()
    return str(get_config("model.provider", "ollama") or "ollama").strip().lower()


def create_model(**extra_kwargs: Any):
    provider = _get_provider()
    if provider == "dashscope":
        return _create_dashscope_model(**extra_kwargs)
    if provider == "ollama":
        return _create_ollama_model(**extra_kwargs)
    if provider == "openai":
        return _create_openai_model(**extra_kwargs)
    raise ValueError(f"Unsupported model provider: {provider}. Expected one of: ollama, dashscope, openai")


def _create_ollama_model(**extra_kwargs: Any):
    from agentscope.model import OllamaChatModel

    override = get_model_override()
    base = {
        "model_name": get_config("model.ollama.model_name"),
        "host": get_config("model.ollama.host"),
        "stream": get_config("model.ollama.stream", True),
        "keep_alive": get_config("model.ollama.keep_alive", "5m"),
        "enable_thinking": get_config("model.ollama.enable_thinking", False),
    }
    config = _merge_non_none(base, override)
    config["host"] = _normalize_ollama_host(config.get("host"))
    config.pop("provider", None)
    config.pop("mode", None)
    config.pop("name", None)
    config.pop("api_key", None)
    config.pop("base_url", None)
    config.pop("organization", None)
    config.pop("client_type", None)
    config.update(extra_kwargs)
    return OllamaChatModel(**config)


def _create_dashscope_model(**extra_kwargs: Any):
    from agentscope.model import DashScopeChatModel

    override = get_model_override()
    base = {
        "model_name": get_config("model.dashscope.model_name"),
        "api_key": get_config("model.dashscope.api_key"),
        "stream": get_config("model.dashscope.stream", True),
        "enable_thinking": get_config("model.dashscope.enable_thinking", False),
        "base_url": get_config("model.dashscope.base_url"),
    }
    config = _merge_non_none(base, override)
    model_kwargs = {
        "model_name": config.get("model_name"),
        "api_key": config.get("api_key"),
        "stream": config.get("stream", True),
        "enable_thinking": config.get("enable_thinking", False),
    }
    if config.get("base_url"):
        model_kwargs["base_http_api_url"] = config["base_url"]
    model_kwargs.update(extra_kwargs)
    return DashScopeChatModel(**model_kwargs)


def _create_openai_model(**extra_kwargs: Any):
    from agentscope.model import OpenAIChatModel

    override = get_model_override()
    base = {
        "model_name": get_config("model.openai.model_name"),
        "api_key": get_config("model.openai.api_key"),
        "stream": get_config("model.openai.stream", True),
        "base_url": get_config("model.openai.base_url"),
        "organization": get_config("model.openai.organization"),
        "client_type": get_config("model.openai.client_type", "openai"),
    }
    config = _merge_non_none(base, override)
    client_kwargs: dict[str, Any] = {}
    if config.get("base_url"):
        client_kwargs["base_url"] = config["base_url"]

    model_kwargs = {
        "model_name": config.get("model_name"),
        "api_key": config.get("api_key"),
        "stream": config.get("stream", True),
        "organization": config.get("organization"),
        "client_type": config.get("client_type", "openai"),
    }
    if client_kwargs:
        model_kwargs["client_kwargs"] = client_kwargs
    model_kwargs.update(extra_kwargs)
    return OpenAIChatModel(**model_kwargs)


def get_formatter():
    provider = _get_provider()
    if provider == "dashscope":
        from agentscope.formatter import DashScopeChatFormatter

        return DashScopeChatFormatter()
    if provider == "openai":
        from agentscope.formatter import OpenAIChatFormatter

        return OpenAIChatFormatter()

    from agentscope.formatter import OllamaChatFormatter

    return OllamaChatFormatter()
