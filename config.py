"""
Configuration management using pydantic-settings.
Supports environment variables, .env file, and CLI arguments.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""
    api_key: str = Field(
        default="",
        description="OpenAI API key (can be empty, will be validated at runtime)"
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for OpenAI-compatible API"
    )

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Ensure base URL ends without trailing slash."""
        return v.rstrip('/')


class ServerConfig(BaseSettings):
    """Server configuration."""
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")


class AppConfig(BaseSettings):
    """Main application configuration."""
    model_name: str = Field(
        default="gpt-4o-mini",
        description="Model to use for generating the search page"
    )
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    # Load from .env file and environment variables
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore',
    )


# Global config instance - lazy loaded to avoid issues when api_key not set
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


# For backward compatibility, expose config as a property
class _ConfigProxy:
    @property
    def config(self) -> AppConfig:
        return get_config()


_config_proxy = _ConfigProxy()


def update_config_from_cli(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> None:
    """Update configuration from CLI arguments."""
    cfg = get_config()
    if api_key:
        cfg.openai.api_key = api_key
    if base_url:
        cfg.openai.base_url = base_url.rstrip('/')
    if model_name:
        cfg.model_name = model_name
    if host:
        cfg.server.host = host
    if port:
        cfg.server.port = port
