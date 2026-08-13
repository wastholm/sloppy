"""
Configuration management using pydantic-settings.
Supports environment variables, .env file, and CLI arguments.

Environment variables:
- OPENAI_API_KEY: API key for OpenAI-compatible endpoint
- OPENAI_BASE_URL: Base URL for the API (default: https://api.openai.com/v1)
- MODEL_NAME: Model to use (default: gpt-4o-mini)
- HOST: Server host (default: 0.0.0.0)
- PORT: Server port (default: 8000)
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Main application configuration."""
    openai_api_key: str = Field(
        default="",
        alias="OPENAI_API_KEY",
        description="OpenAI API key (can be empty for local endpoints without auth)"
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="OPENAI_BASE_URL",
        description="Base URL for OpenAI-compatible API"
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        alias="MODEL_NAME",
        description="Model to use for generating the search page"
    )
    host: str = Field(
        default="0.0.0.0",
        alias="HOST",
        description="Server host"
    )
    port: int = Field(
        default=8000,
        alias="PORT",
        description="Server port"
    )

    @field_validator('openai_base_url')
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Ensure base URL ends without trailing slash."""
        return v.rstrip('/')

    # Load from .env file and environment variables
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore',
        populate_by_name=True,
    )

    @property
    def openai(self) -> "OpenAIConfig":
        """Backward compatibility for openai config."""
        from pydantic import BaseModel
        class OpenAIConfig(BaseModel):
            api_key: str = self.openai_api_key
            base_url: str = self.openai_base_url
        return OpenAIConfig(api_key=self.openai_api_key, base_url=self.openai_base_url)

    @property
    def server(self) -> "ServerConfig":
        """Backward compatibility for server config."""
        from pydantic import BaseModel
        class ServerConfig(BaseModel):
            host: str = self.host
            port: int = self.port
        return ServerConfig(host=self.host, port=self.port)


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


def update_config_from_cli(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> None:
    """Update configuration from CLI arguments."""
    cfg = get_config()
    if api_key is not None:
        cfg.openai_api_key = api_key
    if base_url:
        cfg.openai_base_url = base_url.rstrip('/')
    if model_name:
        cfg.model_name = model_name
    if host:
        cfg.host = host
    if port:
        cfg.port = port
