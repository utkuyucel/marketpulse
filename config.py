"""
Configuration management for MarketPulse API.
All configuration in one place for better maintainability.
Only sensitive API keys are loaded from environment variables.
"""
from functools import lru_cache
from typing import Dict, Any, List
from pydantic import field_validator, ConfigDict
from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings with validation and environment variable support."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=()
    )
    
    # API Keys (loaded from environment)
    alpha_vantage_api_key: str = ""
    fred_api_key: str = ""
    coinlayer_api_key: str = ""
    
    # API Endpoints
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"
    fred_base_url: str = "https://api.stlouisfed.org/fred"
    coinlayer_base_url: str = "http://api.coinlayer.com"
    
    # Database Configuration
    database_url: str = "postgresql://marketpulse_user:marketpulse_pass@localhost:5432/marketpulse"
    
    # Application Settings
    api_version: str = "v1"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    debug: bool = False
    
    # Data Fetching Configuration
    data_fetch_interval: int = 300  # seconds
    refresh_interval: int = 3600  # seconds
    max_concurrent_requests: int = 10
    request_timeout: int = 30  # seconds
    
    # ML Model Parameters
    anomaly_contamination: float = 0.1
    volatility_window: int = 30
    correlation_window: int = 60
    
    # API Rate Limiting
    requests_per_minute: int = 60
    
    # Default data symbols
    default_stock_symbols: str = "AAPL,GOOGL,MSFT,AMZN,TSLA"
    default_economic_indicators: str = "GDP,UNRATE,CPIAUCSL,FEDFUNDS"
    default_crypto_symbols: str = "BTC,ETH,ADA,DOT,LINK"
    
    @field_validator('anomaly_contamination')
    @classmethod
    def validate_contamination(cls, v):
        if not 0 < v < 1:
            raise ValueError('Contamination must be between 0 and 1')
        return v
    
    @field_validator('volatility_window', 'correlation_window')
    @classmethod
    def validate_windows(cls, v):
        if v < 1:
            raise ValueError('Window size must be positive')
        return v
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Extract database configuration parameters (excluding URL)."""
        return {
            "echo": self.debug,
            "pool_size": 20,
            "max_overflow": 30,
            "pool_pre_ping": True,
            "pool_recycle": 3600
        }
    
    @property
    def api_keys(self) -> Dict[str, str]:
        """Get all API keys as dictionary."""
        return {
            "alpha_vantage": self.alpha_vantage_api_key,
            "fred": self.fred_api_key,
            "coinlayer": self.coinlayer_api_key
        }
    
    @property
    def api_endpoints(self) -> Dict[str, str]:
        """Get all API endpoints as dictionary."""
        return {
            "alpha_vantage": self.alpha_vantage_base_url,
            "fred": self.fred_base_url,
            "coinlayer": self.coinlayer_base_url
        }
    
    @property
    def default_symbols(self) -> Dict[str, List[str]]:
        """Get default symbols for data fetching."""
        return {
            "stocks": self.default_stock_symbols.split(","),
            "economic": self.default_economic_indicators.split(","),
            "crypto": self.default_crypto_symbols.split(",")
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Project paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)