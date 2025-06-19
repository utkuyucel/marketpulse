"""
Pydantic schemas for request/response validation.
Ensures type safety and automatic API documentation.
"""
from datetime import date as Date, datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class StockPriceBase(BaseModel):
    """Base stock price schema."""
    symbol: str = Field(..., max_length=10, description="Stock symbol")
    date: Date = Field(..., description="Trading date")
    open_price: Optional[Decimal] = Field(None, ge=0, description="Opening price")
    high_price: Optional[Decimal] = Field(None, ge=0, description="Highest price")
    low_price: Optional[Decimal] = Field(None, ge=0, description="Lowest price")
    close_price: Optional[Decimal] = Field(None, ge=0, description="Closing price")
    volume: Optional[int] = Field(None, ge=0, description="Trading volume")


class StockPriceResponse(StockPriceBase):
    """Stock price response schema."""
    id: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class EconomicIndicatorBase(BaseModel):
    """Base economic indicator schema."""
    indicator_code: str = Field(..., max_length=50)
    indicator_name: Optional[str] = Field(None, max_length=200)
    date: Date
    value: Optional[Decimal] = Field(None, description="Indicator value")


class EconomicIndicatorResponse(EconomicIndicatorBase):
    """Economic indicator response schema."""
    id: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class CryptoPriceBase(BaseModel):
    """Base cryptocurrency price schema."""
    symbol: str = Field(..., max_length=10, description="Crypto symbol")
    date: Date = Field(..., description="Price date")
    price: Optional[Decimal] = Field(None, ge=0, description="Price in USD")
    market_cap: Optional[int] = Field(None, ge=0, description="Market capitalization")


class CryptoPriceResponse(CryptoPriceBase):
    """Cryptocurrency price response schema."""
    id: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class PredictionBase(BaseModel):
    """Base prediction schema."""
    model_type: str = Field(..., max_length=50)
    asset_symbol: str = Field(..., max_length=10)
    prediction_value: Decimal = Field(..., description="Predicted value")
    confidence_score: Decimal = Field(..., ge=0, le=1, description="Confidence score")


class PredictionResponse(PredictionBase):
    """Prediction response schema."""
    id: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class AnomalyDetectionRequest(BaseModel):
    """Anomaly detection request schema."""
    symbol: str = Field(..., max_length=10, description="Asset symbol")
    lookback_days: int = Field(30, ge=1, le=365, description="Days to analyze")


class AnomalyDetectionResponse(BaseModel):
    """Anomaly detection response schema."""
    symbol: str
    anomaly_score: float = Field(..., ge=0, le=1)
    is_anomaly: bool
    analysis_period: str
    detected_anomalies: List[Dict[str, Any]]


class VolatilityPredictionRequest(BaseModel):
    """Volatility prediction request schema."""
    symbol: str = Field(..., max_length=10)
    forecast_days: int = Field(7, ge=1, le=30, description="Days to forecast")


class VolatilityPredictionResponse(BaseModel):
    """Volatility prediction response schema."""
    symbol: str
    forecast_days: int
    predicted_volatility: float = Field(..., ge=0)
    confidence_interval: Dict[str, float]
    forecast_data: List[Dict[str, Any]]


class CorrelationAnalysisRequest(BaseModel):
    """Correlation analysis request schema."""
    symbols: List[str] = Field(..., min_length=2, max_length=10)
    analysis_window: int = Field(60, ge=30, le=365)


class CorrelationAnalysisResponse(BaseModel):
    """Correlation analysis response schema."""
    symbols: List[str]
    correlation_matrix: Dict[str, Dict[str, float]]
    analysis_window: int
    breakpoint_detected: bool
    analysis_date: datetime


class DataStatusResponse(BaseModel):
    """Data status response schema."""
    total_records: Dict[str, int]
    latest_data_dates: Dict[str, Optional[Date]]
    data_freshness_hours: Dict[str, Optional[float]]
    api_status: Dict[str, str]


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str
    timestamp: datetime
    database_connected: bool
    api_keys_configured: bool
    models_loaded: bool
