"""
SQLAlchemy database models for MarketPulse.
Follows repository pattern for clean data access.
"""
from datetime import datetime
from decimal import Decimal
from sqlalchemy import Column, Integer, String, Date, DECIMAL, BigInteger, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class StockPrice(Base):
    """Stock market price data model."""
    __tablename__ = "stock_prices"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    open_price = Column(DECIMAL(10, 2))
    high_price = Column(DECIMAL(10, 2))
    low_price = Column(DECIMAL(10, 2))
    close_price = Column(DECIMAL(10, 2))
    volume = Column(BigInteger)
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_stock_symbol_date', 'symbol', 'date'),
        Index('idx_stock_created_at', 'created_at'),
    )


class EconomicIndicator(Base):
    """Economic indicators data model."""
    __tablename__ = "economic_indicators"
    
    id = Column(Integer, primary_key=True)
    indicator_code = Column(String(50), nullable=False)
    indicator_name = Column(String(200))
    date = Column(Date, nullable=False)
    value = Column(DECIMAL(15, 4))
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_econ_code_date', 'indicator_code', 'date'),
        Index('idx_econ_created_at', 'created_at'),
    )


class CryptoPrice(Base):
    """Cryptocurrency price data model."""
    __tablename__ = "crypto_prices"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    price = Column(DECIMAL(15, 8))
    market_cap = Column(BigInteger)
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_crypto_symbol_date', 'symbol', 'date'),
        Index('idx_crypto_created_at', 'created_at'),
    )


class Prediction(Base):
    """ML model predictions storage."""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True)
    model_type = Column(String(50))
    asset_symbol = Column(String(10))
    prediction_value = Column(DECIMAL(10, 4))
    confidence_score = Column(DECIMAL(3, 2))
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_pred_model_symbol', 'model_type', 'asset_symbol'),
        Index('idx_pred_created_at', 'created_at'),
    )
