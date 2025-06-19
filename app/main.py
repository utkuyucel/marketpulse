"""
MarketPulse FastAPI application.
Main application with all endpoints and middleware configuration.
"""
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Dict, Any
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db, get_db_session
from data_fetcher import DataFetcher
from ml_engine import MLEngine
from models import StockPrice, EconomicIndicator, CryptoPrice, Prediction
from schemas import (
    StockPriceResponse, EconomicIndicatorResponse, CryptoPriceResponse,
    AnomalyDetectionRequest, AnomalyDetectionResponse,
    VolatilityPredictionRequest, VolatilityPredictionResponse,
    CorrelationAnalysisRequest, CorrelationAnalysisResponse,
    DataStatusResponse, HealthCheckResponse
)
from config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    app.state.ml_engine = MLEngine()
    yield
    # Shutdown cleanup if needed


app = FastAPI(
    title="MarketPulse Financial ML API",
    description="Real-time financial machine learning API for market analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health Check Endpoints
@app.get("/health", response_model=HealthCheckResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Application health check."""
    try:
        # Check database connection
        await db.execute("SELECT 1")
        db_connected = True
    except Exception:
        db_connected = False
    
    # Check API keys
    api_keys_configured = all([
        settings.alpha_vantage_api_key,
        settings.fred_api_key,
        settings.coinlayer_api_key
    ])
    
    # Check models
    models_loaded = bool(hasattr(app.state, 'ml_engine'))
    
    return HealthCheckResponse(
        status="healthy" if all([db_connected, api_keys_configured, models_loaded]) else "degraded",
        timestamp=datetime.now(),
        database_connected=db_connected,
        api_keys_configured=api_keys_configured,
        models_loaded=models_loaded
    )


# Data Management Endpoints
@app.post("/api/v1/data/fetch")
async def fetch_data(
    background_tasks: BackgroundTasks,
    stock_symbols: List[str] = None,
    economic_indicators: List[str] = None,
    crypto_symbols: List[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Trigger data collection from all APIs."""
    if not stock_symbols:
        stock_symbols = settings.default_symbols["stocks"]
    if not economic_indicators:
        economic_indicators = settings.default_symbols["economic"]
    if not crypto_symbols:
        crypto_symbols = settings.default_symbols["crypto"]
    
    background_tasks.add_task(
        fetch_and_store_data,
        stock_symbols,
        economic_indicators,
        crypto_symbols
    )
    
    return {
        "message": "Data fetching initiated",
        "stock_symbols": stock_symbols,
        "economic_indicators": economic_indicators,
        "crypto_symbols": crypto_symbols
    }


async def fetch_and_store_data(stock_symbols: List[str], economic_indicators: List[str], crypto_symbols: List[str]):
    """Background task to fetch and store data."""
    async with DataFetcher() as fetcher:
        async with get_db_session() as db:
            try:
                data = await fetcher.fetch_all_data(stock_symbols, economic_indicators, crypto_symbols)
                
                # Store stock data
                for stock_data in data["stocks"]:
                    stock_price = StockPrice(**stock_data)
                    db.add(stock_price)
                
                # Store economic data
                for econ_data in data["economic"]:
                    indicator = EconomicIndicator(**econ_data)
                    db.add(indicator)
                
                # Store crypto data
                for crypto_data in data["crypto"]:
                    crypto_price = CryptoPrice(**crypto_data)
                    db.add(crypto_price)
                
                await db.commit()
                
            except Exception as e:
                await db.rollback()
                raise e


@app.get("/api/v1/data/stocks/{symbol}", response_model=List[StockPriceResponse])
async def get_stock_data(symbol: str, limit: int = 100, db: AsyncSession = Depends(get_db)):
    """Retrieve stock data for a specific symbol."""
    from sqlalchemy import select
    
    query = select(StockPrice).where(StockPrice.symbol == symbol.upper()).limit(limit)
    result = await db.execute(query)
    stocks = result.scalars().all()
    
    if not stocks:
        raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
    
    return stocks


@app.get("/api/v1/data/economic", response_model=List[EconomicIndicatorResponse])
async def get_economic_data(indicator_code: str = None, limit: int = 100, db: AsyncSession = Depends(get_db)):
    """Get economic indicators."""
    from sqlalchemy import select
    
    query = select(EconomicIndicator)
    if indicator_code:
        query = query.where(EconomicIndicator.indicator_code == indicator_code)
    query = query.limit(limit)
    
    result = await db.execute(query)
    indicators = result.scalars().all()
    
    return indicators


@app.get("/api/v1/data/crypto", response_model=List[CryptoPriceResponse])
async def get_crypto_data(symbol: str = None, limit: int = 100, db: AsyncSession = Depends(get_db)):
    """Get cryptocurrency data."""
    from sqlalchemy import select
    
    query = select(CryptoPrice)
    if symbol:
        query = query.where(CryptoPrice.symbol == symbol.upper())
    query = query.limit(limit)
    
    result = await db.execute(query)
    cryptos = result.scalars().all()
    
    return cryptos


@app.get("/api/v1/data/status", response_model=DataStatusResponse)
async def get_data_status(db: AsyncSession = Depends(get_db)):
    """Data freshness and coverage stats."""
    from sqlalchemy import select, func
    
    # Count records
    stock_count = await db.scalar(select(func.count(StockPrice.id)))
    econ_count = await db.scalar(select(func.count(EconomicIndicator.id)))
    crypto_count = await db.scalar(select(func.count(CryptoPrice.id)))
    
    # Get latest dates
    latest_stock = await db.scalar(select(func.max(StockPrice.date)))
    latest_econ = await db.scalar(select(func.max(EconomicIndicator.date)))
    latest_crypto = await db.scalar(select(func.max(CryptoPrice.date)))
    
    return DataStatusResponse(
        total_records={
            "stocks": stock_count or 0,
            "economic": econ_count or 0,
            "crypto": crypto_count or 0
        },
        latest_data_dates={
            "stocks": latest_stock,
            "economic": latest_econ,
            "crypto": latest_crypto
        },
        data_freshness_hours={
            "stocks": (datetime.now().date() - latest_stock).days * 24 if latest_stock else None,
            "economic": (datetime.now().date() - latest_econ).days * 24 if latest_econ else None,
            "crypto": (datetime.now().date() - latest_crypto).days * 24 if latest_crypto else None
        },
        api_status={
            "alpha_vantage": "configured" if settings.alpha_vantage_api_key else "missing",
            "fred": "configured" if settings.fred_api_key else "missing",
            "coinlayer": "configured" if settings.coinlayer_api_key else "missing"
        }
    )


# ML Endpoints
@app.post("/api/v1/ml/train")
async def train_models(
    background_tasks: BackgroundTasks,
    model_types: List[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Trigger model training."""
    if not model_types:
        model_types = ["anomaly", "volatility"]
    
    background_tasks.add_task(train_ml_models, model_types)
    
    return {
        "message": "Model training initiated",
        "model_types": model_types
    }


async def train_ml_models(model_types: List[str]):
    """Background task to train ML models."""
    import pandas as pd
    from sqlalchemy import select
    
    async with get_db_session() as db:
        try:
            # Get training data
            stock_query = select(StockPrice).order_by(StockPrice.date.desc()).limit(1000)
            result = await db.execute(stock_query)
            stocks = result.scalars().all()
            
            if not stocks:
                raise ValueError("No stock data available for training")
            
            # Convert to DataFrame
            stock_data = pd.DataFrame([
                {
                    "date": stock.date,
                    "symbol": stock.symbol,
                    "open_price": float(stock.open_price or 0),
                    "high_price": float(stock.high_price or 0),
                    "low_price": float(stock.low_price or 0),
                    "close_price": float(stock.close_price or 0),
                    "volume": stock.volume or 0
                }
                for stock in stocks
            ])
            
            # Train models
            ml_engine = MLEngine()
            for model_type in model_types:
                try:
                    await ml_engine.train_model(model_type, stock_data)
                except Exception as e:
                    print(f"Failed to train {model_type} model: {e}")
                    
        except Exception as e:
            print(f"Training failed: {e}")


@app.post("/api/v1/predict/anomaly", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    db: AsyncSession = Depends(get_db)
):
    """Detect price anomalies."""
    import pandas as pd
    from sqlalchemy import select
    from datetime import date, timedelta
    
    # Get recent data for the symbol
    end_date = date.today()
    start_date = end_date - timedelta(days=request.lookback_days)
    
    query = select(StockPrice).where(
        StockPrice.symbol == request.symbol.upper(),
        StockPrice.date >= start_date,
        StockPrice.date <= end_date
    ).order_by(StockPrice.date)
    
    result = await db.execute(query)
    stocks = result.scalars().all()
    
    if not stocks:
        raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
    
    # Convert to DataFrame
    stock_data = pd.DataFrame([
        {
            "close_price": float(stock.close_price or 0),
            "volume": stock.volume or 0,
            "high_price": float(stock.high_price or 0),
            "low_price": float(stock.low_price or 0)
        }
        for stock in stocks
    ])
    
    # Generate prediction
    ml_engine = app.state.ml_engine
    try:
        prediction = await ml_engine.predict("anomaly", stock_data)
        
        return AnomalyDetectionResponse(
            symbol=request.symbol.upper(),
            anomaly_score=prediction["anomaly_score"],
            is_anomaly=prediction["anomaly_rate"] > 0.1,  # 10% threshold
            analysis_period=f"{start_date} to {end_date}",
            detected_anomalies=prediction["detected_anomalies"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/v1/predict/volatility", response_model=VolatilityPredictionResponse)
async def predict_volatility(
    request: VolatilityPredictionRequest,
    db: AsyncSession = Depends(get_db)
):
    """Forecast volatility."""
    import pandas as pd
    from sqlalchemy import select
    from datetime import date, timedelta
    
    # Get data for volatility prediction
    end_date = date.today()
    start_date = end_date - timedelta(days=90)  # 3 months of data
    
    query = select(StockPrice).where(
        StockPrice.symbol == request.symbol.upper(),
        StockPrice.date >= start_date,
        StockPrice.date <= end_date
    ).order_by(StockPrice.date)
    
    result = await db.execute(query)
    stocks = result.scalars().all()
    
    if not stocks:
        raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
    
    # Convert to DataFrame
    stock_data = pd.DataFrame([
        {
            "close_price": float(stock.close_price or 0),
            "high_price": float(stock.high_price or 0),
            "low_price": float(stock.low_price or 0),
            "volume": stock.volume or 0
        }
        for stock in stocks
    ])
    
    # Generate prediction
    ml_engine = app.state.ml_engine
    try:
        prediction = await ml_engine.predict("volatility", stock_data)
        
        return VolatilityPredictionResponse(
            symbol=request.symbol.upper(),
            forecast_days=request.forecast_days,
            predicted_volatility=prediction["predicted_volatility"],
            confidence_interval=prediction["confidence_interval"],
            forecast_data=[{
                "day": i + 1,
                "volatility": prediction["predicted_volatility"] * (1 + 0.1 * i)  # Simple forecast
            } for i in range(request.forecast_days)]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/v1/models/status")
async def get_model_status():
    """Model performance metrics."""
    ml_engine = app.state.ml_engine
    return ml_engine.get_model_status()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
