"""
Data fetching module for external APIs.
Implements async HTTP requests with proper error handling and rate limiting.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from functools import wraps
import httpx
from loguru import logger
from config import get_settings

settings = get_settings()


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed API calls."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator


class DataFetcher:
    """Unified data fetcher for all external APIs."""
    
    def __init__(self):
        self.session = None
        self.api_keys = settings.api_keys
        self.api_endpoints = settings.api_endpoints
        self._rate_limiters = {}
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = httpx.AsyncClient(
            timeout=settings.request_timeout,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=settings.max_concurrent_requests * 2
            )
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()
    
    @retry_on_failure(max_retries=3)
    async def fetch_stock_data(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Fetch stock price data from Alpha Vantage."""
        if not self.api_keys.get("alpha_vantage"):
            logger.warning("Alpha Vantage API key not configured")
            return []
        
        url = self.api_endpoints["alpha_vantage"]
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.api_keys["alpha_vantage"],
            "outputsize": "compact"
        }
        
        response = await self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            logger.warning(f"No stock data found for {symbol}")
            return []
        
        time_series = data["Time Series (Daily)"]
        return [
            {
                "symbol": symbol,
                "date": date_str,
                "open_price": float(values["1. open"]),
                "high_price": float(values["2. high"]),
                "low_price": float(values["3. low"]),
                "close_price": float(values["4. close"]),
                "volume": int(values["5. volume"])
            }
            for date_str, values in list(time_series.items())[:days]
        ]
    
    @retry_on_failure(max_retries=3)
    async def fetch_economic_data(self, indicators: List[str]) -> List[Dict[str, Any]]:
        """Fetch economic indicators from FRED API."""
        if not self.api_keys.get("fred"):
            logger.warning("FRED API key not configured")
            return []
        
        results = []
        for indicator in indicators:
            url = f"{self.api_endpoints['fred']}/series/observations"
            params = {
                "series_id": indicator,
                "api_key": self.api_keys["fred"],
                "file_type": "json",
                "limit": 100,
                "sort_order": "desc"
            }
            
            response = await self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "observations" in data:
                for obs in data["observations"]:
                    if obs["value"] != ".":  # Skip missing values
                        results.append({
                            "indicator_code": indicator,
                            "indicator_name": data.get("title", indicator),
                            "date": obs["date"],
                            "value": float(obs["value"])
                        })
        
        return results
    
    @retry_on_failure(max_retries=3)
    async def fetch_crypto_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch cryptocurrency data from CoinLayer API."""
        if not self.api_keys.get("coinlayer"):
            logger.warning("CoinLayer API key not configured")
            return []
        
        url = f"{self.api_endpoints['coinlayer']}/live"
        params = {
            "access_key": self.api_keys["coinlayer"],
            "symbols": ",".join(symbols)
        }
        
        response = await self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success", False):
            logger.error(f"CoinLayer API error: {data.get('error', 'Unknown error')}")
            return []
        
        rates = data.get("rates", {})
        current_date = datetime.now().date()
        
        return [
            {
                "symbol": symbol,
                "date": current_date,
                "price": float(price),
                "market_cap": None  # CoinLayer doesn't provide market cap
            }
            for symbol, price in rates.items()
            if symbol in symbols
        ]
    
    async def fetch_all_data(self, 
                           stock_symbols: List[str] = None,
                           economic_indicators: List[str] = None,
                           crypto_symbols: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch data from all sources concurrently."""
        tasks = []
        
        if stock_symbols:
            for symbol in stock_symbols:
                tasks.append(("stocks", symbol, self.fetch_stock_data(symbol)))
        
        if economic_indicators:
            tasks.append(("economic", "all", self.fetch_economic_data(economic_indicators)))
        
        if crypto_symbols:
            tasks.append(("crypto", "all", self.fetch_crypto_data(crypto_symbols)))
        
        results = {"stocks": [], "economic": [], "crypto": []}
        
        if tasks:
            completed_tasks = await asyncio.gather(
                *[task[2] for task in tasks],
                return_exceptions=True
            )
            
            for (data_type, symbol, _), result in zip(tasks, completed_tasks):
                if isinstance(result, Exception):
                    logger.error(f"Failed to fetch {data_type} data for {symbol}: {result}")
                    continue
                
                if data_type == "stocks":
                    results["stocks"].extend(result)
                elif data_type == "economic":
                    results["economic"].extend(result)
                elif data_type == "crypto":
                    results["crypto"].extend(result)
        
        return results


