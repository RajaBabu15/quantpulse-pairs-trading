"""
Data Management System
=====================

Professional market data handling with multiple providers, caching, and validation.
"""

import os
import pickle
import hashlib
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
import time

# Configure logging
import logging
logger = logging.getLogger(__name__)

@dataclass
class DataRequest:
    """Data request specification."""
    symbols: List[str]
    start_date: str
    end_date: str
    interval: str = "1d"
    provider: str = "yfinance"
    fields: List[str] = None
    
    def __post_init__(self):
        if self.fields is None:
            self.fields = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    
    def cache_key(self) -> str:
        """Generate cache key for this request."""
        key_data = f"{'-'.join(sorted(self.symbols))}_{self.start_date}_{self.end_date}_{self.interval}_{self.provider}"
        return hashlib.md5(key_data.encode()).hexdigest()

class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""
    
    @abstractmethod
    def fetch_data(self, request: DataRequest) -> Dict[str, pd.DataFrame]:
        """Fetch market data for given request."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

class YahooFinanceProvider(MarketDataProvider):
    """Yahoo Finance data provider using yfinance."""
    
    @property
    def name(self) -> str:
        return "yfinance"
    
    def is_available(self) -> bool:
        """Check if Yahoo Finance is available."""
        try:
            # Test with a simple request
            test = yf.Ticker("AAPL")
            info = test.info
            return 'symbol' in info or 'shortName' in info
        except Exception as e:
            logger.warning(f"Yahoo Finance provider unavailable: {e}")
            return False
    
    def fetch_data(self, request: DataRequest) -> Dict[str, pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        logger.info(f"🔄 Fetching data for {request.symbols} from Yahoo Finance")
        
        data = {}
        failed_symbols = []
        
        for symbol in request.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=request.start_date,
                    end=request.end_date,
                    interval=request.interval
                )
                
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                # Ensure we have the required columns
                missing_cols = [col for col in request.fields if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns {missing_cols} for {symbol}")
                
                # Add metadata
                df.attrs['symbol'] = symbol
                df.attrs['provider'] = self.name
                df.attrs['fetch_time'] = datetime.now()
                
                data[symbol] = df
                logger.info(f"✅ Successfully fetched {len(df)} records for {symbol}")
                
            except Exception as e:
                logger.error(f"❌ Error fetching {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            logger.warning(f"Failed to fetch data for: {failed_symbols}")
        
        return data

class AlphaVantageProvider(MarketDataProvider):
    """Alpha Vantage data provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # seconds between requests (free tier: 5 calls/min)
    
    @property
    def name(self) -> str:
        return "alpha_vantage"
    
    def is_available(self) -> bool:
        """Check if Alpha Vantage is available."""
        if not self.api_key:
            logger.warning("Alpha Vantage API key not found")
            return False
        
        try:
            # Test API with a simple request
            response = requests.get(
                self.base_url,
                params={
                    'function': 'GLOBAL_QUOTE',
                    'symbol': 'AAPL',
                    'apikey': self.api_key
                },
                timeout=10
            )
            return response.status_code == 200 and 'Error Message' not in response.text
        except Exception as e:
            logger.warning(f"Alpha Vantage provider unavailable: {e}")
            return False
    
    def fetch_data(self, request: DataRequest) -> Dict[str, pd.DataFrame]:
        """Fetch data from Alpha Vantage."""
        if not self.is_available():
            raise RuntimeError("Alpha Vantage provider not available")
        
        logger.info(f"🔄 Fetching data for {request.symbols} from Alpha Vantage")
        
        data = {}
        
        for i, symbol in enumerate(request.symbols):
            if i > 0:  # Rate limiting
                time.sleep(self.rate_limit_delay)
            
            try:
                params = {
                    'function': 'TIME_SERIES_DAILY_ADJUSTED',
                    'symbol': symbol,
                    'outputsize': 'full',
                    'apikey': self.api_key
                }
                
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                json_data = response.json()
                
                if 'Error Message' in json_data:
                    raise ValueError(f"API Error: {json_data['Error Message']}")
                
                if 'Note' in json_data:
                    raise ValueError(f"Rate limit exceeded: {json_data['Note']}")
                
                time_series = json_data.get('Time Series (Daily)', {})
                
                if not time_series:
                    logger.warning(f"No time series data for {symbol}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Rename columns
                column_mapping = {
                    '1. open': 'Open',
                    '2. high': 'High', 
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. adjusted close': 'Adj Close',
                    '6. volume': 'Volume'
                }
                df = df.rename(columns=column_mapping)
                
                # Convert to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Filter date range
                start_date = pd.to_datetime(request.start_date)
                end_date = pd.to_datetime(request.end_date)
                df = df[start_date:end_date]
                
                # Add metadata
                df.attrs['symbol'] = symbol
                df.attrs['provider'] = self.name
                df.attrs['fetch_time'] = datetime.now()
                
                data[symbol] = df
                logger.info(f"✅ Successfully fetched {len(df)} records for {symbol}")
                
            except Exception as e:
                logger.error(f"❌ Error fetching {symbol}: {e}")
        
        return data

class DataCache:
    """Intelligent caching system for market data."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def get(self, request: DataRequest) -> Optional[Dict[str, pd.DataFrame]]:
        """Get cached data if available and valid."""
        cache_key = request.cache_key()
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Check metadata
        metadata = self.metadata.get(cache_key)
        if not metadata:
            logger.debug(f"No metadata for cache key {cache_key}")
            return None
        
        # Check if cache is expired (default: 1 day for daily data)
        cache_age = datetime.now() - datetime.fromisoformat(metadata['created_at'])
        max_age = timedelta(hours=24) if request.interval == '1d' else timedelta(hours=1)
        
        if cache_age > max_age:
            logger.debug(f"Cache expired for {cache_key}")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"✅ Loaded cached data for {request.symbols}")
            return data
        
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
            return None
    
    def put(self, request: DataRequest, data: Dict[str, pd.DataFrame]):
        """Cache the data."""
        cache_key = request.cache_key()
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Update metadata
            self.metadata[cache_key] = {
                'symbols': request.symbols,
                'start_date': request.start_date,
                'end_date': request.end_date,
                'interval': request.interval,
                'provider': request.provider,
                'created_at': datetime.now().isoformat(),
                'size_mb': cache_file.stat().st_size / (1024 * 1024)
            }
            self._save_metadata()
            
            logger.info(f"💾 Cached data for {request.symbols}")
        
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def clear_expired(self):
        """Clear expired cache entries."""
        expired_keys = []
        
        for cache_key, metadata in self.metadata.items():
            try:
                created_at = datetime.fromisoformat(metadata['created_at'])
                age = datetime.now() - created_at
                
                # Default expiry: 7 days
                if age > timedelta(days=7):
                    expired_keys.append(cache_key)
            except Exception:
                expired_keys.append(cache_key)  # Invalid metadata
        
        for cache_key in expired_keys:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                if cache_file.exists():
                    cache_file.unlink()
                del self.metadata[cache_key]
                logger.info(f"🗑️ Removed expired cache: {cache_key}")
            except Exception as e:
                logger.error(f"Error removing cache {cache_key}: {e}")
        
        if expired_keys:
            self._save_metadata()
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_size = 0
        total_entries = len(self.metadata)
        
        for cache_key in self.metadata:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                total_size += cache_file.stat().st_size
        
        return {
            'total_entries': total_entries,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }

class DataManager:
    """Main data management interface."""
    
    def __init__(self, cache_enabled: bool = True, cache_dir: str = "data/cache"):
        print(f"🔄 ENTERING DataManager.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        # Initialize providers
        self.providers = {
            'yfinance': YahooFinanceProvider(),
            'alpha_vantage': AlphaVantageProvider()
        }
        
        # Initialize cache
        self.cache_enabled = cache_enabled
        self.cache = DataCache(cache_dir) if cache_enabled else None
        
        # Check provider availability
        self.available_providers = {}
        for name, provider in self.providers.items():
            self.available_providers[name] = provider.is_available()
            logger.info(f"Provider {name}: {'✅ Available' if self.available_providers[name] else '❌ Unavailable'}")
        
        print(f"✅ EXITING DataManager.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def get_pairs_data(self, 
                      symbols: List[str], 
                      start_date: str, 
                      end_date: str,
                      interval: str = "1d",
                      provider: str = "auto") -> Dict[str, pd.DataFrame]:
        """Get market data for pairs trading."""
        print(f"🔄 ENTERING get_pairs_data({symbols}, {start_date}, {end_date}) at {datetime.now().strftime('%H:%M:%S')}")
        
        # Validate inputs
        if len(symbols) < 2:
            raise ValueError("At least 2 symbols required for pairs trading")
        
        # Create request
        request = DataRequest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            provider=provider
        )
        
        # Check cache first
        if self.cache_enabled:
            cached_data = self.cache.get(request)
            if cached_data:
                print(f"✅ EXITING get_pairs_data() [cached] at {datetime.now().strftime('%H:%M:%S')}")
                return cached_data
        
        # Determine best provider
        if provider == "auto":
            provider = self._select_best_provider()
        
        if provider not in self.available_providers or not self.available_providers[provider]:
            raise RuntimeError(f"Provider {provider} is not available")
        
        # Fetch data
        request.provider = provider
        data = self.providers[provider].fetch_data(request)
        
        if not data:
            raise RuntimeError(f"No data returned from {provider}")
        
        # Validate data quality
        data = self._validate_and_clean_data(data)
        
        # Cache the results
        if self.cache_enabled:
            self.cache.put(request, data)
        
        print(f"✅ EXITING get_pairs_data() at {datetime.now().strftime('%H:%M:%S')}")
        return data
    
    def _select_best_provider(self) -> str:
        """Select the best available provider."""
        # Priority order
        priority = ['yfinance', 'alpha_vantage']
        
        for provider in priority:
            if self.available_providers.get(provider, False):
                return provider
        
        raise RuntimeError("No data providers available")
    
    def _validate_and_clean_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Validate and clean market data."""
        print(f"🔄 ENTERING _validate_and_clean_data() at {datetime.now().strftime('%H:%M:%S')}")
        
        cleaned_data = {}
        
        for symbol, df in data.items():
            if df.empty:
                logger.warning(f"Empty data for {symbol}")
                continue
            
            # Remove NaN values
            original_len = len(df)
            df = df.dropna()
            
            if len(df) != original_len:
                logger.warning(f"Removed {original_len - len(df)} NaN rows from {symbol}")
            
            # Ensure minimum data requirements
            if len(df) < 30:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                continue
            
            # Basic data validation
            if 'Close' in df.columns:
                if (df['Close'] <= 0).any():
                    logger.warning(f"Invalid price data for {symbol}")
                    df = df[df['Close'] > 0]
            
            # Add technical indicators if needed
            if 'Close' in df.columns:
                df['Returns'] = df['Close'].pct_change()
                df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            cleaned_data[symbol] = df
        
        print(f"✅ EXITING _validate_and_clean_data() at {datetime.now().strftime('%H:%M:%S')}")
        return cleaned_data
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if not self.cache:
            return {'cache_enabled': False}
        
        stats = self.cache.get_stats()
        stats['cache_enabled'] = True
        return stats
    
    def clear_cache(self):
        """Clear all cached data."""
        if self.cache:
            expired = self.cache.clear_expired()
            logger.info(f"Cleared {expired} expired cache entries")
    
    def refresh_provider_status(self):
        """Refresh provider availability status."""
        for name, provider in self.providers.items():
            self.available_providers[name] = provider.is_available()
            status = "✅ Available" if self.available_providers[name] else "❌ Unavailable"
            logger.info(f"Provider {name}: {status}")

# Backward compatibility with existing code
def load_or_download_data(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Backward compatibility function."""
    print(f"🔄 ENTERING load_or_download_data({symbols}, {start_date}, {end_date}) at {datetime.now().strftime('%H:%M:%S')}")
    
    data_manager = DataManager()
    result = data_manager.get_pairs_data(symbols, start_date, end_date)
    
    print(f"✅ EXITING load_or_download_data() at {datetime.now().strftime('%H:%M:%S')}")
    return result
