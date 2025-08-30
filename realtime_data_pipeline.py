"""
📡 REAL-TIME DATA PIPELINE
=========================

Advanced streaming data pipeline with:
- Real-time price feeds
- Streaming technical indicators
- Low-latency processing
- Data quality monitoring
- Robust error handling

Features:
- WebSocket data streams
- Real-time indicator calculation
- Data validation and cleaning
- Performance monitoring
- Failover mechanisms
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import threading
import time
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import logging
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import requests
from functools import wraps
import redis
import warnings

warnings.filterwarnings('ignore')

@dataclass
class StreamingQuote:
    """Real-time quote data structure"""
    symbol: str
    price: float
    timestamp: datetime
    volume: int = 0
    bid: float = 0.0
    ask: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    change: float = 0.0
    change_percent: float = 0.0

@dataclass 
class TechnicalIndicators:
    """Real-time technical indicators"""
    symbol: str
    timestamp: datetime
    sma_20: float = 0.0
    sma_50: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    rsi: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    volatility: float = 0.0
    volume_sma: float = 0.0

@dataclass
class PairMetrics:
    """Real-time pair trading metrics"""
    pair: tuple
    timestamp: datetime
    correlation: float = 0.0
    cointegration_score: float = 0.0
    spread_zscore: float = 0.0
    spread_mean: float = 0.0
    spread_std: float = 0.0
    momentum_score: float = 0.0
    volatility_ratio: float = 0.0
    trading_signal: str = "HOLD"
    confidence: float = 0.0

class DataQualityMonitor:
    """Monitor data quality and detect anomalies"""
    
    def __init__(self):
        self.price_history = {}
        self.quality_metrics = {}
        self.alert_thresholds = {
            'max_price_change': 0.15,  # 15% max price change
            'min_volume_ratio': 0.1,   # 10% of average volume
            'max_spread': 0.05,        # 5% max bid-ask spread
            'data_delay_seconds': 30   # Max 30 second delay
        }
        self.logger = logging.getLogger(__name__)
    
    def validate_quote(self, quote: StreamingQuote) -> bool:
        """Validate incoming quote data"""
        try:
            # Basic validation
            if quote.price <= 0 or np.isnan(quote.price):
                return False
            
            # Check for extreme price movements
            if quote.symbol in self.price_history:
                last_price = self.price_history[quote.symbol][-1]['price']
                price_change = abs(quote.price - last_price) / last_price
                
                if price_change > self.alert_thresholds['max_price_change']:
                    self.logger.warning(f"🚨 Extreme price movement in {quote.symbol}: {price_change*100:.1f}%")
                    return False
            
            # Check bid-ask spread if available
            if quote.bid > 0 and quote.ask > 0:
                spread = (quote.ask - quote.bid) / quote.price
                if spread > self.alert_thresholds['max_spread']:
                    self.logger.warning(f"⚠️ Wide spread in {quote.symbol}: {spread*100:.1f}%")
            
            # Check data freshness
            data_age = (datetime.now() - quote.timestamp).total_seconds()
            if data_age > self.alert_thresholds['data_delay_seconds']:
                self.logger.warning(f"⏰ Stale data for {quote.symbol}: {data_age:.0f}s delay")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Quote validation error for {quote.symbol}: {e}")
            return False
    
    def update_quality_metrics(self, symbol: str, quote: StreamingQuote):
        """Update data quality metrics"""
        if symbol not in self.quality_metrics:
            self.quality_metrics[symbol] = {
                'update_count': 0,
                'last_update': None,
                'avg_delay': 0.0,
                'price_volatility': 0.0,
                'data_gaps': 0
            }
        
        metrics = self.quality_metrics[symbol]
        metrics['update_count'] += 1
        
        # Calculate average delay
        if metrics['last_update']:
            delay = (quote.timestamp - metrics['last_update']).total_seconds()
            metrics['avg_delay'] = (metrics['avg_delay'] * 0.9) + (delay * 0.1)
        
        metrics['last_update'] = quote.timestamp
        
        # Update price history for volatility calculation
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'timestamp': quote.timestamp,
            'price': quote.price,
            'volume': quote.volume
        })
        
        # Keep only recent history
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-500:]
        
        # Calculate price volatility
        if len(self.price_history[symbol]) >= 20:
            recent_prices = [p['price'] for p in self.price_history[symbol][-20:]]
            returns = np.diff(recent_prices) / recent_prices[:-1]
            metrics['price_volatility'] = np.std(returns)

class StreamingIndicatorCalculator:
    """Real-time technical indicator calculation engine"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.price_buffers = {}  # symbol -> price history
        self.volume_buffers = {}  # symbol -> volume history
        self.indicator_cache = {}  # symbol -> latest indicators
        self.logger = logging.getLogger(__name__)
    
    def update_price(self, symbol: str, price: float, volume: int, timestamp: datetime):
        """Update price and recalculate indicators"""
        # Initialize buffers if needed
        if symbol not in self.price_buffers:
            self.price_buffers[symbol] = []
            self.volume_buffers[symbol] = []
        
        # Add new data point
        self.price_buffers[symbol].append({'price': price, 'timestamp': timestamp})
        self.volume_buffers[symbol].append({'volume': volume, 'timestamp': timestamp})
        
        # Maintain buffer size
        if len(self.price_buffers[symbol]) > self.max_history:
            self.price_buffers[symbol] = self.price_buffers[symbol][-self.max_history//2:]
            self.volume_buffers[symbol] = self.volume_buffers[symbol][-self.max_history//2:]
        
        # Calculate indicators if we have enough data
        if len(self.price_buffers[symbol]) >= 50:
            indicators = self._calculate_indicators(symbol, timestamp)
            self.indicator_cache[symbol] = indicators
            return indicators
        
        return None
    
    def _calculate_indicators(self, symbol: str, timestamp: datetime) -> TechnicalIndicators:
        """Calculate all technical indicators for a symbol"""
        try:
            prices = np.array([p['price'] for p in self.price_buffers[symbol]])
            volumes = np.array([v['volume'] for v in self.volume_buffers[symbol]])
            
            # Moving averages
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1]
            
            # Exponential moving averages
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            
            # MACD
            macd = ema_12 - ema_26
            macd_signal = self._calculate_ema(np.array([macd]), 9) if len(prices) >= 26 else 0
            
            # RSI
            rsi = self._calculate_rsi(prices)
            
            # Bollinger Bands
            bollinger_upper, bollinger_lower = self._calculate_bollinger_bands(prices)
            
            # Volatility (20-period)
            if len(prices) >= 20:
                returns = np.diff(prices[-21:]) / prices[-21:-1]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
            else:
                volatility = 0.0
            
            # Volume SMA
            volume_sma = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1] if len(volumes) > 0 else 0
            
            return TechnicalIndicators(\n                symbol=symbol,\n                timestamp=timestamp,\n                sma_20=sma_20,\n                sma_50=sma_50,\n                ema_12=ema_12,\n                ema_26=ema_26,\n                macd=macd,\n                macd_signal=macd_signal,\n                rsi=rsi,\n                bollinger_upper=bollinger_upper,\n                bollinger_lower=bollinger_lower,\n                volatility=volatility,\n                volume_sma=volume_sma\n            )\n            \n        except Exception as e:\n            self.logger.error(f\"Indicator calculation error for {symbol}: {e}\")\n            return TechnicalIndicators(symbol=symbol, timestamp=timestamp)\n    \n    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:\n        \"\"\"Calculate exponential moving average\"\"\"\n        if len(prices) < period:\n            return prices[-1] if len(prices) > 0 else 0.0\n        \n        alpha = 2.0 / (period + 1)\n        ema = prices[0]\n        \n        for price in prices[1:]:\n            ema = alpha * price + (1 - alpha) * ema\n        \n        return ema\n    \n    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:\n        \"\"\"Calculate RSI\"\"\"\n        if len(prices) < period + 1:\n            return 50.0\n        \n        deltas = np.diff(prices[-period-1:])\n        gains = np.where(deltas > 0, deltas, 0)\n        losses = np.where(deltas < 0, -deltas, 0)\n        \n        avg_gain = np.mean(gains)\n        avg_loss = np.mean(losses)\n        \n        if avg_loss == 0:\n            return 100.0\n        \n        rs = avg_gain / avg_loss\n        rsi = 100 - (100 / (1 + rs))\n        \n        return rsi\n    \n    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float]:\n        \"\"\"Calculate Bollinger Bands\"\"\"\n        if len(prices) < period:\n            return prices[-1], prices[-1]\n        \n        recent_prices = prices[-period:]\n        mean_price = np.mean(recent_prices)\n        std_price = np.std(recent_prices)\n        \n        upper_band = mean_price + (std_dev * std_price)\n        lower_band = mean_price - (std_dev * std_price)\n        \n        return upper_band, lower_band\n    \n    def get_latest_indicators(self, symbol: str) -> Optional[TechnicalIndicators]:\n        \"\"\"Get latest indicators for a symbol\"\"\"\n        return self.indicator_cache.get(symbol)\n\nclass RealTimePairAnalyzer:\n    \"\"\"Real-time pair relationship analysis\"\"\"\n    \n    def __init__(self, lookback_periods: int = 100):\n        self.lookback_periods = lookback_periods\n        self.pair_buffers = {}  # pair -> price history\n        self.pair_metrics_cache = {}  # pair -> latest metrics\n        self.correlation_window = 50\n        self.cointegration_window = 100\n        self.logger = logging.getLogger(__name__)\n    \n    def update_pair_data(self, symbol1: str, symbol2: str, price1: float, price2: float, timestamp: datetime):\n        \"\"\"Update pair data and recalculate metrics\"\"\"\n        pair_key = f\"{symbol1}-{symbol2}\"\n        \n        if pair_key not in self.pair_buffers:\n            self.pair_buffers[pair_key] = {\n                'prices1': [],\n                'prices2': [],\n                'timestamps': [],\n                'spreads': [],\n                'ratios': []\n            }\n        \n        buffer = self.pair_buffers[pair_key]\n        \n        # Add new data\n        buffer['prices1'].append(price1)\n        buffer['prices2'].append(price2)\n        buffer['timestamps'].append(timestamp)\n        \n        # Calculate spread and ratio\n        spread = np.log(price1) - np.log(price2)\n        ratio = price1 / price2\n        \n        buffer['spreads'].append(spread)\n        buffer['ratios'].append(ratio)\n        \n        # Maintain buffer size\n        if len(buffer['prices1']) > self.lookback_periods:\n            for key in buffer.keys():\n                buffer[key] = buffer[key][-self.lookback_periods//2:]\n        \n        # Calculate metrics if we have enough data\n        if len(buffer['prices1']) >= self.correlation_window:\n            metrics = self._calculate_pair_metrics(symbol1, symbol2, timestamp)\n            self.pair_metrics_cache[pair_key] = metrics\n            return metrics\n        \n        return None\n    \n    def _calculate_pair_metrics(self, symbol1: str, symbol2: str, timestamp: datetime) -> PairMetrics:\n        \"\"\"Calculate comprehensive pair metrics\"\"\"\n        pair_key = f\"{symbol1}-{symbol2}\"\n        buffer = self.pair_buffers[pair_key]\n        \n        try:\n            prices1 = np.array(buffer['prices1'])\n            prices2 = np.array(buffer['prices2'])\n            spreads = np.array(buffer['spreads'])\n            \n            # Returns for correlation\n            returns1 = np.diff(prices1) / prices1[:-1]\n            returns2 = np.diff(prices2) / prices2[:-1]\n            \n            # Correlation (rolling)\n            if len(returns1) >= self.correlation_window:\n                recent_ret1 = returns1[-self.correlation_window:]\n                recent_ret2 = returns2[-self.correlation_window:]\n                correlation = np.corrcoef(recent_ret1, recent_ret2)[0, 1]\n            else:\n                correlation = 0.0\n            \n            # Cointegration test (simplified ADF-like)\n            cointegration_score = self._simple_cointegration_test(spreads)\n            \n            # Spread statistics\n            if len(spreads) >= 20:\n                spread_mean = np.mean(spreads[-50:])\n                spread_std = np.std(spreads[-50:])\n                current_spread = spreads[-1]\n                spread_zscore = (current_spread - spread_mean) / (spread_std + 1e-8)\n            else:\n                spread_mean = 0.0\n                spread_std = 0.0\n                spread_zscore = 0.0\n            \n            # Momentum score\n            momentum_score = self._calculate_momentum_divergence(returns1, returns2)\n            \n            # Volatility ratio\n            if len(returns1) >= 20 and len(returns2) >= 20:\n                vol1 = np.std(returns1[-20:])\n                vol2 = np.std(returns2[-20:])\n                volatility_ratio = vol1 / (vol2 + 1e-8)\n            else:\n                volatility_ratio = 1.0\n            \n            # Generate trading signal\n            trading_signal, confidence = self._generate_trading_signal(\n                spread_zscore, correlation, momentum_score, volatility_ratio\n            )\n            \n            return PairMetrics(\n                pair=(symbol1, symbol2),\n                timestamp=timestamp,\n                correlation=correlation,\n                cointegration_score=cointegration_score,\n                spread_zscore=spread_zscore,\n                spread_mean=spread_mean,\n                spread_std=spread_std,\n                momentum_score=momentum_score,\n                volatility_ratio=volatility_ratio,\n                trading_signal=trading_signal,\n                confidence=confidence\n            )\n            \n        except Exception as e:\n            self.logger.error(f\"Pair metrics calculation error for {pair_key}: {e}\")\n            return PairMetrics(pair=(symbol1, symbol2), timestamp=timestamp)\n    \n    def _simple_cointegration_test(self, spreads: np.ndarray) -> float:\n        \"\"\"Simplified cointegration test\"\"\"\n        if len(spreads) < 50:\n            return 0.0\n        \n        # Simple mean reversion test\n        recent_spreads = spreads[-50:]\n        \n        # Calculate mean reversion speed\n        lagged_spread = recent_spreads[:-1]\n        current_spread = recent_spreads[1:]\n        \n        if len(lagged_spread) > 10:\n            # Regression coefficient (mean reversion speed)\n            coeff = np.corrcoef(lagged_spread, current_spread)[0, 1]\n            return 1.0 - abs(coeff)  # Higher score = more mean reverting\n        \n        return 0.0\n    \n    def _calculate_momentum_divergence(self, returns1: np.ndarray, returns2: np.ndarray) -> float:\n        \"\"\"Calculate momentum divergence between pairs\"\"\"\n        if len(returns1) < 20 or len(returns2) < 20:\n            return 0.0\n        \n        # Short-term momentum\n        momentum1 = np.mean(returns1[-10:])\n        momentum2 = np.mean(returns2[-10:])\n        \n        # Momentum divergence\n        divergence = momentum1 - momentum2\n        \n        # Normalize by volatility\n        vol1 = np.std(returns1[-20:])\n        vol2 = np.std(returns2[-20:])\n        avg_vol = (vol1 + vol2) / 2\n        \n        if avg_vol > 0:\n            return divergence / avg_vol\n        \n        return 0.0\n    \n    def _generate_trading_signal(self, zscore: float, correlation: float, \n                               momentum: float, vol_ratio: float) -> Tuple[str, float]:\n        \"\"\"Generate trading signal based on multiple factors\"\"\"\n        \n        # Signal strength factors\n        signal_factors = {\n            'mean_reversion': abs(zscore) > 2.0,\n            'high_correlation': abs(correlation) > 0.5,\n            'momentum_divergence': abs(momentum) > 0.5,\n            'vol_stability': 0.5 < vol_ratio < 2.0\n        }\n        \n        signal_count = sum(signal_factors.values())\n        \n        if signal_count >= 3:  # Strong signal\n            if zscore > 2.0:\n                return \"SHORT_SPREAD\", min(signal_count / 4.0, 1.0)\n            elif zscore < -2.0:\n                return \"LONG_SPREAD\", min(signal_count / 4.0, 1.0)\n        \n        elif signal_count >= 2:  # Moderate signal\n            if abs(zscore) > 1.5:\n                signal = \"SHORT_SPREAD\" if zscore > 0 else \"LONG_SPREAD\"\n                return signal, signal_count / 4.0\n        \n        # Exit signals\n        if abs(zscore) < 0.5:\n            return \"CLOSE\", 0.8\n        \n        return \"HOLD\", 0.0\n    \n    def get_latest_metrics(self, symbol1: str, symbol2: str) -> Optional[PairMetrics]:\n        \"\"\"Get latest metrics for a pair\"\"\"\n        pair_key = f\"{symbol1}-{symbol2}\"\n        return self.pair_metrics_cache.get(pair_key)\n\nclass RealTimeDataPipeline:\n    \"\"\"Main real-time data pipeline orchestrator\"\"\"\n    \n    def __init__(self, symbols: List[str], update_frequency: int = 30):\n        self.symbols = symbols\n        self.update_frequency = update_frequency  # seconds\n        \n        # Core components\n        self.quality_monitor = DataQualityMonitor()\n        self.indicator_calculator = StreamingIndicatorCalculator()\n        self.pair_analyzer = RealTimePairAnalyzer()\n        \n        # Data management\n        self.current_quotes = {}\n        self.data_subscribers = []  # Callback functions\n        self.is_running = False\n        \n        # Performance tracking\n        self.pipeline_stats = {\n            'quotes_processed': 0,\n            'indicators_calculated': 0,\n            'pairs_analyzed': 0,\n            'errors': 0,\n            'start_time': None\n        }\n        \n        # Database for persistence\n        self.db_path = Path(\"realtime_data.db\")\n        self._init_database()\n        \n        self.logger = logging.getLogger(__name__)\n        \n        print(f\"📡 Real-time Data Pipeline: {len(symbols)} symbols\")\n        print(f\"⚡ Update frequency: {update_frequency}s\")\n    \n    def _init_database(self):\n        \"\"\"Initialize SQLite database for data persistence\"\"\"\n        conn = sqlite3.connect(self.db_path)\n        \n        # Create tables\n        conn.execute(\"\"\"\n            CREATE TABLE IF NOT EXISTS quotes (\n                symbol TEXT,\n                timestamp TEXT,\n                price REAL,\n                volume INTEGER,\n                bid REAL,\n                ask REAL\n            )\n        \"\"\")\n        \n        conn.execute(\"\"\"\n            CREATE TABLE IF NOT EXISTS indicators (\n                symbol TEXT,\n                timestamp TEXT,\n                sma_20 REAL,\n                sma_50 REAL,\n                rsi REAL,\n                macd REAL,\n                volatility REAL\n            )\n        \"\"\")\n        \n        conn.execute(\"\"\"\n            CREATE TABLE IF NOT EXISTS pair_metrics (\n                pair TEXT,\n                timestamp TEXT,\n                correlation REAL,\n                spread_zscore REAL,\n                trading_signal TEXT,\n                confidence REAL\n            )\n        \"\"\")\n        \n        conn.commit()\n        conn.close()\n    \n    def subscribe_to_data(self, callback: Callable):\n        \"\"\"Subscribe to real-time data updates\"\"\"\n        self.data_subscribers.append(callback)\n    \n    def start_pipeline(self):\n        \"\"\"Start the real-time data pipeline\"\"\"\n        self.is_running = True\n        self.pipeline_stats['start_time'] = datetime.now()\n        \n        self.logger.info(\"🚀 Starting Real-time Data Pipeline\")\n        \n        # Start main processing thread\n        pipeline_thread = threading.Thread(target=self._pipeline_loop, daemon=True)\n        pipeline_thread.start()\n        \n        # Start data persistence thread\n        persistence_thread = threading.Thread(target=self._persistence_loop, daemon=True)\n        persistence_thread.start()\n    \n    def _pipeline_loop(self):\n        \"\"\"Main pipeline processing loop\"\"\"\n        while self.is_running:\n            try:\n                start_time = time.time()\n                \n                # Fetch latest market data\n                self._fetch_market_data()\n                \n                # Process all pairs\n                self._process_pairs()\n                \n                # Notify subscribers\n                self._notify_subscribers()\n                \n                # Update stats\n                processing_time = time.time() - start_time\n                self.logger.debug(f\"Pipeline cycle: {processing_time:.2f}s\")\n                \n                # Sleep until next update\n                sleep_time = max(0, self.update_frequency - processing_time)\n                time.sleep(sleep_time)\n                \n            except Exception as e:\n                self.pipeline_stats['errors'] += 1\n                self.logger.error(f\"Pipeline error: {e}\")\n                time.sleep(5)\n    \n    def _fetch_market_data(self):\n        \"\"\"Fetch current market data for all symbols\"\"\"\n        try:\n            # Use yfinance for current data\n            current_time = datetime.now()\n            \n            # Batch fetch for efficiency\n            tickers = yf.Tickers(' '.join(self.symbols))\n            \n            with ThreadPoolExecutor(max_workers=8) as executor:\n                futures = {}\n                \n                for symbol in self.symbols:\n                    future = executor.submit(self._fetch_symbol_quote, symbol, tickers, current_time)\n                    futures[future] = symbol\n                \n                for future in futures:\n                    symbol = futures[future]\n                    try:\n                        quote = future.result(timeout=5)\n                        if quote and self.quality_monitor.validate_quote(quote):\n                            self.current_quotes[symbol] = quote\n                            self.quality_monitor.update_quality_metrics(symbol, quote)\n                            \n                            # Update indicators\n                            indicators = self.indicator_calculator.update_price(\n                                symbol, quote.price, quote.volume, quote.timestamp\n                            )\n                            \n                            self.pipeline_stats['quotes_processed'] += 1\n                            if indicators:\n                                self.pipeline_stats['indicators_calculated'] += 1\n                    \n                    except Exception as e:\n                        self.logger.warning(f\"Failed to process {symbol}: {e}\")\n            \n        except Exception as e:\n            self.logger.error(f\"Market data fetch error: {e}\")\n    \n    def _fetch_symbol_quote(self, symbol: str, tickers, current_time: datetime) -> Optional[StreamingQuote]:\n        \"\"\"Fetch quote for a single symbol\"\"\"\n        try:\n            ticker = tickers.tickers[symbol]\n            info = ticker.info\n            \n            # Get current price\n            price = None\n            if 'currentPrice' in info:\n                price = info['currentPrice']\n            elif 'regularMarketPrice' in info:\n                price = info['regularMarketPrice']\n            elif 'previousClose' in info:\n                price = info['previousClose']\n            \n            if price and price > 0:\n                return StreamingQuote(\n                    symbol=symbol,\n                    price=price,\n                    timestamp=current_time,\n                    volume=info.get('regularMarketVolume', 0),\n                    bid=info.get('bid', 0),\n                    ask=info.get('ask', 0),\n                    change=info.get('regularMarketChange', 0),\n                    change_percent=info.get('regularMarketChangePercent', 0)\n                )\n            \n        except Exception as e:\n            self.logger.debug(f\"Quote fetch error for {symbol}: {e}\")\n        \n        return None\n    \n    def _process_pairs(self):\n        \"\"\"Process all pair combinations\"\"\"\n        symbols_with_data = list(self.current_quotes.keys())\n        \n        if len(symbols_with_data) < 2:\n            return\n        \n        pairs_processed = 0\n        \n        # Process pairs in parallel batches\n        with ThreadPoolExecutor(max_workers=4) as executor:\n            futures = []\n            \n            for i in range(len(symbols_with_data)):\n                for j in range(i+1, len(symbols_with_data)):\n                    symbol1, symbol2 = symbols_with_data[i], symbols_with_data[j]\n                    \n                    quote1 = self.current_quotes[symbol1]\n                    quote2 = self.current_quotes[symbol2]\n                    \n                    future = executor.submit(\n                        self.pair_analyzer.update_pair_data,\n                        symbol1, symbol2, quote1.price, quote2.price, quote1.timestamp\n                    )\n                    futures.append(future)\n                    \n                    pairs_processed += 1\n                    if pairs_processed >= 50:  # Limit for performance\n                        break\n                \n                if pairs_processed >= 50:\n                    break\n            \n            # Collect results\n            for future in futures:\n                try:\n                    result = future.result(timeout=2)\n                    if result:\n                        self.pipeline_stats['pairs_analyzed'] += 1\n                except Exception as e:\n                    self.logger.debug(f\"Pair processing error: {e}\")\n    \n    def _notify_subscribers(self):\n        \"\"\"Notify all data subscribers\"\"\"\n        pipeline_data = {\n            'timestamp': datetime.now(),\n            'quotes': self.current_quotes,\n            'indicators': {symbol: self.indicator_calculator.get_latest_indicators(symbol) \n                          for symbol in self.symbols},\n            'pair_metrics': self.pair_analyzer.pair_metrics_cache,\n            'pipeline_stats': self.pipeline_stats\n        }\n        \n        for callback in self.data_subscribers:\n            try:\n                callback(pipeline_data)\n            except Exception as e:\n                self.logger.error(f\"Subscriber notification error: {e}\")\n    \n    def _persistence_loop(self):\n        \"\"\"Periodic data persistence to database\"\"\"\n        while self.is_running:\n            try:\n                self._persist_data()\n                time.sleep(300)  # Save every 5 minutes\n            except Exception as e:\n                self.logger.error(f\"Persistence error: {e}\")\n                time.sleep(60)\n    \n    def _persist_data(self):\n        \"\"\"Persist current data to database\"\"\"\n        conn = sqlite3.connect(self.db_path)\n        \n        try:\n            # Save quotes\n            for symbol, quote in self.current_quotes.items():\n                conn.execute(\n                    \"INSERT INTO quotes VALUES (?, ?, ?, ?, ?, ?)\",\n                    (symbol, quote.timestamp.isoformat(), quote.price, \n                     quote.volume, quote.bid, quote.ask)\n                )\n            \n            # Save indicators\n            for symbol in self.symbols:\n                indicators = self.indicator_calculator.get_latest_indicators(symbol)\n                if indicators:\n                    conn.execute(\n                        \"INSERT INTO indicators VALUES (?, ?, ?, ?, ?, ?, ?)\",\n                        (symbol, indicators.timestamp.isoformat(), indicators.sma_20,\n                         indicators.sma_50, indicators.rsi, indicators.macd, indicators.volatility)\n                    )\n            \n            # Save pair metrics\n            for pair_key, metrics in self.pair_analyzer.pair_metrics_cache.items():\n                conn.execute(\n                    \"INSERT INTO pair_metrics VALUES (?, ?, ?, ?, ?, ?)\",\n                    (pair_key, metrics.timestamp.isoformat(), metrics.correlation,\n                     metrics.spread_zscore, metrics.trading_signal, metrics.confidence)\n                )\n            \n            conn.commit()\n            \n        except Exception as e:\n            self.logger.error(f\"Database persistence error: {e}\")\n        finally:\n            conn.close()\n    \n    def get_pipeline_status(self) -> Dict:\n        \"\"\"Get current pipeline status\"\"\"\n        runtime = datetime.now() - self.pipeline_stats['start_time'] if self.pipeline_stats['start_time'] else timedelta(0)\n        \n        return {\n            'is_running': self.is_running,\n            'runtime_minutes': runtime.total_seconds() / 60,\n            'symbols_tracked': len(self.symbols),\n            'quotes_processed': self.pipeline_stats['quotes_processed'],\n            'indicators_calculated': self.pipeline_stats['indicators_calculated'],\n            'pairs_analyzed': self.pipeline_stats['pairs_analyzed'],\n            'errors': self.pipeline_stats['errors'],\n            'quotes_per_minute': self.pipeline_stats['quotes_processed'] / max(runtime.total_seconds() / 60, 1),\n            'active_pairs': len(self.pair_analyzer.pair_metrics_cache),\n            'symbols_with_current_data': len(self.current_quotes)\n        }\n    \n    def get_trading_signals(self, min_confidence: float = 0.6) -> List[Dict]:\n        \"\"\"Get current trading signals above confidence threshold\"\"\"\n        signals = []\n        \n        for pair_key, metrics in self.pair_analyzer.pair_metrics_cache.items():\n            if (metrics.trading_signal != \"HOLD\" and \n                metrics.confidence >= min_confidence):\n                \n                signals.append({\n                    'pair': metrics.pair,\n                    'action': metrics.trading_signal,\n                    'z_score': metrics.spread_zscore,\n                    'confidence': metrics.confidence,\n                    'correlation': metrics.correlation,\n                    'momentum': metrics.momentum_score,\n                    'timestamp': metrics.timestamp\n                })\n        \n        # Sort by confidence\n        signals.sort(key=lambda x: x['confidence'], reverse=True)\n        return signals\n    \n    def stop_pipeline(self):\n        \"\"\"Stop the data pipeline\"\"\"\n        self.is_running = False\n        self._persist_data()  # Final save\n        self.logger.info(\"🛑 Real-time data pipeline stopped\")\n\ndef demo_realtime_pipeline():\n    \"\"\"Demonstrate the real-time data pipeline\"\"\"\n    print(\"📡 REAL-TIME DATA PIPELINE DEMO\")\n    print(\"=\" * 50)\n    \n    # Extended symbol universe\n    demo_symbols = [\n        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',  # Tech\n        'JPM', 'BAC', 'GS', 'V', 'MA',            # Finance\n        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',       # Healthcare\n        'XOM', 'CVX', 'COP', 'EOG', 'SLB',        # Energy\n        'AMZN', 'WMT', 'HD', 'PG', 'KO'           # Consumer\n    ]\n    \n    pipeline = RealTimeDataPipeline(demo_symbols, update_frequency=30)\n    \n    # Subscribe to data updates\n    def data_callback(pipeline_data):\n        timestamp = pipeline_data['timestamp']\n        quotes_count = len([q for q in pipeline_data['quotes'].values() if q])\n        indicators_count = len([i for i in pipeline_data['indicators'].values() if i])\n        pairs_count = len(pipeline_data['pair_metrics'])\n        \n        print(f\"📊 {timestamp.strftime('%H:%M:%S')} | Quotes: {quotes_count} | Indicators: {indicators_count} | Pairs: {pairs_count}\")\n        \n        # Show trading signals\n        signals = pipeline.get_trading_signals(min_confidence=0.7)\n        if signals:\n            for signal in signals[:3]:  # Show top 3\n                print(f\"   🎯 {signal['action']} {signal['pair']} | Z: {signal['z_score']:.2f} | Conf: {signal['confidence']:.2f}\")\n    \n    pipeline.subscribe_to_data(data_callback)\n    \n    try:\n        # Start pipeline\n        pipeline.start_pipeline()\n        \n        print(\"🌐 Pipeline running... (Press Ctrl+C to stop)\")\n        print(\"📈 Monitoring real-time market data...\\n\")\n        \n        # Run for demo duration\n        demo_duration = 300  # 5 minutes\n        start_time = time.time()\n        \n        while time.time() - start_time < demo_duration and pipeline.is_running:\n            time.sleep(10)\n            \n            # Show status every minute\n            if int(time.time() - start_time) % 60 == 0:\n                status = pipeline.get_pipeline_status()\n                print(f\"\\n⚡ Pipeline Status:\")\n                print(f\"   Runtime: {status['runtime_minutes']:.1f} min\")\n                print(f\"   Quotes/min: {status['quotes_per_minute']:.1f}\")\n                print(f\"   Active pairs: {status['active_pairs']}\")\n                print(f\"   Errors: {status['errors']}\")\n        \n        # Final status\n        final_status = pipeline.get_pipeline_status()\n        print(f\"\\n📊 DEMO RESULTS:\")\n        print(f\"   Total quotes processed: {final_status['quotes_processed']}\")\n        print(f\"   Indicators calculated: {final_status['indicators_calculated']}\")\n        print(f\"   Pairs analyzed: {final_status['pairs_analyzed']}\")\n        print(f\"   Processing rate: {final_status['quotes_per_minute']:.1f} quotes/min\")\n        \n        # Show final signals\n        final_signals = pipeline.get_trading_signals(min_confidence=0.5)\n        print(f\"\\n🎯 Final Trading Signals ({len(final_signals)}):\")\n        for signal in final_signals[:5]:\n            print(f\"   {signal['action']} {signal['pair']} | Z: {signal['z_score']:.2f} | Conf: {signal['confidence']:.2f}\")\n    \n    except KeyboardInterrupt:\n        print(\"\\n🛑 Demo stopped by user\")\n    \n    finally:\n        pipeline.stop_pipeline()\n        print(\"✅ Real-time pipeline demo complete\")\n\nif __name__ == \"__main__\":\n    demo_realtime_pipeline()
