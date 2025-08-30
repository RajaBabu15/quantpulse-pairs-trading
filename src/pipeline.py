
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
from typing import Dict, List, Optional, Callable, Any, Tuple
import logging
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import requests
from functools import wraps
import warnings

warnings.filterwarnings('ignore')

@dataclass
class StreamingQuote:
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
    
    def __init__(self):
        self.price_history = {}
        self.quality_metrics = {}
        self.alert_thresholds = {
            'max_price_change': 0.15,
            'min_volume_ratio': 0.1,
            'max_spread': 0.05,
            'data_delay_seconds': 30
        }
        self.logger = logging.getLogger(__name__)
    
    def validate_quote(self, quote: StreamingQuote) -> bool:
        try:
            if quote.price <= 0 or np.isnan(quote.price):
                return False
            
            if quote.symbol in self.price_history:
                last_price = self.price_history[quote.symbol][-1]['price']
                price_change = abs(quote.price - last_price) / last_price
                
                if price_change > self.alert_thresholds['max_price_change']:
                    self.logger.warning(f"🚨 Extreme price movement in {quote.symbol}: {price_change*100:.1f}%")
                    return False
            
            if quote.bid > 0 and quote.ask > 0:
                spread = (quote.ask - quote.bid) / quote.price
                if spread > self.alert_thresholds['max_spread']:
                    self.logger.warning(f"⚠️ Wide spread in {quote.symbol}: {spread*100:.1f}%")
            
            data_age = (datetime.now() - quote.timestamp).total_seconds()
            if data_age > self.alert_thresholds['data_delay_seconds']:
                self.logger.warning(f"⏰ Stale data for {quote.symbol}: {data_age:.0f}s delay")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Quote validation error for {quote.symbol}: {e}")
            return False
    
    def update_quality_metrics(self, symbol: str, quote: StreamingQuote):
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
        
        if metrics['last_update']:
            delay = (quote.timestamp - metrics['last_update']).total_seconds()
            metrics['avg_delay'] = (metrics['avg_delay'] * 0.9) + (delay * 0.1)
        
        metrics['last_update'] = quote.timestamp
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'timestamp': quote.timestamp,
            'price': quote.price,
            'volume': quote.volume
        })
        
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-500:]
        
        if len(self.price_history[symbol]) >= 20:
            recent_prices = [p['price'] for p in self.price_history[symbol][-20:]]
            returns = np.diff(recent_prices) / recent_prices[:-1]
            metrics['price_volatility'] = np.std(returns)

class StreamingIndicatorCalculator:
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.price_buffers = {}
        self.volume_buffers = {}
        self.indicator_cache = {}
        self.logger = logging.getLogger(__name__)
    
    def update_price(self, symbol: str, price: float, volume: int, timestamp: datetime):
        if symbol not in self.price_buffers:
            self.price_buffers[symbol] = []
            self.volume_buffers[symbol] = []
        
        self.price_buffers[symbol].append({'price': price, 'timestamp': timestamp})
        self.volume_buffers[symbol].append({'volume': volume, 'timestamp': timestamp})
        
        if len(self.price_buffers[symbol]) > self.max_history:
            self.price_buffers[symbol] = self.price_buffers[symbol][-self.max_history//2:]
            self.volume_buffers[symbol] = self.volume_buffers[symbol][-self.max_history//2:]
        
        if len(self.price_buffers[symbol]) >= 50:
            indicators = self._calculate_indicators(symbol, timestamp)
            self.indicator_cache[symbol] = indicators
            return indicators
        
        return None
    
    def _calculate_indicators(self, symbol: str, timestamp: datetime) -> TechnicalIndicators:
        try:
            prices = np.array([p['price'] for p in self.price_buffers[symbol]])
            volumes = np.array([v['volume'] for v in self.volume_buffers[symbol]])
            
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1]
            
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            
            macd = ema_12 - ema_26
            macd_signal = self._calculate_ema(np.array([macd]), 9) if len(prices) >= 26 else 0
            
            rsi = self._calculate_rsi(prices)
            
            bollinger_upper, bollinger_lower = self._calculate_bollinger_bands(prices)
            
            if len(prices) >= 20:
                returns = np.diff(prices[-21:]) / prices[-21:-1]
                volatility = np.std(returns) * np.sqrt(252)
            else:
                volatility = 0.0
            
            volume_sma = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1] if len(volumes) > 0 else 0
            
            return TechnicalIndicators(
                symbol=symbol,
                timestamp=timestamp,
                sma_20=sma_20,
                sma_50=sma_50,
                ema_12=ema_12,
                ema_26=ema_26,
                macd=macd,
                macd_signal=macd_signal,
                rsi=rsi,
                bollinger_upper=bollinger_upper,
                bollinger_lower=bollinger_lower,
                volatility=volatility,
                volume_sma=volume_sma
            )
            
        except Exception as e:
            self.logger.error(f"Indicator calculation error for {symbol}: {e}")
            return TechnicalIndicators(symbol=symbol, timestamp=timestamp)
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0.0
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float]:
        if len(prices) < period:
            return prices[-1], prices[-1]
        
        recent_prices = prices[-period:]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        upper_band = mean_price + (std_dev * std_price)
        lower_band = mean_price - (std_dev * std_price)
        
        return upper_band, lower_band
    
    def get_latest_indicators(self, symbol: str) -> Optional[TechnicalIndicators]:
        return self.indicator_cache.get(symbol)

class RealTimePairAnalyzer:
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.pair_buffers = {}
        self.pair_metrics_cache = {}
        self.correlation_window = 50
        self.cointegration_window = 100
        self.logger = logging.getLogger(__name__)
    
    def update_pair_data(self, symbol1: str, symbol2: str, price1: float, price2: float, timestamp: datetime):
        pair_key = f"{symbol1}-{symbol2}"
        
        if pair_key not in self.pair_buffers:
            self.pair_buffers[pair_key] = {
                'prices1': [],
                'prices2': [],
                'timestamps': [],
                'spreads': [],
                'ratios': []
            }
        
        buffer = self.pair_buffers[pair_key]
        
        buffer['prices1'].append(price1)
        buffer['prices2'].append(price2)
        buffer['timestamps'].append(timestamp)
        
        spread = np.log(price1) - np.log(price2)
        ratio = price1 / price2
        
        buffer['spreads'].append(spread)
        buffer['ratios'].append(ratio)
        
        if len(buffer['prices1']) > self.lookback_periods:
            for key in buffer.keys():
                buffer[key] = buffer[key][-self.lookback_periods//2:]
        
        if len(buffer['prices1']) >= self.correlation_window:
            metrics = self._calculate_pair_metrics(symbol1, symbol2, timestamp)
            self.pair_metrics_cache[pair_key] = metrics
            return metrics
        
        return None
    
    def _calculate_pair_metrics(self, symbol1: str, symbol2: str, timestamp: datetime) -> PairMetrics:
        pair_key = f"{symbol1}-{symbol2}"
        buffer = self.pair_buffers[pair_key]
        
        try:
            prices1 = np.array(buffer['prices1'])
            prices2 = np.array(buffer['prices2'])
            spreads = np.array(buffer['spreads'])
            
            returns1 = np.diff(prices1) / prices1[:-1]
            returns2 = np.diff(prices2) / prices2[:-1]
            
            if len(returns1) >= self.correlation_window:
                recent_ret1 = returns1[-self.correlation_window:]
                recent_ret2 = returns2[-self.correlation_window:]
                correlation = np.corrcoef(recent_ret1, recent_ret2)[0, 1]
            else:
                correlation = 0.0
            
            cointegration_score = self._simple_cointegration_test(spreads)
            
            if len(spreads) >= 20:
                spread_mean = np.mean(spreads[-50:])
                spread_std = np.std(spreads[-50:])
                current_spread = spreads[-1]
                spread_zscore = (current_spread - spread_mean) / (spread_std + 1e-8)
            else:
                spread_mean = 0.0
                spread_std = 0.0
                spread_zscore = 0.0
            
            momentum_score = self._calculate_momentum_divergence(returns1, returns2)
            
            if len(returns1) >= 20 and len(returns2) >= 20:
                vol1 = np.std(returns1[-20:])
                vol2 = np.std(returns2[-20:])
                volatility_ratio = vol1 / (vol2 + 1e-8)
            else:
                volatility_ratio = 1.0
            
            trading_signal, confidence = self._generate_trading_signal(
                spread_zscore, correlation, momentum_score, volatility_ratio
            )
            
            return PairMetrics(
                pair=(symbol1, symbol2),
                timestamp=timestamp,
                correlation=correlation,
                cointegration_score=cointegration_score,
                spread_zscore=spread_zscore,
                spread_mean=spread_mean,
                spread_std=spread_std,
                momentum_score=momentum_score,
                volatility_ratio=volatility_ratio,
                trading_signal=trading_signal,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Pair metrics calculation error for {pair_key}: {e}")
            return PairMetrics(pair=(symbol1, symbol2), timestamp=timestamp)
    
    def _simple_cointegration_test(self, spreads: np.ndarray) -> float:
        if len(spreads) < 50:
            return 0.0
        
        recent_spreads = spreads[-50:]
        
        lagged_spread = recent_spreads[:-1]
        current_spread = recent_spreads[1:]
        
        if len(lagged_spread) > 10:
            coeff = np.corrcoef(lagged_spread, current_spread)[0, 1]
            return 1.0 - abs(coeff)
        
        return 0.0
    
    def _calculate_momentum_divergence(self, returns1: np.ndarray, returns2: np.ndarray) -> float:
        if len(returns1) < 20 or len(returns2) < 20:
            return 0.0
        
        momentum1 = np.mean(returns1[-10:])
        momentum2 = np.mean(returns2[-10:])
        
        divergence = momentum1 - momentum2
        
        vol1 = np.std(returns1[-20:])
        vol2 = np.std(returns2[-20:])
        avg_vol = (vol1 + vol2) / 2
        
        if avg_vol > 0:
            return divergence / avg_vol
        
        return 0.0
    
    def _generate_trading_signal(self, zscore: float, correlation: float, 
                               momentum: float, vol_ratio: float) -> Tuple[str, float]:
        
        signal_factors = {
            'mean_reversion': abs(zscore) > 2.0,
            'high_correlation': abs(correlation) > 0.5,
            'momentum_divergence': abs(momentum) > 0.5,
            'vol_stability': 0.5 < vol_ratio < 2.0
        }
        
        signal_count = sum(signal_factors.values())
        
        if signal_count >= 3:
            if zscore > 2.0:
                return "SHORT_SPREAD", min(signal_count / 4.0, 1.0)
            elif zscore < -2.0:
                return "LONG_SPREAD", min(signal_count / 4.0, 1.0)
        
        elif signal_count >= 2:
            if abs(zscore) > 1.5:
                signal = "SHORT_SPREAD" if zscore > 0 else "LONG_SPREAD"
                return signal, signal_count / 4.0
        
        if abs(zscore) < 0.5:
            return "CLOSE", 0.8
        
        return "HOLD", 0.0
    
    def get_latest_metrics(self, symbol1: str, symbol2: str) -> Optional[PairMetrics]:
        pair_key = f"{symbol1}-{symbol2}"
        return self.pair_metrics_cache.get(pair_key)

class RealTimeDataPipeline:
    
    def __init__(self, symbols: List[str], update_frequency: int = 30):
        self.symbols = symbols
        self.update_frequency = update_frequency
        
        self.quality_monitor = DataQualityMonitor()
        self.indicator_calculator = StreamingIndicatorCalculator()
        self.pair_analyzer = RealTimePairAnalyzer()
        
        self.current_quotes = {}
        self.data_subscribers = []
        self.is_running = False
        
        self.pipeline_stats = {
            'quotes_processed': 0,
            'indicators_calculated': 0,
            'pairs_analyzed': 0,
            'errors': 0,
            'start_time': None
        }
        
        self.db_path = Path("realtime_data.db")
        self._init_database()
        
        self.logger = logging.getLogger(__name__)
        
        print(f"📡 Real-time Data Pipeline: {len(symbols)} symbols")
        print(f"⚡ Update frequency: {update_frequency}s")
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quotes (
                symbol TEXT,
                timestamp TEXT,
                price REAL,
                volume INTEGER,
                bid REAL,
                ask REAL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS indicators (
                symbol TEXT,
                timestamp TEXT,
                sma_20 REAL,
                sma_50 REAL,
                rsi REAL,
                macd REAL,
                volatility REAL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pair_metrics (
                pair TEXT,
                timestamp TEXT,
                correlation REAL,
                spread_zscore REAL,
                trading_signal TEXT,
                confidence REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def subscribe_to_data(self, callback: Callable):
        self.data_subscribers.append(callback)
    
    def start_pipeline(self):
        self.is_running = True
        self.pipeline_stats['start_time'] = datetime.now()
        
        self.logger.info("🚀 Starting Real-time Data Pipeline")
        
        pipeline_thread = threading.Thread(target=self._pipeline_loop, daemon=True)
        pipeline_thread.start()
        
        persistence_thread = threading.Thread(target=self._persistence_loop, daemon=True)
        persistence_thread.start()
    
    def _pipeline_loop(self):
        while self.is_running:
            try:
                start_time = time.time()
                
                self._fetch_market_data()
                
                self._process_pairs()
                
                self._notify_subscribers()
                
                processing_time = time.time() - start_time
                self.logger.debug(f"Pipeline cycle: {processing_time:.2f}s")
                
                sleep_time = max(0, self.update_frequency - processing_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.pipeline_stats['errors'] += 1
                self.logger.error(f"Pipeline error: {e}")
                time.sleep(5)
    
    def _fetch_market_data(self):
        try:
            current_time = datetime.now()
            
            tickers = yf.Tickers(' '.join(self.symbols))
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {}
                
                for symbol in self.symbols:
                    future = executor.submit(self._fetch_symbol_quote, symbol, tickers, current_time)
                    futures[future] = symbol
                
                for future in futures:
                    symbol = futures[future]
                    try:
                        quote = future.result(timeout=5)
                        if quote and self.quality_monitor.validate_quote(quote):
                            self.current_quotes[symbol] = quote
                            self.quality_monitor.update_quality_metrics(symbol, quote)
                            
                            indicators = self.indicator_calculator.update_price(
                                symbol, quote.price, quote.volume, quote.timestamp
                            )
                            
                            self.pipeline_stats['quotes_processed'] += 1
                            if indicators:
                                self.pipeline_stats['indicators_calculated'] += 1
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to process {symbol}: {e}")
            
        except Exception as e:
            self.logger.error(f"Market data fetch error: {e}")
    
    def _fetch_symbol_quote(self, symbol: str, tickers, current_time: datetime) -> Optional[StreamingQuote]:
        try:
            ticker = tickers.tickers[symbol]
            info = ticker.info
            
            price = None
            if 'currentPrice' in info:
                price = info['currentPrice']
            elif 'regularMarketPrice' in info:
                price = info['regularMarketPrice']
            elif 'previousClose' in info:
                price = info['previousClose']
            
            if price and price > 0:
                return StreamingQuote(
                    symbol=symbol,
                    price=price,
                    timestamp=current_time,
                    volume=info.get('regularMarketVolume', 0),
                    bid=info.get('bid', 0),
                    ask=info.get('ask', 0),
                    change=info.get('regularMarketChange', 0),
                    change_percent=info.get('regularMarketChangePercent', 0)
                )
            
        except Exception as e:
            self.logger.debug(f"Quote fetch error for {symbol}: {e}")
        
        return None
    
    def _process_pairs(self):
        symbols_with_data = list(self.current_quotes.keys())
        
        if len(symbols_with_data) < 2:
            return
        
        pairs_processed = 0
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for i in range(len(symbols_with_data)):
                for j in range(i+1, len(symbols_with_data)):
                    symbol1, symbol2 = symbols_with_data[i], symbols_with_data[j]
                    
                    quote1 = self.current_quotes[symbol1]
                    quote2 = self.current_quotes[symbol2]
                    
                    future = executor.submit(
                        self.pair_analyzer.update_pair_data,
                        symbol1, symbol2, quote1.price, quote2.price, quote1.timestamp
                    )
                    futures.append(future)
                    
                    pairs_processed += 1
                    if pairs_processed >= 50:
                        break
                
                if pairs_processed >= 50:
                    break
            
            for future in futures:
                try:
                    result = future.result(timeout=2)
                    if result:
                        self.pipeline_stats['pairs_analyzed'] += 1
                except Exception as e:
                    self.logger.debug(f"Pair processing error: {e}")
    
    def _notify_subscribers(self):
        pipeline_data = {
            'timestamp': datetime.now(),
            'quotes': self.current_quotes,
            'indicators': {symbol: self.indicator_calculator.get_latest_indicators(symbol) 
                          for symbol in self.symbols},
            'pair_metrics': self.pair_analyzer.pair_metrics_cache,
            'pipeline_stats': self.pipeline_stats
        }
        
        for callback in self.data_subscribers:
            try:
                callback(pipeline_data)
            except Exception as e:
                self.logger.error(f"Subscriber notification error: {e}")
    
    def _persistence_loop(self):
        while self.is_running:
            try:
                self._persist_data()
                time.sleep(300)
            except Exception as e:
                self.logger.error(f"Persistence error: {e}")
                time.sleep(60)
    
    def _persist_data(self):
        conn = sqlite3.connect(self.db_path)
        
        try:
            for symbol, quote in self.current_quotes.items():
                conn.execute(
                    "INSERT INTO quotes VALUES (?, ?, ?, ?, ?, ?)",
                    (symbol, quote.timestamp.isoformat(), quote.price, 
                     quote.volume, quote.bid, quote.ask)
                )
            
            for symbol in self.symbols:
                indicators = self.indicator_calculator.get_latest_indicators(symbol)
                if indicators:
                    conn.execute(
                        "INSERT INTO indicators VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (symbol, indicators.timestamp.isoformat(), indicators.sma_20,
                         indicators.sma_50, indicators.rsi, indicators.macd, indicators.volatility)
                    )
            
            for pair_key, metrics in self.pair_analyzer.pair_metrics_cache.items():
                conn.execute(
                    "INSERT INTO pair_metrics VALUES (?, ?, ?, ?, ?, ?)",
                    (pair_key, metrics.timestamp.isoformat(), metrics.correlation,
                     metrics.spread_zscore, metrics.trading_signal, metrics.confidence)
                )
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Database persistence error: {e}")
        finally:
            conn.close()
    
    def get_pipeline_status(self) -> Dict:
        runtime = datetime.now() - self.pipeline_stats['start_time'] if self.pipeline_stats['start_time'] else timedelta(0)
        
        return {
            'is_running': self.is_running,
            'runtime_minutes': runtime.total_seconds() / 60,
            'symbols_tracked': len(self.symbols),
            'quotes_processed': self.pipeline_stats['quotes_processed'],
            'indicators_calculated': self.pipeline_stats['indicators_calculated'],
            'pairs_analyzed': self.pipeline_stats['pairs_analyzed'],
            'errors': self.pipeline_stats['errors'],
            'quotes_per_minute': self.pipeline_stats['quotes_processed'] / max(runtime.total_seconds() / 60, 1),
            'active_pairs': len(self.pair_analyzer.pair_metrics_cache),
            'symbols_with_current_data': len(self.current_quotes)
        }
    
    def get_trading_signals(self, min_confidence: float = 0.6) -> List[Dict]:
        signals = []
        
        for pair_key, metrics in self.pair_analyzer.pair_metrics_cache.items():
            if (metrics.trading_signal != "HOLD" and 
                metrics.confidence >= min_confidence):
                
                signals.append({
                    'pair': metrics.pair,
                    'action': metrics.trading_signal,
                    'z_score': metrics.spread_zscore,
                    'confidence': metrics.confidence,
                    'correlation': metrics.correlation,
                    'momentum': metrics.momentum_score,
                    'timestamp': metrics.timestamp
                })
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def stop_pipeline(self):
        self.is_running = False
        self._persist_data()
        self.logger.info("🛑 Real-time data pipeline stopped")

def demo_realtime_pipeline():
    print("📡 REAL-TIME DATA PIPELINE DEMO")
    print("=" * 50)
    
    demo_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',  # Tech
        'JPM', 'BAC', 'GS', 'V', 'MA',            # Finance
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',       # Healthcare
        'XOM', 'CVX', 'COP', 'EOG', 'SLB',        # Energy
        'AMZN', 'WMT', 'HD', 'PG', 'KO'           # Consumer
    ]
    
    pipeline = RealTimeDataPipeline(demo_symbols, update_frequency=30)
    
    def data_callback(pipeline_data):
        timestamp = pipeline_data['timestamp']
        quotes_count = len([q for q in pipeline_data['quotes'].values() if q])
        indicators_count = len([i for i in pipeline_data['indicators'].values() if i])
        pairs_count = len(pipeline_data['pair_metrics'])
        
        print(f"📊 {timestamp.strftime('%H:%M:%S')} | Quotes: {quotes_count} | Indicators: {indicators_count} | Pairs: {pairs_count}")
        
        signals = pipeline.get_trading_signals(min_confidence=0.7)
        if signals:
            for signal in signals[:3]:
                print(f"   🎯 {signal['action']} {signal['pair']} | Z: {signal['z_score']:.2f} | Conf: {signal['confidence']:.2f}")
    
    pipeline.subscribe_to_data(data_callback)
    
    try:
        pipeline.start_pipeline()
        
        print("🌐 Pipeline running... (Press Ctrl+C to stop)")
        print("📈 Monitoring real-time market data...\n")
        
        demo_duration = 300
        start_time = time.time()
        
        while time.time() - start_time < demo_duration and pipeline.is_running:
            time.sleep(10)
            
            if int(time.time() - start_time) % 60 == 0:
                status = pipeline.get_pipeline_status()
                print(f"\n⚡ Pipeline Status:")
                print(f"   Runtime: {status['runtime_minutes']:.1f} min")
                print(f"   Quotes/min: {status['quotes_per_minute']:.1f}")
                print(f"   Active pairs: {status['active_pairs']}")
                print(f"   Errors: {status['errors']}")
        
        final_status = pipeline.get_pipeline_status()
        print(f"\n📊 DEMO RESULTS:")
        print(f"   Total quotes processed: {final_status['quotes_processed']}")
        print(f"   Indicators calculated: {final_status['indicators_calculated']}")
        print(f"   Pairs analyzed: {final_status['pairs_analyzed']}")
        print(f"   Processing rate: {final_status['quotes_per_minute']:.1f} quotes/min")
        
        final_signals = pipeline.get_trading_signals(min_confidence=0.5)
        print(f"\n🎯 Final Trading Signals ({len(final_signals)}):")
        for signal in final_signals[:5]:
            print(f"   {signal['action']} {signal['pair']} | Z: {signal['z_score']:.2f} | Conf: {signal['confidence']:.2f}")
    
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    
    finally:
        pipeline.stop_pipeline()
        print("✅ Real-time pipeline demo complete")

if __name__ == "__main__":
    demo_realtime_pipeline()
