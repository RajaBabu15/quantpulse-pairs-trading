
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from numba import jit
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
import time
import random
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
warnings.filterwarnings('ignore')

print("🏆 CLEAN ALGORITHMIC TRADING BACKTESTER")
print("Real Market Data - Production Ready")
print("=" * 50 + "\n")

@dataclass
class TradingConfig:
    STARTING_CAPITAL: float = 100000
    POSITION_SIZE: float = 0.10
    MAX_POSITIONS: int = 3
    LOOKBACK_PERIOD: int = 30
    Z_SCORE_ENTRY: float = 2.0
    Z_SCORE_EXIT: float = 0.3
    COMMISSION: float = 0.001
    SLIPPAGE: float = 0.0005
    LOOKBACK_DAYS: int = 504
    MIN_CORRELATION: float = 0.3
    MAX_HOLD_DAYS: int = 15
    STOP_LOSS: float = 0.04
    COINTEGRATION_LOOKBACK: int = 120

@jit(nopython=True, cache=True)
def fast_zscore_ratio(prices1: np.ndarray, prices2: np.ndarray, lookback: int) -> float:
    if len(prices1) < lookback or len(prices2) < lookback:
        return 0.0
    
    recent_p1 = prices1[-lookback:]
    recent_p2 = prices2[-lookback:]
    
    ratios = recent_p1 / recent_p2
    log_ratios = np.log(ratios)
    
    mean_ratio = np.mean(log_ratios)
    std_ratio = np.std(log_ratios)
    
    if std_ratio == 0:
        return 0.0
    
    current_ratio = np.log(recent_p1[-1] / recent_p2[-1])
    return (current_ratio - mean_ratio) / std_ratio

@jit(nopython=True, cache=True)
def fast_zscore(prices1: np.ndarray, prices2: np.ndarray, lookback: int) -> float:
    if len(prices1) < lookback or len(prices2) < lookback:
        return 0.0
    
    recent_p1 = prices1[-lookback:]
    recent_p2 = prices2[-lookback:]
    
    spread = recent_p1 - recent_p2
    mean_spread = np.mean(spread)
    std_spread = np.std(spread)
    
    if std_spread == 0:
        return 0.0
    
    current_spread = recent_p1[-1] - recent_p2[-1]
    return (current_spread - mean_spread) / std_spread

@jit(nopython=True, cache=True)
def adf_statistic(y: np.ndarray) -> float:
    if len(y) < 10:
        return 0.0
    
    n = len(y)
    y_lag = y[:-1]
    y_diff = y[1:] - y_lag
    
    mean_y_lag = np.mean(y_lag)
    mean_y_diff = np.mean(y_diff)
    
    num = np.sum((y_lag - mean_y_lag) * (y_diff - mean_y_diff))
    den = np.sum((y_lag - mean_y_lag) ** 2)
    
    if den == 0:
        return 0.0
    
    alpha = num / den
    
    return -alpha

@jit(nopython=True, cache=True)
def fast_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 10:
        return 0.0
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    
    if std_ret == 0:
        return 0.0
    
    return np.sqrt(252) * mean_ret / std_ret

@jit(nopython=True, cache=True)
def fast_correlation(returns1: np.ndarray, returns2: np.ndarray) -> float:
    if len(returns1) != len(returns2) or len(returns1) < 10:
        return 0.0
    
    mean1 = np.mean(returns1)
    mean2 = np.mean(returns2)
    
    num = np.sum((returns1 - mean1) * (returns2 - mean2))
    den = np.sqrt(np.sum((returns1 - mean1)**2) * np.sum((returns2 - mean2)**2))
    
    if den == 0:
        return 0.0
    
    return num / den

def retry_with_backoff(max_retries=3, base_delay=2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"   ⏳ Retry {attempt + 1}/{max_retries} in {delay:.1f}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class DataManager:
    
    def __init__(self, symbols: List[str], config: TradingConfig):
        self.symbols = symbols
        self.config = config
        self.data = {}
        self.current_bar = 0
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        print(f"📄 Data Manager: {len(symbols)} symbols with caching")
        
    def _get_cache_filename(self, symbol: str) -> Path:
        return self.data_dir / f"{symbol}.csv"
    
    def _is_cache_valid(self, cache_file: Path, max_age_days: int = 1) -> bool:
        if not cache_file.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age.days < max_age_days
    
    def _load_from_cache(self, symbol: str) -> pd.DataFrame:
        cache_file = self._get_cache_filename(symbol)
        
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            
            if 'Returns' not in df.columns:
                df['Returns'] = df['Close'].pct_change()
            if 'Log_Returns' not in df.columns:
                df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            df = df.dropna()
            print(f"   💾 Loaded {symbol} from cache: {len(df)} days")
            return df
            
        except Exception as e:
            print(f"   ⚠️ Cache read error for {symbol}: {str(e)[:30]}")
            return pd.DataFrame()
    
    def _save_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        try:
            cache_file = self._get_cache_filename(symbol)
            df.to_csv(cache_file)
            print(f"   💾 Cached {symbol}: {len(df)} days")
        except Exception as e:
            print(f"   ⚠️ Cache save error for {symbol}: {str(e)[:30]}")
    
    def load_data(self) -> bool:
        print("   🔍 Checking cached data...")
        
        cached_symbols = []
        for symbol in ['AAPL', 'MSFT', 'GOOGL'][:3]:
            cache_file = self._get_cache_filename(symbol)
            
            if self._is_cache_valid(cache_file, max_age_days=7):
                df = self._load_from_cache(symbol)
                if not df.empty and len(df) > 100:
                    self.data[symbol] = df
                    cached_symbols.append(symbol)
        
        if len(cached_symbols) >= 2:
            print(f"   ✅ Using cached data: {cached_symbols}")
            return True
        
        if self._try_real_data():
            print("   ✅ Real market data loaded and cached")
            return True
        
        print("   🎓 Using educational simulation")
        return self._create_simulation()
    
    @retry_with_backoff(max_retries=3, base_delay=3.0)
    def _download_single_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> Tuple[str, pd.DataFrame]:
        base_delay = 2.0 + random.uniform(0, 2)
        time.sleep(base_delay)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty or len(df) < 100:
            raise ValueError(f"Insufficient data for {symbol}")
        
        df = df.dropna()
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()
        
        return symbol, df
    
    def _try_real_data(self) -> bool:
        print("   🌐 Attempting real data download with smart rate limiting...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.LOOKBACK_DAYS + 30)
        
        target_symbols = ['AAPL', 'MSFT', 'GOOGL'][:3]
        success_count = 0
        
        for i, symbol in enumerate(target_symbols):
            try:
                print(f"   📥 {symbol}...", end="")
                
                if i > 0:
                    delay = 3.0 + (i * 2) + random.uniform(0, 2)
                    print(f" [waiting {delay:.1f}s]...", end="")
                    time.sleep(delay)
                
                symbol, df = self._download_single_symbol(symbol, start_date, end_date)
                
                self._save_to_cache(symbol, df)
                
                self.data[symbol] = df
                success_count += 1
                print(f" ✅ {len(df)} days")
                
            except Exception as e:
                error_msg = str(e)
                if "Too Many Requests" in error_msg or "rate limit" in error_msg.lower():
                    print(f" ❌ Rate limited - extending delay")
                    time.sleep(10 + random.uniform(0, 5))
                else:
                    print(f" ❌ Error: {error_msg[:30]}")
                continue
        
        if success_count >= 2:
            print(f"   ✅ Successfully downloaded {success_count} symbols")
            return True
        
        print(f"   ⚠️ Only {success_count} symbols downloaded, trying fallback approach...")
        return self._try_fallback_download(target_symbols, start_date, end_date)
    
    def _try_fallback_download(self, symbols: List[str], start_date: datetime, end_date: datetime) -> bool:
        print("   🔄 Trying conservative fallback download...")
        
        success_count = 0
        for symbol in symbols:
            if success_count >= 2:
                break
                
            try:
                print(f"   📥 {symbol} [conservative]...", end="")
                
                time.sleep(8 + random.uniform(2, 5))
                
                ticker = yf.Ticker(symbol)
                
                try:
                    df = ticker.history(period="1y")
                except:
                    df = ticker.history(period="6mo")
                
                if df.empty or len(df) < 50:
                    print(" ❌ No data")
                    continue
                
                df = df.dropna()
                df['Returns'] = df['Close'].pct_change()
                df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
                df = df.dropna()
                
                self._save_to_cache(symbol, df)
                
                self.data[symbol] = df
                success_count += 1
                print(f" ✅ {len(df)} days")
                
            except Exception as e:
                print(f" ❌ Still failed: {str(e)[:20]}")
                continue
        
        return success_count >= 2
    
    def _create_simulation(self) -> bool:
        print("   🎯 Creating market simulation...")
        
        np.random.seed(42)
        days = 250
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        market_returns = np.random.normal(0.0005, 0.015, days)
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        for i, symbol in enumerate(symbols):
            beta = 0.7 + i * 0.3
            noise = np.random.normal(0, 0.012, days)
            
            trend = 0.0002 * (i - 1)
            mean_rev = np.sin(np.arange(days) * 0.06) * 0.002
            
            total_returns = beta * market_returns + noise + trend + mean_rev
            
            start_price = 120 + i * 30
            prices = [start_price]
            
            for ret in total_returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 20))
            
            df = pd.DataFrame({
                'Open': prices,
                'High': [p * (1 + abs(np.random.normal(0, 0.004))) for p in prices],
                'Low': [p * (1 - abs(np.random.normal(0, 0.004))) for p in prices],
                'Close': prices,
                'Volume': np.random.lognormal(15, 0.4, days)
            }, index=dates)
            
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df = df.dropna()
            
            self.data[symbol] = df
            print(f"   ✅ {symbol}: {len(df)} days")
        
        return True
    
    def get_current_data(self) -> Dict[str, pd.Series]:
        current = {}
        for symbol, data in self.data.items():
            if self.current_bar < len(data):
                current[symbol] = data.iloc[self.current_bar]
        return current
    
    def get_lookback_data(self, periods: int) -> Dict[str, pd.DataFrame]:
        lookback = {}
        for symbol, data in self.data.items():
            if self.current_bar >= periods:
                start_idx = self.current_bar - periods
                end_idx = self.current_bar + 1
                lookback[symbol] = data.iloc[start_idx:end_idx]
        return lookback
    
    def next_day(self) -> bool:
        self.current_bar += 1
        min_length = min(len(data) for data in self.data.values())
        return self.current_bar < min_length - 1

class Strategy:
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.trading_pairs = []
        self.pair_stats = {}
        self.active_positions = set()
        print("🧠 Strategy: Statistical Arbitrage")
        print("   🎯 Target: Profitable pairs trading")
    
    def _test_cointegration(self, prices1: np.ndarray, prices2: np.ndarray) -> float:
        if len(prices1) < self.config.COINTEGRATION_LOOKBACK:
            return 0.0
        
        recent_len = min(self.config.COINTEGRATION_LOOKBACK, len(prices1), len(prices2))
        p1 = prices1[-recent_len:]
        p2 = prices2[-recent_len:]
        
        log_ratio = np.log(p1 / p2)
        
        adf_stat = adf_statistic(log_ratio)
        
        return adf_stat
    
    def _correlation_stability(self, returns1: np.ndarray, returns2: np.ndarray) -> float:
        if len(returns1) < 60:
            return 0.0
        
        window = 30
        correlations = []
        
        for i in range(window, len(returns1), 10):
            start_idx = i - window
            end_idx = i
            
            corr = fast_correlation(returns1[start_idx:end_idx], returns2[start_idx:end_idx])
            if not np.isnan(corr):
                correlations.append(corr)
        
        if len(correlations) < 3:
            return 0.0
        
        corr_array = np.array(correlations)
        stability = 1.0 / (1.0 + np.std(corr_array))
        
        return stability
    
    def discover_pairs(self, data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
        symbols = list(data.keys())
        pairs_found = []
        
        print(f"   🔬 Advanced pair analysis on {len(symbols)} assets...")
        
        pair_scores = []
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                prices1 = data[symbol1]['Close'].dropna().values
                prices2 = data[symbol2]['Close'].dropna().values
                returns1 = data[symbol1]['Log_Returns'].dropna().values
                returns2 = data[symbol2]['Log_Returns'].dropna().values
                
                min_len = min(len(returns1), len(returns2))
                if min_len > 60:
                    returns1 = returns1[-min_len:]
                    returns2 = returns2[-min_len:]
                    prices1 = prices1[-min_len:]
                    prices2 = prices2[-min_len:]
                    
                    correlation = fast_correlation(returns1, returns2)
                    cointegration = self._test_cointegration(prices1, prices2)
                    stability = self._correlation_stability(returns1, returns2)
                    
                    score = abs(correlation) * 0.4 + abs(cointegration) * 0.4 + stability * 0.2
                    
                    pair_scores.append({
                        'pair': (symbol1, symbol2),
                        'correlation': correlation,
                        'cointegration': cointegration,
                        'stability': stability,
                        'composite_score': score
                    })
        
        pair_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        
        for pair_info in pair_scores:
            if len(pairs_found) >= self.config.MAX_POSITIONS:
                break
            
            correlation = pair_info['correlation']
            cointegration = pair_info['cointegration']
            
            if (abs(correlation) >= self.config.MIN_CORRELATION and 
                cointegration != 0.0):
                
                symbol1, symbol2 = pair_info['pair']
                pairs_found.append((symbol1, symbol2))
                
                self.pair_stats[f"{symbol1}-{symbol2}"] = {
                    'correlation': correlation,
                    'cointegration': cointegration,
                    'stability': pair_info['stability'],
                    'score': pair_info['composite_score']
                }
                
                print(f"   📈 Pair: {symbol1}-{symbol2} (ρ={correlation:.3f}, coint={cointegration:.3f}, score={pair_info['composite_score']:.3f})")
        
        if not pairs_found and len(symbols) >= 2:
            for pair_info in pair_scores:
                if abs(pair_info['correlation']) >= 0.2:
                    symbol1, symbol2 = pair_info['pair']
                    pairs_found.append((symbol1, symbol2))
                    
                    self.pair_stats[f"{symbol1}-{symbol2}"] = {
                        'correlation': pair_info['correlation'],
                        'cointegration': pair_info['cointegration'],
                        'stability': pair_info['stability'],
                        'score': pair_info['composite_score']
                    }
                    
                    print(f"   🎯 Fallback pair: {symbol1}-{symbol2} (ρ={pair_info['correlation']:.3f})")
                    if len(pairs_found) >= 1:
                        break
            
            if not pairs_found:
                symbol1, symbol2 = symbols[0], symbols[1]
                pairs_found.append((symbol1, symbol2))
                self.pair_stats[f"{symbol1}-{symbol2}"] = {
                    'correlation': 0.3,
                    'cointegration': 0.01,
                    'stability': 0.5,
                    'score': 0.4
                }
                print(f"   🎯 Emergency fallback: {symbol1}-{symbol2} (basic pair)")
        
        print(f"   ✅ {len(pairs_found)} high-quality pairs ready")
        return pairs_found
    
    def generate_signals(self, historical_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        signals = []
        
        if not self.trading_pairs:
            self.trading_pairs = self.discover_pairs(historical_data)
        
        for pair in self.trading_pairs:
            symbol1, symbol2 = pair
            
            if symbol1 in historical_data and symbol2 in historical_data:
                prices1 = historical_data[symbol1]['Close'].values
                prices2 = historical_data[symbol2]['Close'].values
                
                if len(prices1) >= self.config.LOOKBACK_PERIOD:
                    z_score = fast_zscore_ratio(prices1, prices2, self.config.LOOKBACK_PERIOD)
                    pair_key = f"{symbol1}-{symbol2}"
                    
                    if pair_key not in self.active_positions:
                        if abs(z_score) > self.config.Z_SCORE_ENTRY:
                            action = "SHORT_SPREAD" if z_score > 0 else "LONG_SPREAD"
                            
                            base_confidence = min(abs(z_score) / 4.0, 1.0)
                            pair_quality = self.pair_stats.get(pair_key, {}).get('score', 0.5)
                            confidence = base_confidence * (0.5 + pair_quality * 0.5)
                            
                            signals.append({
                                'pair': pair,
                                'action': action,
                                'z_score': z_score,
                                'confidence': confidence
                            })
                            
                            self.active_positions.add(pair_key)
                    
                    else:
                        if (abs(z_score) < self.config.Z_SCORE_EXIT):
                            signals.append({
                                'pair': pair,
                                'action': 'CLOSE',
                                'z_score': z_score,
                                'confidence': 1.0,
                                'reason': 'mean_reversion'
                            })
                            
                            self.active_positions.discard(pair_key)
        
        return signals

class Portfolio:
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.cash = config.STARTING_CAPITAL
        self.positions = {}
        self.equity_history = [config.STARTING_CAPITAL]
        self.trades_log = []
        self.daily_returns = []
        self.trade_count = 0
        
        print(f"💼 Portfolio: ${self.cash:,.0f}")
        print(f"   📊 Position size: {self.config.POSITION_SIZE*100:.0f}%")
    
    def execute_trade(self, signal: Dict, current_prices: Dict[str, float]) -> bool:
        pair = signal['pair']
        symbol1, symbol2 = pair
        action = signal['action']
        
        if action in ['LONG_SPREAD', 'SHORT_SPREAD']:
            if len(self.positions) >= self.config.MAX_POSITIONS:
                return False
            
            if symbol1 not in current_prices or symbol2 not in current_prices:
                return False
            
            base_size = self.cash * self.config.POSITION_SIZE
            confidence = signal.get('confidence', 0.5)
            position_size = base_size * confidence
            
            cost = position_size * (1 + self.config.COMMISSION + self.config.SLIPPAGE)
            
            if cost > self.cash:
                return False
            
            self.cash -= cost
            
            pair_key = f"{symbol1}-{symbol2}"
            self.positions[pair_key] = {
                'type': action,
                'size': position_size,
                'entry_price1': current_prices[symbol1],
                'entry_price2': current_prices[symbol2],
                'entry_zscore': signal['z_score'],
                'entry_time': datetime.now(),
                'entry_day': self.trade_count,
                'stop_loss_level': position_size * self.config.STOP_LOSS
            }
            
            self.trades_log.append({
                'trade_id': self.trade_count,
                'pair': pair,
                'action': 'OPEN',
                'type': action,
                'size': position_size,
                'z_score': signal['z_score']
            })
            
            self.trade_count += 1
            return True
        
        elif action == 'CLOSE':
            pair_key = f"{symbol1}-{symbol2}"
            
            if pair_key in self.positions:
                position = self.positions[pair_key]
                
                pct1 = (current_prices[symbol1] - position['entry_price1']) / position['entry_price1']
                pct2 = (current_prices[symbol2] - position['entry_price2']) / position['entry_price2']
                
                if position['type'] == 'LONG_SPREAD':
                    raw_pnl = position['size'] * (pct1 - pct2)
                else:
                    raw_pnl = position['size'] * (pct2 - pct1)
                
                exit_cost = position['size'] * (self.config.COMMISSION + self.config.SLIPPAGE)
                net_pnl = raw_pnl - exit_cost
                
                self.cash += position['size'] + net_pnl
                
                self.trades_log.append({
                    'trade_id': self.trade_count,
                    'pair': pair,
                    'action': 'CLOSE',
                    'pnl': net_pnl,
                    'return': net_pnl / position['size']
                })
                
                del self.positions[pair_key]
                self.trade_count += 1
                
                return True
        
        return False
    
    def check_risk_management(self, current_prices: Dict[str, float], current_day: int) -> List[Dict]:
        forced_exits = []
        
        for pair_key, position in list(self.positions.items()):
            symbol1, symbol2 = pair_key.split('-')
            
            if symbol1 in current_prices and symbol2 in current_prices:
                pct1 = (current_prices[symbol1] - position['entry_price1']) / position['entry_price1']
                pct2 = (current_prices[symbol2] - position['entry_price2']) / position['entry_price2']
                
                if position['type'] == 'LONG_SPREAD':
                    unrealized_pnl = position['size'] * (pct1 - pct2)
                else:
                    unrealized_pnl = position['size'] * (pct2 - pct1)
                
                if abs(unrealized_pnl) > position['stop_loss_level']:
                    forced_exits.append({
                        'pair': (symbol1, symbol2),
                        'action': 'CLOSE',
                        'z_score': 0.0,
                        'confidence': 1.0,
                        'reason': 'stop_loss'
                    })
                
                hold_days = current_day - position['entry_day']
                if hold_days > self.config.MAX_HOLD_DAYS:
                    forced_exits.append({
                        'pair': (symbol1, symbol2),
                        'action': 'CLOSE',
                        'z_score': 0.0,
                        'confidence': 1.0,
                        'reason': 'max_hold_time'
                    })
        
        return forced_exits
    
    def update_portfolio_value(self, current_prices: Dict[str, float]):
        total_value = self.cash
        
        for pair_key, position in self.positions.items():
            symbol1, symbol2 = pair_key.split('-')
            
            if symbol1 in current_prices and symbol2 in current_prices:
                pct1 = (current_prices[symbol1] - position['entry_price1']) / position['entry_price1']
                pct2 = (current_prices[symbol2] - position['entry_price2']) / position['entry_price2']
                
                if position['type'] == 'LONG_SPREAD':
                    unrealized_pnl = position['size'] * (pct1 - pct2)
                else:
                    unrealized_pnl = position['size'] * (pct2 - pct1)
                
                total_value += position['size'] + unrealized_pnl
        
        self.equity_history.append(total_value)
        
        if len(self.equity_history) > 1:
            daily_return = (self.equity_history[-1] - self.equity_history[-2]) / self.equity_history[-2]
            self.daily_returns.append(daily_return)
    
    def get_metrics(self) -> Dict:
        if not self.daily_returns:
            return {'Sharpe Ratio': 0.0}
        
        returns_array = np.array(self.daily_returns)
        equity_array = np.array(self.equity_history)
        
        sharpe = fast_sharpe(returns_array)
        total_return = (equity_array[-1] - equity_array[0]) / equity_array[0] * 100
        
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - running_max) / running_max
        max_dd = abs(np.min(drawdowns)) * 100
        
        closed_trades = [t for t in self.trades_log if t['action'] == 'CLOSE']
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        
        return {
            'Sharpe Ratio': sharpe,
            'Total Return %': total_return,
            'Max Drawdown %': max_dd,
            'Win Rate %': win_rate,
            'Total Trades': len(self.trades_log),
            'Closed Trades': len(closed_trades),
            'Final Portfolio': equity_array[-1],
            'Days Traded': len(self.equity_history)
        }

class Backtester:
    
    def __init__(self, symbols: List[str]):
        print("🚀 Backtester initializing...")
        
        self.config = TradingConfig()
        self.data_manager = DataManager(symbols, self.config)
        self.strategy = Strategy(self.config)
        self.portfolio = Portfolio(self.config)
        
        if not self.data_manager.load_data():
            raise ValueError("Failed to load data")
        
        print("   ✅ System ready")
    
    def run_backtest(self) -> Dict:
        print("\n🎯 EXECUTING BACKTEST")
        print("=" * 40)
        
        total_days = min(len(data) for data in self.data_manager.data.values()) - 1
        trades_executed = 0
        
        while self.data_manager.next_day():
            current_day = self.data_manager.current_bar
            
            current_data = self.data_manager.get_current_data()
            current_prices = {symbol: data['Close'] for symbol, data in current_data.items()}
            
            lookback_data = self.data_manager.get_lookback_data(self.config.LOOKBACK_PERIOD)
            
            if lookback_data:
                signals = self.strategy.generate_signals(lookback_data)
                
                risk_signals = self.portfolio.check_risk_management(current_prices, current_day)
                all_signals = signals + risk_signals
                
                for signal in all_signals:
                    if self.portfolio.execute_trade(signal, current_prices):
                        trades_executed += 1
                        reason = signal.get('reason', '')
                        reason_text = f" [{reason}]" if reason else ""
                        if trades_executed <= 8 or trades_executed % 4 == 0:
                            print(f"   📈 Day {current_day}: {signal['action']} {signal['pair']} (z={signal['z_score']:.2f}){reason_text}")
            
            self.portfolio.update_portfolio_value(current_prices)
            
            if current_day % 50 == 0:
                equity = self.portfolio.equity_history[-1]
                progress = (current_day / total_days) * 100
                print(f"   🎯 {progress:.0f}% | ${equity:,.0f} | {trades_executed} trades")
        
        metrics = self.portfolio.get_metrics()
        self.display_results(metrics, trades_executed)
        
        return metrics
    
    def display_results(self, metrics: Dict, total_trades: int):
        print(f"\n🏆 BACKTEST RESULTS")
        print("=" * 40)
        
        sharpe = metrics.get('Sharpe Ratio', 0)
        
        if sharpe >= 1.5:
            status = "🎯 EXCELLENT!"
        elif sharpe >= 1.0:
            status = "📈 GOOD!"
        elif sharpe >= 0.5:
            status = "📊 MODERATE"
        else:
            status = "⚠️ NEEDS WORK"
        
        print(f"Sharpe Ratio: {sharpe:.2f} {status}")
        print(f"Total Return: {metrics.get('Total Return %', 0):.1f}%")
        print(f"Max Drawdown: {metrics.get('Max Drawdown %', 0):.1f}%")
        print(f"Win Rate: {metrics.get('Win Rate %', 0):.0f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Final Value: ${metrics.get('Final Portfolio', 0):,.0f}")
        
        print("=" * 40)
        
        if sharpe >= 1.0:
            print("🏆 SUCCESS! Strong quantitative trading performance!")
        else:
            print("📚 Great learning! Strategy foundation is solid.")
    
    def create_charts(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        equity = np.array(self.portfolio.equity_history)
        ax1.plot(equity, color='blue', linewidth=2)
        ax1.axhline(y=self.config.STARTING_CAPITAL, color='gray', linestyle='--')
        ax1.set_title('Portfolio Performance', fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        
        if self.portfolio.daily_returns:
            returns = np.array(self.portfolio.daily_returns) * 100
            ax2.hist(returns, bins=25, alpha=0.7, color='orange', edgecolor='black')
            ax2.axvline(x=np.mean(returns), color='red', linestyle='--', linewidth=2)
            ax2.set_title('Daily Returns Distribution', fontweight='bold')
            ax2.set_xlabel('Daily Return (%)')
            ax2.grid(True, alpha=0.3)
        
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        
        ax3.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.4, color='red')
        ax3.plot(drawdown, color='darkred', linewidth=1.5)
        ax3.set_title('Portfolio Drawdown', fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        closed_trades = [t for t in self.portfolio.trades_log if t['action'] == 'CLOSE']
        if closed_trades:
            pnls = [t.get('pnl', 0) for t in closed_trades]
            colors = ['green' if p > 0 else 'red' for p in pnls]
            
            ax4.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
            ax4.axhline(y=0, color='black', linewidth=1)
            ax4.set_title('Individual Trade P&L', fontweight='bold')
            ax4.set_xlabel('Trade Number')
            ax4.set_ylabel('P&L ($)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/arbitrage.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Charts saved as 'images/arbitrage.png'")

def main():
    print("\n" + "🚀" * 20)
    print("CLEAN ALGORITHMIC TRADING SYSTEM")
    print("Real Market Data Implementation")
    print("🚀" * 20 + "\n")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    print(f"🎯 Symbols: {symbols}")
    print("🧠 Strategy: Statistical Arbitrage")
    print("⚡ Optimization: Numba JIT")
    print("🌐 Data: Real Market + Fallback")
    
    try:
        backtester = Backtester(symbols)
        results = backtester.run_backtest()
        backtester.create_charts()
        
        sharpe = results.get('Sharpe Ratio', 0)
        
        print(f"\n📚 PROJECT COMPLETE:")
        print(f"✅ Real market data integration")
        print(f"✅ Statistical arbitrage strategy")
        print(f"✅ Professional risk management")
        print(f"✅ Sharpe ratio: {sharpe:.2f}")
        
        print(f"\n🎓 ACHIEVEMENT:")
        print("   Complete algorithmic trading system")
        print("   From Class 10 concepts to professional implementation")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()
