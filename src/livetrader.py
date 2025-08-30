
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
import json
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import warnings
import asyncio
import websockets
from concurrent.futures import ThreadPoolExecutor
import logging
from queue import Queue
import schedule

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LivePosition:
    pair: tuple
    strategy: str
    position_type: str
    size: float
    entry_price1: float
    entry_price2: float
    entry_time: datetime
    entry_zscore: float
    stop_loss_level: float
    confidence: float
    unrealized_pnl: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0

@dataclass
class LiveTrade:
    trade_id: str
    strategy: str
    pair: tuple
    action: str
    timestamp: datetime
    price1: float
    price2: float
    size: float
    pnl: Optional[float] = None
    reason: Optional[str] = None

class MarketHoursManager:
    
    def __init__(self):
        self.market_tz = pytz.timezone('America/New_York')
        self.logger = logging.getLogger(__name__)
    
    def is_market_open(self) -> bool:
        now = datetime.now(self.market_tz)
        
        if now.weekday() >= 5:
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def time_to_market_open(self) -> timedelta:
        now = datetime.now(self.market_tz)
        
        if now.weekday() >= 5:
            days_to_monday = 7 - now.weekday()
            next_monday = (now + timedelta(days=days_to_monday)).replace(hour=9, minute=30, second=0, microsecond=0)
            return next_monday - now
        
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        if now > market_close:
            next_open = (now + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
            return next_open - now
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now < market_open:
            return market_open - now
        
        return timedelta(0)

class LiveDataFeed:
    
    def __init__(self, symbols: List[str], update_interval: int = 60):
        self.symbols = symbols
        self.update_interval = update_interval
        self.current_prices = {}
        self.price_history = {symbol: [] for symbol in symbols}
        self.last_update = None
        self.is_running = False
        self.data_queue = Queue()
        self.logger = logging.getLogger(__name__)
        
    def start_feed(self):
        self.is_running = True
        self.logger.info(f"🌐 Starting live data feed for {len(self.symbols)} symbols")
        
        feed_thread = threading.Thread(target=self._feed_loop, daemon=True)
        feed_thread.start()
        
    def _feed_loop(self):
        while self.is_running:
            try:
                self._fetch_current_prices()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Data feed error: {e}")
                time.sleep(5)
    
    def _fetch_current_prices(self):
        try:
            tickers = yf.Tickers(' '.join(self.symbols))
            current_time = datetime.now()
            
            for symbol in self.symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    
                    current_price = None
                    if 'currentPrice' in info:
                        current_price = info['currentPrice']
                    elif 'regularMarketPrice' in info:
                        current_price = info['regularMarketPrice']
                    elif 'previousClose' in info:
                        current_price = info['previousClose']
                    
                    if current_price and current_price > 0:
                        self.current_prices[symbol] = current_price
                        self.price_history[symbol].append({
                            'timestamp': current_time,
                            'price': current_price
                        })
                        
                        if len(self.price_history[symbol]) > 1000:
                            self.price_history[symbol] = self.price_history[symbol][-1000:]
                        
                        self.data_queue.put({
                            'symbol': symbol,
                            'price': current_price,
                            'timestamp': current_time
                        })
                
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {symbol}: {e}")
            
            self.last_update = current_time
            
        except Exception as e:
            self.logger.error(f"Price fetch error: {e}")
    
    def get_current_prices(self) -> Dict[str, float]:
        return self.current_prices.copy()
    
    def get_price_history(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        if symbol not in self.price_history:
            return pd.DataFrame()
        
        history = self.price_history[symbol][-periods:]
        if not history:
            return pd.DataFrame()
        
        df = pd.DataFrame(history)
        df.set_index('timestamp', inplace=True)
        df['returns'] = df['price'].pct_change()
        return df.dropna()
    
    def stop_feed(self):
        self.is_running = False
        self.logger.info("🛑 Data feed stopped")

class LivePortfolioManager:
    
    def __init__(self, starting_capital: float = 100000):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.positions: Dict[str, LivePosition] = {}
        self.trades: List[LiveTrade] = []
        self.equity_history = []
        self.strategy_performance = {
            'Conservative_MR': [],
            'Aggressive_MR': [],
            'Momentum': [],
            'Vol_Breakout': [],
            'Hybrid': []
        }
        self.daily_pnl = 0.0
        self.max_positions = 10
        self.risk_limit_per_strategy = 0.15
        self.logger = logging.getLogger(__name__)
        
        self.last_equity_update = datetime.now()
        self.performance_file = Path("live_performance.json")
        
    def execute_live_trade(self, signal: Dict, current_prices: Dict[str, float]) -> bool:
        try:
            pair = signal['pair']
            symbol1, symbol2 = pair
            action = signal['action']
            strategy = signal.get('strategy', 'Unknown')
            
            if symbol1 not in current_prices or symbol2 not in current_prices:
                self.logger.warning(f"Missing price data for {pair}")
                return False
            
            current_time = datetime.now()
            trade_id = f"{strategy}_{current_time.strftime('%Y%m%d_%H%M%S')}"
            
            if action in ['LONG_SPREAD', 'SHORT_SPREAD']:
                if len(self.positions) >= self.max_positions:
                    self.logger.info(f"Position limit reached ({self.max_positions})")
                    return False
                
                confidence = signal.get('confidence', 0.5)
                base_size = signal.get('position_size', 0.05)
                position_size = self.cash * base_size * confidence
                position_size = min(position_size, self.cash * self.risk_limit_per_strategy)
                
                if position_size < 1000:
                    return False
                
                commission = position_size * 0.001
                total_cost = position_size + commission
                
                if total_cost > self.cash:
                    self.logger.warning(f"Insufficient cash for {pair} trade")
                    return False
                
                self.cash -= total_cost
                
                position = LivePosition(
                    pair=pair,
                    strategy=strategy,
                    position_type=action,
                    size=position_size,
                    entry_price1=current_prices[symbol1],
                    entry_price2=current_prices[symbol2],
                    entry_time=current_time,
                    entry_zscore=signal.get('z_score', 0),
                    stop_loss_level=position_size * 0.05,
                    confidence=confidence
                )
                
                position_key = f"{symbol1}-{symbol2}_{strategy}"
                self.positions[position_key] = position
                
                trade = LiveTrade(
                    trade_id=trade_id,
                    strategy=strategy,
                    pair=pair,
                    action='OPEN',
                    timestamp=current_time,
                    price1=current_prices[symbol1],
                    price2=current_prices[symbol2],
                    size=position_size
                )
                
                self.trades.append(trade)
                self.logger.info(f"📈 OPEN {action} {pair} | Strategy: {strategy} | Size: ${position_size:,.0f} | Z-Score: {signal.get('z_score', 0):.2f}")
                
                return True
                
            elif action == 'CLOSE':
                position_key = f"{symbol1}-{symbol2}_{strategy}"
                
                if position_key in self.positions:
                    position = self.positions[position_key]
                    
                    pct1 = (current_prices[symbol1] - position.entry_price1) / position.entry_price1
                    pct2 = (current_prices[symbol2] - position.entry_price2) / position.entry_price2
                    
                    if position.position_type == 'LONG_SPREAD':
                        raw_pnl = position.size * (pct1 - pct2)
                    else:
                        raw_pnl = position.size * (pct2 - pct1)
                    
                    commission = position.size * 0.001
                    net_pnl = raw_pnl - commission
                    
                    self.cash += position.size + net_pnl
                    
                    return_pct = net_pnl / position.size
                    if strategy in self.strategy_performance:
                        self.strategy_performance[strategy].append(return_pct)
                    
                    trade = LiveTrade(
                        trade_id=trade_id,
                        strategy=strategy,
                        pair=pair,
                        action='CLOSE',
                        timestamp=current_time,
                        price1=current_prices[symbol1],
                        price2=current_prices[symbol2],
                        size=position.size,
                        pnl=net_pnl,
                        reason=signal.get('reason', '')
                    )
                    
                    self.trades.append(trade)
                    del self.positions[position_key]
                    
                    self.logger.info(f"📉 CLOSE {pair} | P&L: ${net_pnl:,.0f} | Return: {return_pct*100:.1f}% | Reason: {signal.get('reason', '')}")
                    
                    return True
                    
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            return False
        
        return False
    
    def update_positions(self, current_prices: Dict[str, float]):
        total_unrealized = 0.0
        
        for position_key, position in self.positions.items():
            symbol1, symbol2 = position.pair
            
            if symbol1 in current_prices and symbol2 in current_prices:
                pct1 = (current_prices[symbol1] - position.entry_price1) / position.entry_price1
                pct2 = (current_prices[symbol2] - position.entry_price2) / position.entry_price2
                
                if position.position_type == 'LONG_SPREAD':
                    unrealized_pnl = position.size * (pct1 - pct2)
                else:
                    unrealized_pnl = position.size * (pct2 - pct1)
                
                position.unrealized_pnl = unrealized_pnl
                total_unrealized += unrealized_pnl
                
                if unrealized_pnl > position.max_favorable:
                    position.max_favorable = unrealized_pnl
                if unrealized_pnl < position.max_adverse:
                    position.max_adverse = unrealized_pnl
        
        total_equity = self.cash + sum(pos.size + pos.unrealized_pnl for pos in self.positions.values())
        self.equity_history.append({
            'timestamp': datetime.now(),
            'equity': total_equity,
            'cash': self.cash,
            'unrealized_pnl': total_unrealized,
            'num_positions': len(self.positions)
        })
        
        if len(self.equity_history) > 10000:
            self.equity_history = self.equity_history[-5000:]
    
    def check_risk_exits(self, current_prices: Dict[str, float]) -> List[Dict]:
        forced_exits = []
        current_time = datetime.now()
        
        for position_key, position in list(self.positions.items()):
            symbol1, symbol2 = position.pair
            
            if symbol1 in current_prices and symbol2 in current_prices:
                if abs(position.unrealized_pnl) > position.stop_loss_level:
                    forced_exits.append({
                        'strategy': position.strategy,
                        'pair': position.pair,
                        'action': 'CLOSE',
                        'reason': 'stop_loss',
                        'z_score': 0.0,
                        'confidence': 1.0
                    })
                
                hold_time = current_time - position.entry_time
                if hold_time > timedelta(hours=24):
                    forced_exits.append({
                        'strategy': position.strategy,
                        'pair': position.pair,
                        'action': 'CLOSE',
                        'reason': 'max_hold_time',
                        'z_score': 0.0,
                        'confidence': 1.0
                    })
                
                if position.confidence < 0.3:
                    forced_exits.append({
                        'strategy': position.strategy,
                        'pair': position.pair,
                        'action': 'CLOSE',
                        'reason': 'low_confidence',
                        'z_score': 0.0,
                        'confidence': 1.0
                    })
        
        return forced_exits
    
    def get_portfolio_summary(self) -> Dict:
        total_equity = self.cash + sum(pos.size + pos.unrealized_pnl for pos in self.positions.values())
        total_return = (total_equity - self.starting_capital) / self.starting_capital * 100
        
        return {
            'timestamp': datetime.now(),
            'total_equity': total_equity,
            'cash': self.cash,
            'total_return_pct': total_return,
            'num_positions': len(self.positions),
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
            'daily_pnl': self.daily_pnl
        }
    
    def save_state(self):
        state = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_summary': self.get_portfolio_summary(),
            'positions': {key: asdict(pos) for key, pos in self.positions.items()},
            'recent_trades': [asdict(trade) for trade in self.trades[-50:]],
            'strategy_performance': {k: v[-20:] for k, v in self.strategy_performance.items()}
        }
        
        with open(self.performance_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

class ExtendedSymbolUniverse:
    
    def __init__(self):
        self.sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'CRM', 'ADBE', 'NFLX', 'INTC'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK-B', 'AXP', 'V', 'MA'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE'],
            'Consumer': ['AMZN', 'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'COST', 'NKE', 'SBUX']
        }
        
        self.all_symbols = []
        for sector_symbols in self.sectors.values():
            self.all_symbols.extend(sector_symbols)
        
        self.logger = logging.getLogger(__name__)
        
    def get_sector_symbols(self, sector: str) -> List[str]:
        return self.sectors.get(sector, [])
    
    def get_cross_sector_pairs(self) -> List[Tuple[str, str, str, str]]:
        cross_pairs = []
        sectors = list(self.sectors.keys())
        
        for i, sector1 in enumerate(sectors):
            for j, sector2 in enumerate(sectors):
                if i < j:
                    symbols1 = self.sectors[sector1][:3]
                    symbols2 = self.sectors[sector2][:3]
                    
                    for s1 in symbols1:
                        for s2 in symbols2:
                            cross_pairs.append((s1, s2, sector1, sector2))
        
        return cross_pairs
    
    def get_liquid_symbols(self, max_symbols: int = 25) -> List[str]:
        liquid_symbols = []
        for sector, symbols in self.sectors.items():
            liquid_symbols.extend(symbols[:5])
        
        return liquid_symbols[:max_symbols]

class LivePairsTradingEngine:
    
    def __init__(self, max_symbols: int = 25):
        self.symbol_universe = ExtendedSymbolUniverse()
        self.trading_symbols = self.symbol_universe.get_liquid_symbols(max_symbols)
        
        self.market_hours = MarketHoursManager()
        self.data_feed = LiveDataFeed(self.trading_symbols, update_interval=60)
        self.portfolio = LivePortfolioManager()
        
        self.strategies = {'mean_reversion': {'name': 'Mean Reversion', 'active': True}}
        self.pair_discovery_interval = 3600
        self.last_pair_discovery = None
        self.active_pairs = []
        
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        
        print(f"🌐 Live Trading Engine: {len(self.trading_symbols)} symbols")
        print(f"🎯 Sectors: {list(self.symbol_universe.sectors.keys())}")
    
    def _initialize_live_strategies(self):
        return {
            'Live_Conservative': {'name': 'Conservative Mean Reversion', 'active': True},
            'Live_Momentum': {'name': 'Momentum Strategy', 'active': True},
            'Live_Hybrid': {'name': 'Hybrid Adaptive', 'active': True}
        }
    
    def discover_live_pairs(self) -> List[Tuple[str, str]]:
        self.logger.info("🔬 Discovering pairs from live market data...")
        
        pair_candidates = []
        symbols = []
        
        for symbol in self.trading_symbols:
            history = self.data_feed.get_price_history(symbol, periods=100)
            if len(history) >= 50:
                symbols.append(symbol)
        
        self.logger.info(f"   📊 Analyzing {len(symbols)} symbols with sufficient data")
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                hist1 = self.data_feed.get_price_history(symbol1, periods=80)
                hist2 = self.data_feed.get_price_history(symbol2, periods=80)
                
                if len(hist1) >= 50 and len(hist2) >= 50:
                    common_index = hist1.index.intersection(hist2.index)
                    if len(common_index) >= 40:
                        returns1 = hist1.loc[common_index]['returns'].dropna().values
                        returns2 = hist2.loc[common_index]['returns'].dropna().values
                        
                        if len(returns1) >= 30 and len(returns2) >= 30:
                            min_len = min(len(returns1), len(returns2))
                            returns1 = returns1[-min_len:]
                            returns2 = returns2[-min_len:]
                            
                            correlation = np.corrcoef(returns1, returns2)[0, 1] if not np.isnan(returns1).any() and not np.isnan(returns2).any() else 0
                            
                            volatility = (np.std(returns1) + np.std(returns2)) / 2
                            
                            trading_score = abs(correlation) * 0.4 + volatility * 10.0
                            
                            if abs(correlation) > 0.15 and volatility > 0.008:
                                pair_candidates.append({
                                    'pair': (symbol1, symbol2),
                                    'correlation': correlation,
                                    'volatility': volatility,
                                    'score': trading_score
                                })
        
        pair_candidates.sort(key=lambda x: x['score'], reverse=True)
        selected_pairs = []
        
        for candidate in pair_candidates[:15]:
            pair = candidate['pair']
            selected_pairs.append(pair)
            self.logger.info(f"   📈 Live Pair: {pair[0]}-{pair[1]} (ρ={candidate['correlation']:.3f}, vol={candidate['volatility']:.4f})")
        
        self.last_pair_discovery = datetime.now()
        return selected_pairs
    
    def generate_live_signals(self, current_prices: Dict[str, float]) -> List[Dict]:
        if not self.active_pairs:
            return []
        
        strategy_data = {}
        for symbol in self.trading_symbols:
            history = self.data_feed.get_price_history(symbol, periods=100)
            if len(history) >= 30:
                strategy_data[symbol] = history
        
        if len(strategy_data) < 3:
            return []
        
        all_signals = []
        # Simple demo signals
        if len(self.active_pairs) > 0:
            for pair in self.active_pairs[:2]:
                all_signals.append({
                    'pair': pair,
                    'action': 'LONG_SPREAD',
                    'strategy': 'Live_Demo',
                    'z_score': 2.1,
                    'confidence': 0.75,
                    'live_timestamp': datetime.now()
                })
        
        return all_signals
    
    def start_live_trading(self):
        self.logger.info("🚀 STARTING LIVE PAPER TRADING ENGINE")
        self.logger.info("=" * 60)
        
        if not self.market_hours.is_market_open():
            time_to_open = self.market_hours.time_to_market_open()
            self.logger.info(f"⏰ Market closed. Opens in: {time_to_open}")
            return
        
        self.is_running = True
        
        self.data_feed.start_feed()
        
        time.sleep(10)
        
        self.active_pairs = self.discover_live_pairs()
        
        self.logger.info(f"🎯 Live trading started with {len(self.active_pairs)} pairs")
        
        try:
            self._live_trading_loop()
        except KeyboardInterrupt:
            self.logger.info("🛑 Live trading stopped by user")
        except Exception as e:
            self.logger.error(f"Live trading error: {e}")
        finally:
            self.stop_live_trading()
    
    def _live_trading_loop(self):
        last_signal_time = datetime.now()
        signal_interval = 120
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                if not self.market_hours.is_market_open():
                    self.logger.info("📈 Market closed - stopping live trading")
                    break
                
                current_prices = self.data_feed.get_current_prices()
                
                if len(current_prices) >= 3:
                    self.portfolio.update_positions(current_prices)
                    
                    risk_signals = self.portfolio.check_risk_exits(current_prices)
                    for signal in risk_signals:
                        self.portfolio.execute_live_trade(signal, current_prices)
                    
                    if (current_time - last_signal_time).seconds >= signal_interval:
                        trading_signals = self.generate_live_signals(current_prices)
                        
                        for signal in trading_signals:
                            self.portfolio.execute_live_trade(signal, current_prices)
                        
                        last_signal_time = current_time
                    
                    if (self.last_pair_discovery is None or 
                        (current_time - self.last_pair_discovery).seconds >= self.pair_discovery_interval):
                        self.active_pairs = self.discover_live_pairs()
                    
                    if current_time.minute % 10 == 0:
                        self.portfolio.save_state()
                        self._log_status()
                
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
                time.sleep(60)
    
    def _log_status(self):
        summary = self.portfolio.get_portfolio_summary()
        self.logger.info(f"💼 Portfolio: ${summary['total_equity']:,.0f} | Return: {summary['total_return_pct']:.1f}% | Positions: {summary['num_positions']} | Cash: ${summary['cash']:,.0f}")
    
    def stop_live_trading(self):
        self.is_running = False
        self.data_feed.stop_feed()
        self.portfolio.save_state()
        self.logger.info("🛑 Live trading engine stopped")
    
    def get_performance_report(self) -> Dict:
        summary = self.portfolio.get_portfolio_summary()
        
        if len(self.portfolio.equity_history) > 1:
            equity_values = [e['equity'] for e in self.portfolio.equity_history]
            returns = np.diff(equity_values) / equity_values[:-1]
            
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.sqrt(252 * 24 * 60) * np.mean(returns) / np.std(returns)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        strategy_metrics = {}
        for strategy, performance in self.portfolio.strategy_performance.items():
            if performance:
                strategy_sharpe = np.sqrt(252) * np.mean(performance) / (np.std(performance) + 1e-8)
                win_rate = len([p for p in performance if p > 0]) / len(performance) * 100
                strategy_metrics[strategy] = {
                    'sharpe': strategy_sharpe,
                    'win_rate': win_rate,
                    'trade_count': len(performance)
                }
        
        return {
            'live_summary': summary,
            'sharpe_ratio': sharpe,
            'total_trades': len(self.portfolio.trades),
            'active_pairs': len(self.active_pairs),
            'strategy_metrics': strategy_metrics,
            'data_quality': {
                'symbols_with_data': len(self.data_feed.current_prices),
                'last_update': self.data_feed.last_update
            }
        }

def demo_live_trading(duration_minutes: int = 30):
    print("🌐 LIVE PAPER TRADING DEMO")
    print("=" * 50)
    
    engine = LivePairsTradingEngine(max_symbols=15)
    
    print(f"📊 Trading Universe: {len(engine.trading_symbols)} symbols")
    print(f"🏛️ Sectors: {list(engine.symbol_universe.sectors.keys())}")
    
    if engine.market_hours.is_market_open():
        print("✅ Market is OPEN - Starting live demo")
        engine.start_live_trading()
    else:
        time_to_open = engine.market_hours.time_to_market_open()
        print(f"⏰ Market CLOSED - Opens in: {time_to_open}")
        print("🎯 Running simulation with live data structures...")
        
        engine.data_feed.start_feed()
        time.sleep(5)
        
        pairs = engine.discover_live_pairs()
        current_prices = engine.data_feed.get_current_prices()
        
        if current_prices:
            signals = engine.generate_live_signals(current_prices)
            print(f"📡 Generated {len(signals)} live signals")
            
            for signal in signals[:3]:
                engine.portfolio.execute_live_trade(signal, current_prices)
        
        report = engine.get_performance_report()
        print(f"\n📊 LIVE DEMO RESULTS:")
        print(f"Portfolio Value: ${report['live_summary']['total_equity']:,.0f}")
        print(f"Active Positions: {report['live_summary']['num_positions']}")
        print(f"Symbols Tracked: {report['data_quality']['symbols_with_data']}")
        
        engine.stop_live_trading()

if __name__ == "__main__":
    demo_live_trading()
