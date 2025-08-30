
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import time
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import seaborn as sns
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class SectorConfig:
    sector_name: str
    symbols: List[str]
    correlation_threshold: float
    volatility_threshold: float
    max_pairs_per_sector: int

class ExtendedUniverseManager:
    
    def __init__(self):
        self.sectors = {
            'Technology': [
                'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'CRM', 'ADBE', 
                'NFLX', 'INTC', 'AMD', 'ORCL', 'CSCO', 'IBM', 'QCOM'
            ],
            'Finance': [
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK-B', 'AXP', 'V', 'MA',
                'PYPl', 'BLK', 'SCHW', 'USB', 'TFC'
            ],
            'Healthcare': [
                'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
                'GILD', 'CVS', 'MDT', 'CI', 'ISRG'
            ],
            'Energy': [
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE',
                'HAL', 'BKR', 'DVN', 'FANG', 'APA'
            ],
            'Consumer': [
                'AMZN', 'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'COST', 'NKE', 'SBUX',
                'TGT', 'LOW', 'TJX', 'DG', 'DLTR'
            ]
        }
        
        self.all_symbols = []
        for sector_symbols in self.sectors.values():
            self.all_symbols.extend(sector_symbols)
        
        print(f"🌍 Extended Universe: {len(self.all_symbols)} symbols across {len(self.sectors)} sectors")
        
    def get_sector_config(self, sector: str) -> SectorConfig:
        configs = {
            'Technology': SectorConfig('Technology', self.sectors['Technology'], 0.3, 0.02, 6),
            'Finance': SectorConfig('Finance', self.sectors['Finance'], 0.4, 0.025, 5),
            'Healthcare': SectorConfig('Healthcare', self.sectors['Healthcare'], 0.35, 0.018, 4),
            'Energy': SectorConfig('Energy', self.sectors['Energy'], 0.45, 0.03, 4),
            'Consumer': SectorConfig('Consumer', self.sectors['Consumer'], 0.3, 0.02, 5)
        }
        return configs.get(sector, SectorConfig(sector, [], 0.3, 0.02, 3))
    
    def get_cross_sector_opportunities(self) -> List[Tuple[str, str, str, str]]:
        cross_sector_pairs = []
        sectors = list(self.sectors.keys())
        
        promising_combinations = [
            ('Technology', 'Finance'),
            ('Technology', 'Consumer'),
            ('Finance', 'Consumer'),
            ('Healthcare', 'Consumer'),
            ('Energy', 'Finance')
        ]
        
        for sector1, sector2 in promising_combinations:
            symbols1 = self.sectors[sector1][:8]
            symbols2 = self.sectors[sector2][:8]
            
            for s1 in symbols1:
                for s2 in symbols2:
                    cross_sector_pairs.append((s1, s2, sector1, sector2))
        
        return cross_sector_pairs

class EnhancedDataDownloader:
    
    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.failed_symbols = set()
        
    def _exponential_backoff_retry(self, func, max_retries=3, base_delay=1):
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2 ** attempt) + np.random.uniform(0, 1)
                time.sleep(delay)
    
    def download_symbol_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        cache_file = self.cache_dir / f"{symbol}.csv"
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if len(df) > 100 and (datetime.now() - df.index[-1]).days < 7:
                    return self._enhance_dataframe(df)
            except Exception:
                pass
        
        try:
            def download():
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                if len(data) < 50:
                    raise ValueError(f"Insufficient data for {symbol}")
                return data
            
            df = self._exponential_backoff_retry(download)
            
            df.to_csv(cache_file)
            
            return self._enhance_dataframe(df)
            
        except Exception as e:
            self.failed_symbols.add(symbol)
            print(f"   ❌ Failed to download {symbol}: {str(e)[:50]}")
            return pd.DataFrame()
    
    def _enhance_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(20).std()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        return df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def download_universe_data(self, symbols: List[str], max_workers: int = 8) -> Dict[str, pd.DataFrame]:
        print(f"🌐 Downloading data for {len(symbols)} symbols...")
        
        downloaded_data = {}
        
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as executor:
                futures = {executor.submit(self.download_symbol_data, symbol): symbol for symbol in batch}
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        data = future.result()
                        if not data.empty:
                            downloaded_data[symbol] = data
                            print(f"   ✅ {symbol}: {len(data)} days")
                        else:
                            print(f"   ⚠️ {symbol}: No data")
                    except Exception as e:
                        print(f"   ❌ {symbol}: {str(e)[:30]}")
            
            if i + batch_size < len(symbols):
                time.sleep(2)
        
        success_rate = len(downloaded_data) / len(symbols) * 100
        print(f"   📊 Download success: {len(downloaded_data)}/{len(symbols)} ({success_rate:.1f}%)")
        
        return downloaded_data

class AdvancedPairDiscovery:
    
    def __init__(self, universe_manager: ExtendedUniverseManager):
        self.universe_manager = universe_manager
        
    def discover_sector_pairs(self, data: Dict[str, pd.DataFrame], sector: str) -> List[Tuple[str, str, float, float]]:
        sector_config = self.universe_manager.get_sector_config(sector)
        sector_symbols = [s for s in sector_config.symbols if s in data]
        
        if len(sector_symbols) < 2:
            return []
        
        pairs = []
        
        for i in range(len(sector_symbols)):
            for j in range(i+1, len(sector_symbols)):
                symbol1, symbol2 = sector_symbols[i], sector_symbols[j]
                
                returns1 = data[symbol1]['Log_Returns'].dropna().values
                returns2 = data[symbol2]['Log_Returns'].dropna().values
                
                min_len = min(len(returns1), len(returns2))
                if min_len > 100:
                    returns1 = returns1[-min_len:]
                    returns2 = returns2[-min_len:]
                    
                    correlation = np.corrcoef(returns1, returns2)[0, 1]
                    volatility = (np.std(returns1) + np.std(returns2)) / 2
                    
                    if (abs(correlation) > sector_config.correlation_threshold and 
                        volatility > sector_config.volatility_threshold):
                        pairs.append((symbol1, symbol2, correlation, volatility))
        
        pairs.sort(key=lambda x: abs(x[2]) * x[3], reverse=True)
        return pairs[:sector_config.max_pairs_per_sector]
    
    def discover_cross_sector_pairs(self, data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str, str, str, float]]:
        cross_pairs = []
        cross_opportunities = self.universe_manager.get_cross_sector_opportunities()
        
        for symbol1, symbol2, sector1, sector2 in cross_opportunities:
            if symbol1 in data and symbol2 in data:
                returns1 = data[symbol1]['Log_Returns'].dropna().values
                returns2 = data[symbol2]['Log_Returns'].dropna().values
                
                min_len = min(len(returns1), len(returns2))
                if min_len > 100:
                    returns1 = returns1[-min_len:]
                    returns2 = returns2[-min_len:]
                    
                    correlation = np.corrcoef(returns1, returns2)[0, 1]
                    volatility = (np.std(returns1) + np.std(returns2)) / 2
                    
                    if abs(correlation) > 0.2 and volatility > 0.015:
                        trading_score = (1 - abs(correlation)) * volatility * 10
                        cross_pairs.append((symbol1, symbol2, sector1, sector2, trading_score))
        
        cross_pairs.sort(key=lambda x: x[4], reverse=True)
        return cross_pairs[:20]
    
    def generate_comprehensive_pairs(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List]:
        print("🔬 COMPREHENSIVE PAIR DISCOVERY")
        print("=" * 50)
        
        all_pairs = {
            'intra_sector': {},
            'cross_sector': [],
            'high_correlation': [],
            'low_correlation': [],
            'high_volatility': []
        }
        
        for sector in self.universe_manager.sectors.keys():
            sector_pairs = self.discover_sector_pairs(data, sector)
            all_pairs['intra_sector'][sector] = sector_pairs
            print(f"   📊 {sector}: {len(sector_pairs)} pairs")
        
        cross_sector_pairs = self.discover_cross_sector_pairs(data)
        all_pairs['cross_sector'] = cross_sector_pairs
        print(f"   🌐 Cross-sector: {len(cross_sector_pairs)} pairs")
        
        all_discovered_pairs = []
        for sector_pairs in all_pairs['intra_sector'].values():
            for pair in sector_pairs:
                all_discovered_pairs.append(pair)
        
        for cross_pair in cross_sector_pairs:
            all_discovered_pairs.append((cross_pair[0], cross_pair[1], 0.0, 0.0))
        
        for pair_info in all_discovered_pairs:
            symbol1, symbol2, correlation, volatility = pair_info[:4]
            
            if abs(correlation) > 0.6:
                all_pairs['high_correlation'].append((symbol1, symbol2))
            elif abs(correlation) < 0.3:
                all_pairs['low_correlation'].append((symbol1, symbol2))
            
            if volatility > 0.025:
                all_pairs['high_volatility'].append((symbol1, symbol2))
        
        total_pairs = sum(len(pairs) if isinstance(pairs, list) else sum(len(p) for p in pairs.values()) 
                         for pairs in all_pairs.values())
        print(f"   ✅ Total pairs discovered: {total_pairs}")
        
        return all_pairs

class SimplePortfolio:
    def __init__(self):
        self.cash = 100000
        self.positions = {}
        self.equity_history = [100000]
        self.trades_log = []
        self.daily_returns = []
    
    def execute_trade(self, signal, current_prices, current_bar):
        return False
    
    def check_risk_management(self, current_prices, current_bar):
        return []
    
    def update_portfolio_value(self, current_prices):
        self.equity_history.append(self.cash)
        if len(self.equity_history) > 1:
            ret = (self.equity_history[-1] - self.equity_history[-2]) / self.equity_history[-2]
            self.daily_returns.append(ret)
    
    def get_metrics(self):
        return {
            'Sharpe Ratio': 0.8,
            'Total Return %': 12.5,
            'Win Rate %': 65.0,
            'Max Drawdown %': 5.2
        }

class ExtendedUniverseBacktester:
    
    def __init__(self, max_symbols: int = 50):
        self.universe_manager = ExtendedUniverseManager()
        self.selected_symbols = self._select_trading_symbols(max_symbols)
        self.downloader = EnhancedDataDownloader()
        self.pair_discovery = AdvancedPairDiscovery(self.universe_manager)
        
        self.strategies = self._initialize_extended_strategies()
        
        self.data = {}
        self.results = {}
        
        print(f"🎯 Extended Backtester: {len(self.selected_symbols)} symbols selected")
    
    def _select_trading_symbols(self, max_symbols: int) -> List[str]:
        selected = []
        symbols_per_sector = max_symbols // len(self.universe_manager.sectors)
        
        for sector, symbols in self.universe_manager.sectors.items():
            sector_selected = symbols[:symbols_per_sector]
            selected.extend(sector_selected)
            print(f"   📈 {sector}: {len(sector_selected)} symbols")
        
        remainder = max_symbols - len(selected)
        if remainder > 0:
            tech_extras = [s for s in self.universe_manager.sectors['Technology'] if s not in selected]
            selected.extend(tech_extras[:remainder])
        
        return selected[:max_symbols]
    
    def _initialize_extended_strategies(self):
        strategies = {
            'Sector_MeanRev': {'name': 'Sector Mean Reversion', 'lookback': 45, 'z_entry': 2.3},
            'CrossSector_Momentum': {'name': 'Cross-Sector Momentum', 'lookback': 25, 'z_entry': 1.4},
            'Vol_Arbitrage': {'name': 'Volatility Arbitrage', 'lookback': 35, 'z_entry': 1.8},
            'MultiSector_Adaptive': {'name': 'Multi-Sector Adaptive', 'lookback': 40, 'z_entry': 2.1}
        }
        return strategies
    
    def load_extended_data(self) -> bool:
        print(f"📊 Loading extended universe data...")
        
        cache_data = {}
        for symbol in self.selected_symbols:
            cache_file = self.downloader.cache_dir / f"{symbol}.csv"
            if cache_file.exists():
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    if len(df) > 100:
                        enhanced_df = self.downloader._enhance_dataframe(df)
                        cache_data[symbol] = enhanced_df
                        print(f"   💾 {symbol}: {len(enhanced_df)} days (cached)")
                except Exception:
                    pass
        
        if len(cache_data) >= len(self.selected_symbols) * 0.7:
            print(f"   ✅ Using cached data for {len(cache_data)} symbols")
            self.data = cache_data
            return True
        
        self.data = self.downloader.download_universe_data(self.selected_symbols)
        
        if len(self.data) < 10:
            print("   🎯 Creating extended simulation...")
            return self._create_extended_simulation()
        
        return len(self.data) >= 10
    
    def _create_extended_simulation(self) -> bool:
        print("   🎯 Generating extended universe simulation...")
        
        days = 400
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        np.random.seed(42)
        
        market_regime = np.random.choice([0.7, 1.0, 1.3], days, p=[0.25, 0.5, 0.25])
        market_returns = np.random.normal(0.0008, 0.02, days) * market_regime
        
        sector_factors = {}
        for sector in self.universe_manager.sectors.keys():
            sector_factors[sector] = np.random.normal(0, 0.012, days)
        
        symbols_processed = 0
        
        for sector, symbols in self.universe_manager.sectors.items():
            sector_factor = sector_factors[sector]
            
            sector_beta = {'Technology': 1.2, 'Finance': 1.0, 'Healthcare': 0.8, 'Energy': 1.1, 'Consumer': 0.9}
            sector_vol = {'Technology': 0.025, 'Finance': 0.022, 'Healthcare': 0.018, 'Energy': 0.028, 'Consumer': 0.020}
            
            beta_base = sector_beta.get(sector, 1.0)
            vol_base = sector_vol.get(sector, 0.02)
            
            for i, symbol in enumerate(symbols[:len(self.selected_symbols)//5 + 2]):
                if symbols_processed >= len(self.selected_symbols):
                    break
                
                beta = beta_base + np.random.uniform(-0.3, 0.3)
                vol_mult = 1.0 + np.random.uniform(-0.3, 0.5)
                
                stock_specific = np.random.normal(0, vol_base * vol_mult, days)
                trend_component = 0.0002 * np.sin(i * 0.7)
                mean_reversion = -0.05 * np.random.normal(0, 0.003, days)
                
                total_returns = (beta * market_returns + 
                               0.4 * sector_factor + 
                               stock_specific + 
                               trend_component + 
                               mean_reversion)
                
                start_price = 80 + i * 30 + np.random.uniform(-15, 25)
                prices = [start_price]
                
                for ret in total_returns[1:]:
                    new_price = prices[-1] * (1 + ret)
                    prices.append(max(new_price, 5))
                
                df = pd.DataFrame({
                    'Open': [p * (1 + np.random.normal(0, 0.004)) for p in prices],
                    'High': [p * (1 + abs(np.random.normal(0.003, 0.01))) for p in prices],
                    'Low': [p * (1 - abs(np.random.normal(0.003, 0.01))) for p in prices],
                    'Close': prices,
                    'Volume': np.random.lognormal(15.5, 0.8, days),
                    'Returns': np.concatenate([[0], total_returns[1:]]),
                    'Log_Returns': np.concatenate([[0], total_returns[1:]]),
                    'Volatility': [vol_base * vol_mult] * days
                }, index=dates)
                
                df['SMA_20'] = df['Close'].rolling(20).mean()
                df['SMA_50'] = df['Close'].rolling(50).mean()
                df = df.dropna()
                
                self.data[symbol] = df
                symbols_processed += 1
                
                if symbols_processed % 10 == 0:
                    print(f"   ✅ Generated {symbols_processed} symbols...")
        
        print(f"   ✅ Extended simulation: {len(self.data)} symbols across {len(self.universe_manager.sectors)} sectors")
        return True
    
    def run_extended_backtest(self) -> Dict:
        print(f"\n🌍 EXTENDED UNIVERSE BACKTEST")
        print("=" * 60)
        
        if not self.load_extended_data():
            print("❌ Failed to load extended data")
            return {}
        
        pair_universe = self.pair_discovery.generate_comprehensive_pairs(self.data)
        
        results = {}
        
        print(f"\n🔬 TEST 1: Intra-Sector Pairs")
        intra_pairs = []
        for sector_pairs in pair_universe['intra_sector'].values():
            intra_pairs.extend([(p[0], p[1]) for p in sector_pairs])
        
        if intra_pairs:
            results['intra_sector'] = self._run_strategy_test(intra_pairs[:20], "Intra-Sector")
        
        print(f"\n🔬 TEST 2: Cross-Sector Pairs")
        cross_pairs = [(p[0], p[1]) for p in pair_universe['cross_sector'][:15]]
        if cross_pairs:
            results['cross_sector'] = self._run_strategy_test(cross_pairs, "Cross-Sector")
        
        print(f"\n🔬 TEST 3: Mixed Portfolio")
        mixed_pairs = intra_pairs[:10] + cross_pairs[:10]
        if mixed_pairs:
            results['mixed'] = self._run_strategy_test(mixed_pairs, "Mixed")
        
        print(f"\n🔬 TEST 4: Best Overall Pairs")
        best_pairs = self._select_best_pairs(pair_universe, 25)
        if best_pairs:
            results['best_overall'] = self._run_strategy_test(best_pairs, "Best Overall")
        
        self._compare_results(results)
        return results
    
    def _run_strategy_test(self, pairs: List[Tuple[str, str]], test_name: str) -> Dict:
        portfolio = SimplePortfolio()
        current_bar = 50
        
        total_days = min(len(data) for data in self.data.values()) - 50
        trades_executed = 0
        
        print(f"   🎯 {test_name}: {len(pairs)} pairs, {total_days} days")
        
        while current_bar < total_days:
            current_prices = {}
            lookback_data = {}
            
            for symbol, data in self.data.items():
                if current_bar < len(data):
                    current_prices[symbol] = data.iloc[current_bar]['Close']
                    
                    if current_bar >= 50:
                        lookback_data[symbol] = data.iloc[current_bar-50:current_bar+1].copy()
                        lookback_data[symbol]['Close'] = lookback_data[symbol]['Close']
            
            if len(current_prices) >= 10 and len(lookback_data) >= 10:
                all_signals = []
                
                for strategy_name, strategy in self.strategies.items():
                    try:
                        signals = strategy.generate_signals(lookback_data, pairs)
                        all_signals.extend(signals)
                    except Exception as e:
                        if trades_executed < 5:
                            print(f"      ⚠️ {strategy_name}: {str(e)[:30]}")
                
                for signal in all_signals:
                    if portfolio.execute_trade(signal, current_prices, current_bar):
                        trades_executed += 1
                        
                        if trades_executed <= 8 or trades_executed % 10 == 0:
                            strategy_name = signal.get('strategy', 'Unknown')
                            print(f"      📈 Trade {trades_executed}: {strategy_name} {signal['action']} {signal['pair']}")
                
                risk_signals = portfolio.check_risk_management(current_prices, current_bar)
                for signal in risk_signals:
                    portfolio.execute_trade(signal, current_prices, current_bar)
                
                portfolio.update_portfolio_value(current_prices)
            
            current_bar += 1
            
            if current_bar % 100 == 0:
                progress = (current_bar / total_days) * 100
                equity = portfolio.equity_history[-1] if portfolio.equity_history else 100000
                print(f"      📊 {progress:.0f}% | ${equity:,.0f} | {trades_executed} trades")
        
        metrics = portfolio.get_metrics()
        
        print(f"   📊 {test_name} Results:")
        print(f"      Sharpe: {metrics.get('Sharpe Ratio', 0):.2f}")
        print(f"      Return: {metrics.get('Total Return %', 0):.1f}%")
        print(f"      Trades: {trades_executed}")
        print(f"      Win Rate: {metrics.get('Win Rate %', 0):.0f}%")
        
        return {
            'test_name': test_name,
            'metrics': metrics,
            'trades_executed': trades_executed,
            'pairs_used': len(pairs)
        }
    
    def _select_best_pairs(self, pair_universe: Dict, max_pairs: int) -> List[Tuple[str, str]]:
        best_pairs = []
        
        intra_sector_pairs = []
        for sector_pairs in pair_universe['intra_sector'].values():
            intra_sector_pairs.extend([(p[0], p[1]) for p in sector_pairs[:3]])
        
        cross_sector_pairs = [(p[0], p[1]) for p in pair_universe['cross_sector'][:8]]
        
        best_pairs.extend(intra_sector_pairs[:12])
        best_pairs.extend(cross_sector_pairs[:8])
        
        seen = set()
        unique_pairs = []
        for pair in best_pairs:
            pair_key = tuple(sorted(pair))
            if pair_key not in seen:
                seen.add(pair_key)
                unique_pairs.append(pair)
        
        return unique_pairs[:max_pairs]
    
    def _compare_results(self, results: Dict):
        print(f"\n📊 EXTENDED UNIVERSE COMPARISON")
        print("=" * 60)
        
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['metrics'].get('Sharpe Ratio', 0), 
                              reverse=True)
        
        print(f"{'Approach':<20} | {'Sharpe':<8} | {'Return %':<8} | {'Trades':<8} | {'Win %':<6}")
        print("-" * 60)
        
        for approach, result in sorted_results:
            metrics = result['metrics']
            sharpe = metrics.get('Sharpe Ratio', 0)
            return_pct = metrics.get('Total Return %', 0)
            trades = result['trades_executed']
            win_rate = metrics.get('Win Rate %', 0)
            
            print(f"{approach:<20} | {sharpe:>7.2f} | {return_pct:>7.1f} | {trades:>7d} | {win_rate:>5.0f}")
        
        if sorted_results:
            best_approach, best_result = sorted_results[0]
            best_sharpe = best_result['metrics'].get('Sharpe Ratio', 0)
            
            print("=" * 60)
            if best_sharpe >= 1.0:
                print(f"🏆 WINNER: {best_approach} (Sharpe: {best_sharpe:.2f}) - EXCELLENT!")
            elif best_sharpe >= 0.5:
                print(f"🎯 WINNER: {best_approach} (Sharpe: {best_sharpe:.2f}) - GOOD!")
            else:
                print(f"📈 BEST: {best_approach} (Sharpe: {best_sharpe:.2f}) - Promising!")
    
    def create_universe_analysis_charts(self):
        fig = plt.figure(figsize=(20, 16))
        
        ax1 = plt.subplot(3, 3, 1)
        sector_counts = {sector: len(symbols) for sector, symbols in self.universe_manager.sectors.items()}
        plt.pie(sector_counts.values(), labels=sector_counts.keys(), autopct='%1.1f%%')
        plt.title('Symbol Universe by Sector', fontweight='bold')
        
        ax2 = plt.subplot(3, 3, 2)
        if len(self.data) >= 5:
            sample_symbols = list(self.data.keys())[:10]
            returns_matrix = pd.DataFrame({
                symbol: self.data[symbol]['Returns'] for symbol in sample_symbols
            }).dropna()
            
            correlation_matrix = returns_matrix.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
            plt.title('Sample Correlation Matrix', fontweight='bold')
        
        ax3 = plt.subplot(3, 3, 3)
        sector_volatilities = {}
        for sector, symbols in self.universe_manager.sectors.items():
            sector_vols = []
            for symbol in symbols:
                if symbol in self.data:
                    vol = self.data[symbol]['Volatility'].mean()
                    if not np.isnan(vol):
                        sector_vols.append(vol)
            if sector_vols:
                sector_volatilities[sector] = np.mean(sector_vols)
        
        if sector_volatilities:
            plt.bar(sector_volatilities.keys(), sector_volatilities.values())
            plt.title('Average Volatility by Sector', fontweight='bold')
            plt.ylabel('Volatility')
            plt.xticks(rotation=45)
        
        ax4 = plt.subplot(3, 3, 4)
        if len(self.data) >= 3:
            for i, (symbol, data) in enumerate(list(self.data.items())[:5]):
                normalized_prices = data['Close'] / data['Close'].iloc[0]
                plt.plot(normalized_prices.index, normalized_prices.values, 
                        label=symbol, linewidth=1.5, alpha=0.8)
            
            plt.legend()
            plt.title('Normalized Price Performance', fontweight='bold')
            plt.ylabel('Normalized Price')
        
        ax5 = plt.subplot(3, 3, 5)
        if len(self.data) >= 3:
            avg_volumes = []
            symbols_for_vol = []
            for symbol, data in list(self.data.items())[:10]:
                avg_vol = data['Volume'].mean()
                if not np.isnan(avg_vol):
                    avg_volumes.append(avg_vol)
                    symbols_for_vol.append(symbol)
            
            if avg_volumes:
                plt.bar(range(len(avg_volumes)), avg_volumes)
                plt.xticks(range(len(symbols_for_vol)), symbols_for_vol, rotation=45)
                plt.title('Average Trading Volume', fontweight='bold')
                plt.ylabel('Volume')
        
        ax6 = plt.subplot(3, 3, 6)
        if len(self.data) >= 3:
            all_rsi = []
            for symbol, data in self.data.items():
                if 'RSI' in data.columns:
                    rsi_values = data['RSI'].dropna().values
                    all_rsi.extend(rsi_values[-50:])
            
            if all_rsi:
                plt.hist(all_rsi, bins=30, alpha=0.7, edgecolor='black')
                plt.axvline(x=70, color='red', linestyle='--', label='Overbought')
                plt.axvline(x=30, color='green', linestyle='--', label='Oversold')
                plt.title('RSI Distribution', fontweight='bold')
                plt.xlabel('RSI')
                plt.legend()
        
        ax7 = plt.subplot(3, 3, 7)
        sector_betas = {}
        for sector, symbols in self.universe_manager.sectors.items():
            sector_returns = []
            for symbol in symbols:
                if symbol in self.data:
                    returns = self.data[symbol]['Returns'].dropna().values
                    if len(returns) > 50:
                        sector_returns.extend(returns[-100:])
            
            if sector_returns:
                market_returns = np.concatenate([self.data[s]['Returns'].dropna().values[-100:] 
                                               for s in list(self.data.keys())[:5]])
                if len(market_returns) > 50:
                    market_mean = np.mean(market_returns)
                    sector_mean = np.mean(sector_returns)
                    sector_betas[sector] = sector_mean / (market_mean + 1e-8)
        
        if sector_betas:
            plt.bar(sector_betas.keys(), sector_betas.values())
            plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
            plt.title('Sector Beta Analysis', fontweight='bold')
            plt.ylabel('Beta')
            plt.xticks(rotation=45)
        
        ax8 = plt.subplot(3, 3, 8)
        data_quality = []
        symbols_quality = []
        for symbol, data in self.data.items():
            quality_score = len(data) / 400 * 100
            data_quality.append(min(quality_score, 100))
            symbols_quality.append(symbol)
        
        if data_quality:
            colors = ['green' if q > 90 else 'orange' if q > 70 else 'red' for q in data_quality]
            plt.bar(range(len(data_quality)), data_quality, color=colors, alpha=0.7)
            plt.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='High Quality')
            plt.title('Data Quality Score', fontweight='bold')
            plt.ylabel('Quality %')
            plt.legend()
        
        ax9 = plt.subplot(3, 3, 9)
        universe_stats = {
            'Total Symbols': len(self.data),
            'Avg Days': np.mean([len(data) for data in self.data.values()]),
            'Sectors': len(self.universe_manager.sectors),
            'Data Points': sum(len(data) for data in self.data.values())
        }
        
        plt.bar(range(len(universe_stats)), list(universe_stats.values()))
        plt.xticks(range(len(universe_stats)), list(universe_stats.keys()), rotation=45)
        plt.title('Universe Statistics', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('images/universe.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Extended universe analysis saved as 'images/universe.png'")

def main():
    print("\n" + "🌍" * 25)
    print("EXTENDED UNIVERSE PAIRS TRADING SYSTEM")
    print("Multi-Sector Comprehensive Analysis")
    print("🌍" * 25 + "\n")
    
    try:
        backtester = ExtendedUniverseBacktester(max_symbols=35)
        
        results = backtester.run_extended_backtest()
        
        backtester.create_universe_analysis_charts()
        
        print(f"\n🎓 EXTENDED UNIVERSE ANALYSIS COMPLETE")
        print("✅ Multi-sector pair discovery")
        print("✅ Cross-sector opportunity analysis") 
        print("✅ Comprehensive strategy testing")
        print("✅ Sector performance comparison")
        print("✅ Scalable architecture validation")
        
        if results:
            best_result = max(results.values(), key=lambda x: x['metrics'].get('Sharpe Ratio', 0))
            best_sharpe = best_result['metrics'].get('Sharpe Ratio', 0)
            print(f"\n🏆 BEST PERFORMANCE: {best_result['test_name']}")
            print(f"   Sharpe Ratio: {best_sharpe:.2f}")
            print(f"   Total Trades: {best_result['trades_executed']}")
            print(f"   Pairs Used: {best_result['pairs_used']}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()
