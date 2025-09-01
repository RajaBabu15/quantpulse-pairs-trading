"""
Trading Strategies Module
========================

Professional trading strategies with statistical arbitrage, mean reversion,
and machine learning enhanced signals.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, shapiro
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

import logging
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Trading signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0

class StrategyType(Enum):
    """Strategy classification."""
    PAIRS_TRADING = "pairs_trading"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"  
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ML_ENHANCED = "ml_enhanced"

@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    timestamp: datetime
    symbol1: str
    symbol2: str
    signal_type: SignalType
    strength: float
    confidence: float
    entry_price1: float
    entry_price2: float
    hedge_ratio: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class StrategyParameters:
    """Base strategy parameters."""
    lookback_window: int = 60
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    stop_loss: float = 0.05
    take_profit: float = 0.03
    min_half_life: int = 5
    max_half_life: int = 60
    min_correlation: float = 0.7
    max_position_size: float = 0.1

class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, parameters: StrategyParameters):
        self.parameters = parameters
        self.strategy_type = StrategyType.PAIRS_TRADING
        self.signals_history = []
        self.performance_metrics = {}
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Generate trading signals from market data."""
        pass
    
    @abstractmethod
    def calculate_hedge_ratio(self, price1: pd.Series, price2: pd.Series) -> float:
        """Calculate optimal hedge ratio between two assets."""
        pass
    
    def validate_pair(self, price1: pd.Series, price2: pd.Series) -> bool:
        """Validate if pair is suitable for trading."""
        print(f"🔄 ENTERING validate_pair() at {datetime.now().strftime('%H:%M:%S')}")
        
        # Check data length
        if len(price1) < self.parameters.lookback_window or len(price2) < self.parameters.lookback_window:
            logger.warning("Insufficient data for validation")
            return False
        
        # Check correlation
        correlation = price1.corr(price2)
        if abs(correlation) < self.parameters.min_correlation:
            logger.warning(f"Low correlation: {correlation:.3f}")
            return False
        
        # Check for cointegration (Augmented Dickey-Fuller test)
        try:
            from statsmodels.tsa.stattools import coint
            score, pvalue, _ = coint(price1, price2)
            
            if pvalue > 0.05:
                logger.warning(f"Pair not cointegrated: p-value={pvalue:.3f}")
                return False
        except ImportError:
            logger.warning("Statsmodels not available, skipping cointegration test")
        except Exception as e:
            logger.warning(f"Cointegration test failed: {e}")
        
        print(f"✅ EXITING validate_pair() at {datetime.now().strftime('%H:%M:%S')}")
        return True
    
    def calculate_spread(self, price1: pd.Series, price2: pd.Series, hedge_ratio: float) -> pd.Series:
        """Calculate price spread."""
        return price1 - hedge_ratio * price2
    
    def calculate_zscore(self, spread: pd.Series, window: int = None) -> pd.Series:
        """Calculate rolling z-score of spread."""
        if window is None:
            window = self.parameters.lookback_window
        
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        
        return (spread - rolling_mean) / rolling_std

class PairsTradingStrategy(BaseStrategy):
    """Classic pairs trading strategy with statistical arbitrage."""
    
    def __init__(self, parameters: StrategyParameters = None):
        print(f"🔄 ENTERING PairsTradingStrategy.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        if parameters is None:
            parameters = StrategyParameters()
        
        super().__init__(parameters)
        self.strategy_type = StrategyType.PAIRS_TRADING
        
        print(f"✅ EXITING PairsTradingStrategy.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def calculate_hedge_ratio(self, price1: pd.Series, price2: pd.Series) -> float:
        """Calculate hedge ratio using Ordinary Least Squares."""
        print(f"🔄 ENTERING calculate_hedge_ratio() at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Remove any NaN values
            data = pd.DataFrame({'price1': price1, 'price2': price2}).dropna()
            
            if len(data) < 10:
                logger.warning("Insufficient data for hedge ratio calculation")
                return 1.0
            
            # OLS regression: price1 = beta * price2 + alpha
            X = data['price2'].values.reshape(-1, 1)
            y = data['price1'].values
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            hedge_ratio = model.coef_[0]
            r_squared = model.score(X, y)
            
            logger.info(f"Hedge ratio: {hedge_ratio:.4f}, R²: {r_squared:.4f}")
            
            print(f"✅ EXITING calculate_hedge_ratio() at {datetime.now().strftime('%H:%M:%S')}")
            return hedge_ratio
            
        except Exception as e:
            logger.error(f"Error calculating hedge ratio: {e}")
            return 1.0
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate mean reversion half-life."""
        print(f"🔄 ENTERING calculate_half_life() at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Calculate lagged spread
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            
            # Align series
            common_index = spread_lag.index.intersection(spread_diff.index)
            spread_lag = spread_lag.loc[common_index]
            spread_diff = spread_diff.loc[common_index]
            
            if len(spread_lag) < 10:
                return np.inf
            
            # Regression: Δspread = α + β * spread_lag
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            X = spread_lag.values.reshape(-1, 1)
            y = spread_diff.values
            
            model.fit(X, y)
            beta = model.coef_[0]
            
            if beta >= 0:
                return np.inf  # No mean reversion
            
            half_life = -np.log(2) / beta
            
            print(f"✅ EXITING calculate_half_life() at {datetime.now().strftime('%H:%M:%S')}")
            return half_life
            
        except Exception as e:
            logger.error(f"Error calculating half-life: {e}")
            return np.inf
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Generate pairs trading signals."""
        print(f"🔄 ENTERING generate_signals() at {datetime.now().strftime('%H:%M:%S')}")
        
        signals = []
        
        # Get all possible pairs
        symbols = list(data.keys())
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                try:
                    df1, df2 = data[symbol1], data[symbol2]
                    
                    # Align data
                    common_index = df1.index.intersection(df2.index)
                    if len(common_index) < self.parameters.lookback_window:
                        continue
                    
                    price1 = df1.loc[common_index]['Close']
                    price2 = df2.loc[common_index]['Close']
                    
                    # Validate pair
                    if not self.validate_pair(price1, price2):
                        continue
                    
                    # Calculate hedge ratio and spread
                    hedge_ratio = self.calculate_hedge_ratio(price1, price2)
                    spread = self.calculate_spread(price1, price2, hedge_ratio)
                    
                    # Check mean reversion properties
                    half_life = self.calculate_half_life(spread)
                    if not (self.parameters.min_half_life <= half_life <= self.parameters.max_half_life):
                        logger.debug(f"Half-life {half_life:.2f} outside acceptable range for {symbol1}-{symbol2}")
                        continue
                    
                    # Calculate z-scores
                    zscores = self.calculate_zscore(spread)
                    
                    # Generate signals for each timestamp
                    for timestamp in zscores.index[-30:]:  # Last 30 days
                        if pd.isna(zscores.loc[timestamp]):
                            continue
                        
                        zscore = zscores.loc[timestamp]
                        
                        # Entry signals
                        if abs(zscore) >= self.parameters.entry_threshold:
                            signal_type = SignalType.SELL if zscore > 0 else SignalType.BUY
                            confidence = min(abs(zscore) / self.parameters.entry_threshold, 3.0) / 3.0
                            
                            signal = TradingSignal(
                                timestamp=timestamp,
                                symbol1=symbol1,
                                symbol2=symbol2,
                                signal_type=signal_type,
                                strength=abs(zscore),
                                confidence=confidence,
                                entry_price1=price1.loc[timestamp],
                                entry_price2=price2.loc[timestamp],
                                hedge_ratio=hedge_ratio,
                                stop_loss=self.parameters.stop_loss,
                                take_profit=self.parameters.take_profit,
                                metadata={
                                    'spread': spread.loc[timestamp],
                                    'zscore': zscore,
                                    'half_life': half_life,
                                    'correlation': price1.corr(price2)
                                }
                            )
                            
                            signals.append(signal)
                
                except Exception as e:
                    logger.error(f"Error generating signals for {symbol1}-{symbol2}: {e}")
        
        logger.info(f"Generated {len(signals)} trading signals")
        print(f"✅ EXITING generate_signals() at {datetime.now().strftime('%H:%M:%S')}")
        return signals

class StatisticalArbitrageStrategy(BaseStrategy):
    """Advanced statistical arbitrage with multiple models."""
    
    def __init__(self, parameters: StrategyParameters = None):
        print(f"🔄 ENTERING StatisticalArbitrageStrategy.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        if parameters is None:
            parameters = StrategyParameters()
        
        super().__init__(parameters)
        self.strategy_type = StrategyType.STATISTICAL_ARBITRAGE
        
        # Advanced parameters
        self.kalman_enabled = True
        self.regime_detection = True
        self.volatility_adjustment = True
        
        print(f"✅ EXITING StatisticalArbitrageStrategy.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def calculate_hedge_ratio(self, price1: pd.Series, price2: pd.Series) -> float:
        """Calculate hedge ratio using Kalman Filter or rolling regression."""
        print(f"🔄 ENTERING calculate_hedge_ratio() [StatArb] at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            if self.kalman_enabled:
                return self._kalman_hedge_ratio(price1, price2)
            else:
                return self._rolling_hedge_ratio(price1, price2)
        except Exception as e:
            logger.error(f"Error in advanced hedge ratio calculation: {e}")
            # Fallback to simple OLS
            return super().calculate_hedge_ratio(price1, price2)
    
    def _kalman_hedge_ratio(self, price1: pd.Series, price2: pd.Series) -> float:
        """Kalman filter hedge ratio estimation."""
        try:
            from pykalman import KalmanFilter
            
            # Prepare data
            observations = np.column_stack([price1.values, price2.values])
            
            # Initialize Kalman Filter
            kf = KalmanFilter(
                transition_matrices=np.eye(2),
                observation_matrices=np.eye(2)
            )
            
            # Fit and get state estimates
            state_means, _ = kf.em(observations).smooth()
            
            # Calculate hedge ratio from final state
            hedge_ratio = state_means[-1, 0] / state_means[-1, 1]
            
            print(f"✅ EXITING _kalman_hedge_ratio() at {datetime.now().strftime('%H:%M:%S')}")
            return hedge_ratio
            
        except ImportError:
            logger.warning("PyKalman not available, using rolling regression")
            return self._rolling_hedge_ratio(price1, price2)
        except Exception as e:
            logger.error(f"Kalman filter error: {e}")
            return self._rolling_hedge_ratio(price1, price2)
    
    def _rolling_hedge_ratio(self, price1: pd.Series, price2: pd.Series) -> float:
        """Rolling window hedge ratio."""
        window = min(30, len(price1) // 4)
        hedge_ratios = []
        
        for i in range(window, len(price1)):
            p1_window = price1.iloc[i-window:i]
            p2_window = price2.iloc[i-window:i]
            
            try:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                X = p2_window.values.reshape(-1, 1)
                y = p1_window.values
                model.fit(X, y)
                
                hedge_ratios.append(model.coef_[0])
            except:
                continue
        
        return np.median(hedge_ratios) if hedge_ratios else 1.0
    
    def detect_regime(self, price1: pd.Series, price2: pd.Series) -> str:
        """Detect market regime (trending vs mean-reverting)."""
        print(f"🔄 ENTERING detect_regime() at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Calculate correlation over different windows
            short_corr = price1.rolling(20).corr(price2).iloc[-1]
            long_corr = price1.rolling(60).corr(price2).iloc[-1]
            
            # Calculate volatility
            returns1 = price1.pct_change()
            returns2 = price2.pct_change()
            vol1 = returns1.rolling(20).std().iloc[-1]
            vol2 = returns2.rolling(20).std().iloc[-1]
            
            # Simple regime classification
            if short_corr > long_corr and vol1 < returns1.std() and vol2 < returns2.std():
                regime = "mean_reverting"
            elif short_corr < long_corr or vol1 > 1.5 * returns1.std():
                regime = "trending"
            else:
                regime = "neutral"
            
            print(f"✅ EXITING detect_regime() [{regime}] at {datetime.now().strftime('%H:%M:%S')}")
            return regime
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return "neutral"
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Generate statistical arbitrage signals with regime awareness."""
        print(f"🔄 ENTERING generate_signals() [StatArb] at {datetime.now().strftime('%H:%M:%S')}")
        
        signals = []
        symbols = list(data.keys())
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                try:
                    df1, df2 = data[symbol1], data[symbol2]
                    
                    # Align data
                    common_index = df1.index.intersection(df2.index)
                    if len(common_index) < self.parameters.lookback_window:
                        continue
                    
                    price1 = df1.loc[common_index]['Close']
                    price2 = df2.loc[common_index]['Close']
                    
                    # Validate pair
                    if not self.validate_pair(price1, price2):
                        continue
                    
                    # Detect regime
                    regime = self.detect_regime(price1, price2)
                    
                    # Skip if in trending regime (not suitable for mean reversion)
                    if regime == "trending":
                        continue
                    
                    # Calculate dynamic hedge ratio
                    hedge_ratio = self.calculate_hedge_ratio(price1, price2)
                    spread = self.calculate_spread(price1, price2, hedge_ratio)
                    
                    # Volatility-adjusted z-scores
                    if self.volatility_adjustment:
                        vol_window = 20
                        vol_spread = spread.rolling(vol_window).std()
                        rolling_mean = spread.rolling(self.parameters.lookback_window).mean()
                        zscores = (spread - rolling_mean) / vol_spread
                    else:
                        zscores = self.calculate_zscore(spread)
                    
                    # Regime-adjusted thresholds
                    entry_threshold = self.parameters.entry_threshold
                    if regime == "mean_reverting":
                        entry_threshold *= 0.8  # Lower threshold for strong mean reversion
                    
                    # Generate signals
                    for timestamp in zscores.index[-10:]:  # Last 10 days
                        if pd.isna(zscores.loc[timestamp]):
                            continue
                        
                        zscore = zscores.loc[timestamp]
                        
                        if abs(zscore) >= entry_threshold:
                            signal_type = SignalType.SELL if zscore > 0 else SignalType.BUY
                            confidence = min(abs(zscore) / entry_threshold, 2.5) / 2.5
                            
                            signal = TradingSignal(
                                timestamp=timestamp,
                                symbol1=symbol1,
                                symbol2=symbol2,
                                signal_type=signal_type,
                                strength=abs(zscore),
                                confidence=confidence,
                                entry_price1=price1.loc[timestamp],
                                entry_price2=price2.loc[timestamp],
                                hedge_ratio=hedge_ratio,
                                stop_loss=self.parameters.stop_loss,
                                take_profit=self.parameters.take_profit,
                                metadata={
                                    'spread': spread.loc[timestamp],
                                    'zscore': zscore,
                                    'regime': regime,
                                    'vol_adj': self.volatility_adjustment,
                                    'kalman': self.kalman_enabled
                                }
                            )
                            
                            signals.append(signal)
                
                except Exception as e:
                    logger.error(f"Error in StatArb signals for {symbol1}-{symbol2}: {e}")
        
        logger.info(f"Generated {len(signals)} statistical arbitrage signals")
        print(f"✅ EXITING generate_signals() [StatArb] at {datetime.now().strftime('%H:%M:%S')}")
        return signals

class MeanReversionStrategy(BaseStrategy):
    """Pure mean reversion strategy with ML enhancement."""
    
    def __init__(self, parameters: StrategyParameters = None):
        print(f"🔄 ENTERING MeanReversionStrategy.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        if parameters is None:
            parameters = StrategyParameters()
        
        super().__init__(parameters)
        self.strategy_type = StrategyType.MEAN_REVERSION
        
        # ML models for signal enhancement
        self.ml_enabled = True
        self.models = {}
        self.feature_scaler = StandardScaler()
        
        print(f"✅ EXITING MeanReversionStrategy.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def calculate_hedge_ratio(self, price1: pd.Series, price2: pd.Series) -> float:
        """Use cointegration-based hedge ratio."""
        print(f"🔄 ENTERING calculate_hedge_ratio() [MeanRev] at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            from statsmodels.tsa.vector_error_correction_model import coint_johansen
            
            # Prepare data matrix
            data_matrix = np.column_stack([price1.values, price2.values])
            
            # Johansen cointegration test
            result = coint_johansen(data_matrix, det_order=0, k_ar_diff=1)
            
            # Extract cointegrating vector (normalized)
            coint_vector = result.evec[:, 0]
            hedge_ratio = -coint_vector[1] / coint_vector[0]
            
            print(f"✅ EXITING calculate_hedge_ratio() [Cointegration] at {datetime.now().strftime('%H:%M:%S')}")
            return hedge_ratio
            
        except ImportError:
            logger.warning("Statsmodels not available, using OLS")
            return super().calculate_hedge_ratio(price1, price2)
        except Exception as e:
            logger.error(f"Cointegration test failed: {e}")
            return super().calculate_hedge_ratio(price1, price2)
    
    def create_ml_features(self, price1: pd.Series, price2: pd.Series, spread: pd.Series) -> pd.DataFrame:
        """Create ML features for signal enhancement."""
        print(f"🔄 ENTERING create_ml_features() at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            features = pd.DataFrame(index=spread.index)
            
            # Technical indicators
            features['spread'] = spread
            features['spread_ma5'] = spread.rolling(5).mean()
            features['spread_ma20'] = spread.rolling(20).mean()
            features['spread_std'] = spread.rolling(20).std()
            features['spread_rsi'] = self._calculate_rsi(spread)
            
            # Price momentum features
            features['price1_momentum'] = price1.pct_change(5)
            features['price2_momentum'] = price2.pct_change(5)
            features['price_ratio'] = price1 / price2
            features['price_ratio_ma'] = features['price_ratio'].rolling(10).mean()
            
            # Volatility features
            returns1 = price1.pct_change()
            returns2 = price2.pct_change()
            features['vol1'] = returns1.rolling(20).std()
            features['vol2'] = returns2.rolling(20).std()
            features['vol_ratio'] = features['vol1'] / features['vol2']
            
            # Correlation features
            features['rolling_corr'] = returns1.rolling(30).corr(returns2)
            features['corr_change'] = features['rolling_corr'].diff()
            
            # Clean features
            features = features.fillna(method='ffill').fillna(0)
            
            print(f"✅ EXITING create_ml_features() at {datetime.now().strftime('%H:%M:%S')}")
            return features
            
        except Exception as e:
            logger.error(f"Error creating ML features: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_ml_model(self, features: pd.DataFrame, target: pd.Series, pair_key: str):
        """Train ML model for signal prediction."""
        print(f"🔄 ENTERING train_ml_model({pair_key}) at {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Prepare data
            X = features.dropna()
            y = target.loc[X.index]
            
            if len(X) < 100:  # Need sufficient data
                logger.warning(f"Insufficient data for ML training: {len(X)}")
                return
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Create ensemble model
            models = {
                'rf': RandomForestRegressor(n_estimators=50, random_state=42),
                'gbm': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'ridge': Ridge(alpha=1.0)
            }
            
            # Train models with time series CV
            tscv = TimeSeriesSplit(n_splits=3)
            best_model = None
            best_score = -np.inf
            
            for name, model in models.items():
                scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
                avg_score = scores.mean()
                
                logger.info(f"Model {name} CV score: {avg_score:.4f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
            
            # Train best model on full data
            best_model.fit(X_scaled, y)
            self.models[pair_key] = best_model
            
            logger.info(f"Trained ML model for {pair_key}, best score: {best_score:.4f}")
            
            print(f"✅ EXITING train_ml_model() at {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Generate ML-enhanced mean reversion signals."""
        print(f"🔄 ENTERING generate_signals() [MeanRev] at {datetime.now().strftime('%H:%M:%S')}")
        
        signals = []
        symbols = list(data.keys())
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                pair_key = f"{symbol1}-{symbol2}"
                
                try:
                    df1, df2 = data[symbol1], data[symbol2]
                    
                    # Align data
                    common_index = df1.index.intersection(df2.index)
                    if len(common_index) < self.parameters.lookback_window:
                        continue
                    
                    price1 = df1.loc[common_index]['Close']
                    price2 = df2.loc[common_index]['Close']
                    
                    # Validate pair
                    if not self.validate_pair(price1, price2):
                        continue
                    
                    # Calculate hedge ratio and spread
                    hedge_ratio = self.calculate_hedge_ratio(price1, price2)
                    spread = self.calculate_spread(price1, price2, hedge_ratio)
                    
                    # ML enhancement
                    ml_confidence = 0.5  # Default confidence
                    
                    if self.ml_enabled:
                        try:
                            features = self.create_ml_features(price1, price2, spread)
                            
                            if len(features) > 100:
                                # Create target (future spread change)
                                target = spread.shift(-5) - spread  # 5-day forward return
                                
                                # Train model if not exists
                                if pair_key not in self.models:
                                    self.train_ml_model(features, target, pair_key)
                                
                                # Predict if model available
                                if pair_key in self.models:
                                    recent_features = features.iloc[-1:].fillna(0)
                                    X_scaled = self.feature_scaler.transform(recent_features)
                                    prediction = self.models[pair_key].predict(X_scaled)[0]
                                    
                                    # Convert prediction to confidence
                                    ml_confidence = 1 / (1 + np.exp(-abs(prediction)))  # Sigmoid
                        
                        except Exception as e:
                            logger.warning(f"ML enhancement failed for {pair_key}: {e}")
                    
                    # Calculate z-scores
                    zscores = self.calculate_zscore(spread)
                    
                    # Generate signals with ML enhancement
                    for timestamp in zscores.index[-5:]:  # Last 5 days
                        if pd.isna(zscores.loc[timestamp]):
                            continue
                        
                        zscore = zscores.loc[timestamp]
                        
                        if abs(zscore) >= self.parameters.entry_threshold:
                            signal_type = SignalType.SELL if zscore > 0 else SignalType.BUY
                            
                            # Combined confidence from statistical and ML signals
                            stat_confidence = min(abs(zscore) / self.parameters.entry_threshold, 2.0) / 2.0
                            combined_confidence = 0.7 * stat_confidence + 0.3 * ml_confidence
                            
                            signal = TradingSignal(
                                timestamp=timestamp,
                                symbol1=symbol1,
                                symbol2=symbol2,
                                signal_type=signal_type,
                                strength=abs(zscore),
                                confidence=combined_confidence,
                                entry_price1=price1.loc[timestamp],
                                entry_price2=price2.loc[timestamp],
                                hedge_ratio=hedge_ratio,
                                stop_loss=self.parameters.stop_loss,
                                take_profit=self.parameters.take_profit,
                                metadata={
                                    'spread': spread.loc[timestamp],
                                    'zscore': zscore,
                                    'ml_confidence': ml_confidence,
                                    'stat_confidence': stat_confidence,
                                    'strategy': 'mean_reversion'
                                }
                            )
                            
                            signals.append(signal)
                
                except Exception as e:
                    logger.error(f"Error generating MeanRev signals for {symbol1}-{symbol2}: {e}")
        
        logger.info(f"Generated {len(signals)} mean reversion signals")
        print(f"✅ EXITING generate_signals() [MeanRev] at {datetime.now().strftime('%H:%M:%S')}")
        return signals

# Factory function for creating strategies
def create_strategy(strategy_type: str, parameters: StrategyParameters = None) -> BaseStrategy:
    """Factory function to create trading strategies."""
    print(f"🔄 ENTERING create_strategy({strategy_type}) at {datetime.now().strftime('%H:%M:%S')}")
    
    strategy_map = {
        'pairs_trading': PairsTradingStrategy,
        'statistical_arbitrage': StatisticalArbitrageStrategy,
        'mean_reversion': MeanReversionStrategy,
    }
    
    if strategy_type not in strategy_map:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    strategy = strategy_map[strategy_type](parameters)
    
    print(f"✅ EXITING create_strategy() at {datetime.now().strftime('%H:%M:%S')}")
    return strategy
