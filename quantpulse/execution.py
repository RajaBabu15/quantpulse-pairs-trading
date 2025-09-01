"""
Execution and Risk Management System
===================================

Professional order management, position tracking, and risk controls for
algorithmic trading operations.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import uuid
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PositionType(Enum):
    """Position types."""
    LONG = "long"
    SHORT = "short"

@dataclass
class Order:
    """Trading order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    parent_strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining unfilled quantity."""
        return self.quantity - self.filled_quantity
    
    @property
    def notional_value(self) -> float:
        """Get notional value of the order."""
        price = self.price or self.avg_fill_price
        return self.quantity * price if price else 0.0

@dataclass
class Fill:
    """Trade fill representation."""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    exchange: str = "simulation"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Position representation."""
    symbol: str
    quantity: float
    avg_price: float
    market_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def position_type(self) -> Optional[PositionType]:
        """Get position type."""
        if self.quantity > 0:
            return PositionType.LONG
        elif self.quantity < 0:
            return PositionType.SHORT
        return None
    
    @property
    def market_value(self) -> float:
        """Get current market value."""
        return self.quantity * self.market_price
    
    @property
    def total_pnl(self) -> float:
        """Get total P&L."""
        return self.unrealized_pnl + self.realized_pnl
    
    def update_market_price(self, price: float):
        """Update market price and unrealized P&L."""
        self.market_price = price
        if self.quantity != 0:
            self.unrealized_pnl = (price - self.avg_price) * self.quantity

@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.02    # 2% daily loss limit
    max_drawdown: float = 0.10      # 10% max drawdown
    max_concentration: float = 0.25  # 25% max sector concentration
    max_leverage: float = 1.0       # No leverage by default
    var_limit: float = 0.05         # 5% VaR limit
    correlation_limit: float = 0.8   # Max correlation between positions

class OrderManager:
    """Professional order management system."""
    
    def __init__(self, commission_rate: float = 0.001):
        print(f"🔄 ENTERING OrderManager.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        self.commission_rate = commission_rate
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.order_queue = queue.Queue()
        
        # Order processing
        self.processing_thread = None
        self.is_running = False
        self.slippage_model = self._default_slippage_model
        
        print(f"✅ EXITING OrderManager.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def submit_order(self, 
                    symbol: str,
                    side: OrderSide, 
                    quantity: float,
                    order_type: OrderType = OrderType.MARKET,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    strategy: Optional[str] = None) -> str:
        """Submit a new order."""
        print(f"🔄 ENTERING submit_order({symbol}, {side}, {quantity}) at {datetime.now().strftime('%H:%M:%S')}")
        
        order_id = str(uuid.uuid4())
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=abs(quantity),  # Ensure positive quantity
            price=price,
            stop_price=stop_price,
            parent_strategy=strategy
        )
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            logger.error(f"Order rejected: {order_id}")
            return order_id
        
        # Store order
        self.orders[order_id] = order
        
        # Queue for processing
        self.order_queue.put(order_id)
        
        logger.info(f"Order submitted: {order_id} - {symbol} {side.value} {quantity}")
        
        print(f"✅ EXITING submit_order() [{order_id}] at {datetime.now().strftime('%H:%M:%S')}")
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        print(f"🔄 ENTERING cancel_order({order_id}) at {datetime.now().strftime('%H:%M:%S')}")
        
        if order_id not in self.orders:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            logger.warning(f"Cannot cancel order in status: {order.status}")
            return False
        
        order.status = OrderStatus.CANCELLED
        logger.info(f"Order cancelled: {order_id}")
        
        print(f"✅ EXITING cancel_order() at {datetime.now().strftime('%H:%M:%S')}")
        return True
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_orders(self, status: Optional[OrderStatus] = None, symbol: Optional[str] = None) -> List[Order]:
        """Get orders with optional filtering."""
        orders = list(self.orders.values())
        
        if status:
            orders = [o for o in orders if o.status == status]
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        return orders
    
    def process_market_data(self, symbol: str, price: float):
        """Process market data update for order execution."""
        print(f"🔄 ENTERING process_market_data({symbol}, {price}) at {datetime.now().strftime('%H:%M:%S')}")
        
        # Find orders that can be executed
        executable_orders = [
            order for order in self.orders.values()
            if (order.symbol == symbol and 
                order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED] and
                self._can_execute_order(order, price))
        ]
        
        for order in executable_orders:
            self._execute_order(order, price)
        
        print(f"✅ EXITING process_market_data() at {datetime.now().strftime('%H:%M:%S')}")
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order before submission."""
        if order.quantity <= 0:
            logger.error("Invalid quantity")
            return False
        
        if order.order_type == OrderType.LIMIT and order.price is None:
            logger.error("Limit order requires price")
            return False
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            logger.error("Stop order requires stop price")
            return False
        
        return True
    
    def _can_execute_order(self, order: Order, market_price: float) -> bool:
        """Check if order can be executed at current market price."""
        if order.order_type == OrderType.MARKET:
            return True
        
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and market_price <= order.price:
                return True
            elif order.side == OrderSide.SELL and market_price >= order.price:
                return True
        
        if order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and market_price >= order.stop_price:
                return True
            elif order.side == OrderSide.SELL and market_price <= order.stop_price:
                return True
        
        return False
    
    def _execute_order(self, order: Order, market_price: float):
        """Execute an order."""
        print(f"🔄 ENTERING _execute_order({order.order_id}) at {datetime.now().strftime('%H:%M:%S')}")
        
        # Calculate execution price with slippage
        execution_price = self.slippage_model(market_price, order)
        
        # Execute full quantity for simplicity
        fill_quantity = order.remaining_quantity
        commission = fill_quantity * execution_price * self.commission_rate
        
        # Create fill
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=execution_price,
            timestamp=datetime.now(),
            commission=commission
        )
        
        # Update order
        order.filled_quantity += fill_quantity
        order.avg_fill_price = ((order.avg_fill_price * (order.filled_quantity - fill_quantity)) + 
                               (execution_price * fill_quantity)) / order.filled_quantity
        order.commission += commission
        order.status = OrderStatus.FILLED
        
        # Store fill
        self.fills.append(fill)
        
        logger.info(f"Order executed: {order.order_id} - {fill_quantity}@{execution_price:.4f}")
        
        print(f"✅ EXITING _execute_order() at {datetime.now().strftime('%H:%M:%S')}")
    
    def _default_slippage_model(self, market_price: float, order: Order) -> float:
        """Default slippage model."""
        slippage_bps = 5  # 0.5 basis points
        slippage_factor = 1 + (slippage_bps / 10000)
        
        if order.side == OrderSide.BUY:
            return market_price * slippage_factor
        else:
            return market_price / slippage_factor
    
    def get_fill_history(self, symbol: Optional[str] = None) -> List[Fill]:
        """Get fill history."""
        fills = self.fills
        
        if symbol:
            fills = [f for f in fills if f.symbol == symbol]
        
        return fills

class PositionManager:
    """Position tracking and management."""
    
    def __init__(self):
        print(f"🔄 ENTERING PositionManager.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = 0.0
        self.cash_balance = 0.0
        self.total_equity = 0.0
        
        print(f"✅ EXITING PositionManager.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def process_fill(self, fill: Fill):
        """Process a trade fill and update positions."""
        print(f"🔄 ENTERING process_fill({fill.symbol}, {fill.quantity}) at {datetime.now().strftime('%H:%M:%S')}")
        
        symbol = fill.symbol
        
        if symbol not in self.positions:
            # New position
            quantity = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=fill.price,
                market_price=fill.price
            )
        else:
            # Update existing position
            position = self.positions[symbol]
            fill_quantity = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
            
            if (position.quantity > 0 and fill_quantity < 0) or (position.quantity < 0 and fill_quantity > 0):
                # Closing or reducing position
                if abs(fill_quantity) >= abs(position.quantity):
                    # Closing entire position
                    realized_pnl = (fill.price - position.avg_price) * position.quantity
                    position.realized_pnl += realized_pnl
                    
                    # Any remaining quantity starts new position
                    remaining = abs(fill_quantity) - abs(position.quantity)
                    if remaining > 0:
                        position.quantity = remaining * np.sign(fill_quantity)
                        position.avg_price = fill.price
                    else:
                        position.quantity = 0
                else:
                    # Partial close
                    closed_quantity = fill_quantity
                    realized_pnl = (fill.price - position.avg_price) * (-closed_quantity)
                    position.realized_pnl += realized_pnl
                    position.quantity += fill_quantity
            else:
                # Adding to position
                old_cost = position.quantity * position.avg_price
                new_cost = fill_quantity * fill.price
                total_quantity = position.quantity + fill_quantity
                
                if total_quantity != 0:
                    position.avg_price = (old_cost + new_cost) / total_quantity
                
                position.quantity = total_quantity
        
        # Update commission
        position = self.positions[symbol]
        position.commission_paid += fill.commission
        
        # Clean up zero positions
        if abs(position.quantity) < 1e-6:
            del self.positions[symbol]
        
        logger.info(f"Position updated: {symbol} - {position.quantity:.2f}@{position.avg_price:.4f}")
        
        print(f"✅ EXITING process_fill() at {datetime.now().strftime('%H:%M:%S')}")
    
    def update_market_prices(self, prices: Dict[str, float]):
        """Update market prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_market_price(price)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """Get all current positions."""
        return list(self.positions.values())
    
    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio-level metrics."""
        total_value = sum(pos.market_value for pos in self.positions.values())
        total_pnl = sum(pos.total_pnl for pos in self.positions.values())
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized = sum(pos.realized_pnl for pos in self.positions.values())
        total_commission = sum(pos.commission_paid for pos in self.positions.values())
        
        return {
            'total_value': total_value,
            'total_pnl': total_pnl,
            'unrealized_pnl': total_unrealized,
            'realized_pnl': total_realized,
            'total_commission': total_commission,
            'position_count': len(self.positions),
            'long_positions': len([p for p in self.positions.values() if p.quantity > 0]),
            'short_positions': len([p for p in self.positions.values() if p.quantity < 0])
        }

class RiskManager:
    """Comprehensive risk management system."""
    
    def __init__(self, risk_limits: RiskLimits = None, initial_capital: float = 100000):
        print(f"🔄 ENTERING RiskManager.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        self.risk_limits = risk_limits or RiskLimits()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Risk tracking
        self.daily_pnl_history = deque(maxlen=252)  # 1 year
        self.position_history = deque(maxlen=1000)
        self.violation_history = []
        
        print(f"✅ EXITING RiskManager.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def check_order_risk(self, order: Order, current_positions: Dict[str, Position], 
                        portfolio_value: float) -> Tuple[bool, str]:
        """Check if order violates risk limits."""
        print(f"🔄 ENTERING check_order_risk({order.symbol}) at {datetime.now().strftime('%H:%M:%S')}")
        
        # Position size check
        order_value = order.notional_value
        position_size_pct = order_value / portfolio_value if portfolio_value > 0 else 0
        
        if position_size_pct > self.risk_limits.max_position_size:
            message = f"Position size {position_size_pct:.2%} exceeds limit {self.risk_limits.max_position_size:.2%}"
            self._log_violation("position_size", order.symbol, message)
            print(f"❌ EXITING check_order_risk() [VIOLATION] at {datetime.now().strftime('%H:%M:%S')}")
            return False, message
        
        # Concentration check
        symbol_exposure = order_value
        existing_position = current_positions.get(order.symbol)
        if existing_position:
            symbol_exposure += abs(existing_position.market_value)
        
        concentration_pct = symbol_exposure / portfolio_value if portfolio_value > 0 else 0
        if concentration_pct > self.risk_limits.max_concentration:
            message = f"Concentration {concentration_pct:.2%} exceeds limit {self.risk_limits.max_concentration:.2%}"
            self._log_violation("concentration", order.symbol, message)
            print(f"❌ EXITING check_order_risk() [VIOLATION] at {datetime.now().strftime('%H:%M:%S')}")
            return False, message
        
        # Leverage check
        total_exposure = sum(abs(pos.market_value) for pos in current_positions.values()) + order_value
        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        if leverage > self.risk_limits.max_leverage:
            message = f"Leverage {leverage:.2f}x exceeds limit {self.risk_limits.max_leverage:.2f}x"
            self._log_violation("leverage", order.symbol, message)
            print(f"❌ EXITING check_order_risk() [VIOLATION] at {datetime.now().strftime('%H:%M:%S')}")
            return False, message
        
        print(f"✅ EXITING check_order_risk() [OK] at {datetime.now().strftime('%H:%M:%S')}")
        return True, "Order approved"
    
    def check_portfolio_risk(self, positions: Dict[str, Position], portfolio_value: float) -> List[str]:
        """Check portfolio-level risk metrics."""
        print(f"🔄 ENTERING check_portfolio_risk() at {datetime.now().strftime('%H:%M:%S')}")
        
        violations = []
        
        # Daily loss check
        if len(self.daily_pnl_history) > 0:
            today_pnl = portfolio_value - self.current_capital
            daily_loss_pct = today_pnl / self.current_capital if self.current_capital > 0 else 0
            
            if daily_loss_pct < -self.risk_limits.max_daily_loss:
                message = f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.risk_limits.max_daily_loss:.2%}"
                violations.append(message)
                self._log_violation("daily_loss", "PORTFOLIO", message)
        
        # Drawdown check
        peak_value = max([self.initial_capital] + list(self.daily_pnl_history))
        current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
        
        if current_drawdown > self.risk_limits.max_drawdown:
            message = f"Drawdown {current_drawdown:.2%} exceeds limit {self.risk_limits.max_drawdown:.2%}"
            violations.append(message)
            self._log_violation("drawdown", "PORTFOLIO", message)
        
        # VaR calculation (simplified)
        if len(self.daily_pnl_history) >= 30:
            returns = np.array(self.daily_pnl_history) / self.current_capital
            var_95 = np.percentile(returns, 5)  # 5th percentile
            
            if abs(var_95) > self.risk_limits.var_limit:
                message = f"VaR {abs(var_95):.2%} exceeds limit {self.risk_limits.var_limit:.2%}"
                violations.append(message)
                self._log_violation("var", "PORTFOLIO", message)
        
        # Correlation check (if multiple positions)
        position_list = list(positions.values())
        if len(position_list) >= 2:
            correlations = self._calculate_position_correlations(position_list)
            max_corr = np.max(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))
            
            if max_corr > self.risk_limits.correlation_limit:
                message = f"Max correlation {max_corr:.2f} exceeds limit {self.risk_limits.correlation_limit:.2f}"
                violations.append(message)
                self._log_violation("correlation", "PORTFOLIO", message)
        
        print(f"✅ EXITING check_portfolio_risk() [{len(violations)} violations] at {datetime.now().strftime('%H:%M:%S')}")
        return violations
    
    def update_daily_pnl(self, portfolio_value: float):
        """Update daily P&L tracking."""
        self.daily_pnl_history.append(portfolio_value)
        self.current_capital = portfolio_value
    
    def _calculate_position_correlations(self, positions: List[Position]) -> np.ndarray:
        """Calculate correlation matrix for positions (simplified)."""
        # This is a simplified version - in practice, you'd use historical returns
        n = len(positions)
        correlations = np.eye(n)  # Identity matrix as placeholder
        return correlations
    
    def _log_violation(self, violation_type: str, symbol: str, message: str):
        """Log a risk violation."""
        violation = {
            'timestamp': datetime.now(),
            'type': violation_type,
            'symbol': symbol,
            'message': message
        }
        
        self.violation_history.append(violation)
        logger.warning(f"Risk violation: {message}")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        return {
            'risk_limits': self.risk_limits.__dict__,
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'violations_today': len([v for v in self.violation_history 
                                   if v['timestamp'].date() == datetime.now().date()]),
            'total_violations': len(self.violation_history),
            'recent_violations': self.violation_history[-10:]  # Last 10 violations
        }

class TradingEngine:
    """Main trading engine orchestrating all components."""
    
    def __init__(self, initial_capital: float = 100000, risk_limits: RiskLimits = None):
        print(f"🔄 ENTERING TradingEngine.__init__() at {datetime.now().strftime('%H:%M:%S')}")
        
        self.initial_capital = initial_capital
        
        # Initialize components
        self.order_manager = OrderManager()
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(risk_limits, initial_capital)
        
        # Engine state
        self.is_running = False
        self.current_prices = {}
        
        print(f"✅ EXITING TradingEngine.__init__() at {datetime.now().strftime('%H:%M:%S')}")
    
    def submit_trade(self, symbol: str, side: OrderSide, quantity: float, 
                    order_type: OrderType = OrderType.MARKET, price: Optional[float] = None) -> str:
        """Submit a trade with risk checks."""
        print(f"🔄 ENTERING submit_trade({symbol}, {side}, {quantity}) at {datetime.now().strftime('%H:%M:%S')}")
        
        # Create order
        order_id = str(uuid.uuid4())
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=abs(quantity),
            price=price
        )
        
        # Risk check
        portfolio_metrics = self.position_manager.calculate_portfolio_metrics()
        portfolio_value = portfolio_metrics['total_value'] + self.initial_capital
        
        risk_ok, risk_message = self.risk_manager.check_order_risk(
            order, self.position_manager.positions, portfolio_value
        )
        
        if not risk_ok:
            logger.error(f"Trade rejected due to risk: {risk_message}")
            return ""
        
        # Submit order
        order_id = self.order_manager.submit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price
        )
        
        print(f"✅ EXITING submit_trade() [{order_id}] at {datetime.now().strftime('%H:%M:%S')}")
        return order_id
    
    def update_market_data(self, prices: Dict[str, float]):
        """Update market data and process orders."""
        print(f"🔄 ENTERING update_market_data() at {datetime.now().strftime('%H:%M:%S')}")
        
        self.current_prices.update(prices)
        
        # Update position market prices
        self.position_manager.update_market_prices(prices)
        
        # Process orders
        for symbol, price in prices.items():
            self.order_manager.process_market_data(symbol, price)
        
        # Process any new fills
        for fill in self.order_manager.fills:
            if not hasattr(fill, '_processed'):
                self.position_manager.process_fill(fill)
                fill._processed = True
        
        # Update risk manager
        portfolio_metrics = self.position_manager.calculate_portfolio_metrics()
        portfolio_value = portfolio_metrics['total_value'] + self.initial_capital
        self.risk_manager.update_daily_pnl(portfolio_value)
        
        print(f"✅ EXITING update_market_data() at {datetime.now().strftime('%H:%M:%S')}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        portfolio_metrics = self.position_manager.calculate_portfolio_metrics()
        risk_report = self.risk_manager.get_risk_report()
        
        return {
            'positions': self.position_manager.get_all_positions(),
            'portfolio_metrics': portfolio_metrics,
            'risk_report': risk_report,
            'pending_orders': self.order_manager.get_orders(OrderStatus.PENDING),
            'current_prices': self.current_prices
        }
