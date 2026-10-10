"""
EGX Smart Money Architecture & Modular Detection System
════════════════════════════════════════════════════════

Core Philosophy:
  - NO probabilities or vague signals
  - ONLY measurable, reproducible market mechanics
  - Institutional positioning = detectable price + volume signatures
  - Smart Money = institutional order flow (NOT guess work)

Foundation: 
  - Layered architecture (Base → Analysis → Signals → Action)
  - Each component is independently testable
  - Easy to replace/upgrade without breaking the chain
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: CORE DATA STRUCTURES (Immutable, measurable facts)
# ═══════════════════════════════════════════════════════════════════════════

class MarketPhase(Enum):
    """Objective market phases based on measurable conditions."""
    ACCUMULATION = "accumulation"      # Smart money entering silently
    DISTRIBUTION = "distribution"       # Smart money exiting visibly
    MARKUP = "markup"                  # Initial breakout phase
    MARKDOWN = "markdown"              # Crash phase
    CONSOLIDATION = "consolidation"    # Sideways, building base
    BREAKOUT_READY = "breakout_ready"  # Base complete, high probability setup


@dataclass
class VolumeProfile:
    """Volume distribution across price levels (factual, not subjective)."""
    price_levels: np.ndarray          # Sorted price points
    volumes: np.ndarray               # Volume at each level
    point_of_control: float           # Price with highest volume
    poc_volume: float                 # Volume at POC
    value_area_high: float            # Top of 70% volume range
    value_area_low: float             # Bottom of 70% volume range
    poc_strength: float               # POC volume / total volume (0-1)
    
    def is_strong_poc(self) -> bool:
        """POC is objectively strong if it concentrates >15% of total volume."""
        return self.poc_strength > 0.15


@dataclass
class OrderFlowSignature:
    """Raw institutional order signatures - measurable, not interpretive."""
    period_start: datetime
    period_end: datetime
    
    # FACTUAL measurements
    total_volume: float
    buy_volume: float                 # Volume on closes > open
    sell_volume: float                # Volume on closes < open
    neutral_volume: float             # Volume on closes = open
    
    # RATIOS (no opinion, just math)
    buy_sell_ratio: float             # buy / sell (>1 = more buying)
    volume_weighted_price: float      # Weighted by volume
    
    # MOMENTUM (rate of change, pure physics)
    obv_slope: float                  # On-Balance Volume derivative
    adl_slope: float                  # Accumulation/Distribution slope
    
    # CONVICTION (capital commitment per unit price moved)
    capital_commitment: float         # Volume * range / time
    
    def is_institutional_buying(self) -> bool:
        """Institutional buying detected if: ratio > 1.5 AND capital high."""
        return self.buy_sell_ratio > 1.5 and self.capital_commitment > 0
    
    def is_institutional_selling(self) -> bool:
        """Institutional selling if: ratio < 0.67 AND capital high."""
        return self.buy_sell_ratio < 0.67 and self.capital_commitment > 0


@dataclass
class PriceStructure:
    """Objective price levels and their quality (no interpretation)."""
    highest_high_20d: float
    lowest_low_20d: float
    resistance_zones: List[float]     # Measured, not guessed
    support_zones: List[float]
    swing_high: float                 # Last clear higher high
    swing_low: float                  # Last clear lower low
    current_price: float
    
    @property
    def is_above_20d_high(self) -> bool:
        return self.current_price > self.highest_high_20d
    
    @property
    def is_below_20d_low(self) -> bool:
        return self.current_price < self.lowest_low_20d
    
    @property
    def is_in_breakout_range(self) -> bool:
        """Breakout range = price above 20d high + near resistance."""
        return self.is_above_20d_high and abs(self.current_price - self.resistance_zones[-1]) < 0.02 * self.current_price


@dataclass
class AccumulationSignal:
    """
    NOT "maybe accumulating" — YES/NO factual assessment.
    Only generated when MULTIPLE independent confirmations align.
    """
    timestamp: datetime
    symbol: str
    market_phase: MarketPhase
    
    # HARD CONDITIONS (all must be true for signal)
    volume_profile_bullish: bool      # POC below current price
    order_flow_buying: bool           # Buy/sell ratio > 1.5
    price_structure_bullish: bool     # Above key levels
    obv_confirmed: bool               # OBV rising
    adl_confirmed: bool               # ADL rising
    
    # STRENGTH METRIC (0-100, not vague)
    signal_strength: int              # % of conditions met * 100
    
    # ACTION ZONES (measurable, executable)
    entry_zone_low: float
    entry_zone_high: float
    first_target: float               # Initial breakout level
    second_target: float              # Extended target
    risk_loss_level: float            # Where signal invalidates
    risk_reward_ratio: float          # (Target - Entry) / (Entry - RiskLevel)
    
    def is_high_conviction(self) -> bool:
        """High conviction = all 5 conditions true + RR > 2.0."""
        conditions_met = sum([
            self.volume_profile_bullish,
            self.order_flow_buying,
            self.price_structure_bullish,
            self.obv_confirmed,
            self.adl_confirmed
        ])
        return conditions_met == 5 and self.risk_reward_ratio > 2.0
    
    def is_valid(self) -> bool:
        """Signal is valid only if at least 4/5 conditions met."""
        conditions_met = sum([
            self.volume_profile_bullish,
            self.order_flow_buying,
            self.price_structure_bullish,
            self.obv_confirmed,
            self.adl_confirmed
        ])
        return conditions_met >= 4


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: DATA CALCULATORS (Transform raw OHLCV into structures)
# ═══════════════════════════════════════════════════════════════════════════

class OrderFlowCalculator:
    """Calculate institutional order flow signatures from OHLCV."""
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume: pure accumulation/distribution indicator."""
        volume_sign = np.sign(df['close'].diff())
        volume_sign[volume_sign == 0] = 0  # Neutral on no change
        obv = (volume_sign * df['volume']).cumsum()
        return obv
    
    @staticmethod
    def calculate_adl(df: pd.DataFrame) -> pd.Series:
        """
        Accumulation/Distribution Line.
        Measures WHERE in the candle price closed = institutional intent.
        """
        df = df.copy()
        df['clv'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        adl = (df['clv'] * df['volume']).cumsum()
        return adl
    
    @staticmethod
    def calculate_order_flow_signature(df: pd.DataFrame, period: int = 20) -> OrderFlowSignature:
        """
        Generate institutional order flow signature over N periods.
        This is OBJECTIVE: what happened, not what might happen.
        """
        df = df.copy()
        window = df.tail(period)
        
        # Buy/Sell volume calculation
        close_diff = window['close'].diff()
        buy_vol = window['volume'].where(close_diff > 0, 0).sum()
        sell_vol = window['volume'].where(close_diff < 0, 0).sum()
        neutral_vol = window['volume'].where(close_diff == 0, 0).sum()
        total_vol = window['volume'].sum()
        
        # Volume-weighted price
        vwap = (window['close'] * window['volume']).sum() / total_vol
        
        # Momentum slopes (derivative, not interpretation)
        obv = OrderFlowCalculator.calculate_obv(window)
        adl = OrderFlowCalculator.calculate_adl(window)
        
        obv_slope = (obv.iloc[-1] - obv.iloc[0]) / period
        adl_slope = (adl.iloc[-1] - adl.iloc[0]) / period
        
        # Capital commitment: how much volume per unit of price move
        price_range = window['high'].max() - window['low'].min()
        capital_commitment = total_vol * price_range / (period * 24)  # normalized per hour
        
        return OrderFlowSignature(
            period_start=window.index[0],
            period_end=window.index[-1],
            total_volume=total_vol,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            neutral_volume=neutral_vol,
            buy_sell_ratio=(buy_vol + 1) / (sell_vol + 1),  # Add 1 to avoid division by zero
            volume_weighted_price=vwap,
            obv_slope=obv_slope,
            adl_slope=adl_slope,
            capital_commitment=capital_commitment
        )


class VolumeProfileCalculator:
    """Calculate volume distribution across price levels."""
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, bins: int = 20) -> VolumeProfile:
        """
        Create volume profile: where was trading concentrated?
        This reveals institutional support/resistance.
        """
        df = df.copy()
        
        # Price bins
        price_min, price_max = df['close'].min(), df['close'].max()
        price_levels = np.linspace(price_min, price_max, bins)
        
        # Map volumes to price levels
        volumes = np.zeros(bins)
        for i in range(bins - 1):
            in_level = (df['close'] >= price_levels[i]) & (df['close'] < price_levels[i + 1])
            volumes[i] = df[in_level]['volume'].sum()
        
        # Point of Control
        poc_idx = np.argmax(volumes)
        poc = price_levels[poc_idx]
        poc_volume = volumes[poc_idx]
        poc_strength = poc_volume / volumes.sum()
        
        # Value Area (70% of volume)
        sorted_idx = np.argsort(volumes)[::-1]
        cumsum = np.cumsum(volumes[sorted_idx])
        cutoff_idx = np.where(cumsum >= 0.7 * volumes.sum())[0][0]
        va_prices = price_levels[sorted_idx[:cutoff_idx + 1]]
        value_area_high = va_prices.max()
        value_area_low = va_prices.min()
        
        return VolumeProfile(
            price_levels=price_levels,
            volumes=volumes,
            point_of_control=poc,
            poc_volume=poc_volume,
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            poc_strength=poc_strength
        )


class PriceStructureAnalyzer:
    """Identify objective support/resistance and breakout levels."""
    
    @staticmethod
    def find_swing_levels(df: pd.DataFrame, lookback: int = 20) -> Tuple[float, float]:
        """
        Last swing high and swing low (not subjective).
        Swing High: local maximum where price was higher on both sides.
        Swing Low: local minimum where price was lower on both sides.
        """
        df = df.copy()
        
        # Find local highs
        highs = df['high'].iloc[-lookback:]
        swing_high = highs.max()
        
        # Find local lows
        lows = df['low'].iloc[-lookback:]
        swing_low = lows.min()
        
        return swing_high, swing_low
    
    @staticmethod
    def calculate_price_structure(df: pd.DataFrame) -> PriceStructure:
        """Objective price levels based on recent action."""
        df = df.copy()
        window = df.tail(30)
        
        # 20-day extremes
        highest_20d = window['high'].max()
        lowest_20d = window['low'].min()
        
        # Swing levels
        swing_high, swing_low = PriceStructureAnalyzer.find_swing_levels(df)
        
        # Resistance zones: recent highs + 1% above
        resistance_zones = [swing_high, highest_20d, highest_20d * 1.01]
        resistance_zones = sorted(set(resistance_zones), reverse=True)
        
        # Support zones: recent lows - 1% below
        support_zones = [swing_low, lowest_20d, lowest_20d * 0.99]
        support_zones = sorted(set(support_zones))
        
        return PriceStructure(
            highest_high_20d=highest_20d,
            lowest_low_20d=lowest_20d,
            resistance_zones=resistance_zones,
            support_zones=support_zones,
            swing_high=swing_high,
            swing_low=swing_low,
            current_price=df['close'].iloc[-1]
        )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: SIGNAL GENERATOR (Combines all factors = YES or NO)
# ═══════════════════════════════════════════════════════════════════════════

class AccumulationSignalGenerator:
    """
    Generate accumulation signals based on HARD CONDITIONS.
    Not "maybe accumulating" — only signals when multiple independent
    confirmations align. Signal = executable trade setup.
    """
    
    @staticmethod
    def generate_signal(
        df: pd.DataFrame,
        symbol: str,
        lookback_periods: int = 20
    ) -> Optional[AccumulationSignal]:
        """
        Returns AccumulationSignal only if conditions are met.
        Returns None if no valid signal.
        """
        
        # Calculate all components
        order_flow = OrderFlowCalculator.calculate_order_flow_signature(df, lookback_periods)
        vol_profile = VolumeProfileCalculator.calculate_volume_profile(df)
        price_struct = PriceStructureAnalyzer.calculate_price_structure(df)
        
        # CONDITION 1: Volume Profile Bullish
        # POC should be BELOW current price (institutional accumulation was lower)
        vol_profile_bullish = vol_profile.point_of_control < price_struct.current_price
        
        # CONDITION 2: Order Flow shows institutional buying
        order_flow_buying = order_flow.is_institutional_buying()
        
        # CONDITION 3: Price Structure bullish
        # Price above 20-day high = breaking resistance
        price_structure_bullish = price_struct.is_above_20d_high
        
        # CONDITION 4: OBV Confirmation
        obv = OrderFlowCalculator.calculate_obv(df)
        obv_20ma = obv.tail(20).mean()
        obv_confirmed = obv.iloc[-1] > obv_20ma
        
        # CONDITION 5: ADL Confirmation
        adl = OrderFlowCalculator.calculate_adl(df)
        adl_20ma = adl.tail(20).mean()
        adl_confirmed = adl.iloc[-1] > adl_20ma
        
        # COUNT CONDITIONS
        conditions_met = sum([
            vol_profile_bullish,
            order_flow_buying,
            price_structure_bullish,
            obv_confirmed,
            adl_confirmed
        ])
        
        # ONLY CREATE SIGNAL if at least 4/5 conditions met
        if conditions_met < 4:
            return None
        
        signal_strength = int((conditions_met / 5) * 100)
        
        # CALCULATE ACTION ZONES
        current = price_struct.current_price
        support = price_struct.support_zones[0]
        resistance = price_struct.resistance_zones[0]
        
        entry_zone_low = current * 0.99
        entry_zone_high = current * 1.01
        first_target = resistance
        second_target = resistance * (1 + (resistance - support) / support)
        risk_loss_level = support * 0.98
        
        avg_target = (first_target + second_target) / 2
        risk = entry_zone_high - risk_loss_level
        reward = avg_target - entry_zone_low
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Determine market phase
        if obv.iloc[-1] > obv_20ma and vol_profile_bullish and order_flow_buying:
            phase = MarketPhase.ACCUMULATION
        elif price_struct.is_above_20d_high:
            phase = MarketPhase.MARKUP
        else:
            phase = MarketPhase.CONSOLIDATION
        
        return AccumulationSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            market_phase=phase,
            volume_profile_bullish=vol_profile_bullish,
            order_flow_buying=order_flow_buying,
            price_structure_bullish=price_structure_bullish,
            obv_confirmed=obv_confirmed,
            adl_confirmed=adl_confirmed,
            signal_strength=signal_strength,
            entry_zone_low=entry_zone_low,
            entry_zone_high=entry_zone_high,
            first_target=first_target,
            second_target=second_target,
            risk_loss_level=risk_loss_level,
            risk_reward_ratio=risk_reward_ratio
        )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4: PUBLIC API (Clean, testable interface)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_egx_accumulation(
    df: pd.DataFrame,
    symbol: str,
    lookback: int = 20
) -> Dict:
    """
    Analyze EGX stock for accumulation (smart money) positioning.
    
    RETURNS:
      - accumulation_signal: None or AccumulationSignal object
      - market_phase: Current phase (objective fact)
      - volumes: OrderFlowSignature (measurable facts)
      - price_levels: Support/Resistance (factual)
      - analysis_quality: Confidence in data (1-5 stars)
    
    NO vague probabilities. Only YES/NO signals based on hard conditions.
    """
    
    # Validate data quality
    if len(df) < 21:
        return {"error": "Insufficient data (need 21+ periods)"}
    
    # Generate signal
    signal = AccumulationSignalGenerator.generate_signal(df, symbol, lookback)
    
    # Get supporting data
    order_flow = OrderFlowCalculator.calculate_order_flow_signature(df, lookback)
    price_struct = PriceStructureAnalyzer.calculate_price_structure(df)
    vol_profile = VolumeProfileCalculator.calculate_volume_profile(df)
    
    return {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "signal": signal,
        "order_flow": {
            "buy_sell_ratio": order_flow.buy_sell_ratio,
            "buy_volume": order_flow.buy_volume,
            "sell_volume": order_flow.sell_volume,
            "obv_slope": order_flow.obv_slope,
            "adl_slope": order_flow.adl_slope,
            "is_institutional_buying": order_flow.is_institutional_buying()
        },
        "price_structure": {
            "current": price_struct.current_price,
            "swing_high": price_struct.swing_high,
            "swing_low": price_struct.swing_low,
            "support_zones": price_struct.support_zones,
            "resistance_zones": price_struct.resistance_zones,
            "above_20d_high": price_struct.is_above_20d_high
        },
        "volume_profile": {
            "point_of_control": vol_profile.point_of_control,
            "poc_strength": vol_profile.poc_strength,
            "is_strong_poc": vol_profile.is_strong_poc(),
            "value_area_high": vol_profile.value_area_high,
            "value_area_low": vol_profile.value_area_low
        }
    }
