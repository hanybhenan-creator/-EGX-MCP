"""
EGX Accumulation Detection Engine & Smart Money Analyzer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Intelligent detection of:
  1. Accumulation phases (silent buying by institutions)
  2. Distribution zones (pre-crash selling)
  3. Banker/Smart money positioning
  4. Volume/price confirmation signals
  5. Sector rotation patterns
"""

from __future__ import annotations
import yfinance as yf
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta
from tradingview_mcp.core.errors import exception_to_envelope, ErrorCode


class AccumulationDetector:
    """
    Multi-indicator accumulation detection system for EGX stocks.
    
    Uses:
      - On-Balance Volume (OBV)
      - Volume Profile & Point of Control
      - Accumulation/Distribution Line (ADL)
      - Price Action Range (PVP)
      - Buy/Sell Volume Ratio
    """

    @staticmethod
    def detect_accumulation(
        symbol: str,
        period: str = "6mo",
        interval: str = "1d",
    ) -> dict:
        """
        Detect accumulation patterns for an EGX stock.
        
        Args:
            symbol: EGX stock symbol (e.g., "COMI.CA")
            period: Historical period (1mo, 3mo, 6mo, 1y, 2y)
            interval: Timeframe (1d, 1h)
            
        Returns:
            {
                "symbol": str,
                "accumulation_score": 0-100,
                "status": "STRONG_ACCUMULATION" | "MILD_ACCUMULATION" | "NEUTRAL" | "DISTRIBUTION",
                "indicators": {
                    "obv_trend": float,  # -100 to 100
                    "adl_trend": float,  # -100 to 100
                    "volume_ratio": float,  # buy vol / sell vol
                    "price_vol_confirmation": bool,
                    "pvp_level": float,  # Point of control price
                    "accumulation_zone": {
                        "bottom": float,
                        "top": float,
                        "strength": 0-100
                    }
                },
                "institutional_signals": [...],
                "recommendation": str,
                "next_target": float,
                "risk_level": "LOW" | "MEDIUM" | "HIGH"
            }
        """
        try:
            # Fetch OHLCV data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return {"error": f"No data for {symbol}"}
            
            # Normalize columns
            hist = hist.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Calculate indicators
            obv = AccumulationDetector._calculate_obv(hist)
            adl = AccumulationDetector._calculate_adl(hist)
            volume_ratio = AccumulationDetector._calculate_volume_ratio(hist)
            pvp = AccumulationDetector._calculate_point_of_control(hist)
            
            # Detect accumulation zones
            accum_zone = AccumulationDetector._detect_accumulation_zones(hist)
            
            # Generate score
            score = AccumulationDetector._calculate_accumulation_score(
                obv, adl, volume_ratio, accum_zone
            )
            
            # Status classification
            if score >= 75:
                status = "STRONG_ACCUMULATION"
                recommendation = "🟢 HIGH CONVICTION BUY - Strong institutional accumulation detected"
            elif score >= 55:
                status = "MILD_ACCUMULATION"
                recommendation = "🟡 WATCH - Mild accumulation signals present"
            elif score <= 25:
                status = "DISTRIBUTION"
                recommendation = "🔴 CAUTION - Distribution phase detected. Potential pullback incoming"
            else:
                status = "NEUTRAL"
                recommendation = "⚪ NEUTRAL - No clear accumulation or distribution"
            
            return {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "timestamp": datetime.now().isoformat(),
                "accumulation_score": score,
                "status": status,
                "indicators": {
                    "obv_trend": obv["trend_score"],
                    "obv_strength": obv["strength"],
                    "adl_trend": adl["trend_score"],
                    "adl_confirmation": adl["confirmation"],
                    "volume_ratio": volume_ratio,
                    "buy_volume_pct": (100 * sum(hist['volume'].where(hist['close'].diff() > 0)) / hist['volume'].sum()),
                    "sell_volume_pct": (100 * sum(hist['volume'].where(hist['close'].diff() < 0)) / hist['volume'].sum()),
                    "price_vol_confirmation": obv["confirmation"] and adl["confirmation"],
                    "pvp_level": pvp["level"],
                    "price_to_pvp_distance": ((hist['close'].iloc[-1] - pvp["level"]) / pvp["level"]),
                },
                "accumulation_zone": accum_zone,
                "institutional_signals": AccumulationDetector._generate_signals(obv, adl, volume_ratio, accum_zone),
                "recent_price_action": {
                    "current_price": hist['close'].iloc[-1],
                    "support_level": hist['low'].rolling(20).min().iloc[-1],
                    "resistance_level": hist['high'].rolling(20).max().iloc[-1],
                    "current_trend": "UPTREND" if hist['close'].iloc[-1] > hist['close'].iloc[-20] else "DOWNTREND"
                },
                "next_target": AccumulationDetector._calculate_target(hist, accum_zone),
                "stop_loss": hist['low'].rolling(20).min().iloc[-1] * 0.95,
                "risk_level": AccumulationDetector._assess_risk(obv, adl, volume_ratio),
                "recommendation": recommendation,
                "conviction_level": "HIGH" if score >= 75 else ("MEDIUM" if score >= 55 else "LOW")
            }
            
        except Exception as e:
            return exception_to_envelope(e, context="detect_accumulation")

    @staticmethod
    def _calculate_obv(df: pd.DataFrame) -> dict:
        """On-Balance Volume calculation with trend analysis."""
        df = df.copy()
        obv = np.where(df['close'].diff() > 0, df['volume'], 
                      np.where(df['close'].diff() < 0, -df['volume'], 0))
        obv = np.cumsum(obv)
        
        # Trend: compare recent OBV to 20-day average
        recent_obv = obv[-20:]
        obv_20ma = np.mean(recent_obv)
        obv_current = obv[-1]
        trend_score = 100 * (obv_current - obv_20ma) / abs(obv_20ma + 1)
        trend_score = max(-100, min(100, trend_score))
        
        return {
            "values": obv,
            "trend_score": trend_score,
            "strength": "STRONG" if abs(trend_score) > 50 else "MODERATE",
            "confirmation": trend_score > 20  # Positive OBV trend
        }

    @staticmethod
    def _calculate_adl(df: pd.DataFrame) -> dict:
        """Accumulation/Distribution Line."""
        df = df.copy()
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        adl = (clv * df['volume']).cumsum()
        
        # Trend check
        adl_20ma = adl[-20:].mean()
        adl_current = adl.iloc[-1]
        trend_score = 100 * (adl_current - adl_20ma) / abs(adl_20ma + 1)
        trend_score = max(-100, min(100, trend_score))
        
        return {
            "values": adl.values,
            "trend_score": trend_score,
            "strength": "STRONG" if abs(trend_score) > 50 else "MODERATE",
            "confirmation": trend_score > 20
        }

    @staticmethod
    def _calculate_volume_ratio(df: pd.DataFrame) -> float:
        """Buy volume / Sell volume ratio."""
        df = df.copy()
        buy_vol = df['volume'].where(df['close'].diff() > 0).fillna(0).sum()
        sell_vol = df['volume'].where(df['close'].diff() < 0).fillna(0).sum()
        return (buy_vol + 1) / (sell_vol + 1)

    @staticmethod
    def _calculate_point_of_control(df: pd.DataFrame) -> dict:
        """Volume-weighted price level (Point of Control)."""
        df = df.copy()
        price_bins = pd.cut(df['close'], bins=20)
        volume_by_price = df.groupby(price_bins)['volume'].sum()
        pvp_bin = volume_by_price.idxmax()
        pvp_level = pvp_bin.mid
        
        return {
            "level": pvp_level,
            "volume_at_pvp": volume_by_price.max()
        }

    @staticmethod
    def _detect_accumulation_zones(df: pd.DataFrame) -> dict:
        """Identify areas of price consolidation with high volume."""
        df = df.copy()
        
        # Find ranges with low volatility and high volume
        rolling_std = df['close'].rolling(20).std()
        rolling_vol = df['volume'].rolling(20).mean()
        
        # Zone: low volatility + high volume
        zone_signal = (rolling_std < rolling_std.quantile(0.3)) & (rolling_vol > rolling_vol.quantile(0.7))
        
        if zone_signal.sum() > 0:
            zone_prices = df[zone_signal]['close']
            bottom = zone_prices.min()
            top = zone_prices.max()
            strength = int(100 * zone_signal.sum() / len(df))
        else:
            bottom = df['close'].min()
            top = df['close'].max()
            strength = 0
        
        return {
            "bottom": bottom,
            "top": top,
            "midpoint": (bottom + top) / 2,
            "range": top - bottom,
            "strength": strength,  # 0-100
            "type": "ACCUMULATION" if df['close'].iloc[-1] > (bottom + top) / 2 else "DISTRIBUTION"
        }

    @staticmethod
    def _calculate_accumulation_score(obv: dict, adl: dict, vol_ratio: float, zone: dict) -> int:
        """Composite accumulation score (0-100)."""
        score = 0
        
        # OBV component (30 points)
        if obv["confirmation"]:
            score += 30
        elif obv["strength"] == "MODERATE":
            score += 15
        
        # ADL component (30 points)
        if adl["confirmation"]:
            score += 30
        elif adl["strength"] == "MODERATE":
            score += 15
        
        # Volume ratio (20 points) - buy > 1.5x sell
        if vol_ratio > 1.5:
            score += 20
        elif vol_ratio > 1.2:
            score += 10
        
        # Zone strength (20 points)
        score += int(0.2 * zone["strength"])
        
        return max(0, min(100, score))

    @staticmethod
    def _generate_signals(obv: dict, adl: dict, vol_ratio: float, zone: dict) -> list:
        """Generate actionable trading signals."""
        signals = []
        
        if obv["confirmation"] and adl["confirmation"]:
            signals.append({
                "signal": "DUAL_CONFIRMATION",
                "strength": "VERY_HIGH",
                "description": "Both OBV and ADL confirm accumulation - institutional buying evident"
            })
        
        if vol_ratio > 2.0:
            signals.append({
                "signal": "EXTREME_BUY_VOLUME",
                "strength": "HIGH",
                "description": "Buy volume > 2x sell volume - strong accumulation phase"
            })
        
        if zone["strength"] > 70:
            signals.append({
                "signal": "STRONG_BASE_FORMATION",
                "strength": "HIGH",
                "description": f"Price consolidating in {zone['bottom']:.2f}-{zone['top']:.2f} range"
            })
        
        if obv["trend_score"] > 50 and adl["trend_score"] > 50:
            signals.append({
                "signal": "BREAKOUT_IMMINENT",
                "strength": "MEDIUM",
                "description": "Multiple indicators suggest breakout potential"
            })
        
        return signals

    @staticmethod
    def _calculate_target(df: pd.DataFrame, zone: dict) -> float:
        """Calculate potential breakout target based on zone range."""
        range_size = zone["range"]
        current = df['close'].iloc[-1]
        target = current + (range_size * 1.5)  # 1.5x range projection
        return target

    @staticmethod
    def _assess_risk(obv: dict, adl: dict, vol_ratio: float) -> str:
        """Assess risk level based on indicators."""
        if obv["confirmation"] and adl["confirmation"] and vol_ratio > 1.5:
            return "LOW"
        elif obv["strength"] == "STRONG" or vol_ratio > 1.2:
            return "MEDIUM"
        else:
            return "HIGH"


def detect_egx_accumulation(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
) -> dict:
    """
    Public entry point for accumulation detection.
    """
    return AccumulationDetector.detect_accumulation(symbol, period, interval)
