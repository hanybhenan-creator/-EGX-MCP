"""
EGX Data Source Abstraction Layer
════════════════════════════════════════════════════════════════════════════

Abstract base classes for EGX data sources.
Allows swapping sources without changing analysis code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict
import pandas as pd


@dataclass
class EGXStock:
    """Represents an EGX stock with current data."""
    symbol: str
    name: str
    price: float
    change_percent: float
    volume: int
    high_52w: float
    low_52w: float
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class EGXQuote:
    """Real-time market quote."""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: datetime


@dataclass
class EGXIndex:
    """EGX Index data."""
    name: str  # EGX30, EGX70, EGX100, etc.
    value: float
    change_percent: float
    constituents: List[str]
    timestamp: datetime


class DataSourceException(Exception):
    """Base exception for data source errors."""
    pass


class DataValidationException(DataSourceException):
    """Data validation failed."""
    pass


class EGXDataSource(ABC):
    """
    Abstract base class for EGX data sources.
    
    Contract:
      - Must implement all abstract methods
      - Must raise DataSourceException on failure
      - Must return validated data or raise DataValidationException
      - Must be thread-safe
    """
    
    @abstractmethod
    def get_stock(self, symbol: str) -> EGXStock:
        """Get real-time stock quote."""
        pass
    
    @abstractmethod
    def get_stocks(self, symbols: List[str]) -> List[EGXStock]:
        """Get multiple stock quotes."""
        pass
    
    @abstractmethod
    def get_all_stocks(self) -> List[EGXStock]:
        """Get all EGX listed stocks."""
        pass
    
    @abstractmethod
    def get_index(self, index_name: str) -> EGXIndex:
        """Get index data (EGX30, EGX70, EGX100)."""
        pass
    
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        Returns DataFrame with columns:
          - open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    def get_sector_data(self, sector: str) -> List[EGXStock]:
        """Get all stocks in a sector."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if data source is available/responsive."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get data source name."""
        pass


class EGXDataValidator:
    """Validate EGX data for quality issues."""
    
    @staticmethod
    def validate_stock(stock: EGXStock) -> bool:
        """Validate stock data integrity."""
        if stock.price <= 0:
            raise DataValidationException(f"Invalid price: {stock.price}")
        if stock.volume < 0:
            raise DataValidationException(f"Invalid volume: {stock.volume}")
        if not stock.symbol or len(stock.symbol) == 0:
            raise DataValidationException("Empty symbol")
        return True
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> bool:
        """Validate OHLCV dataframe."""
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(df.columns):
            raise DataValidationException(f"Missing columns: {required_cols - set(df.columns)}")
        
        # Check logical relationships
        if (df['high'] < df['low']).any():
            raise DataValidationException("High < Low detected")
        if (df['close'] < 0).any() or (df['open'] < 0).any():
            raise DataValidationException("Negative prices detected")
        if (df['volume'] < 0).any():
            raise DataValidationException("Negative volume detected")
        
        return True
    
    @staticmethod
    def validate_index(index: EGXIndex) -> bool:
        """Validate index data."""
        if index.value <= 0:
            raise DataValidationException(f"Invalid index value: {index.value}")
        if not index.name:
            raise DataValidationException("Empty index name")
        if len(index.constituents) == 0:
            raise DataValidationException("No constituents")
        return True
