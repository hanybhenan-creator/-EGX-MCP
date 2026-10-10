"""Data source adapters for the EGX acquisition layer."""

from .base import (
    Capability,
    HealthSnapshot,
    MarketDataSource,
    SourceDataError,
    SourceDescriptor,
    SourceHealth,
    SourceParseError,
    SourceRegistry,
    SourceUnavailable,
    OFFICIAL_PREFERRED,
)
from .egx_official import EGXOfficialSource, ParseReport, ParsedRows
from .yahoo import YahooFinanceSource, bare_symbol, yahoo_symbol
from .tradingview import TradingViewSource, normalise_tv_row, normalise_tv_rows

__all__ = [
    "Capability",
    "HealthSnapshot",
    "MarketDataSource",
    "SourceDataError",
    "SourceDescriptor",
    "SourceHealth",
    "SourceParseError",
    "SourceRegistry",
    "SourceUnavailable",
    "OFFICIAL_PREFERRED",
    "EGXOfficialSource",
    "ParseReport",
    "ParsedRows",
    "YahooFinanceSource",
    "bare_symbol",
    "yahoo_symbol",
    "TradingViewSource",
    "normalise_tv_row",
    "normalise_tv_rows",
]
