"""
EGX Data Cache Layer - Local SQLite Storage
════════════════════════════════════════════════════════════════════════════

Stores real-time market data locally for:
  - Fast access
  - Offline capability
  - Historical tracking
  - Disaster recovery
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
from threading import Lock


class EGXDataCache:
    """Thread-safe SQLite cache for EGX market data."""
    
    def __init__(self, db_path: str = "egx_market_data.db"):
        self.db_path = Path(db_path)
        self.lock = Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite tables."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Stocks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    symbol TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    price REAL NOT NULL,
                    change_percent REAL,
                    volume INTEGER,
                    high_52w REAL,
                    low_52w REAL,
                    sector TEXT,
                    market_cap REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Historical OHLCV table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            # Intraday data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS intraday (
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    price REAL NOT NULL,
                    volume INTEGER,
                    bid REAL,
                    ask REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            
            # Sectors table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sectors (
                    name TEXT PRIMARY KEY,
                    stocks TEXT NOT NULL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indices (
                    name TEXT PRIMARY KEY,
                    value REAL NOT NULL,
                    change_percent REAL,
                    constituents TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Data quality log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    symbol TEXT,
                    status TEXT,
                    message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
    
    def insert_stock(self, symbol: str, name: str, price: float, 
                    change_percent: float, volume: int, high_52w: float,
                    low_52w: float, sector: Optional[str] = None,
                    market_cap: Optional[float] = None) -> bool:
        """Insert or update stock data."""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO stocks
                    (symbol, name, price, change_percent, volume, high_52w, low_52w, sector, market_cap, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """, (symbol, name, price, change_percent, volume, high_52w, low_52w, sector, market_cap))
                
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            self._log_error("insert_stock", symbol, str(e))
            return False
    
    def get_stock(self, symbol: str) -> Optional[Dict]:
        """Get stock data from cache."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM stocks WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return dict(row)
            return None
    
    def get_all_stocks(self) -> List[Dict]:
        """Get all stocks from cache."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM stocks ORDER BY symbol")
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
    
    def insert_ohlcv(self, symbol: str, date: str, open_: float, high: float,
                    low: float, close: float, volume: int) -> bool:
        """Insert daily OHLCV data."""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO ohlcv
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (symbol, date, open_, high, low, close, volume))
                
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            self._log_error("insert_ohlcv", symbol, str(e))
            return False
    
    def get_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get OHLCV data for date range."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT * FROM ohlcv 
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
            conn.close()
            
            return df
    
    def insert_intraday(self, symbol: str, timestamp: str, price: float,
                       volume: Optional[int] = None, bid: Optional[float] = None,
                       ask: Optional[float] = None) -> bool:
        """Insert intraday data point."""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO intraday
                    (symbol, timestamp, price, volume, bid, ask)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (symbol, timestamp, price, volume, bid, ask))
                
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            self._log_error("insert_intraday", symbol, str(e))
            return False
    
    def get_intraday(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """Get intraday data for last N hours."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT * FROM intraday 
                WHERE symbol = ? AND timestamp > datetime('now', ? || ' hours')
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, -hours))
            conn.close()
            
            return df
    
    def insert_sector(self, name: str, stocks: List[str]) -> bool:
        """Insert sector data."""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO sectors (name, stocks)
                    VALUES (?, ?)
                """, (name, json.dumps(stocks)))
                
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            self._log_error("insert_sector", name, str(e))
            return False
    
    def get_sector(self, name: str) -> Optional[List[str]]:
        """Get stocks in a sector."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT stocks FROM sectors WHERE name = ?", (name,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return json.loads(row[0])
            return None
    
    def insert_index(self, name: str, value: float, change_percent: float,
                    constituents: List[str]) -> bool:
        """Insert index data."""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO indices
                    (name, value, change_percent, constituents)
                    VALUES (?, ?, ?, ?)
                """, (name, value, change_percent, json.dumps(constituents)))
                
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            self._log_error("insert_index", name, str(e))
            return False
    
    def get_index(self, name: str) -> Optional[Dict]:
        """Get index data."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM indices WHERE name = ?", (name,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                result = dict(row)
                result['constituents'] = json.loads(result['constituents'])
                return result
            return None
    
    def cleanup_old_intraday(self, days: int = 30):
        """Remove intraday data older than N days."""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM intraday 
                    WHERE timestamp < datetime('now', ? || ' days')
                """, (-days,))
                
                conn.commit()
                conn.close()
        except Exception as e:
            self._log_error("cleanup", None, str(e))
    
    def _log_error(self, operation: str, symbol: Optional[str], message: str):
        """Log data quality issues."""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO data_quality_log (source, symbol, status, message)
                    VALUES (?, ?, 'ERROR', ?)
                """, (operation, symbol, message))
                
                conn.commit()
                conn.close()
        except:
            pass  # Prevent recursive errors
