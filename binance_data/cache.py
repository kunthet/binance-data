import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

class DataCache:
    """Manages caching of downloaded Binance futures data."""
    
    def __init__(self, cache_dir="hlocv_cache"):
        """Initialize the cache manager.
        
        Args:
            cache_dir (str): Directory to store cached data files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _align_to_daily_boundaries(self, dt: datetime) -> tuple[datetime, datetime]:
        """Align datetime to daily boundaries.
        
        Args:
            dt (datetime): Datetime to align
            
        Returns:
            tuple[datetime, datetime]: Start and end of the day
        """
        day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = (day_start + timedelta(days=1)) - timedelta(microseconds=1)
        return day_start, day_end
    
    def get_cache_filename(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> str:
        """Generate cache filename based on parameters.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            timeframe (str): Data timeframe (e.g., '1m', '1h')
            start_time (datetime): Start time
            end_time (datetime): End time
            
        Returns:
            str: Cache filename
        """
        # Align to daily boundaries
        day_start, _ = self._align_to_daily_boundaries(start_time)
        filename = f"{symbol}_{timeframe}_{day_start.strftime('%Y%m%d')}_{day_start.strftime('%Y%m%d')}.npy"
        return str(self.cache_dir / filename)
    
    def save_to_cache(self, data: np.ndarray, symbol: str, timeframe: str, 
                     start_time: datetime, end_time: datetime) -> None:
        """Save data to cache file.
        
        Args:
            data (np.ndarray): Data to cache
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            start_time (datetime): Start time
            end_time (datetime): End time
        """
        filename = self.get_cache_filename(symbol, timeframe, start_time, end_time)
        np.save(filename, data)
    
    def load_from_cache(self, symbol: str, timeframe: str, 
                       start_time: datetime, end_time: datetime) -> np.ndarray:
        """Load data from cache file.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            start_time (datetime): Start time
            end_time (datetime): End time
            
        Returns:
            np.ndarray: Cached data if available, None otherwise
        """
        filename = self.get_cache_filename(symbol, timeframe, start_time, end_time)
        if os.path.exists(filename):
            return np.load(filename)
        return None
    
    def get_cached_files(self, symbol: str, timeframe: str) -> list:
        """Get list of cached files for a symbol and timeframe.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            
        Returns:
            list: List of cached file paths
        """
        pattern = f"{symbol}_{timeframe}_*.npy"
        return list(self.cache_dir.glob(pattern))
    
    def clear_cache(self, symbol: str = None, timeframe: str = None) -> None:
        """Clear cache files matching the specified criteria.
        
        Args:
            symbol (str, optional): Trading pair symbol
            timeframe (str, optional): Data timeframe
        """
        pattern = f"{'*' if symbol is None else symbol}_{'*' if timeframe is None else timeframe}_*.npy"
        for file in self.cache_dir.glob(pattern):
            file.unlink() 