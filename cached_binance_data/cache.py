import os
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

class DataCache:
    """Manages caching of downloaded Binance futures market data."""
    
    def __init__(self, cache_dir=None):
        """Initialize the cache manager.
        
        Args:
            cache_dir (str, optional): Directory to store cached data files. If None, uses system's document folder.
        """
        if cache_dir is None:
            # Use system's document folder
            home = Path.home()
            if os.name == 'nt':  # Windows
                docs = home / 'Documents'
            elif os.name == 'darwin':  # macOS
                docs = home / 'Documents'
            else:  # Linux and others
                docs = home / 'Documents'
                if not docs.exists():
                    docs = home
            
            self.cache_dir = docs / 'binance_data'
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
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
        filename = f"{symbol}_{timeframe}_{day_start.strftime('%Y%m%d')}_{day_start.strftime('%Y%m%d')}.csv"
        return str(self.cache_dir / filename)
    
    def save_to_cache(self, data: list, symbol: str, timeframe: str, 
                     start_time: datetime, end_time: datetime) -> None:
        """Save data to cache file.
        
        Args:
            data (list): List of lists containing [timestamp, high, low, open, close, volume]
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            start_time (datetime): Start time
            end_time (datetime): End time
        """
        filename = self.get_cache_filename(symbol, timeframe, start_time, end_time)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'high', 'low', 'open', 'close', 'volume'])
            writer.writerows(data)
    
    def load_from_cache(self, symbol: str, timeframe: str, 
                       start_time: datetime, end_time: datetime) -> list:
        """Load data from cache file.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            start_time (datetime): Start time
            end_time (datetime): End time
            
        Returns:
            list: List of lists containing [timestamp, high, low, open, close, volume], or None if not found
        """
        filename = self.get_cache_filename(symbol, timeframe, start_time, end_time)
        if os.path.exists(filename):
            data = []
            with open(filename, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    data.append([float(x) for x in row])
            return data
        return None
    
    def get_cached_files(self, symbol: str, timeframe: str) -> list:
        """Get list of cached files for a symbol and timeframe.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Data timeframe
            
        Returns:
            list: List of cached file paths
        """
        pattern = f"{symbol}_{timeframe}_*.csv"
        return list(self.cache_dir.glob(pattern))
    
    def clear_cache(self, symbol: str = None, timeframe: str = None) -> None:
        """Clear cache files matching the specified criteria.
        
        Args:
            symbol (str, optional): Trading pair symbol
            timeframe (str, optional): Data timeframe
        """
        pattern = f"{'*' if symbol is None else symbol}_{'*' if timeframe is None else timeframe}_*.csv"
        for file in self.cache_dir.glob(pattern):
            file.unlink()

class CacheManager:
    """Manages cache operations for downloaded data."""
    
    def __init__(self, cache_dir: Optional[str] = None, today_cache_expiry: timedelta = timedelta(minutes=60)):
        """Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cached data files
            today_cache_expiry: Cache expiration time for today's data
        """
        self.cache = DataCache(cache_dir)
        self.today_cache_expiry = today_cache_expiry
    
    def should_use_cache(self, chunk_start: datetime, current_time: datetime, force_refresh: bool) -> bool:
        """Determine if cache should be used for the given chunk.
        
        Args:
            chunk_start: Start time of the chunk
            current_time: Current time
            force_refresh: Whether to force refresh data
            
        Returns:
            bool: True if cache should be used
        """
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        is_today = chunk_start >= today_start
        return not (force_refresh or (is_today and self.today_cache_expiry.total_seconds() == 0))
    
    def is_cache_expired(self, cache_file: str, current_time: datetime, is_today: bool) -> bool:
        """Check if cache is expired for today's data.
        
        Args:
            cache_file: Path to cache file
            current_time: Current time
            is_today: Whether the data is from today
            
        Returns:
            bool: True if cache is expired
        """
        if not is_today:
            return False
        cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        return current_time - cache_mtime > self.today_cache_expiry
    
    def try_get_cached_data(self, symbol: str, timeframe: str, 
                           chunk_start: datetime, chunk_end: datetime,
                           current_time: datetime, force_refresh: bool) -> Optional[List[List[float]]]:
        """Try to get data from cache if possible.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time interval string
            chunk_start: Start time of chunk
            chunk_end: End time of chunk
            current_time: Current time
            force_refresh: Whether to force refresh data
            
        Returns:
            Cached data if available and valid, None otherwise
        """
        if not self.should_use_cache(chunk_start, current_time, force_refresh):
            return None
            
        cached_data = self.cache.load_from_cache(symbol, timeframe, chunk_start, chunk_end)
        if cached_data is None:
            return None
            
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        is_today = chunk_start >= today_start
        
        if is_today:
            cache_file = self.cache.get_cache_filename(symbol, timeframe, chunk_start, chunk_end)
            if self.is_cache_expired(cache_file, current_time, is_today):
                return None
                
        return cached_data
    
    def save_to_cache(self, data: List[List[float]], symbol: str, timeframe: str,
                     chunk_start: datetime, chunk_end: datetime,
                     current_time: datetime) -> None:
        """Save data to cache if appropriate.
        
        Args:
            data: Data to cache
            symbol: Trading pair symbol
            timeframe: Time interval string
            chunk_start: Start time of chunk
            chunk_end: End time of chunk
            current_time: Current time
        """
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        is_today = chunk_start >= today_start
        
        if not is_today or self.today_cache_expiry.total_seconds() > 0:
            self.cache.save_to_cache(data, symbol, timeframe, chunk_start, chunk_end) 