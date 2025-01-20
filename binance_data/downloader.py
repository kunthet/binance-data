from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import time
from enum import Enum
from .cache import DataCache
from typing import Optional, Union, List, Tuple

class TimeFrame(str, Enum):
    """Supported timeframes for data download."""
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

class BinanceDataDownloader:
    """Downloads and manages Binance futures market data."""
    
    BASE_URL = "https://fapi.binance.com"
    VALID_TIMEFRAMES = {
        TimeFrame.MINUTE_1: 60,
        TimeFrame.MINUTE_3: 180,
        TimeFrame.MINUTE_5: 300,
        TimeFrame.MINUTE_15: 900,
        TimeFrame.MINUTE_30: 1800,
        TimeFrame.HOUR_1: 3600,
        TimeFrame.HOUR_2: 7200,
        TimeFrame.HOUR_4: 14400,
        TimeFrame.HOUR_6: 21600,
        TimeFrame.HOUR_8: 28800,
        TimeFrame.HOUR_12: 43200,
        TimeFrame.DAY_1: 86400,
        TimeFrame.DAY_3: 259200,
        TimeFrame.WEEK_1: 604800,
        TimeFrame.MONTH_1: 2592000
    }
    
    def __init__(self, cache_dir: str = "hlocv_cache", requests_per_minute: int = 1200):
        """Initialize the downloader.
        
        Args:
            cache_dir (str): Directory for caching data
            requests_per_minute (int): Maximum number of requests per minute (default: 1200)
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
        })
        self.cache = DataCache(cache_dir)
        self.requests_per_minute = requests_per_minute
        self.last_request_time = 0
        self.min_request_interval = 60.0 / requests_per_minute  # Time in seconds between requests
    
    def validate_timeframe(self, timeframe: str) -> TimeFrame:
        """Validate and convert the timeframe string to TimeFrame enum.
        
        Args:
            timeframe (str): Timeframe to validate
            
        Returns:
            TimeFrame: Validated timeframe enum
            
        Raises:
            ValueError: If timeframe is invalid
        """
        try:
            return TimeFrame(timeframe)
        except ValueError:
            valid_options = [tf.value for tf in TimeFrame]
            raise ValueError(f"Invalid timeframe. Valid options are: {valid_options}")
    
    def _align_to_daily_boundaries(self, dt: datetime) -> Tuple[datetime, datetime]:
        """Align datetime to daily boundaries.
        
        Args:
            dt (datetime): Datetime to align
            
        Returns:
            Tuple[datetime, datetime]: Start and end of the day
        """
        day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = (day_start + timedelta(days=1)) - timedelta(microseconds=1)
        return day_start, day_end
    
    def _get_records_per_day(self, timeframe: TimeFrame) -> int:
        """Calculate number of records per day for a given timeframe.
        
        Args:
            timeframe (TimeFrame): Data timeframe
            
        Returns:
            int: Number of records per day
        """
        seconds_per_day = 24 * 60 * 60
        return seconds_per_day // self.VALID_TIMEFRAMES[timeframe]
    
    def _split_time_range(self, start_time: datetime, end_time: datetime, 
                         timeframe: TimeFrame) -> List[Tuple[datetime, datetime]]:
        """Split time range into daily chunks.
        
        Args:
            start_time (datetime): Start time
            end_time (datetime): End time
            timeframe (TimeFrame): Data timeframe
            
        Returns:
            List[Tuple[datetime, datetime]]: List of (chunk_start, chunk_end) pairs
        """
        chunks = []
        current_start = start_time
        
        # Handle first day (partial if not aligned)
        if current_start.time() != datetime.min.time():
            _, day_end = self._align_to_daily_boundaries(current_start)
            chunks.append((current_start, day_end))
            current_start = day_end + timedelta(microseconds=1)
        
        # Handle full days
        while current_start < end_time:
            day_start, day_end = self._align_to_daily_boundaries(current_start)
            if day_end > end_time:
                chunks.append((day_start, end_time))
                break
            chunks.append((day_start, day_end))
            current_start = day_end + timedelta(microseconds=1)
        
        return chunks
    
    def _download_chunk(self, symbol: str, timeframe: TimeFrame, 
                       start_time: datetime, end_time: datetime) -> np.ndarray:
        """Download a single chunk of data from Binance.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (TimeFrame): Data timeframe
            start_time (datetime): Chunk start time
            end_time (datetime): Chunk end time
            
        Returns:
            np.ndarray: Downloaded data
        """
        # Apply rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        params = {
            'symbol': symbol.upper(),  # Ensure symbol is uppercase
            'interval': timeframe.value,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000),
            'limit': 1500  # Maximum limit as per Binance API docs
        }
        
        response = self.session.get(f"{self.BASE_URL}/fapi/v1/klines", params=params)
        self.last_request_time = time.time()  # Update last request time
        response.raise_for_status()
        
        klines = response.json()
        
        if not klines:
            return np.array([])
        
        valid_data = []
        for k in klines:
            try:
                if len(k) < 6:  # Need at least 6 elements for OHLCV data
                    continue
                valid_data.append([
                    float(k[0]),  # timestamp
                    float(k[2]),  # high
                    float(k[3]),  # low
                    float(k[1]),  # open
                    float(k[4]),  # close
                    float(k[5])   # volume
                ])
            except (IndexError, ValueError, TypeError):
                continue  # Skip invalid entries
        
        if not valid_data:
            return np.array([])
            
        return np.array(valid_data)
    
    def download(self, symbol: str, timeframe: str, 
                start_time: Union[str, datetime], 
                end_time: Union[str, datetime]) -> pd.DataFrame:
        """Download Binance futures data for the specified period.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            timeframe (str): Data timeframe (e.g., '1m', '1h')
            start_time (Union[str, datetime]): Start time (string format: 'YYYY-MM-DD' or datetime)
            end_time (Union[str, datetime]): End time (string format: 'YYYY-MM-DD' or datetime)
            
        Returns:
            pd.DataFrame: Downloaded data with columns [timestamp, high, low, open, close, volume]
        """
        timeframe_enum = self.validate_timeframe(timeframe)
        
        # Convert string dates to datetime if necessary
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, '%Y-%m-%d')
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, '%Y-%m-%d')
            
        # If end_time is just a date, set it to end of day
        if end_time.hour == 0 and end_time.minute == 0 and end_time.second == 0:
            end_time = end_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Return empty DataFrame for future dates
        current_time = datetime.now()
        if start_time > current_time:
            return pd.DataFrame(columns=['timestamp', 'high', 'low', 'open', 'close', 'volume'])
        
        # Adjust end_time if it's in the future
        if end_time > current_time:
            end_time = current_time
        
        chunks = self._split_time_range(start_time, end_time, timeframe_enum)
        all_data = []
        
        for chunk_start, chunk_end in chunks:
            # Try to load from cache first
            cached_data = self.cache.load_from_cache(symbol, timeframe, chunk_start, chunk_end)
            
            if cached_data is not None and len(cached_data) > 0:
                all_data.append(cached_data)
            else:
                # Download and cache if not found
                chunk_data = self._download_chunk(symbol, timeframe_enum, chunk_start, chunk_end)
                if len(chunk_data) > 0:
                    self.cache.save_to_cache(chunk_data, symbol, timeframe, chunk_start, chunk_end)
                    all_data.append(chunk_data)
        
        if not all_data:
            return pd.DataFrame(columns=['timestamp', 'high', 'low', 'open', 'close', 'volume'])
        
        # Combine all chunks and convert to DataFrame
        combined_data = np.concatenate(all_data)
        df = pd.DataFrame(
            combined_data,
            columns=['timestamp', 'high', 'low', 'open', 'close', 'volume']
        )
        
        # Convert timestamp to datetime and filter to requested range
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[
            (df['timestamp'] >= pd.Timestamp(start_time)) & 
            (df['timestamp'] <= pd.Timestamp(end_time))
        ]
        
        # Sort by timestamp and remove duplicates
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first')
        
        # For timeframes larger than 1 minute, ensure proper interval spacing
        if timeframe_enum != TimeFrame.MINUTE_1:
            interval_seconds = self.VALID_TIMEFRAMES[timeframe_enum]
            expected_timestamps = pd.date_range(
                start=df['timestamp'].min(),
                end=df['timestamp'].max(),
                freq=f"{interval_seconds}s"
            )
            df = df.set_index('timestamp').reindex(expected_timestamps).reset_index()
            df = df.rename(columns={'index': 'timestamp'})
            # Forward fill missing values
            df = df.ffill()
        
        return df.reset_index(drop=True) 