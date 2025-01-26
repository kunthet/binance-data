from datetime import datetime, timedelta
from enum import Enum
import requests
import time
from .cache import DataCache
from typing import Optional, Union, List, Tuple
import os

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

class DateHandler:
    """Handles all date-related operations for data downloading."""
    
    @staticmethod
    def normalize_datetime(dt: Union[str, datetime]) -> datetime:
        """Convert string date to datetime or normalize existing datetime.
        
        Args:
            dt: Date string or datetime object
            
        Returns:
            Normalized datetime object
        """
        if isinstance(dt, str):
            return datetime.strptime(dt, '%Y-%m-%d')
        return dt
    
    @staticmethod
    def align_to_daily_boundaries(dt: datetime) -> Tuple[datetime, datetime]:
        """Align datetime to daily boundaries.
        
        Args:
            dt: Datetime to align
            
        Returns:
            Start and end of the day
        """
        day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = (day_start + timedelta(days=1)) - timedelta(microseconds=1)
        return day_start, day_end
    
    @staticmethod
    def is_today(dt: datetime, current_time: datetime) -> bool:
        """Check if datetime is from today.
        
        Args:
            dt: Datetime to check
            current_time: Current time for comparison
            
        Returns:
            True if datetime is from today
        """
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        return dt >= today_start
    
    @staticmethod
    def split_time_range(start_time: datetime, end_time: datetime) -> List[Tuple[datetime, datetime]]:
        """Split time range into daily chunks.
        
        Args:
            start_time: Start time
            end_time: End time
            
        Returns:
            List of (chunk_start, chunk_end) pairs
        """
        chunks = []
        current_start = start_time
        
        # If start and end are on the same day, return a single chunk
        if current_start.date() == end_time.date():
            chunks.append((current_start, end_time))
            return chunks
        
        # Handle first day (partial if not aligned)
        if current_start.time() != datetime.min.time():
            _, day_end = DateHandler.align_to_daily_boundaries(current_start)
            chunks.append((current_start, day_end))
            current_start = day_end + timedelta(microseconds=1)
        
        # Handle full days
        while current_start < end_time:
            day_start, day_end = DateHandler.align_to_daily_boundaries(current_start)
            if day_end > end_time:
                chunks.append((day_start, end_time))
                break
            chunks.append((day_start, day_end))
            current_start = day_end + timedelta(microseconds=1)
        
        return chunks

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
    
    def __init__(self, cache_dir: Optional[str] = None, 
                 requests_per_minute: int = 1200,
                 today_cache_expiry_minutes: int = 60):
        """Initialize the downloader.
        
        Args:
            cache_dir (str, optional): Directory for caching data. If None, uses system's document folder.
            requests_per_minute (int): Maximum number of requests per minute (default: 1200)
            today_cache_expiry_minutes (int): Cache expiration time in minutes for today's data.
                                            Set to 0 to disable caching for today's data.
                                            Default is 60 minutes.
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
        self.today_cache_expiry = timedelta(minutes=today_cache_expiry_minutes)
        self.date_handler = DateHandler()
    
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
    
    def _get_records_per_day(self, timeframe: TimeFrame) -> int:
        """Calculate number of records per day for a given timeframe.
        
        Args:
            timeframe (TimeFrame): Data timeframe
            
        Returns:
            int: Number of records per day
        """
        seconds_per_day = 24 * 60 * 60
        return seconds_per_day // self.VALID_TIMEFRAMES[timeframe]
    
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
            _, day_end = self.date_handler.align_to_daily_boundaries(current_start)
            chunks.append((current_start, day_end))
            current_start = day_end + timedelta(microseconds=1)
        
        # Handle full days
        while current_start < end_time:
            day_start, day_end = self.date_handler.align_to_daily_boundaries(current_start)
            if day_end > end_time:
                chunks.append((day_start, end_time))
                break
            chunks.append((day_start, day_end))
            current_start = day_end + timedelta(microseconds=1)
        
        return chunks
    
    def _download_chunk(self, symbol: str, timeframe: TimeFrame, 
                       start_time: datetime, end_time: datetime) -> list:
        """Download a single chunk of data from Binance.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (TimeFrame): Data timeframe
            start_time (datetime): Chunk start time
            end_time (datetime): Chunk end time
            
        Returns:
            list: List of lists containing [timestamp, high, low, open, close, volume]
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
            return []
        
        valid_data = []
        for k in klines:
            try:
                if len(k) < 6:  # Need at least 6 elements for OHLCV data
                    continue
                # Binance kline format: [open_time, open, high, low, close, volume, ...]
                valid_data.append([
                    float(k[0]),  # timestamp (open_time)
                    float(k[2]),  # high
                    float(k[3]),  # low
                    float(k[1]),  # open
                    float(k[4]),  # close
                    float(k[5])   # volume
                ])
            except (IndexError, ValueError, TypeError):
                continue  # Skip invalid entries
        
        return valid_data
    
    def _filter_unique_data(self, data: List[List[float]], start_time: datetime, end_time: datetime) -> List[List[float]]:
        """Filter data to include only unique timestamps within the specified time range.
        
        Args:
            data: List of OHLCV data points
            start_time: Start of the time range
            end_time: End of the time range
            
        Returns:
            List of unique data points within the time range
        """
        # Convert datetime to millisecond timestamps for comparison
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        # Sort by timestamp and remove duplicates while filtering time range
        data.sort(key=lambda x: x[0])
        seen_timestamps = set()
        unique_data = []
        
        for row in data:
            # Convert timestamp to integer for consistent comparison
            timestamp = int(float(row[0]))
            
            # Check if timestamp is within range and not seen before
            if timestamp not in seen_timestamps and start_ts <= timestamp <= end_ts:
                seen_timestamps.add(timestamp)
                unique_data.append(row)
        
        return sorted(unique_data, key=lambda x: x[0])  # Ensure sorted output
    
    def _ensure_interval_spacing(self, data: List[List[float]], interval_seconds: int) -> List[List[float]]:
        """Ensure proper interval spacing between data points, forward-filling missing values.
        
        Args:
            data: List of OHLCV data points
            interval_seconds: Expected interval between points in seconds
            
        Returns:
            List of data points with proper interval spacing
        """
        if not data:
            return []
            
        interval_ms = interval_seconds * 1000
        expected_data = []
        
        # Align timestamps to interval boundaries
        first_ts = int(float(data[0][0]))
        last_ts = int(float(data[-1][0]))
        
        # Calculate the number of intervals needed
        num_intervals = (last_ts - first_ts) // interval_ms
        if (last_ts - first_ts) % interval_ms > 0:
            num_intervals += 1
        
        # Generate timestamps for each interval
        i = 0  # Index in data array
        for interval in range(int(num_intervals + 1)):
            current_ts = first_ts + (interval * interval_ms)
            
            # Find if we have a data point for this interval
            while i < len(data) and int(float(data[i][0])) < current_ts:
                i += 1
                
            if i < len(data) and abs(int(float(data[i][0])) - current_ts) < interval_ms:
                # Use actual data point
                expected_data.append(data[i])
                i += 1
            else:
                # Forward fill from previous point
                prev_point = list(data[i-1] if i > 0 else data[0])
                prev_point[0] = float(current_ts)
                expected_data.append(prev_point)
        
        return expected_data
    
    def _should_use_cache(self, chunk_start: datetime, current_time: datetime, force_refresh: bool) -> bool:
        """Determine if cache should be used for the given chunk.
        
        Args:
            chunk_start: Start time of the chunk
            current_time: Current time
            force_refresh: Whether to force refresh data
            
        Returns:
            bool: True if cache should be used, False otherwise
        """
        is_today = self.date_handler.is_today(chunk_start, current_time)
        return not (force_refresh or (is_today and self.today_cache_expiry.total_seconds() == 0))
    
    def _is_cache_expired(self, cache_file: str, current_time: datetime, is_today: bool) -> bool:
        """Check if cache is expired for today's data.
        
        Args:
            cache_file: Path to cache file
            current_time: Current time
            is_today: Whether the data is from today
            
        Returns:
            bool: True if cache is expired, False otherwise
        """
        if not is_today:
            return False
        cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        return current_time - cache_mtime > self.today_cache_expiry
    
    def _process_chunk(self, symbol: str, timeframe: str, timeframe_enum: TimeFrame,
                      chunk_start: datetime, chunk_end: datetime, 
                      current_time: datetime, force_refresh: bool) -> List[List[float]]:
        """Process a single chunk of data, handling cache and download logic.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time interval string
            timeframe_enum: TimeFrame enum
            chunk_start: Start time of chunk
            chunk_end: End time of chunk
            current_time: Current time
            force_refresh: Whether to force refresh data
            
        Returns:
            List of OHLCV data points
        """
        if self._should_use_cache(chunk_start, current_time, force_refresh):
            cached_data = self.cache.load_from_cache(symbol, timeframe, chunk_start, chunk_end)
            if cached_data is not None:
                is_today = self.date_handler.is_today(chunk_start, current_time)
                cache_file = self.cache.get_cache_filename(symbol, timeframe, chunk_start, chunk_end)
                if not self._is_cache_expired(cache_file, current_time, is_today):
                    return cached_data

        # Download and cache if needed
        chunk_data = self._download_chunk(symbol, timeframe_enum, chunk_start, chunk_end)
        if chunk_data and self._should_use_cache(chunk_start, current_time, force_refresh):
            self.cache.save_to_cache(chunk_data, symbol, timeframe, chunk_start, chunk_end)
        return chunk_data or []
    
    def download(self, symbol: str, timeframe: str, 
                start_time: Union[str, datetime], 
                end_time: Union[str, datetime],
                force_refresh: bool = False) -> List[List[float]]:
        """Download market data for the specified symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            timeframe: Time interval (e.g., "1m", "1h", "1d")
            start_time: Start time for data download
            end_time: End time for data download
            force_refresh: If True, ignore cache and force download fresh data
            
        Returns:
            List of OHLCV data points
        """
        timeframe_enum = self.validate_timeframe(timeframe)
        
        # Normalize dates
        start_time = self.date_handler.normalize_datetime(start_time)
        end_time = self.date_handler.normalize_datetime(end_time)
        
        # If end_time is just a date, set it to end of day
        if end_time.hour == 0 and end_time.minute == 0 and end_time.second == 0:
            end_time = end_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Return empty list for future dates - do this check first
        current_time = datetime.now().replace(microsecond=0)  # Remove microseconds for stable comparison
        if start_time >= current_time:
            return []
        
        # Adjust end_time if it's in the future and validate time range
        if end_time > current_time:
            end_time = current_time
        
        if end_time <= start_time:
            return []
        
        chunks = self._split_time_range(start_time, end_time, timeframe_enum)
        all_data = []
        
        for chunk_start, chunk_end in chunks:
            chunk_data = self._process_chunk(
                symbol, timeframe, timeframe_enum,
                chunk_start, chunk_end, current_time, force_refresh
            )
            all_data.extend(chunk_data)
        
        if not all_data:
            return []
        
        # Filter unique data points within time range
        unique_data = self._filter_unique_data(all_data, start_time, end_time)
        
        # Ensure proper interval spacing for timeframes larger than 1 minute
        if timeframe_enum != TimeFrame.MINUTE_1 and unique_data:
            unique_data = self._ensure_interval_spacing(
                unique_data, 
                self.VALID_TIMEFRAMES[timeframe_enum]
            )
        
        return unique_data 