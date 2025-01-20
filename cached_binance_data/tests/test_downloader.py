import pytest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from cached_binance_data import BinanceDataDownloader
from unittest.mock import Mock, patch
import os
import shutil
from cached_binance_data.downloader import TimeFrame
import requests

@pytest.fixture
def mock_response():
    """Create a mock response for requests."""
    def create_mock_for_timestamp(timestamp, interval="1m"):
        mock = Mock()
        # Get interval in seconds
        interval_map = {
            "1m": 60,
            "1h": 3600,
            "1d": 86400,
            "1w": 604800,
            "1M": 2592000
        }
        interval_seconds = interval_map.get(interval, 60) * 1000  # Convert to milliseconds
        
        # Generate data points based on the interval
        data = []
        for i in range(100):  # Return 100 points per request
            current_time = timestamp + (i * interval_seconds)
            data.append([
                current_time,        # Open time
                "50000.00",         # Open
                "50100.00",         # High
                "49900.00",         # Low
                "50050.00",         # Close
                "10.5",             # Volume
                current_time + interval_seconds - 1,  # Close time
                "525250.00",        # Quote asset volume
                100,                # Number of trades
                "5.25",             # Taker buy base asset volume
                "262625.00",        # Taker buy quote asset volume
                "0"                 # Ignore
            ])
        mock.json.return_value = data
        mock.raise_for_status.return_value = None
        return mock

    return create_mock_for_timestamp

@pytest.fixture
def downloader(mock_response):
    """Create a downloader instance for testing."""
    with patch('requests.Session') as mock_session:
        # Configure the session mock to return different data for each request
        session_instance = mock_session.return_value
        def get_mock_response(*args, **kwargs):
            # Extract timestamp and interval from the request parameters
            params = kwargs.get('params', {})
            start_time = int(params.get('startTime', 0))
            interval = params.get('interval', '1m')
            return mock_response(start_time, interval)
        
        session_instance.get.side_effect = get_mock_response
        downloader = BinanceDataDownloader(cache_dir="test_cache")
        yield downloader

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up test cache directory after each test."""
    yield
    if os.path.exists("test_cache"):
        shutil.rmtree("test_cache")

def test_validate_timeframe(downloader):
    """Test timeframe validation."""
    assert downloader.validate_timeframe("1m") == TimeFrame.MINUTE_1
    assert downloader.validate_timeframe("1h") == TimeFrame.HOUR_1
    assert downloader.validate_timeframe("1d") == TimeFrame.DAY_1
    
    with pytest.raises(ValueError):
        downloader.validate_timeframe("invalid")

def test_align_to_daily_boundaries(downloader):
    """Test datetime alignment to daily boundaries."""
    # Test mid-day alignment
    dt = datetime(2020, 1, 1, 12, 30, 45)
    start, end = downloader._align_to_daily_boundaries(dt)
    assert start == datetime(2020, 1, 1, 0, 0, 0)
    assert end == datetime(2020, 1, 1, 23, 59, 59, 999999)
    
    # Test start of day
    dt = datetime(2020, 1, 1, 0, 0, 0)
    start, end = downloader._align_to_daily_boundaries(dt)
    assert start == dt
    assert end == datetime(2020, 1, 1, 23, 59, 59, 999999)
    
    # Test end of day
    dt = datetime(2020, 1, 1, 23, 59, 59)
    start, end = downloader._align_to_daily_boundaries(dt)
    assert start == datetime(2020, 1, 1, 0, 0, 0)
    assert end == datetime(2020, 1, 1, 23, 59, 59, 999999)

def test_get_records_per_day(downloader):
    """Test calculation of records per day."""
    # 1-minute data should have 1440 records per day (24 * 60)
    assert downloader._get_records_per_day(TimeFrame.MINUTE_1) == 1440
    
    # 1-hour data should have 24 records per day
    assert downloader._get_records_per_day(TimeFrame.HOUR_1) == 24
    
    # 1-day data should have 1 record per day
    assert downloader._get_records_per_day(TimeFrame.DAY_1) == 1

def test_split_time_range(downloader):
    """Test time range splitting into daily chunks."""
    # Test exact day boundaries
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2020, 1, 3, 23, 59, 59, 999999)
    chunks = downloader._split_time_range(start, end, TimeFrame.MINUTE_1)
    
    assert len(chunks) == 3
    assert chunks[0][0] == datetime(2020, 1, 1, 0, 0, 0)
    assert chunks[0][1] == datetime(2020, 1, 1, 23, 59, 59, 999999)
    assert chunks[1][0] == datetime(2020, 1, 2, 0, 0, 0)
    assert chunks[1][1] == datetime(2020, 1, 2, 23, 59, 59, 999999)
    assert chunks[2][0] == datetime(2020, 1, 3, 0, 0, 0)
    assert chunks[2][1] == end
    
    # Test partial days
    start = datetime(2020, 1, 1, 12, 0, 0)
    end = datetime(2020, 1, 2, 12, 0, 0)
    chunks = downloader._split_time_range(start, end, TimeFrame.MINUTE_1)
    
    assert len(chunks) == 2
    assert chunks[0][0] == start
    assert chunks[0][1] == datetime(2020, 1, 1, 23, 59, 59, 999999)
    assert chunks[1][0] == datetime(2020, 1, 2, 0, 0, 0)
    assert chunks[1][1] == end

def test_download_integration(downloader):
    """Test the download functionality."""
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    
    df = downloader.download("BTCUSDT", "1m", start, end)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    # Verify timestamps are within the requested range
    assert df['timestamp'].min() >= pd.Timestamp(start)
    assert df['timestamp'].max() <= pd.Timestamp(end)

def test_caching(downloader):
    """Test the caching functionality."""
    # Test with a partial day request
    start = datetime(2020, 1, 1, 12, 0)  # Start at noon
    end = datetime(2020, 1, 2, 12, 0)    # End at noon next day
    
    # First download should hit the API
    df1 = downloader.download("BTCUSDT", "1m", start, end)
    
    # Verify we got data for the requested range
    assert len(df1) > 0
    assert df1['timestamp'].min() >= pd.Timestamp(start)
    assert df1['timestamp'].max() <= pd.Timestamp(end)
    
    # Second download should use cache
    df2 = downloader.download("BTCUSDT", "1m", start, end)
    assert df1.equals(df2)
    
    # Verify cache files are aligned to day boundaries
    cache_files = os.listdir("test_cache")
    assert len(cache_files) > 0
    
    # Get the first cache file and verify its format
    cache_file = cache_files[0]
    parts = cache_file.split('_')
    assert len(parts) == 4, f"Invalid cache filename format: {cache_file}"  # symbol_timeframe_date_date.npy
    
    # Extract date parts
    date_str = parts[2]  # Both start and end dates are the same
    assert len(date_str) == 8, f"Invalid date format in cache filename: {date_str}"
    
    # Parse the date
    date = datetime.strptime(date_str, '%Y%m%d').date()
    assert date in [datetime(2020, 1, 1).date(), datetime(2020, 1, 2).date()], \
        f"Cache file date {date} not in expected range"

def test_http_error_handling():
    """Test handling of HTTP errors."""
    with patch('requests.Session') as mock_session:
        # Configure the session mock to raise an error
        session_instance = mock_session.return_value
        session_instance.get.side_effect = requests.exceptions.RequestException("API Error")
        
        # Create a new downloader instance within the patched context
        downloader = BinanceDataDownloader(cache_dir="test_cache")
        
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 2)
        
        with pytest.raises(requests.exceptions.RequestException):
            downloader.download("BTCUSDT", "1m", start, end)

def test_invalid_symbol(downloader):
    """Test handling of invalid trading pair symbols."""
    with patch.object(downloader.session, 'get') as mock_get:
        mock_get.return_value.json.return_value = []
        mock_get.return_value.raise_for_status.return_value = None
        
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 2)
        
        df = downloader.download("INVALIDPAIR", "1m", start, end)
        assert len(df) == 0
        assert list(df.columns) == ['timestamp', 'high', 'low', 'open', 'close', 'volume']

def test_different_timeframes(downloader):
    """Test downloading data with different timeframes."""
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    
    # Test hourly data
    df_1h = downloader.download("BTCUSDT", "1h", start, end)
    assert len(df_1h) > 0
    assert df_1h['timestamp'].diff().mean().total_seconds() == 3600  # 1 hour in seconds
    
    # Test daily data
    df_1d = downloader.download("BTCUSDT", "1d", start, end)
    assert len(df_1d) > 0
    assert df_1d['timestamp'].diff().mean().total_seconds() == 86400  # 24 hours in seconds

def test_empty_response_handling(downloader):
    """Test handling of empty responses from API."""
    with patch.object(downloader.session, 'get') as mock_get:
        mock_get.return_value.json.return_value = []
        mock_get.return_value.raise_for_status.return_value = None
        
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 2)
        
        df = downloader.download("BTCUSDT", "1m", start, end)
        assert len(df) == 0
        assert list(df.columns) == ['timestamp', 'high', 'low', 'open', 'close', 'volume']

def test_cache_invalidation(downloader):
    """Test cache invalidation functionality."""
    start = datetime(2020, 1, 1, 12, 0)
    end = datetime(2020, 1, 2, 12, 0)
    
    # First download to populate cache
    df1 = downloader.download("BTCUSDT", "1m", start, end)
    assert len(df1) > 0
    
    # Clear cache
    downloader.cache.clear_cache("BTCUSDT", "1m")
    
    # Verify cache files are deleted
    cache_files = os.listdir("test_cache")
    assert len([f for f in cache_files if f.startswith("BTCUSDT_1m")]) == 0
    
    # Download again to verify it works after cache clear
    df2 = downloader.download("BTCUSDT", "1m", start, end)
    assert len(df2) > 0
    pd.testing.assert_frame_equal(df1, df2)

def test_future_dates(downloader):
    """Test handling of future date ranges."""
    now = datetime.now()
    future = now + timedelta(days=7)
    
    df = downloader.download("BTCUSDT", "1m", now, future)
    
    # Should return empty DataFrame for future dates
    assert len(df) == 0
    assert list(df.columns) == ['timestamp', 'high', 'low', 'open', 'close', 'volume']

def test_cross_month_download(downloader):
    """Test downloading data across month boundaries."""
    start = datetime(2020, 1, 31)
    end = datetime(2020, 2, 2)
    
    df = downloader.download("BTCUSDT", "1d", start, end)
    assert len(df) > 0
    
    # Verify we have data for both months
    months = df['timestamp'].dt.month.unique()
    assert len(months) == 2
    assert 1 in months  # January
    assert 2 in months  # February

def test_malformed_response(downloader):
    """Test handling of malformed API responses."""
    with patch.object(downloader.session, 'get') as mock_get:
        # Return malformed data
        mock_get.return_value.json.return_value = [
            ["invalid"],  # Missing required fields
            ["also_invalid", "not_enough_fields"]
        ]
        mock_get.return_value.raise_for_status.return_value = None
        
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 2)
        
        # Should handle malformed data gracefully
        df = downloader.download("BTCUSDT", "1m", start, end)
        assert len(df) == 0
        assert list(df.columns) == ['timestamp', 'high', 'low', 'open', 'close', 'volume'] 