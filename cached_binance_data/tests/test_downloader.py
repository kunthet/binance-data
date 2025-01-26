import pytest
from datetime import datetime, timedelta
import os
import shutil
from pathlib import Path
import csv
from cached_binance_data import BinanceDataDownloader
from unittest.mock import Mock, patch
from cached_binance_data.downloader import TimeFrame
import requests

@pytest.fixture
def mock_response():
    """Create a mock response for requests."""
    def create_mock_for_timestamp(timestamp, interval="1m", end_time=None):
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
        
        # Don't generate data for future timestamps
        now_ts = int(datetime.now().timestamp() * 1000)
        if int(float(timestamp)) > now_ts:
            mock.json.return_value = []
            mock.raise_for_status.return_value = None
            return mock
        
        # Generate data points based on the interval
        data = []
        current_time = int(float(timestamp))  # Ensure integer timestamp
        
        # Use provided end_time or default to 100 intervals
        end_timestamp = int(float(end_time)) if end_time is not None else current_time + (100 * interval_seconds)
        
        # Don't generate data beyond current time
        end_timestamp = min(end_timestamp, now_ts)
        
        i = 0
        while current_time <= end_timestamp and i < 100:  # Respect both end time and max points
            point_time = current_time + (i * interval_seconds)
            if point_time > end_timestamp:
                break
                
            data.append([
                float(point_time),  # Open time (will be used as timestamp)
                "50000.00",        # Open
                "50100.00",        # High
                "49900.00",        # Low
                "50050.00",        # Close
                "10.5",            # Volume
                float(point_time + interval_seconds - 1),  # Close time
                "525250.00",       # Quote asset volume
                100,               # Number of trades
                "5.25",            # Taker buy base asset volume
                "262625.00",       # Taker buy quote asset volume
                "0"               # Ignore
            ])
            i += 1
            
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
            start_time = params.get('startTime', 0)
            end_time = params.get('endTime', 0)
            interval = params.get('interval', '1m')
            return mock_response(start_time, interval, end_time)
        
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
    start = datetime(2020, 1, 1)  # Start at midnight Jan 1
    end = datetime(2020, 1, 1, 23, 59, 59, 999999)  # End at last microsecond of Jan 1
    
    data = downloader.download("BTCUSDT", "1m", start, end)
    
    assert isinstance(data, list)
    assert len(data) > 0
    assert len(data[0]) == 6  # [timestamp, high, low, open, close, volume]
    
    # Verify timestamps are within the requested range
    start_ts = start.timestamp() * 1000
    end_ts = end.timestamp() * 1000
    
    # Debug information
    if data:
        first_ts = data[0][0]
        last_ts = data[-1][0]
        print(f"\nTimestamp verification:")
        print(f"Start timestamp: {start_ts}")
        print(f"End timestamp: {end_ts}")
        print(f"First data timestamp: {first_ts}")
        print(f"Last data timestamp: {last_ts}")
        print(f"First data datetime: {datetime.fromtimestamp(first_ts/1000)}")
        print(f"Last data datetime: {datetime.fromtimestamp(last_ts/1000)}")
        
        # Check each timestamp individually
        invalid_timestamps = [(i, row[0]) for i, row in enumerate(data) 
                            if not (start_ts <= row[0] <= end_ts)]
        if invalid_timestamps:
            print("\nInvalid timestamps found:")
            for idx, ts in invalid_timestamps:
                print(f"Row {idx}: {ts} ({datetime.fromtimestamp(ts/1000)})")
    
    assert all(start_ts <= row[0] <= end_ts for row in data)

def test_caching(downloader):
    """Test the caching functionality."""
    # Test with a partial day request
    start = datetime(2020, 1, 1, 12, 0)  # Start at noon
    end = datetime(2020, 1, 2, 12, 0)    # End at noon next day
    
    # First download should hit the API
    data1 = downloader.download("BTCUSDT", "1m", start, end)
    
    # Verify we got data for the requested range
    assert len(data1) > 0
    start_ts = start.timestamp() * 1000
    end_ts = end.timestamp() * 1000
    assert all(start_ts <= row[0] <= end_ts for row in data1)
    
    # Second download should use cache
    data2 = downloader.download("BTCUSDT", "1m", start, end)
    assert len(data1) == len(data2)
    assert all(row1 == row2 for row1, row2 in zip(data1, data2))
    
    # Verify cache files are aligned to day boundaries
    cache_files = os.listdir("test_cache")
    assert len(cache_files) > 0
    
    # Get the first cache file and verify its format
    cache_file = cache_files[0]
    parts = cache_file.split('_')
    assert len(parts) == 4, f"Invalid cache filename format: {cache_file}"  # symbol_timeframe_date_date.csv
    
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
        
        data = downloader.download("INVALIDPAIR", "1m", start, end)
        assert len(data) == 0

def test_different_timeframes(downloader):
    """Test downloading data with different timeframes."""
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    
    # Test hourly data
    data_1h = downloader.download("BTCUSDT", "1h", start, end)
    assert len(data_1h) > 0
    # Verify interval between timestamps
    for i in range(1, len(data_1h)):
        assert data_1h[i][0] - data_1h[i-1][0] == 3600000  # 1 hour in milliseconds
    
    # Test daily data
    data_1d = downloader.download("BTCUSDT", "1d", start, end)
    assert len(data_1d) > 0
    # Verify interval between timestamps
    for i in range(1, len(data_1d)):
        assert data_1d[i][0] - data_1d[i-1][0] == 86400000  # 24 hours in milliseconds

def test_empty_response_handling(downloader):
    """Test handling of empty responses from API."""
    with patch.object(downloader.session, 'get') as mock_get:
        mock_get.return_value.json.return_value = []
        mock_get.return_value.raise_for_status.return_value = None
        
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 2)
        
        data = downloader.download("BTCUSDT", "1m", start, end)
        assert len(data) == 0

def test_future_dates(downloader):
    """Test handling of future date ranges."""
    now = datetime.now()
    future = now + timedelta(days=7)
    
    data = downloader.download("BTCUSDT", "1m", now, future)
    assert len(data) == 0

def test_cross_month_download(downloader):
    """Test downloading data across month boundaries."""
    start = datetime(2020, 1, 31)
    end = datetime(2020, 2, 2)
    
    data = downloader.download("BTCUSDT", "1d", start, end)
    assert len(data) > 0
    
    # Verify we have data for both months
    months = set()
    for row in data:
        dt = datetime.fromtimestamp(row[0] / 1000)
        months.add(dt.month)
    
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
        data = downloader.download("BTCUSDT", "1m", start, end)
        assert len(data) == 0

def test_default_cache_directory():
    """Test that default cache directory is set to system's document folder."""
    with patch('requests.Session'):
        downloader = BinanceDataDownloader()
        expected_path = Path.home() / 'Documents' / 'binance_data'
        if os.name != 'darwin' and os.name != 'nt' and not (Path.home() / 'Documents').exists():
            expected_path = Path.home() / 'binance_data'
        
        assert downloader.cache.cache_dir == expected_path
        assert expected_path.exists()
        
        # Cleanup
        shutil.rmtree(expected_path)

def test_custom_cache_directory():
    """Test that custom cache directory is used when specified."""
    custom_dir = "test_custom_cache"
    with patch('requests.Session'):
        downloader = BinanceDataDownloader(cache_dir=custom_dir)
        expected_path = Path(custom_dir)
        
        assert downloader.cache.cache_dir == expected_path
        assert expected_path.exists()
        
        # Cleanup
        shutil.rmtree(custom_dir)

def test_absolute_cache_directory():
    """Test that absolute cache directory paths are handled correctly."""
    abs_path = os.path.abspath("absolute_test_cache")
    with patch('requests.Session'):
        downloader = BinanceDataDownloader(cache_dir=abs_path)
        expected_path = Path(abs_path)
        
        assert downloader.cache.cache_dir == expected_path
        assert expected_path.exists()
        
        # Cleanup
        shutil.rmtree(abs_path)

def test_cache_directory_creation():
    """Test that nested cache directories are created properly."""
    nested_dir = "test_cache/nested/path"
    with patch('requests.Session'):
        downloader = BinanceDataDownloader(cache_dir=nested_dir)
        expected_path = Path(nested_dir)
        
        assert downloader.cache.cache_dir == expected_path
        assert expected_path.exists()
        
        # Cleanup
        shutil.rmtree("test_cache")

def test_cache_persistence():
    """Test that cache files persist in the specified directory."""
    cache_dir = "test_persistence_cache"
    with patch('requests.Session') as mock_session:
        # Configure mock to return data
        session_instance = mock_session.return_value
        session_instance.get.return_value.json.return_value = [
            [1577836800000, "100", "101", "99", "100.5", "1000", 0, 0, 0, 0, 0, 0]
        ]
        session_instance.get.return_value.raise_for_status.return_value = None
        
        # Create downloader and download some data
        downloader = BinanceDataDownloader(cache_dir=cache_dir)
        start_time = datetime(2020, 1, 1)
        end_time = datetime(2020, 1, 2)
        
        data1 = downloader.download("BTCUSDT", "1m", start_time, end_time)
        
        # Verify cache file exists
        cache_files = list(Path(cache_dir).glob("*.csv"))
        assert len(cache_files) > 0
        
        # Create new downloader instance and verify it uses cached data
        new_downloader = BinanceDataDownloader(cache_dir=cache_dir)
        data2 = new_downloader.download("BTCUSDT", "1m", start_time, end_time)
        
        assert len(data1) == len(data2)
        assert all(row1 == row2 for row1, row2 in zip(data1, data2))
        
        # Cleanup
        shutil.rmtree(cache_dir)

def test_filter_unique_data(downloader):
    """Test filtering of unique data points within a time range."""
    # Create test data with duplicates and out-of-range points
    start_time = datetime(2020, 1, 1)
    end_time = datetime(2020, 1, 1, 23, 59, 59, 999999)
    
    test_data = [
        # Out of range timestamps
        [1577750400000, 98.0, 97.0, 97.5, 98.0, 900.0],    # 2019-12-31 00:00:00 (before start)
        
        # Duplicate timestamps
        [1577836800000, 100.0, 99.0, 99.5, 100.0, 1000.0],  # 2020-01-01 00:00:00
        [1577836800000, 101.0, 98.0, 99.0, 100.5, 1100.0],  # 2020-01-01 00:00:00 (duplicate)
        
        # Valid timestamps
        [1577840400000, 102.0, 101.0, 101.5, 102.0, 1200.0],  # 2020-01-01 01:00:00
        [1577844000000, 103.0, 102.0, 102.5, 103.0, 1300.0],  # 2020-01-01 02:00:00
        
        # Out of range timestamps
        [1577923200000, 105.0, 104.0, 104.5, 105.0, 1500.0]  # 2020-01-02 00:00:00 (after end)
    ]
    
    filtered_data = downloader._filter_unique_data(test_data, start_time, end_time)
    
    # Verify results
    assert len(filtered_data) == 3  # Should only include unique points within range
    
    # Check that timestamps are sorted and within range
    timestamps = [row[0] for row in filtered_data]
    assert timestamps == sorted(timestamps)  # Should be sorted
    assert all(start_time.timestamp() * 1000 <= ts <= end_time.timestamp() * 1000 
              for ts in timestamps)  # Should be within range
    
    # Verify no duplicates
    assert len(set(timestamps)) == len(timestamps)
    
    # Verify first point is the first occurrence of duplicate timestamp
    assert filtered_data[0][1:] == [100.0, 99.0, 99.5, 100.0, 1000.0]  # Should keep first occurrence values

def test_ensure_interval_spacing(downloader):
    """Test ensuring proper interval spacing between data points."""
    # Create test data with missing intervals
    test_data = [
        [1577836800000, 100.0, 99.0, 99.5, 100.0, 1000.0],    # 2020-01-01 00:00:00
        [1577844000000, 102.0, 101.0, 101.5, 102.0, 1200.0],  # 2020-01-01 02:00:00
        [1577847600000, 103.0, 102.0, 102.5, 103.0, 1300.0]   # 2020-01-01 03:00:00
    ]
    
    # Test with 1-hour intervals
    spaced_data = downloader._ensure_interval_spacing(test_data, 3600)  # 3600 seconds = 1 hour
    
    # Should have filled in the missing 01:00:00 point
    assert len(spaced_data) == 4  # Should have 4 hourly points
    
    # Verify timestamps are at exact hour intervals
    timestamps = [row[0] for row in spaced_data]
    expected_timestamps = [
        1577836800000,  # 00:00:00
        1577840400000,  # 01:00:00 (filled)
        1577844000000,  # 02:00:00
        1577847600000   # 03:00:00
    ]
    assert timestamps == expected_timestamps
    
    # Verify forward-filled values
    filled_point = spaced_data[1]  # The 01:00:00 point
    assert filled_point[1:] == test_data[0][1:]  # Should have same values as previous point
    
    # Test with empty data
    assert downloader._ensure_interval_spacing([], 3600) == []
    
    # Test with single point
    single_point = [test_data[0]]
    single_result = downloader._ensure_interval_spacing(single_point, 3600)
    assert len(single_result) == 1
    assert single_result[0] == single_point[0]

def test_filter_and_space_integration(downloader):
    """Test integration of filtering and spacing functionality."""
    start_time = datetime(2020, 1, 1)
    end_time = datetime(2020, 1, 1, 3, 0, 0)

    # Create test data with duplicates, gaps, and out-of-range points
    test_data = [
        # Out of range points
        [1577811540000, 98.0, 97.0, 97.5, 98.0, 900.0],       # 23:59:00 previous day

        # Duplicate and valid points
        [1577811600000, 100.0, 99.0, 99.5, 100.0, 1000.0],    # 00:00:00
        [1577811600000, 101.0, 98.0, 99.0, 100.5, 1100.0],    # 00:00:00 (duplicate)
        [1577819700000, 102.0, 101.0, 101.5, 102.0, 1200.0],  # 02:15:00

        # Out of range points
        [1577822460000, 105.0, 104.0, 104.5, 105.0, 1500.0]   # 03:01:00
    ]

    # First filter unique points
    filtered_data = downloader._filter_unique_data(test_data, start_time, end_time)

    # Debug information
    print("\nDebug information:")
    print(f"Start timestamp: {int(start_time.timestamp() * 1000)}")
    print(f"End timestamp: {int(end_time.timestamp() * 1000)}")
    print("Test data timestamps:")
    for row in test_data:
        print(f"{int(row[0])} ({datetime.fromtimestamp(row[0]/1000)})")
    print("\nFiltered data timestamps:")
    for row in filtered_data:
        print(f"{int(row[0])} ({datetime.fromtimestamp(row[0]/1000)})")

    # Verify filtered data
    assert len(filtered_data) == 2  # Should have 00:00 and 02:15
    assert filtered_data[0][0] == 1577811600000  # 00:00:00
    assert filtered_data[1][0] == 1577819700000  # 02:15:00
    
    # Then ensure proper spacing
    final_data = downloader._ensure_interval_spacing(filtered_data, 3600)
    
    # Verify results
    assert len(final_data) == 4  # Should have points for 00:00, 01:00, 02:15, 03:01
    
    # Check timestamps
    timestamps = [row[0] for row in final_data]
    expected_timestamps = [
        1577811600000,  # 00:00:00
        1577815200000,  # 00:00:00 (filled)
        1577819700000,  # 02:15:00
        1577822400000   # 03:00:00 (filled)
    ]
    assert timestamps == expected_timestamps
    
    # Verify values are properly forward-filled
    assert final_data[1][1:] == final_data[0][1:]  # 00:00 should have same values as 00:00
    assert final_data[3][1:] == final_data[2][1:]  # 03:00 should have same values as 02:15 