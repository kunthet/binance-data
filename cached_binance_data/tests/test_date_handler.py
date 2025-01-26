import pytest
from datetime import datetime, timedelta
from cached_binance_data.downloader import DateHandler

def test_normalize_datetime():
    """Test datetime normalization."""
    handler = DateHandler()
    
    # Test string date
    assert handler.normalize_datetime("2020-01-01") == datetime(2020, 1, 1)
    
    # Test datetime object
    dt = datetime(2020, 1, 1, 12, 30)
    assert handler.normalize_datetime(dt) == dt
    
    # Test invalid format
    with pytest.raises(ValueError):
        handler.normalize_datetime("01/01/2020")

def test_align_to_daily_boundaries():
    """Test datetime alignment to daily boundaries."""
    handler = DateHandler()
    
    # Test mid-day alignment
    dt = datetime(2020, 1, 1, 12, 30, 45)
    start, end = handler.align_to_daily_boundaries(dt)
    assert start == datetime(2020, 1, 1, 0, 0, 0)
    assert end == datetime(2020, 1, 1, 23, 59, 59, 999999)
    
    # Test start of day
    dt = datetime(2020, 1, 1, 0, 0, 0)
    start, end = handler.align_to_daily_boundaries(dt)
    assert start == dt
    assert end == datetime(2020, 1, 1, 23, 59, 59, 999999)
    
    # Test end of day
    dt = datetime(2020, 1, 1, 23, 59, 59)
    start, end = handler.align_to_daily_boundaries(dt)
    assert start == datetime(2020, 1, 1, 0, 0, 0)
    assert end == datetime(2020, 1, 1, 23, 59, 59, 999999)

def test_is_today():
    """Test checking if date is from today."""
    handler = DateHandler()
    
    # Set current time to noon today
    current_time = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    
    # Test earlier today
    dt = current_time.replace(hour=0)
    assert handler.is_today(dt, current_time) is True
    
    # Test later today
    dt = current_time.replace(hour=23)
    assert handler.is_today(dt, current_time) is True
    
    # Test yesterday
    dt = current_time - timedelta(days=1)
    assert handler.is_today(dt, current_time) is False
    
    # Test tomorrow
    dt = current_time + timedelta(days=1)
    assert handler.is_today(dt, current_time) is True  # Future dates count as today

def test_split_time_range():
    """Test splitting time range into daily chunks."""
    handler = DateHandler()
    
    # Test exact day boundaries
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2020, 1, 3, 23, 59, 59, 999999)
    chunks = handler.split_time_range(start, end)
    
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
    chunks = handler.split_time_range(start, end)
    
    assert len(chunks) == 2
    assert chunks[0][0] == start
    assert chunks[0][1] == datetime(2020, 1, 1, 23, 59, 59, 999999)
    assert chunks[1][0] == datetime(2020, 1, 2, 0, 0, 0)
    assert chunks[1][1] == end
    
    # Test single day
    start = datetime(2020, 1, 1, 12, 0, 0)
    end = datetime(2020, 1, 1, 18, 0, 0)
    chunks = handler.split_time_range(start, end)
    
    assert len(chunks) == 1
    assert chunks[0][0] == start
    assert chunks[0][1] == end 