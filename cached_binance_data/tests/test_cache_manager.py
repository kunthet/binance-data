import pytest
from datetime import datetime, timedelta
import os
import shutil
from pathlib import Path
from cached_binance_data.cache import CacheManager

@pytest.fixture
def cache_manager():
    """Create a cache manager for testing."""
    manager = CacheManager(cache_dir="test_cache", today_cache_expiry=timedelta(minutes=60))
    yield manager
    # Cleanup
    if os.path.exists("test_cache"):
        shutil.rmtree("test_cache")

def test_should_use_cache(cache_manager):
    """Test cache usage decision logic."""
    current_time = datetime.now().replace(microsecond=0)
    today_start = current_time.replace(hour=0, minute=0, second=0)
    yesterday = today_start - timedelta(days=1)
    
    # Historical data should use cache
    assert cache_manager.should_use_cache(yesterday, current_time, force_refresh=False) is True
    
    # Today's data should use cache by default
    assert cache_manager.should_use_cache(today_start, current_time, force_refresh=False) is True
    
    # Force refresh should bypass cache
    assert cache_manager.should_use_cache(yesterday, current_time, force_refresh=True) is False
    
    # No cache expiry should skip cache for today's data
    no_cache_manager = CacheManager(cache_dir="test_cache", today_cache_expiry=timedelta(minutes=0))
    assert no_cache_manager.should_use_cache(today_start, current_time, force_refresh=False) is False

def test_is_cache_expired(cache_manager, tmp_path):
    """Test cache expiration check."""
    current_time = datetime.now().replace(microsecond=0)
    
    # Create a test file
    test_file = tmp_path / "test.csv"
    test_file.write_text("")
    
    # Not today's data should never expire
    assert cache_manager.is_cache_expired(str(test_file), current_time, is_today=False) is False
    
    # Today's data should expire after cache_expiry
    assert cache_manager.is_cache_expired(str(test_file), current_time + timedelta(hours=2), is_today=True) is True

def test_try_get_cached_data(cache_manager):
    """Test retrieving data from cache."""
    current_time = datetime.now().replace(microsecond=0)
    chunk_start = current_time.replace(hour=0, minute=0, second=0)
    chunk_end = chunk_start + timedelta(hours=1)
    
    # Test with force refresh
    assert cache_manager.try_get_cached_data(
        "BTCUSDT", "1h", chunk_start, chunk_end,
        current_time, force_refresh=True
    ) is None
    
    # Test with non-existent data
    assert cache_manager.try_get_cached_data(
        "BTCUSDT", "1h", chunk_start, chunk_end,
        current_time, force_refresh=False
    ) is None
    
    # Test with actual data
    test_data = [[float(chunk_start.timestamp() * 1000), 100.0, 99.0, 99.5, 100.0, 1000.0]]
    cache_manager.save_to_cache(test_data, "BTCUSDT", "1h", chunk_start, chunk_end, current_time)
    
    cached_data = cache_manager.try_get_cached_data(
        "BTCUSDT", "1h", chunk_start, chunk_end,
        current_time, force_refresh=False
    )
    assert cached_data == test_data

def test_save_to_cache(cache_manager):
    """Test saving data to cache."""
    current_time = datetime.now().replace(microsecond=0)
    chunk_start = current_time.replace(hour=0, minute=0, second=0)
    chunk_end = chunk_start + timedelta(hours=1)
    test_data = [[float(chunk_start.timestamp() * 1000), 100.0, 99.0, 99.5, 100.0, 1000.0]]
    
    # Save today's data
    cache_manager.save_to_cache(test_data, "BTCUSDT", "1h", chunk_start, chunk_end, current_time)
    
    # Verify data was saved
    cached_data = cache_manager.try_get_cached_data(
        "BTCUSDT", "1h", chunk_start, chunk_end,
        current_time, force_refresh=False
    )
    assert cached_data == test_data
    
    # Test no cache for today with expiry=0
    no_cache_manager = CacheManager(cache_dir="test_cache", today_cache_expiry=timedelta(minutes=0))
    no_cache_manager.save_to_cache(test_data, "BTCUSDT", "1h", chunk_start, chunk_end, current_time)
    
    # Verify data was not saved
    cached_data = no_cache_manager.try_get_cached_data(
        "BTCUSDT", "1h", chunk_start, chunk_end,
        current_time, force_refresh=False
    )
    assert cached_data is None 