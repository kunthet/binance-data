from cached_binance_data import BinanceDataDownloader
from datetime import datetime, timedelta

def main():
    # Example 1: Default settings (60-minute cache expiry for today's data)
    downloader = BinanceDataDownloader()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    print("\nExample 1: Using cached data")
    data1 = downloader.download(
        symbol="BTCUSDT",
        timeframe="1h",
        start_time=start_time,
        end_time=end_time
    )
    print(f"Downloaded {len(data1)} data points")
    
    print("\nExample 2: Force refresh (bypass cache)")
    data2 = downloader.download(
        symbol="BTCUSDT",
        timeframe="1h",
        start_time=start_time,
        end_time=end_time,
        force_refresh=True
    )
    print(f"Downloaded {len(data2)} data points")
    
    print("\nExample 3: No cache for today's data")
    no_cache_downloader = BinanceDataDownloader(today_cache_expiry_minutes=0)
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    data3 = no_cache_downloader.download(
        symbol="BTCUSDT",
        timeframe="1h",
        start_time=today_start,
        end_time=end_time
    )
    print(f"Downloaded {len(data3)} data points")

if __name__ == "__main__":
    main() 