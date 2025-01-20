from binance_data import BinanceDataDownloader
from datetime import datetime, timedelta

def main():
    # Initialize downloader
    downloader = BinanceDataDownloader()
    
    # Set time range for last 7 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    # Download BTCUSDT 1-minute data
    df = downloader.download(
        symbol="BTCUSDT",
        timeframe="1m",
        start_time=start_time,
        end_time=end_time
    )
    
    # Print basic statistics
    print("\nData Summary:")
    print(f"Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Number of records: {len(df)}")
    print("\nPrice Statistics:")
    print(df[['open', 'high', 'low', 'close']].describe())
    
    # Example of downloading different timeframes
    timeframes = ['1m', '1h', '1d']
    for timeframe in timeframes:
        df = downloader.download(
            symbol="BTCUSDT",
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        print(f"\n{timeframe} Data Summary:")
        print(f"Number of records: {len(df)}")

if __name__ == "__main__":
    main() 