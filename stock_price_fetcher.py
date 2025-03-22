# stock_price_fetcher.py
import yfinance as yf
import pandas as pd

class StockPriceFetcher:
    def __init__(self):
        self.current_date = '2025-03-22'

    def get_stock_price(self, symbol, target_date=None):
        stock = yf.Ticker(symbol)
        end_date = target_date if target_date else self.current_date
        start_date = pd.to_datetime(end_date if target_date else '2000-01-01') - pd.Timedelta(days=5)
        data = stock.history(start=start_date, end=end_date, interval='1d')
        if data.empty:
            raise ValueError(f"No data found for {symbol} up to {end_date}")
        if target_date:
            target_date = pd.to_datetime(target_date).date()
            if target_date in data.index.date:
                return data.loc[str(target_date), 'Close']
            else:
                available_dates = data.index.date
                closest_date = max(d for d in available_dates if d <= target_date)
                return data.loc[str(closest_date), 'Close']
        else:
            return data['Close'].iloc[-1]