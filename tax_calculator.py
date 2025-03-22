# tax_calculator.py
import pandas as pd
from stock_price_fetcher import StockPriceFetcher

class TaxCalculator:
    def __init__(self):
        """Initialize the TaxCalculator with a StockPriceFetcher instance and tax rates."""
        self.price_fetcher = StockPriceFetcher()
        self.current_date = pd.to_datetime('2025-03-22')
        self.stcg_rate = 0.20
        self.ltcg_rate = 0.125
        self.ltcg_exemption = 125000
        self.dividend_tax_rate = 0.30

    def generate_tax_report(self, stock_name, purchase_date, quantity, sale_date=None, dividend_income=0):
        """
        Generate a tax report based on Indian tax rules, including Money on Hand.
        """
        # Convert dates to datetime
        purchase_date = pd.to_datetime(purchase_date)
        sale_date = pd.to_datetime(sale_date) if sale_date else None

        # Fetch purchase price
        try:
            purchase_price = self.price_fetcher.get_stock_price(stock_name, purchase_date)
        except ValueError as e:
            return {"Error": str(e)}

        # Fetch sale price (if sold) or current price (if unsold)
        if sale_date:
            try:
                sale_price = self.price_fetcher.get_stock_price(stock_name, sale_date)
            except ValueError as e:
                return {"Error": str(e)}
        else:
            sale_price = self.price_fetcher.get_stock_price(stock_name)
            sale_date = self.current_date

        # Calculate holding period (in months)
        holding_period = (sale_date - purchase_date).days / 30.44

        # Capital Gains Calculation
        capital_gain = (sale_price - purchase_price) * quantity
        gain_type = None
        tax_rate = 0
        taxable_gain = 0
        capital_tax = 0

        if capital_gain != 0:
            if holding_period <= 12:
                gain_type = "Short-Term Capital Gain (STCG)"
                tax_rate = self.stcg_rate
                taxable_gain = capital_gain
                capital_tax = taxable_gain * tax_rate
            else:
                gain_type = "Long-Term Capital Gain (LTCG)"
                tax_rate = self.ltcg_rate
                taxable_gain = max(0, capital_gain - self.ltcg_exemption)
                capital_tax = taxable_gain * tax_rate

        # Dividend Tax Calculation
        dividend_tax = dividend_income * self.dividend_tax_rate
        tds = min(dividend_income * 0.10, dividend_tax) if dividend_income > 5000 else 0

        # Calculate Money on Hand
        if sale_date != self.current_date:  # Sold stock
            money_on_hand = (sale_price * quantity) - capital_tax
        else:  # Unsold stock
            money_on_hand = sale_price * quantity  # Current market value, no tax deducted yet

        # Prepare report
        report = {
            'Stock Name': stock_name,
            'Purchase Date': purchase_date.strftime('%Y-%m-%d'),
            'Quantity': quantity,
            'Purchase Price': round(purchase_price, 2),
            'Sale Date': sale_date.strftime('%Y-%m-%d') if sale_date != self.current_date else 'Not Sold (Current Price Used)',
            'Sale Price': round(sale_price, 2),
            'Holding Period (Months)': round(holding_period, 2),
            'Capital Gain': round(capital_gain, 2),
            'Gain Type': gain_type or 'Not Applicable',
            'Taxable Capital Gain': round(taxable_gain, 2),
            'Capital Gains Tax': round(capital_tax, 2),
            'Dividend Income': dividend_income,
            'Dividend Tax': round(dividend_tax, 2),
            'TDS on Dividend': round(tds, 2),
            'Total Tax Liability': round(capital_tax + dividend_tax - tds, 2),
            'Money on Hand': round(money_on_hand, 2)  # New field
        }

        return report