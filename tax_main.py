# main.py
from tax_calculator import TaxCalculator

def print_report(report):
    """Utility function to print the tax report."""
    for key, value in report.items():
        print(f"{key}: {value}")

# Create an instance of TaxCalculator
tax_calc = TaxCalculator()

# Example usage
# Case 1: Stock sold
report1 = tax_calc.generate_tax_report("HAL.NS", "2023-05-15", 100, "2024-06-20", 5000)
print("Tax Report for Sold Stock:")
print_report(report1)

# Case 2: Stock unsold (use current price as of March 22, 2025)
report2 = tax_calc.generate_tax_report("HAL.NS", "2023-05-15", 100, dividend_income=2000)
print("\nTax Report for Unsold Stock (Current Price):")
print_report(report2)

# Case 3: Nifty 50 Index (^NSEI) sold
report3 = tax_calc.generate_tax_report("^NSEI", "2023-01-01", 10, "2024-03-01")
print("\nTax Report for Nifty 50 Index:")
print_report(report3)