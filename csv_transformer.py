import pandas as pd
import numpy as np
import io

# ==============================================================================
# 1. TRANSFORMATION FUNCTION
# ==============================================================================

def load_and_filter_csv(file_source, start_date=None, end_date=None):
    """
    Reads the raw crypto CSV format (with header) and converts it to a Price Panel.
    Expected columns: coin, datetime, ..., close

    Parameters:
    - file_source: Path to the CSV file
    - start_date: Optional start date (format: 'YYYY-MM-DD' or datetime object)
    - end_date: Optional end date (format: 'YYYY-MM-DD' or datetime object)
    """

    print(">>> Loading CSV data...")
    # Read CSV using the header from the file (default behavior)
    df = pd.read_csv(file_source)

    # Basic validation to ensure column names match your CSV
    required_cols = ['datetime', 'coin', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV is missing one of required columns: {required_cols}")

    # Parse datetime string to actual datetime object
    # UTC=True is safer for crypto data
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

    print(f">>> Raw data loaded: {len(df)} rows.")

    # -----------------------------------------------------
    # DATE FILTERING (before pivot to reduce data size)
    # -----------------------------------------------------
    if start_date is not None or end_date is not None:
        print(">>> Filtering by date range...")
        if start_date is not None:
            start_date = pd.to_datetime(start_date, utc=True)
            df = df[df['datetime'] >= start_date]
        if end_date is not None:
            end_date = pd.to_datetime(end_date, utc=True)
            df = df[df['datetime'] <= end_date]
        print(f">>> After date filtering: {len(df)} rows.")

    # -----------------------------------------------------
    # PIVOT: Transform from Long to Wide
    # Index = datetime
    # Columns = coin (Symbol)
    # Values = close
    # -----------------------------------------------------
    print(">>> Pivoting to Price Panel...")
    price_panel = df.pivot(index='datetime', columns='coin', values='close')

    # -----------------------------------------------------
    # DATA CLEANING
    # -----------------------------------------------------

    # Replace empty strings with NaN
    price_panel = price_panel.replace('', np.nan)
    price_panel = price_panel.replace(r'^\s*$', np.nan, regex=True)

    return price_panel

# ==============================================================================
# 2. EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # -----------------------------------------------------------
    # USER: REPLACE THIS PART WITH YOUR FILE PATH LATER
    # Example: file_source = "path/to/your/file.csv"
    # -----------------------------------------------------------

    # Using StringIO to simulate a file from the string above
    file_source = "/Users/mateofourquet/Desktop/data/data/Hyperliquid_ALL_COINS_1h.csv"

    # Run transformation
    panel = load_and_filter_csv(file_source)

    # Export to CSV
    output_path = "/Users/mateofourquet/Desktop/LeadLag/data/price_panel_test.csv"
    panel.to_csv(output_path)
    print(f"\n>>> Price Panel exported to: {output_path}")
