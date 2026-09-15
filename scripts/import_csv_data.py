"""
import_csv_data.py
------------------
Utility script to import your own customer and transaction data into Aegis.

Usage:
1. Prepare customers.csv and transactions.csv matching the expected schema.
2. Run `python scripts/import_csv_data.py`
"""
import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.data.database_manager import DatabaseManager

def import_data(customers_csv_path, transactions_csv_path, segments_csv_path=None):
    print("Connecting to Aegis Database...")
    db = DatabaseManager()
    
    # 1. Load and insert Customers
    print(f"Loading customers from {customers_csv_path}...")
    customers_df = pd.read_csv(customers_csv_path)
    # Ensure replace clears out the synthetic data
    db.insert_dataframe(customers_df, "customers", if_exists="replace")
    
    # 2. Load and insert Transactions
    print(f"Loading transactions from {transactions_csv_path}...")
    transactions_df = pd.read_csv(transactions_csv_path)
    if 'transaction_timestamp' in transactions_df.columns:
        transactions_df['transaction_timestamp'] = pd.to_datetime(transactions_df['transaction_timestamp'])
    db.insert_dataframe(transactions_df, "transactions", if_exists="replace")
    
    # 3. Load and insert Segments
    if segments_csv_path and os.path.exists(segments_csv_path):
        print(f"Loading segments from {segments_csv_path}...")
        segments_df = pd.read_csv(segments_csv_path)
        db.insert_dataframe(segments_df, "customer_segments", if_exists="replace")
        
    print("\n✅ Data import complete! Restart your FastAPI server to see changes.")

if __name__ == "__main__":
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    cust_file = os.path.join(data_dir, "customers.csv")
    txn_file = os.path.join(data_dir, "transactions.csv")
    seg_file = os.path.join(data_dir, "segments.csv")
    
    if not os.path.exists(cust_file):
        print(f"Creating template CSVs in '{data_dir}' folder...")
        # Customers
        pd.DataFrame(columns=[
            "customer_id", "first_name", "last_name", "email", "registration_date", "age", "gender", "location"
        ]).to_csv(cust_file, index=False)
        # Transactions
        pd.DataFrame(columns=[
            "transaction_id", "customer_id", "transaction_timestamp", "transaction_type", 
            "invoice", "product_id", "gross_amount", "return_amount", "net_amount", "category", "merchant"
        ]).to_csv(txn_file, index=False)
        # Segments
        pd.DataFrame(columns=[
            "customer_id", "segment_id", "segment_name", "last_updated"
        ]).to_csv(seg_file, index=False)
        
        print("Templates created! Please fill them with your real data and run this script again.")
    else:
        import_data(cust_file, txn_file, seg_file)
