import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

def generate_dataset():
    # Load catalog
    catalog = pd.read_parquet(os.path.join(os.path.dirname(__file__), '../artifacts/data/instacart_product_subset.parquet'))
    
    # We want 100 realistic customers
    # Let's generate a list of 100 customer IDs
    customers = [f"CUST_{str(i).zfill(3)}" for i in range(1, 101)]
    
    records = []
    
    start_date = datetime.now() - timedelta(days=365)
    
    for cust in customers:
        # Determine behavior profile
        profile = random.choice(["high_value", "dormant", "one_time", "frequent", "active"])
        
        if profile == "high_value":
            num_orders = random.randint(15, 30)
            avg_basket_size = random.randint(10, 25)
            last_order_days_ago = random.randint(1, 10)
        elif profile == "dormant":
            num_orders = random.randint(2, 10)
            avg_basket_size = random.randint(3, 8)
            last_order_days_ago = random.randint(90, 150) # >90 days
        elif profile == "one_time":
            num_orders = 1
            avg_basket_size = random.randint(1, 5)
            last_order_days_ago = random.randint(10, 200)
        elif profile == "frequent":
            num_orders = random.randint(10, 40)
            avg_basket_size = random.randint(3, 10)
            last_order_days_ago = random.randint(1, 15)
        else: # active
            num_orders = random.randint(3, 10)
            avg_basket_size = random.randint(5, 12)
            last_order_days_ago = random.randint(5, 30)
            
        order_dates = []
        current_date = datetime.now() - timedelta(days=last_order_days_ago)
        for _ in range(num_orders):
            order_dates.append(current_date)
            # go back in time
            current_date -= timedelta(days=random.randint(7, 30))
            
        order_dates.reverse() # chronological
        
        # Customer preferred products
        pref_products = catalog.sample(n=random.randint(3, 10))
        
        invoice_counter = 100000 + random.randint(1, 10000)
        for o_date in order_dates:
            invoice_id = f"INV_{cust}_{invoice_counter}"
            invoice_counter += 1
            
            # Basket size for this order
            b_size = max(1, int(np.random.normal(avg_basket_size, 2)))
            
            # 70% chance to buy preferred products, 30% random
            for _ in range(b_size):
                if random.random() < 0.7:
                    p = pref_products.sample(n=1).iloc[0]
                else:
                    p = catalog.sample(n=1).iloc[0]
                    
                qty = random.randint(1, 3)
                price = round(random.uniform(2.0, 20.0), 2)
                gross_amount = qty * price
                
                records.append({
                    "customer_id": cust,
                    "invoice": invoice_id,
                    "product_id": p["instacart_product_id"], # use real ID
                    "product_name": p["product_name"],
                    "category": p["department"],
                    "transaction_timestamp": o_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "quantity": qty,
                    "unit_price": price,
                    "gross_amount": gross_amount,
                    "transaction_type": "purchase"
                })
                
    df = pd.DataFrame(records)
    # add some duplicates to test idempotency
    dup_indices = df.sample(frac=0.01).index
    df = pd.concat([df, df.loc[dup_indices]])
    
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    out_path = os.path.join(os.path.dirname(__file__), "../artifacts/data/final_demo_dataset.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated dataset with {len(df)} rows at {out_path}")

if __name__ == "__main__":
    generate_dataset()
