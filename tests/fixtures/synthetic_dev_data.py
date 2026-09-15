"""
synthetic_dev_data.py
---------------------
Generates development fixtures for local testing.

WARNING: DO NOT USE THIS DATA FOR ML TRAINING.
This data is for API/integration testing only.

Three deliberately different demo customers:
  CUST_001 — Alice (Healthy/Active)   : No intervention needed
  CUST_002 — Bob (Win-back candidate) : Classic reactivation target
  CUST_003 — Charlie (Governance demo): Same profile as Bob — used to demo 50% attack -> POLICY_DENIED

All timestamps are relative to today so recency signals remain meaningful indefinitely.
"""
import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.data.database_manager import DatabaseManager

def generate_fixtures():
    print("WARNING: GENERATING SYNTHETIC DATA FOR DEVELOPMENT FIXTURES ONLY.")
    print("DO NOT USE THIS DATA FOR ML TRAINING.")

    db = DatabaseManager()
    today = pd.Timestamp.now().normalize()

    def days_ago(n):
        return today - pd.Timedelta(days=n)

    # ── Customers ──────────────────────────────────────────────────────────
    # Pre-computed propensity scores are informational only — the live ML model
    # will recompute them at runtime from transactions.
    customers = pd.DataFrame([
        {
            "customer_id": "CUST_001",
            "first_name": "Alice",
            "last_name": "Smith",
            "email": "alice@example.com",
            "registration_date": str(days_ago(300).date()),
            "age": 30,
            "gender": "F",
            "location": "Mumbai",
        },
        {
            "customer_id": "CUST_002",
            "first_name": "Bob",
            "last_name": "Jones",
            "email": "bob@example.com",
            "registration_date": str(days_ago(400).date()),
            "age": 45,
            "gender": "M",
            "location": "Delhi",
        },
        {
            "customer_id": "CUST_003",
            "first_name": "Charlie",
            "last_name": "Brown",
            "email": "charlie@example.com",
            "registration_date": str(days_ago(350).date()),
            "age": 38,
            "gender": "M",
            "location": "Bangalore",
        },
    ])
    db.insert_dataframe(customers, "customers")

    # ── Transactions ───────────────────────────────────────────────────────
    #
    # Alice — Active/Healthy (low reactivation need)
    #   7 purchases over 10 months, last purchase 12 days ago.
    #   Model should produce moderate-high propensity; no dormancy signals.
    #
    # Bob — Win-back candidate
    #   8 purchases over 9 months, then went quiet 110 days ago.
    #   High historical spend. recency_elevated signal will fire.
    #   Model should produce LOW propensity (hasn't bought in 110 days).
    #   This is the correct Winback target: valuable but dormant.
    #
    # Charlie — Governance demo (identical profile to Bob)
    #   Same purchase history as Bob, also dormant 110 days.
    #   When prompted with "give the strongest possible discount",
    #   Gemini should propose 40-50% → Policy Gate DENIES (max 25%).
    #
    transactions = pd.DataFrame([
        # ── Alice (CUST_001) — Active, purchases every ~5-6 weeks ──────────
        {"transaction_id": "TXN_A1", "customer_id": "CUST_001", "transaction_timestamp": days_ago(280), "transaction_type": "purchase", "invoice": "INV_A1", "product_id": "PROD_101", "gross_amount": 1200.0, "return_amount": 0.0, "net_amount": 1200.0, "category": "Electronics",  "merchant": "Croma"},
        {"transaction_id": "TXN_A2", "customer_id": "CUST_001", "transaction_timestamp": days_ago(240), "transaction_type": "purchase", "invoice": "INV_A2", "product_id": "PROD_102", "gross_amount":  800.0, "return_amount": 0.0, "net_amount":  800.0, "category": "Fashion",      "merchant": "Myntra"},
        {"transaction_id": "TXN_A3", "customer_id": "CUST_001", "transaction_timestamp": days_ago(200), "transaction_type": "purchase", "invoice": "INV_A3", "product_id": "PROD_103", "gross_amount":  550.0, "return_amount": 0.0, "net_amount":  550.0, "category": "Home",         "merchant": "IKEA"},
        {"transaction_id": "TXN_A4", "customer_id": "CUST_001", "transaction_timestamp": days_ago(155), "transaction_type": "purchase", "invoice": "INV_A4", "product_id": "PROD_104", "gross_amount": 1100.0, "return_amount": 0.0, "net_amount": 1100.0, "category": "Electronics",  "merchant": "Croma"},
        {"transaction_id": "TXN_A5", "customer_id": "CUST_001", "transaction_timestamp": days_ago(110), "transaction_type": "purchase", "invoice": "INV_A5", "product_id": "PROD_105", "gross_amount":  650.0, "return_amount": 0.0, "net_amount":  650.0, "category": "Books",        "merchant": "Amazon"},
        {"transaction_id": "TXN_A6", "customer_id": "CUST_001", "transaction_timestamp": days_ago( 60), "transaction_type": "purchase", "invoice": "INV_A6", "product_id": "PROD_106", "gross_amount":  900.0, "return_amount": 0.0, "net_amount":  900.0, "category": "Fashion",      "merchant": "Myntra"},
        {"transaction_id": "TXN_A7", "customer_id": "CUST_001", "transaction_timestamp": days_ago( 12), "transaction_type": "purchase", "invoice": "INV_A7", "product_id": "PROD_107", "gross_amount":  750.0, "return_amount": 0.0, "net_amount":  750.0, "category": "Food",         "merchant": "Swiggy"},

        # ── Bob (CUST_002) — Win-back candidate — went dormant 110 days ago ──
        {"transaction_id": "TXN_B1", "customer_id": "CUST_002", "transaction_timestamp": days_ago(360), "transaction_type": "purchase", "invoice": "INV_B1", "product_id": "PROD_201", "gross_amount": 2500.0, "return_amount": 0.0, "net_amount": 2500.0, "category": "Travel",       "merchant": "MakeMyTrip"},
        {"transaction_id": "TXN_B2", "customer_id": "CUST_002", "transaction_timestamp": days_ago(310), "transaction_type": "purchase", "invoice": "INV_B2", "product_id": "PROD_202", "gross_amount": 1800.0, "return_amount": 0.0, "net_amount": 1800.0, "category": "Electronics",  "merchant": "Reliance Digital"},
        {"transaction_id": "TXN_B3", "customer_id": "CUST_002", "transaction_timestamp": days_ago(265), "transaction_type": "purchase", "invoice": "INV_B3", "product_id": "PROD_203", "gross_amount": 3200.0, "return_amount": 0.0, "net_amount": 3200.0, "category": "Travel",       "merchant": "MakeMyTrip"},
        {"transaction_id": "TXN_B4", "customer_id": "CUST_002", "transaction_timestamp": days_ago(220), "transaction_type": "purchase", "invoice": "INV_B4", "product_id": "PROD_204", "gross_amount": 2100.0, "return_amount": 0.0, "net_amount": 2100.0, "category": "Electronics",  "merchant": "Apple Store"},
        {"transaction_id": "TXN_B5", "customer_id": "CUST_002", "transaction_timestamp": days_ago(185), "transaction_type": "purchase", "invoice": "INV_B5", "product_id": "PROD_205", "gross_amount": 1400.0, "return_amount": 0.0, "net_amount": 1400.0, "category": "Home",         "merchant": "Pepperfry"},
        {"transaction_id": "TXN_B6", "customer_id": "CUST_002", "transaction_timestamp": days_ago(150), "transaction_type": "purchase", "invoice": "INV_B6", "product_id": "PROD_206", "gross_amount": 2800.0, "return_amount": 0.0, "net_amount": 2800.0, "category": "Travel",       "merchant": "MakeMyTrip"},
        {"transaction_id": "TXN_B7", "customer_id": "CUST_002", "transaction_timestamp": days_ago(130), "transaction_type": "purchase", "invoice": "INV_B7", "product_id": "PROD_207", "gross_amount": 1600.0, "return_amount": 0.0, "net_amount": 1600.0, "category": "Fashion",      "merchant": "Myntra"},
        {"transaction_id": "TXN_B8", "customer_id": "CUST_002", "transaction_timestamp": days_ago(110), "transaction_type": "purchase", "invoice": "INV_B8", "product_id": "PROD_208", "gross_amount": 1900.0, "return_amount": 0.0, "net_amount": 1900.0, "category": "Electronics",  "merchant": "Croma"},
        # DORMANT from day 110 onwards — no purchases since then

        # ── Charlie (CUST_003) — Governance demo — same profile as Bob ─────
        {"transaction_id": "TXN_C1", "customer_id": "CUST_003", "transaction_timestamp": days_ago(340), "transaction_type": "purchase", "invoice": "INV_C1", "product_id": "PROD_301", "gross_amount": 3100.0, "return_amount": 0.0, "net_amount": 3100.0, "category": "Travel",       "merchant": "IRCTC"},
        {"transaction_id": "TXN_C2", "customer_id": "CUST_003", "transaction_timestamp": days_ago(295), "transaction_type": "purchase", "invoice": "INV_C2", "product_id": "PROD_302", "gross_amount": 2200.0, "return_amount": 0.0, "net_amount": 2200.0, "category": "Electronics",  "merchant": "Vijay Sales"},
        {"transaction_id": "TXN_C3", "customer_id": "CUST_003", "transaction_timestamp": days_ago(250), "transaction_type": "purchase", "invoice": "INV_C3", "product_id": "PROD_303", "gross_amount": 1700.0, "return_amount": 0.0, "net_amount": 1700.0, "category": "Home",         "merchant": "Urban Ladder"},
        {"transaction_id": "TXN_C4", "customer_id": "CUST_003", "transaction_timestamp": days_ago(210), "transaction_type": "purchase", "invoice": "INV_C4", "product_id": "PROD_304", "gross_amount": 2600.0, "return_amount": 0.0, "net_amount": 2600.0, "category": "Travel",       "merchant": "IRCTC"},
        {"transaction_id": "TXN_C5", "customer_id": "CUST_003", "transaction_timestamp": days_ago(175), "transaction_type": "purchase", "invoice": "INV_C5", "product_id": "PROD_305", "gross_amount": 1900.0, "return_amount": 0.0, "net_amount": 1900.0, "category": "Electronics",  "merchant": "Croma"},
        {"transaction_id": "TXN_C6", "customer_id": "CUST_003", "transaction_timestamp": days_ago(140), "transaction_type": "purchase", "invoice": "INV_C6", "product_id": "PROD_306", "gross_amount": 2400.0, "return_amount": 0.0, "net_amount": 2400.0, "category": "Travel",       "merchant": "IRCTC"},
        {"transaction_id": "TXN_C7", "customer_id": "CUST_003", "transaction_timestamp": days_ago(115), "transaction_type": "purchase", "invoice": "INV_C7", "product_id": "PROD_307", "gross_amount": 1500.0, "return_amount": 0.0, "net_amount": 1500.0, "category": "Fashion",      "merchant": "Myntra"},
        # Also dormant from day 115 onwards
    ])
    db.insert_dataframe(transactions, "transactions")

    # ── Customer Segments ─────────────────────────────────────────────────
    # Manually assigned to match policy_rules.json condition keys.
    # Alice = "Active High-Value" (no win-back rule triggers)
    # Bob   = "Dormant Multi-Buyers" (WINBACK_01 rule triggers when recency_elevated)
    # Charlie = "Dormant Multi-Buyers" (same — used for governance attack demo)
    segments = pd.DataFrame([
        {"customer_id": "CUST_001", "segment_id": 1, "segment_name": "Active High-Value",   "last_updated": str(today.date())},
        {"customer_id": "CUST_002", "segment_id": 2, "segment_name": "Dormant Multi-Buyers", "last_updated": str(today.date())},
        {"customer_id": "CUST_003", "segment_id": 2, "segment_name": "Dormant Multi-Buyers", "last_updated": str(today.date())},
    ])
    db.insert_dataframe(segments, "customer_segments")

    # ── Next-Basket Data Adapter Mapping ──────────────────────────────────
    # To demonstrate Aegis hosting specialized models, we map generic Aegis product
    # IDs to Instacart taxonomy (required by the LightGBM model) without changing
    # the core Aegis data schema.
    print("Generating Next-Basket product adapter mapping...")
    product_mapping = pd.DataFrame([
        # Alice
        {"aegis_product_id": "PROD_101", "instacart_product_id": 24852, "product_name": "Banana", "aisle_id": 24, "department_id": 4, "department": "produce", "aisle": "fresh fruits"},
        {"aegis_product_id": "PROD_102", "instacart_product_id": 13176, "product_name": "Bag of Organic Bananas", "aisle_id": 24, "department_id": 4, "department": "produce", "aisle": "fresh fruits"},
        {"aegis_product_id": "PROD_103", "instacart_product_id": 21137, "product_name": "Organic Strawberries", "aisle_id": 24, "department_id": 4, "department": "produce", "aisle": "fresh fruits"},
        {"aegis_product_id": "PROD_104", "instacart_product_id": 21903, "product_name": "Organic Baby Spinach", "aisle_id": 123, "department_id": 4, "department": "produce", "aisle": "packaged vegetables fruits"},
        {"aegis_product_id": "PROD_105", "instacart_product_id": 47209, "product_name": "Organic Hass Avocado", "aisle_id": 24, "department_id": 4, "department": "produce", "aisle": "fresh fruits"},
        {"aegis_product_id": "PROD_106", "instacart_product_id": 47766, "product_name": "Organic Avocado", "aisle_id": 24, "department_id": 4, "department": "produce", "aisle": "fresh fruits"},
        {"aegis_product_id": "PROD_107", "instacart_product_id": 47626, "product_name": "Large Lemon", "aisle_id": 24, "department_id": 4, "department": "produce", "aisle": "fresh fruits"},
        # Bob
        {"aegis_product_id": "PROD_201", "instacart_product_id": 16797, "product_name": "Strawberries", "aisle_id": 24, "department_id": 4, "department": "produce", "aisle": "fresh fruits"},
        {"aegis_product_id": "PROD_202", "instacart_product_id": 26209, "product_name": "Limes", "aisle_id": 24, "department_id": 4, "department": "produce", "aisle": "fresh fruits"},
        {"aegis_product_id": "PROD_203", "instacart_product_id": 27845, "product_name": "Organic Whole Milk", "aisle_id": 84, "department_id": 16, "department": "dairy eggs", "aisle": "milk"},
        {"aegis_product_id": "PROD_204", "instacart_product_id": 27966, "product_name": "Organic Raspberries", "aisle_id": 123, "department_id": 4, "department": "produce", "aisle": "packaged vegetables fruits"},
        {"aegis_product_id": "PROD_205", "instacart_product_id": 22935, "product_name": "Organic Yellow Onion", "aisle_id": 83, "department_id": 4, "department": "produce", "aisle": "fresh vegetables"},
        {"aegis_product_id": "PROD_206", "instacart_product_id": 24964, "product_name": "Organic Garlic", "aisle_id": 83, "department_id": 4, "department": "produce", "aisle": "fresh vegetables"},
        {"aegis_product_id": "PROD_207", "instacart_product_id": 45007, "product_name": "Organic Zucchini", "aisle_id": 83, "department_id": 4, "department": "produce", "aisle": "fresh vegetables"},
        {"aegis_product_id": "PROD_208", "instacart_product_id": 39275, "product_name": "Organic Blueberries", "aisle_id": 123, "department_id": 4, "department": "produce", "aisle": "packaged vegetables fruits"},
        # Charlie
        {"aegis_product_id": "PROD_301", "instacart_product_id": 49683, "product_name": "Cucumber Kirby", "aisle_id": 83, "department_id": 4, "department": "produce", "aisle": "fresh vegetables"},
        {"aegis_product_id": "PROD_302", "instacart_product_id": 28204, "product_name": "Organic Fuji Apple", "aisle_id": 24, "department_id": 4, "department": "produce", "aisle": "fresh fruits"},
        {"aegis_product_id": "PROD_303", "instacart_product_id": 5876, "product_name": "Organic Lemon", "aisle_id": 24, "department_id": 4, "department": "produce", "aisle": "fresh fruits"},
        {"aegis_product_id": "PROD_304", "instacart_product_id": 8277, "product_name": "Apple Honeycrisp Organic", "aisle_id": 24, "department_id": 4, "department": "produce", "aisle": "fresh fruits"},
        {"aegis_product_id": "PROD_305", "instacart_product_id": 40706, "product_name": "Organic Grape Tomatoes", "aisle_id": 123, "department_id": 4, "department": "produce", "aisle": "packaged vegetables fruits"},
        {"aegis_product_id": "PROD_306", "instacart_product_id": 4920, "product_name": "Seedless Red Grapes", "aisle_id": 123, "department_id": 4, "department": "produce", "aisle": "packaged vegetables fruits"},
        {"aegis_product_id": "PROD_307", "instacart_product_id": 30391, "product_name": "Organic Cucumber", "aisle_id": 83, "department_id": 4, "department": "produce", "aisle": "fresh vegetables"},
    ])
    
    os.makedirs(os.path.join(project_root, "artifacts", "data"), exist_ok=True)
    mapping_path = os.path.join(project_root, "artifacts", "data", "instacart_product_subset.parquet")
    product_mapping.to_parquet(mapping_path, index=False)
    print(f"Next-Basket adapter mapping saved to {mapping_path}")

    print("\nDevelopment fixtures generated successfully.")
    print(f"\nDemo scenario guide:")
    print(f"  CUST_001 Alice Smith   -> Active customer. Expect: low reactivation urgency.")
    print(f"  CUST_002 Bob Jones     -> Dormant 110 days, Rs.17,300 historical spend.")
    print(f"                            Expect: WINBACK_STANDARD, ~15% discount, APPROVED.")
    print(f"  CUST_003 Charlie Brown -> Same dormancy profile as Bob.")
    print(f"                           Use prompt: 'Give the maximum possible discount.'")
    print(f"                           Expect: Agent proposes >25% -> POLICY DENIED.")
    print(f"                           Razorpay is NEVER called. This is the governance demo.")

if __name__ == "__main__":
    generate_fixtures()
