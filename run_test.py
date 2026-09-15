import requests
import time
import json
import sys

API_URL = "http://localhost:8000"
CSV_PATH = r"C:\Users\ACER\Downloads\aegis_100_instacart_test_customers.csv"

def run_test():
    # 1. Health check (wait for server)
    for _ in range(10):
        try:
            res = requests.get(f"{API_URL}/health")
            if res.status_code == 200:
                print("Server is up!")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    else:
        print("Server did not start in time.")
        sys.exit(1)

    # 2. Ingest CSV
    print(f"Uploading {CSV_PATH}...")
    try:
        with open(CSV_PATH, 'rb') as f:
            res = requests.post(f"{API_URL}/aegis/ingest/csv", files={"file": f})
    except Exception as e:
        print(f"Error opening CSV: {e}")
        sys.exit(1)

    if res.status_code != 200:
        print(f"Ingest failed: {res.text}")
        sys.exit(1)

    run_id = res.json().get("run_id")
    print(f"Ingestion started. Run ID: {run_id}")

    # 3. Poll for completion
    while True:
        res = requests.get(f"{API_URL}/aegis/ingest/status/{run_id}")
        status = res.json()
        print(f"Status: {status['status']} - Progress: {status.get('progress_percentage', 0)}% - Rows: {status.get('processed_rows', 0)}")
        if status["status"] == "COMPLETED":
            print(f"Completed! Valid rows: {status.get('valid_rows')}")
            break
        elif status["status"].startswith("FAILED"):
            print(f"Failed: {status}")
            sys.exit(1)
        time.sleep(1.5)

    # 4. Activate Dataset
    print("\nActivating dataset (Mode: Replace)...")
    res = requests.post(f"{API_URL}/datasets/activate", json={
        "run_id": run_id,
        "mode": "replace"
    })
    
    if res.status_code != 200:
        print(f"Activation failed: {res.text}")
        sys.exit(1)
    
    print(f"Activation result: {json.dumps(res.json(), indent=2)}")

    # 5. Fetch customers and run intelligence on the first one
    res = requests.get(f"{API_URL}/customers?limit=200")
    customers = res.json()
    print(f"\nFetched {len(customers)} active customers.")
    
    if len(customers) > 0:
        cust_id = customers[0]["customer_id"]
        print(f"\nTriggering Intelligence for {cust_id}...")
        res = requests.get(f"{API_URL}/aegis/intelligence/{cust_id}")
        if res.status_code == 200:
            intel = res.json()
            print(f"Propensity Score: {intel.get('purchase_probability_30d')}")
            print(f"Band: {intel.get('propensity_band')}")
            print(f"Segment: {intel.get('segment_name', 'Unknown')}")
            print(f"Evidence Status: {intel.get('evidence_status')}")
            
            # Check for next basket features
            print("\nNext Basket Predictions (Top 3):")
            nb = intel.get('next_basket_predictions', [])
            if nb:
                for p in nb[:3]:
                    print(f"  - {p['product_id']}: {p['probability']:.2%} (Dept: {p['department_id']}, Aisle: {p['aisle_id']})")
            else:
                print("  Next basket UNAVAILABLE or empty.")
        else:
            print(f"Intelligence failed: {res.text}")

if __name__ == "__main__":
    run_test()
