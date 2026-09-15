import json
import logging
from typing import Dict, Any

from backend.data.database_manager import DatabaseManager
from backend.intelligence.engine import IntelligenceEngine

logger = logging.getLogger("AgentTools")


class AgentTools:
    """
    Tools exposed to the LLM agent via google-genai function calling.
    """
    def __init__(self, db: DatabaseManager, intelligence_engine: IntelligenceEngine):
        self.db = db
        self.ie = intelligence_engine

    def get_customer_intelligence(self, customer_id: str) -> str:
        """
        Retrieves canonical intelligence (propensity, segment, signals) for a given customer.
        Returns JSON string.
        """
        logger.info(f"Tool called: get_customer_intelligence({customer_id})")
        
        # 1. Get customer info
        cust_df = self.db.get_customer_data(customer_id)
        if cust_df.empty:
            return json.dumps({"error": f"Customer {customer_id} not found."})
            
        segment = None
        if "segment_name" in cust_df.columns:
            segment = cust_df.iloc[0].get("segment_name")

        # 2. Get all transactions to compute features
        query = "SELECT * FROM transactions WHERE customer_id = ?"
        tx_df = self.db.query_to_dataframe(query, (customer_id,))

        if tx_df.empty:
            return json.dumps({"error": f"No transactions found for customer {customer_id}."})

        # 3. Process via Intelligence Engine
        try:
            intel = self.ie.get_intelligence(customer_id, tx_df, segment=segment)
            return intel.model_dump_json()
        except Exception as e:
            logger.error(f"Intelligence processing failed: {e}")
            return json.dumps({"error": f"Failed to generate intelligence: {str(e)}"})

    def get_customer_history(self, customer_id: str, limit: int = 10) -> str:
        """
        Retrieves recent transaction history for a customer.
        Returns JSON string.
        """
        logger.info(f"Tool called: get_customer_history({customer_id}, limit={limit})")
        
        query = """
            SELECT transaction_timestamp, transaction_type, gross_amount, net_amount, category, merchant 
            FROM transactions 
            WHERE customer_id = ? 
            ORDER BY transaction_timestamp DESC 
            LIMIT ?
        """
        df = self.db.query_to_dataframe(query, (customer_id, limit))
        if df.empty:
            return json.dumps({"message": "No transaction history found."})
            
        # Convert timestamp to string for JSON serialization
        if 'transaction_timestamp' in df.columns:
            df['transaction_timestamp'] = df['transaction_timestamp'].astype(str)
            
        return df.to_json(orient='records')
        
    def get_campaign_eligibility(self) -> str:
        """
        Returns the list of available campaigns and their eligibility criteria.
        Returns JSON string.
        """
        logger.info("Tool called: get_campaign_eligibility()")
        try:
            with open("config/campaigns.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            return json.dumps(config.get("campaigns", {}))
        except Exception as e:
            logger.error(f"Failed to load campaigns config: {e}")
            return json.dumps({"error": "Campaigns configuration unavailable."})
