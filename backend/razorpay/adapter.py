import os
import hmac
import time
import uuid
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

logger = logging.getLogger("RazorpayAdapter")

# Razorpay rejects descriptions over 2048 chars; the text is LLM-written, so keep it short.
_MAX_DESCRIPTION = 250


def _clip(text: str, limit: int = _MAX_DESCRIPTION) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def verify_webhook_signature(body: bytes, signature: str, secret: str) -> bool:
    """Razorpay signs the raw webhook body with HMAC-SHA256 using the webhook secret."""
    if not (body is not None and signature and secret):
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)

class RazorpayAdapter(ABC):
    """
    Abstract interface for Razorpay execution.
    """
    @abstractmethod
    def create_payment_link(self, amount: int, customer_id: str, reference_id: str, description: str, currency: str = "INR") -> Dict[str, Any]:
        pass

    @abstractmethod
    def fetch_payment_link_status(self, link_id: str) -> str:
        pass

class MockRazorpayAdapter(RazorpayAdapter):
    """
    Used for local development, automated tests, and when credentials are missing.
    """
    mode = "mock"

    def __init__(self):
        self._links = {}
        logger.info("Initialized MockRazorpayAdapter")

    def create_payment_link(self, amount: int, customer_id: str, reference_id: str, description: str, currency: str = "INR") -> Dict[str, Any]:
        link_id = f"plink_mock_{uuid.uuid4().hex[:8]}"
        
        # Simulate network latency
        time.sleep(0.1)
        
        response = {
            "id": link_id,
            "reference_id": reference_id,
            "customer_id": customer_id,
            "amount": amount,
            "currency": currency,
            "description": _clip(description),
            "short_url": f"https://rzp.io/i/{link_id}",
            "status": "created"
        }
        
        self._links[link_id] = response
        return response

    def fetch_payment_link_status(self, link_id: str) -> str:
        # Mock links stay 'created' until a (simulated) webhook marks them otherwise.
        link = self._links.get(link_id)
        if not link:
            return "unknown"
        return link.get("status", "unknown")

class RazorpayTestAdapter(RazorpayAdapter):
    """
    Used for actual execution against Razorpay Test API.
    Requires RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in environment.
    """
    mode = "test"

    def __init__(self):
        try:
            import razorpay
        except ImportError:
            raise RuntimeError("razorpay package not installed. Cannot use RazorpayTestAdapter.")
            
        key_id = os.getenv("RAZORPAY_KEY_ID")
        key_secret = os.getenv("RAZORPAY_KEY_SECRET")
        
        if not key_id or not key_secret:
            raise ValueError("Razorpay test credentials not found in environment")
        # A live key would create real, payable links. Require an explicit opt-in.
        if key_id.startswith("rzp_live_") and os.getenv("RAZORPAY_ENV", "test").lower() != "live":
            raise ValueError("Refusing a rzp_live_ key while RAZORPAY_ENV != 'live'")
        if key_id.startswith("rzp_live_"):
            self.mode = "live"
            
        self.client = razorpay.Client(auth=(key_id, key_secret))
        logger.info("Initialized RazorpayTestAdapter")

    def create_payment_link(self, amount: int, customer_id: str, reference_id: str, description: str, currency: str = "INR") -> Dict[str, Any]:
        try:
            # amount is already in subunits (paise for INR)
            data = {
                "amount": amount,
                "currency": currency,
                "accept_partial": False,
                "reference_id": reference_id,
                "description": _clip(description),
                "customer": {
                    "contact": "+919876543210", # Valid placeholder for India
                    "email": "customer@example.com",
                    "name": customer_id
                },
                "notify": {
                    "sms": False,
                    "email": False
                },
                "reminder_enable": False,
            }
            
            response = self.client.payment_link.create(data)
            return {
                "id": response.get("id"),
                "reference_id": response.get("reference_id"),
                "customer_id": customer_id,
                "amount": amount,
                "currency": response.get("currency"),
                "description": response.get("description"),
                "short_url": response.get("short_url"),
                "status": response.get("status")
            }
            
        except Exception as e:
            logger.error(f"Razorpay API Error: {str(e)}")
            raise RuntimeError(f"Razorpay Test API failed: {str(e)}")

    def fetch_payment_link_status(self, link_id: str) -> str:
        try:
            response = self.client.payment_link.fetch(link_id)
            return response.get("status", "unknown")
        except Exception as e:
            logger.error(f"Razorpay Fetch Error: {str(e)}")
            return "unknown"

def get_razorpay_adapter() -> RazorpayAdapter:
    """
    Factory function. Returns Test adapter if credentials exist, otherwise Mock.
    """
    key_id = os.getenv("RAZORPAY_KEY_ID")
    if key_id and not key_id.startswith("rzp_test_example"):
        try:
            return RazorpayTestAdapter()
        except Exception as e:
            logger.warning(f"Failed to initialize RazorpayTestAdapter, falling back to Mock. Error: {e}")
            return MockRazorpayAdapter()
    
    return MockRazorpayAdapter()
