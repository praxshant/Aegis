from pydantic import BaseModel, Field
from typing import Optional

class ActionProposal(BaseModel):
    intent: str = Field(description="The general intent, e.g., 'CREATE_WINBACK_OFFER'")
    customer_id: str = Field(description="The customer's ID")
    policy_rule_id: str = Field(description="The ID of the policy rule you are invoking, e.g., 'WINBACK_01'")
    reasoning_summary: str = Field(description="A brief summary explaining the decision based on evidence")
    requested_discount_pct: float = Field(description="The requested discount percentage, e.g., 20.0")
    confidence: str = Field(description="Confidence level: High, Medium, or Low")
    action_type: str = Field(description="The specific action type, e.g., 'CREATE_PAYMENT_LINK'")
