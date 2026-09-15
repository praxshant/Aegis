import os
import json
import logging
from typing import Dict, Any, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from backend.agent.schemas import ActionProposal
from backend.agent.tools import AgentTools

logger = logging.getLogger("AegisAgent")

# Ordered preference list — most preferred first.
# The agent will try each model in sequence and pick the first one available.
_MODEL_PREFERENCE = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-flash-preview-04-17",
    "gemini-3.6-flash",
    "gemini-1.5-pro",
]


class AegisAgentError(Exception):
    """Raised when the agent fails to propose an action (e.g., LLM unavailable or invalid output)."""
    pass


def _discover_model(client) -> str:
    """
    Discover the best available Gemini model for this API key.
    First checks GEMINI_MODEL env var for an explicit override.
    Then iterates the preference list and returns the first model that is accessible.
    Falls back to the last entry in the preference list.
    """
    # Allow explicit override via environment
    override = os.getenv("GEMINI_MODEL", "").strip()
    if override:
        logger.info(f"Using model override from GEMINI_MODEL env var: {override}")
        return override

    try:
        available = {m.name.replace("models/", "") for m in client.models.list()}
        logger.info(f"Available Gemini models ({len(available)} total): {sorted(available)}")
        for preferred in _MODEL_PREFERENCE:
            if preferred in available:
                logger.info(f"Selected model: {preferred}")
                return preferred
        # If none matched the preference list, pick any flash model
        flash_models = sorted([m for m in available if "flash" in m.lower()])
        if flash_models:
            logger.info(f"No preferred model found; using first flash model: {flash_models[0]}")
            return flash_models[0]
    except Exception as e:
        logger.warning(f"Model discovery failed ({e}); falling back to preference list default.")

    fallback = _MODEL_PREFERENCE[0]
    logger.warning(f"Using fallback model: {fallback}")
    return fallback


class AegisAgent:
    """
    Governed LLM Agent using google-genai.
    
    Design decisions:
    - Intelligence is pre-computed and embedded in the prompt, not fetched via tool.
      This reduces API calls from 4–5 per invocation to 1–2.
    - get_campaign_eligibility is kept as a tool so the agent can verify constraints.
    - Response schema (structured output) is requested in a single final call.
    - Model is auto-discovered from what's available on the API key.
    """

    def __init__(self, tools: AgentTools):
        self.tools = tools
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name: Optional[str] = None

        if self.api_key and genai:
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = _discover_model(self.client)
            logger.info(f"AegisAgent ready. Model: {self.model_name}")
        else:
            self.client = None
            logger.warning("Google GenAI client not initialized. Check GEMINI_API_KEY and package.")

    def propose_action(
        self,
        customer_id: str,
        prompt: str,
        intelligence_context: Optional[Dict[str, Any]] = None,
    ) -> ActionProposal:
        """
        Executes the agent loop to propose an action for a customer.

        Optimised for minimum latency:
        - All intelligence AND campaign data are embedded in the prompt — no tool calls.
        - A single structured-output API call produces the final ActionProposal.
        - Falls back to Ollama on 503 / quota errors.
        """
        if not self.client:
            return self._fallback_ollama(customer_id, prompt, intelligence_context)

        # ── Load campaign data once (static file, negligible cost) ────────
        campaigns_block = ""
        try:
            import pathlib, json as _json
            cfg_path = pathlib.Path("config/campaigns.json")
            if cfg_path.exists():
                cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
                campaigns_block = (
                    "\n\n## Available Campaigns (pre-loaded)\n"
                    f"```json\n{_json.dumps(cfg.get('campaigns', {}), indent=2)}\n```\n"
                    "Use this data to determine which campaign applies. "
                    "Do NOT call any tool to fetch campaigns.\n"
                )
        except Exception:
            pass

        system_instruction = (
            "You are Aegis, a governed AI commerce agent. "
            "Your sole job is to reason over customer evidence and produce a structured ActionProposal. "
            "RULES:\n"
            "1. If intelligence status is INCOMPLETE, set action_type=NO_ACTION, requested_discount_pct=0.\n"
            "2. If the customer is active with no dormancy signals, keep requested_discount_pct <= 10 "
            "unless the operator explicitly requests otherwise.\n"
            "3. All data you need (intelligence + campaigns) is embedded below. "
            "Do not call any external tools.\n"
            "4. Base your reasoning strictly on the evidence provided. Do not hallucinate data."
        )

        # ── Build single prompt with all context ──────────────────────────
        context_block = ""
        if intelligence_context:
            ctx = dict(intelligence_context)
            next_basket = ctx.pop("next_basket", None)
            context_block = (
                "\n\n## Pre-computed Customer Intelligence\n"
                f"```json\n{json.dumps(ctx, indent=2, default=str)}\n```\n"
            )
            if next_basket and next_basket.get("available"):
                top_products = "\n".join(
                    [f"{i+1}. {p['product_name']} ({p['probability']*100:.0f}%)"
                     for i, p in enumerate(next_basket.get("top_products", []))]
                )
                context_block += (
                    "\n## Next-Basket Prediction (ML)\n"
                    f"Top likely purchases:\n{top_products}\n"
                    f"Signal: {next_basket.get('signal', 'unknown')}\n"
                )
            context_block += "\nThis intelligence was computed by the Aegis ML pipeline. Use it as ground truth.\n"

        full_message = (
            f"Evaluate customer {customer_id}.\n"
            f"Operator instruction: {prompt}"
            f"{context_block}"
            f"{campaigns_block}"
            "\nProduce the ActionProposal JSON now."
        )

        logger.info(f"Starting agent reasoning for customer {customer_id} using model {self.model_name}")

        # ── Single API call with structured output — no tool calls ────────
        import time
        last_err = None
        for attempt in range(2):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=ActionProposal,
                    )
                )
                raw_json = response.text
                try:
                    proposal_dict = json.loads(raw_json)
                    proposal = ActionProposal(**proposal_dict)
                except (json.JSONDecodeError, ValueError) as e:
                    raise AegisAgentError(f"LLM returned invalid structured output: {e}. Raw: {raw_json[:300]}")

                logger.info(f"Agent proposed: intent={proposal.intent}, discount={proposal.requested_discount_pct}%")
                return proposal

            except AegisAgentError:
                raise
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "503" in err_str or "unavailable" in err_str or "demand" in err_str:
                    if attempt == 0:
                        logger.warning(f"Gemini 503 on attempt {attempt+1}, retrying after 3s...")
                        time.sleep(3)
                        continue
                    # Second failure → fallback
                    logger.warning(f"Gemini 503 persists. Falling back to Ollama.")
                    return self._fallback_ollama(customer_id, prompt, intelligence_context)
                if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
                    logger.warning(f"Gemini quota exhausted. Falling back to Ollama.")
                    return self._fallback_ollama(customer_id, prompt, intelligence_context)
                break

        logger.error(f"Agent execution failed: {last_err}")
        raise AegisAgentError(f"Agent execution failed: {last_err}") from last_err

    def _fallback_ollama(
        self,
        customer_id: str,
        prompt: str,
        intelligence_context: Optional[Dict[str, Any]] = None,
    ) -> ActionProposal:
        import requests
        
        model = "llama3.2:3b"
        logger.info(f"Falling back to local Ollama model: {model}")
        
        system_instruction = (
            "You are Aegis, a governed AI commerce agent.\n"
            "Produce ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "intent": "string (e.g. CREATE_WINBACK_OFFER)",\n'
            '  "customer_id": "string",\n'
            '  "policy_rule_id": "string (e.g. WINBACK_01)",\n'
            '  "reasoning_summary": "string",\n'
            '  "requested_discount_pct": float,\n'
            '  "confidence": "string (High, Medium, or Low)",\n'
            '  "action_type": "string (e.g. CREATE_PAYMENT_LINK)"\n'
            "}\n"
            "RULES:\n"
            "1. If intelligence status is INCOMPLETE, set action_type=NO_ACTION, requested_discount_pct=0.\n"
            "2. If the customer is active with no dormancy signals, keep requested_discount_pct <= 10 unless the operator explicitly requests a different action.\n"
            "3. Output JSON only."
        )

        context_block = ""
        if intelligence_context:
            context_block = (
                f"\n\n## Pre-computed Customer Intelligence\n"
                f"```json\n{json.dumps(intelligence_context, indent=2, default=str)}\n```\n"
            )

        user_prompt = f"Evaluate customer {customer_id}.\nInstruction: {prompt}\n{context_block}\nReturn JSON."
        
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": f"{system_instruction}\n\n{user_prompt}",
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=120
            )
            resp.raise_for_status()
            raw_json = resp.json().get("response", "")
            
            try:
                proposal_dict = json.loads(raw_json)
                proposal_dict.setdefault("customer_id", customer_id)
                proposal = ActionProposal(**proposal_dict)
                logger.info(f"Ollama proposed: intent={proposal.intent}, discount={proposal.requested_discount_pct}%")
                return proposal
            except (json.JSONDecodeError, ValueError) as e:
                raise AegisAgentError(f"Ollama returned invalid structured output: {e}. Raw: {raw_json[:300]}")
                
        except Exception as e:
            raise AegisAgentError(f"Ollama fallback failed: {e}")
