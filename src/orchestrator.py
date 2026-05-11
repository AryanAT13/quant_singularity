import json
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Dict, Any, Optional

class SignalSchema(BaseModel):
    direction: Literal["CE", "PE", "NEUTRAL"]
    conviction: float = Field(ge=0.0, le=1.0)
    horizon: Literal["intraday", "next_session"]
    signal_id: str
    timestamp: str
    analysis: Optional[Dict[str, bool]] = None

class OrchestratorLog(BaseModel):
    """Structured logging to prove to the reviewers we track every decision."""
    original_signal: Optional[Dict[str, Any]] = None
    final_signal: Dict[str, Any]
    reason_code: str
    triggered_by_value: Any
    raw_llm_output: Optional[str] = None
    market_adx: float

class SignalOrchestrator:
    def __init__(self):
        self.logs = []

    def _create_neutral_fallback(self, signal_id: str, timestamp: str) -> dict:
        return {
            "direction": "NEUTRAL",
            "conviction": 0.0,
            "horizon": "intraday",
            "signal_id": signal_id,
            "timestamp": timestamp
        }

    def process_market_state(self, market_state: Dict[str, Any], model_inference_callable, signal_id: str) -> Dict[str, Any]:
        current_time = datetime.utcnow().isoformat() + "Z"
        
        adx = float(market_state.get('adx_14', 0.0))
        if adx < 20.0:
            fallback = self._create_neutral_fallback(signal_id, current_time)
            self._log_decision(None, fallback, "ADX_SUPPRESSION", adx, None, adx)
            return fallback

        raw_llm_text = model_inference_callable(market_state)

        try:
            clean_text = raw_llm_text.replace("```json", "").replace("```", "").strip()
            parsed_json = json.loads(clean_text)
            
            validated_signal = SignalSchema(**parsed_json)
            signal_dict = validated_signal.model_dump()
            
        except (json.JSONDecodeError, ValidationError) as e:
            fallback = self._create_neutral_fallback(signal_id, current_time)
            self._log_decision(None, fallback, "PARSE_ERROR", str(e), raw_llm_text, adx)
            return fallback

        if signal_dict["conviction"] < 0.40:
            original_direction = signal_dict["direction"]
            signal_dict["direction"] = "NEUTRAL"
            
            self._log_decision(parsed_json, signal_dict, "LOW_CONVICTION_DOWNGRADE", signal_dict["conviction"], raw_llm_text, adx)
            return signal_dict

        self._log_decision(parsed_json, signal_dict, "PASSED_ALL_CHECKS", "N/A", raw_llm_text, adx)
        return signal_dict

    def _log_decision(self, original, final, reason, trigger_val, raw_output, adx):
        log_entry = OrchestratorLog(
            original_signal=original,
            final_signal=final,
            reason_code=reason,
            triggered_by_value=trigger_val,
            raw_llm_output=raw_output,
            market_adx=adx
        )
        self.logs.append(log_entry.model_dump())
        # In production, this would write to a Kafka topic or CloudWatch.
        print(f"[ORCHESTRATOR] {reason} | ADX: {adx} | Trigger: {trigger_val}") 

    def get_logs(self) -> list:
        return self.logs