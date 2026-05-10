import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class EvalThresholds:
    """
    Hard thresholds committed BEFORE training. 
    """
    min_schema_pass_rate: float = 1.0 
    min_directional_accuracy: float = 0.45 

    expected_suppression_rate_min: float = 0.10
    expected_suppression_rate_max: float = 0.30
    expected_downgrade_rate_min: float = 0.05

class WalkForwardEvaluator:
    def __init__(self, thresholds: EvalThresholds = EvalThresholds()):
        self.thresholds = thresholds

    def evaluate_5_day_window(self, df_window: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluates a 5 day walk forward window against our strict thresholds.
        """
        total_signals = len(df_window)
        if total_signals == 0:
            return {}

        schema_pass_rate = df_window['parse_success'].mean()
        suppression_rate = df_window['suppressed'].mean()
        downgrade_rate = df_window['downgraded'].mean()

        active_trades = df_window[df_window['final_orchestrator_direction'] != 'NEUTRAL']
        
        if len(active_trades) > 0:
            directional_accuracy = (
                active_trades['final_orchestrator_direction'] == active_trades['actual_direction']
            ).mean()
        else:
            directional_accuracy = 0.0

        high_vix_mask = df_window['vix_level'] > df_window['vix_level'].median()
        high_vix_accuracy = (
            df_window.loc[high_vix_mask, 'final_orchestrator_direction'] == 
            df_window.loc[high_vix_mask, 'actual_direction']
        ).mean() if sum(high_vix_mask) > 0 else 0.0

        return {
            "schema_pass_rate": schema_pass_rate,
            "directional_accuracy": directional_accuracy,
            "suppression_rate": suppression_rate,
            "downgrade_rate": downgrade_rate,
            "high_vix_accuracy": high_vix_accuracy,
            "passed_safety_checks": schema_pass_rate >= self.thresholds.min_schema_pass_rate
        }

    def evaluate_conviction_validity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If accuracy doesn't monotonically increase with conviction, the conviction score is garbage
        """
        bins = [0.0, 0.4, 0.6, 0.8, 1.0]
        df['conviction_bin'] = pd.cut(df['conviction_score'], bins=bins)
        
        validity = df.groupby('conviction_bin').apply(
            lambda x: (x['final_orchestrator_direction'] == x['actual_direction']).mean() 
            if len(x) > 0 else 0.0
        )
        return validity