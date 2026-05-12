import pandas as pd
import numpy as np
import mlflow
import json
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.orchestrator import SignalOrchestrator
from src.evaluation import WalkForwardEvaluator, EvalThresholds

sys.path.append(os.path.abspath('data/raw'))
try:
    from retrieve import retrieve
except ImportError:
    print("Warning: retrieve.py not found in data/raw/. RAG will be disabled.")
    retrieve = None

class InferencePipeline:
    def __init__(self, use_rag: bool = False):
        self.use_rag = use_rag
        self.orchestrator = SignalOrchestrator()
        self.evaluator = WalkForwardEvaluator(EvalThresholds())
        self.model = None
        self.tokenizer = None
        
    def load_model(self, adapter_path: str = "./quant_slm_adapter"):
        print(f"Loading Base Model and LoRA Adapter from {adapter_path}")
        base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="mps", #for silicon chipp
            torch_dtype=torch.float16
        )
        
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval() 

    def model_inference(self, market_state: dict) -> str:
        raw_prompt = self.construct_prompt(market_state)
        
        instruction = raw_prompt.split("Market State:")[0].strip()
        state_str = json.dumps(market_state)
        formatted_prompt = f"<|system|>\n{instruction}</s>\n<|user|>\n{state_str}</s>\n<|assistant|>\n"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("mps")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=150,
                temperature=0.3,       
                top_p=0.9,            
                repetition_penalty=1.1, 
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text.strip()

    def construct_prompt(self, market_state: dict) -> str:
        base_instruction = "Analyze the provided NIFTY options market state. First output an 'analysis' block with boolean values for trend_aligned, volatility_support, and pcr_aligned. Then output the final directional signal, horizon, signal_id, timestamp, and a strictly calculated conviction score derived from the analysis block."
        
        context_str = ""
        if self.use_rag and retrieve is not None:
            episodes = retrieve(market_state, k=3)
            context_str = "\nHistorical Context:\n" + "\n".join([ep['summary'] for ep in episodes])
            
        return f"{base_instruction}{context_str}\nMarket State: {json.dumps(market_state)}"

    def run_walk_forward_eval(self, parquet_path: str):
        print(f"Starting Walk-Forward Eval. RAG Enabled: {self.use_rag}")
        self.load_model() 
        
        df = pd.read_parquet(parquet_path)
        
        if 'day' not in df.columns:
            print("Warning: 'day' column missing. Synthetically generating days 1-60 based on row count.")
            df['day'] = np.repeat(np.arange(1, 61), len(df) // 60 + 1)[:len(df)]
            
        eval_df = df[(df['day'] >= 31) & (df['day'] <= 60)].copy()
        results_log = []
        
        with mlflow.start_run(run_name=f"WalkForward_RAG_{self.use_rag}"):
            mlflow.log_param("use_rag", self.use_rag)
            
            for window_start in range(31, 61, 5):
                window_end = window_start + 4
                window_df = eval_df[(eval_df['day'] >= window_start) & (eval_df['day'] <= window_end)].copy()
                
                print(f"\nEvaluating Window: Days {window_start}-{window_end} ({len(window_df)} events)")
                
                window_results = []
                for count, (idx, row) in enumerate(window_df.iterrows(), 1):
                    print(f" Processing {count}/{len(window_df)}", end="", flush=True)
                    market_state = row.to_dict()
                    signal_id = f"sig_{window_start}_{idx}"
                    
                    final_signal = self.orchestrator.process_market_state(
                        market_state=market_state, 
                        model_inference_callable=self.model_inference, 
                        signal_id=signal_id
                    )
                    print(f"Result: {final_signal['direction']} (Conviction: {final_signal['conviction']})")
                    
                    window_results.append({
                        'actual_direction': row.get('target_direction', 'NEUTRAL'), 
                        'raw_model_direction': final_signal.get('direction', 'NEUTRAL'), 
                        'final_orchestrator_direction': final_signal['direction'],
                        'conviction_score': final_signal['conviction'],
                        'parse_success': 1 if final_signal['direction'] != 'NEUTRAL' or final_signal['conviction'] > 0 else 0, 
                        'suppressed': 1 if self.orchestrator.logs[-1]['reason_code'] == 'ADX_SUPPRESSION' else 0,
                        'downgraded': 1 if self.orchestrator.logs[-1]['reason_code'] == 'LOW_CONVICTION_DOWNGRADE' else 0,
                        'vix_level': market_state.get('vix_india', 15.0)
                    })
                
                res_df = pd.DataFrame(window_results)
                metrics = self.evaluator.evaluate_5_day_window(res_df)
                
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"win_{window_start}_{k}", v)
                        
                results_log.append(metrics)
                
            print("\nWalk-forward evaluation complete. Data logged to MLflow.")

if __name__ == "__main__":
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)

    print("NO RAG control exp")
    pipeline_no_rag = InferencePipeline(use_rag=False)
    pipeline_no_rag.run_walk_forward_eval('data/raw/market_states.parquet')
    
    del pipeline_no_rag
    import gc
    gc.collect()

    print("\ntreatment exp with RAG")
    pipeline_rag = InferencePipeline(use_rag=True)
    pipeline_rag.run_walk_forward_eval('data/raw/market_states.parquet')