import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def run_sanity_check():
    adapter_path = "./quant_slm_adapter"
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("Loading model onto MPS...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="mps", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    test_cases = {
        "1. EXTREME BULL (Should output CE)": {
            "adx_14": 45.0, "vix_india": 11.5, "pcr": 0.55, "dte_nearest": 2, "nifty_spot": 24000,
            "trend_signal": "STRONG", "volatility_regime": "FAVORABLE", "pcr_sentiment": "ALIGNED"
        },
        "2. EXTREME BEAR (Should output PE)": {
            "adx_14": 40.0, "vix_india": 32.0, "pcr": 1.65, "dte_nearest": 1, "nifty_spot": 22000,
            "trend_signal": "STRONG", "volatility_regime": "FAVORABLE", "pcr_sentiment": "ALIGNED"
        },
        "3. DEAD MARKET (Should output NEUTRAL)": {
            "adx_14": 12.0, "vix_india": 14.0, "pcr": 1.05, "dte_nearest": 4, "nifty_spot": 23000,
            "trend_signal": "WEAK", "volatility_regime": "HOSTILE", "pcr_sentiment": "MIXED"
        }
    }

    base_instruction = "Analyze the NIFTY options state and the provided text hints (trend_signal, volatility_regime, pcr_sentiment). First output an 'analysis' block with booleans mapping exactly to the hints. Then output the final directional signal, horizon, signal_id, timestamp, and conviction."

    for name, state in test_cases.items():
        print(f"--- {name} ---")
        prompt = f"<|system|>\n{base_instruction}</s>\n<|user|>\n{json.dumps(state)}</s>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.3, top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            
        print(f"Model Output:\n{tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)}\n")

if __name__ == "__main__":
    run_sanity_check()