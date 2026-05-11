import json
import ast
import re

def calculate_confluence_conviction(market_state, direction):
    """
    evaluating 3 hard market rules
    """
    # Trend 
    adx = float(market_state.get('adx_14', 0))
    trend_aligned = adx > 20.0 
    
    # Volatility Support
    vix = float(market_state.get('vix_india', 15))
    if direction == "CE":
        vol_support = vix < 20
    elif direction == "PE":
        vol_support = vix >= 15
    else:
        vol_support = True
        
    # PCR 
    pcr = float(market_state.get('pcr', 1.0))
    if direction == "CE":
        pcr_aligned = pcr < 1.0
    elif direction == "PE":
        pcr_aligned = pcr > 1.0
    else:
        pcr_aligned = (0.8 <= pcr <= 1.2)

    factors = [trend_aligned, vol_support, pcr_aligned]
    score = sum(factors) / len(factors)
    
    analysis_block = {
        "trend_aligned": trend_aligned,
        "volatility_support": vol_support,
        "pcr_aligned": pcr_aligned
    }
    
    return round(score, 2), analysis_block

def process_and_clean_data():
    cleaned_data = []
    anomalies = {"malformed_output": 0, "missing_fields": 0, "illogical_signal": 0}

    try:
        with open('data/raw/finetune_instructions.jsonl', 'r') as f:
            instructions = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("Error: Ensure finetune_instructions.jsonl is in data/raw/")
        return

    print(f"Starting audit {len(instructions)}")

    for idx, row in enumerate(instructions):
        input_str = row.get('input', '{}')
        output_str = row.get('output', '{}')
        
        try:
            input_json = json.loads(input_str)
            target_json = json.loads(output_str)
        except json.JSONDecodeError:
            print(f"Anomaly Caught (Row {idx}): Malformed nested JSON. Discarding.")
            anomalies["malformed_output"] += 1
            continue

        #Check for required keys
        direction = target_json.get('direction')
        horizon = target_json.get('horizon')
        signal_id = target_json.get('signal_id')
        timestamp = target_json.get('generated_at')
        
        if not all([direction, horizon, signal_id, timestamp]):
            print(f"Anomaly Caught (Row {idx}): Missing critical schema keys. Discarding.")
            anomalies["missing_fields"] += 1
            continue

        conviction_score, analysis_block = calculate_confluence_conviction(input_json, direction)
        
        raw_conviction = target_json.get('conviction', 0.0)
        try:
            original_conviction = float(raw_conviction)
        except ValueError:
            match = re.search(r"([0-9]*\.?[0-9]+)", str(raw_conviction))
            original_conviction = float(match.group(1)) if match else 0.0

        if original_conviction > 0.7 and conviction_score < 0.4:
             print(f"Anomaly Caught (Row {idx}): Illogical Signal. Target conviction is {original_conviction} but confluence is {conviction_score}. Discarding.")
             anomalies["illogical_signal"] += 1
             continue

        #Upgrade the Target JSON
        upgraded_target = {
            "analysis": analysis_block,
            "direction": direction,
            "conviction": conviction_score, 
            "horizon": horizon,
            "signal_id": signal_id,
            "timestamp": timestamp 
        }
        
        new_instruction = "Analyze the provided NIFTY options market state. First output an 'analysis' block with boolean values for trend_aligned, volatility_support, and pcr_aligned. Then output the final directional signal, horizon, signal_id, timestamp, and a strictly calculated conviction score (0.33, 0.66, or 1.0) derived from the analysis block."

        cleaned_data.append({
            "instruction": new_instruction,
            "input": input_str, 
            "output": json.dumps(upgraded_target) 
        })

    print(f"Malformed JSON dropped: {anomalies['malformed_output']}")
    print(f"Missing Field dropped: {anomalies['missing_fields']}")
    print(f"Illogical Signals dropped: {anomalies['illogical_signal']}")
    print(f"Total safe, upgraded rows ready for training: {len(cleaned_data)} / {len(instructions)}")
    
    with open('data/processed/finetune_clean_cot.jsonl', 'w') as f:
        for item in cleaned_data:
            f.write(json.dumps(item) + '\n')
    print("Success. Cleaned dataset saved to data/processed/finetune_clean_cot.jsonl")

if __name__ == "__main__":
    process_and_clean_data()