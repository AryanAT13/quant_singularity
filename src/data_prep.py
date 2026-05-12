import json
import re
import random

def calculate_confluence_conviction(market_state, direction):
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
    anomalies = {"malformed_output": 0, "missing_fields": 0, "illogical_signal": 0}
    
    directional_samples = []
    neutral_samples = []

    try:
        with open('data/raw/finetune_instructions.jsonl', 'r') as f:
            instructions = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("Error: Ensure finetune_instructions.jsonl is in data/raw/")
        return

    print(f"Starting audit on {len(instructions)}")

    for idx, row in enumerate(instructions):
        input_str = row.get('input', '{}')
        output_str = row.get('output', '{}')
        
        try:
            input_json = json.loads(input_str)
            target_json = json.loads(output_str)
        except json.JSONDecodeError:
            anomalies["malformed_output"] += 1
            continue

        direction = target_json.get('direction')
        horizon = target_json.get('horizon')
        signal_id = target_json.get('signal_id')
        timestamp = target_json.get('generated_at')
        
        if not all([direction, horizon, signal_id, timestamp]):
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
             anomalies["illogical_signal"] += 1
             continue

        final_direction = direction
        if conviction_score < 0.4:
            final_direction = "NEUTRAL"
            conviction_score = 1.0 # High conviction in staying cash
            analysis_block = calculate_confluence_conviction(input_json, final_direction)[1]

        upgraded_target = {
            "analysis": analysis_block,
            "direction": final_direction,
            "conviction": conviction_score, 
            "horizon": horizon,
            "signal_id": signal_id,
            "timestamp": timestamp 
        }
        
        new_instruction = "Analyze the provided NIFTY options market state. First output an 'analysis' block with boolean values for trend_aligned, volatility_support, and pcr_aligned. Then output the final directional signal, horizon, signal_id, timestamp, and a strictly calculated conviction score derived from the analysis block."

        clean_row = {
            "instruction": new_instruction,
            "input": input_str, 
            "output": json.dumps(upgraded_target) 
        }
        
        if final_direction in ["CE", "PE"]:
            directional_samples.append(clean_row)
        else:
            neutral_samples.append(clean_row)

    # We cap NEUTRALs to roughly match the directional samples to prevent Mode Collapse
    max_neutrals = len(directional_samples) + 20 
    random.shuffle(neutral_samples)
    balanced_neutral_samples = neutral_samples[:max_neutrals]
    
    final_dataset = directional_samples + balanced_neutral_samples
    random.shuffle(final_dataset) # Shuffle so it doesn't learn a pattern of CE CE PE NEUTRAL NEUTRAL

    ce_count = sum(1 for x in final_dataset if '"direction": "CE"' in x['output'])
    pe_count = sum(1 for x in final_dataset if '"direction": "PE"' in x['output'])
    neutral_count = sum(1 for x in final_dataset if '"direction": "NEUTRAL"' in x['output'])

    print(f"Malformed JSON dropped: {anomalies['malformed_output']}")
    print(f"Missing Field dropped: {anomalies['missing_fields']}")
    print(f"Illogical Signals dropped: {anomalies['illogical_signal']}")

    print(f"CE: {ce_count} | PE: {pe_count} | NEUTRAL: {neutral_count}")
    print(f"Total rows ready for training: {len(final_dataset)}")
    
    with open('data/processed/finetune_clean_cot.jsonl', 'w') as f:
        for item in final_dataset:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    process_and_clean_data()