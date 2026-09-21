import json
import os

def combine_jsonl_dedup(file1, file2, output_file):
    models = {}
    counts = {file1: 0, file2: 0}

    # Process both files
    for filename in [file1, file2]:
        if not os.path.exists(filename):
            print(f"Warning: File not found, skipping: {filename}")
            continue

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                
                try:
                    model = json.loads(line)
                    
                    # --- FIX: Robust Key Detection ---
                    # This checks for 'model_id' first, then 'id' as a fallback
                    key = model.get('model_id') or model.get('id')
                    
                    if key:
                        models[key] = model
                        # Update counter for this specific file
                        # We use the index of the filename in our list to track
                        current_file = file1 if filename == file1 else file2
                        counts[current_file] += 1
                    else:
                        print(f"Warning: Model missing both 'id' and 'model_id' in {filename}")
                
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in {filename}")

    # Write the unique models back to a file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for model in models.values():
            f_out.write(json.dumps(model) + '\n')

    # Print a summary so you can verify it's actually working
    print("-" * 30)
    print(f"Models loaded from {os.path.basename(file1)}: {counts[file1]}")
    print(f"Models loaded from {os.path.basename(file2)}: {counts[file2]}")
    print(f"Total unique models saved: {len(models)}")
    print("-" * 30)

if __name__ == "__main__":
    # --- Configuration ---
    DATA_DIR = 'clean_license_data' 
    
    FILE_1 = 'clean_licenses.jsonl'
    FILE_2 = 'deep_resolved_licenses.jsonl'
    OUTPUT_FILE = 'final_combined_licenses.jsonl'
    # ---------------------

    path1 = os.path.join(DATA_DIR, FILE_1)
    path2 = os.path.join(DATA_DIR, FILE_2)
    path_out = os.path.join(DATA_DIR, OUTPUT_FILE)

    os.makedirs(DATA_DIR, exist_ok=True)

    combine_jsonl_dedup(
        file1=path1, 
        file2=path2, 
        output_file=path_out
    )
    
    print(f"Success! Combined files saved to: {path_out}")