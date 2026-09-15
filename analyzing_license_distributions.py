'''
Counting how many models use each license, and how many models have no license specified. 
This is useful for understanding the distribution of licenses across models on Hugging Face.
'''
import json
from collections import Counter

# This will store our counts: {'apache-2.0': 15, 'mit': 10, ...}
license_counts = Counter()

def analyze_license_distributions(data): 
    """
    Analyze the license distributions in the provided JSONL file.
    """
    with open(data, 'r') as f:
        for line in f:
            model = json.loads(line)
            
            # Get the license, or label it 'Unknown' if the key doesn't exist
            lic = license_resolver(model)
            license_counts[lic] += 1

    # Print the results sorted by the most common
    print("License Distribution:")
    for license, count in license_counts.most_common():
        print(f"\t{license}: {count}")


def license_resolver(model):
        # Define what we consider an "unresolved" or "ambiguous" license
        ambiguous_licenses = {'unknown', 'other', 'none', None}
        
        license_val = model.get('license', 'unknown')
    
        # If the license is ambiguous, try to find a specific one in the tags
        if license_val in ambiguous_licenses:
            tags = model.get('tags', [])
            for tag in tags:
                if tag.startswith('license:'):
                    license_val = tag.split(':', 1)[1]
                    break
                    
        return license_val
    
def filter_unresolved_licenses(input_file, output_file):
    # The licenses we want to track/flag for review
    flag_licenses = {'unknown', 'other', None}
    unresolved_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
                
            model = json.loads(line)
            resolved_license = license_resolver(model)
            
            # Update the model object with the resolved license if you want 
            # the output file to show the "best guess"
            model['resolved_license'] = resolved_license
            
            # If the final result is still ambiguous, write to the tracking file
            if resolved_license in flag_licenses:
                f_out.write(json.dumps(model) + '\n')
                unresolved_count += 1

    print(f"Processing complete. Found {unresolved_count} models with unresolved licenses.")
    print(f"Results written to: {output_file}")

# need to get license from readme file when license is other or uknwon


if __name__ == "__main__":
    DATA_FILE = "raw_data/hf_models_09_15_26.jsonl"  # Path to the JSONL file containing model metadata
    # analyze_licsense_distributions(DATA_FILE)
    filter_unresolved_licenses(DATA_FILE, 'unresolved_licenses.jsonl')