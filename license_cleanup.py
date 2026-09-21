import json
import os
import re
from huggingface_hub import hf_hub_download

def fetch_license_info_from_readme(model_id, token=None):
    """
    Downloads the README and extracts the prioritized license name and the license link.
    """
    try:
        # Download README via official Hub API
        readme_path = hf_hub_download(
            repo_id=model_id, 
            filename="README.md", 
            repo_type="model", 
            token=token
        )
        
        with open(readme_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 1. Extract License Name (Priority: license_name > license)
        name_match = re.search(r'(?i)license_name:\s*(.+)', text)
        if not name_match:
            name_match = re.search(r'(?i)license:\s*(.+)', text)
        
        license_name = name_match.group(1).strip().strip('"\'') if name_match else None
        
        # 2. Extract License Link
        link_match = re.search(r'(?i)license_link:\s*(.+)', text)
        license_link = link_match.group(1).strip().strip('"\'') if link_match else None

        return license_name, license_link

    except Exception as e:
        # Silently fail for specific models to keep the pipeline moving, 
        # but you can print(e) here for debugging.
        return None, None

def license_resolver(model, hf_token=None):
    """
    Determines the best license name and link using a 3-tier system.
    """
    
    # Initial values from the model metadata
    current_license = model.get('license', 'unknown')
    current_link = model.get('license_link', None)

    # Tier 1: If the license is already good, just return it
    ambiguous_licenses = {'unknown', 'other', 'none', None, "null"}
    try:
        if current_license not in ambiguous_licenses:
            return current_license, current_link

    #fail gracefully if license is not a string or is malformed, and print the model id for debugging
    except:
        print(f"Error processing model {model.get('id')}: {model}")
    
    # if current_license in ambiguous_licenses:
    #     print(model.get('id'), "has ambiguous license:", current_license)

    # Tier 2: Check tags if license is ambiguous
    tags = model.get('tags', [])
    for tag in tags:
        if tag.startswith('license:'):
            current_license = tag.split(':', 1)[1]
            break

    # Tier 3: If still ambiguous, hit the Hugging Face API
    if current_license in ambiguous_licenses:
        model_id = model.get('id')
        if model_id:
            name, link = fetch_license_info_from_readme(model_id, hf_token)
            if name:
                current_license = name
            if link:
                current_link = link

    return current_license, current_link

def process_full_dataset(input_file, cleaned_file, unresolved_file, hf_token=None):
    """
    Reads input JSONL, cleans licenses, and writes to cleaned and unresolved files.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(cleaned_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    flag_licenses = {'unknown', 'other', 'none'}
    processed_count = 0
    unresolved_count = 0

    print(f"Starting processing: {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(cleaned_file, 'w', encoding='utf-8') as f_cleaned, \
            open(unresolved_file, 'w', encoding='utf-8') as f_unresolved:
        
        for line in f_in:
            if not line.strip(): continue
            
            model = json.loads(line)

            #check if the model id exists in the cleaned file already and skip it, saving API calls and processing time
            with open(cleaned_file, 'r', encoding='utf-8') as f_check:
                if any(model.get('id') == json.loads(existing_line).get('id') for existing_line in f_check):
                    print(f"Skipping {model.get('id')} as it already exists in the cleaned file.")
                    continue
            
            # Resolve the license and link
            resolved_name, resolved_link = license_resolver(model, hf_token)
            
            # Update model object
            model['license'] = resolved_name
            model['license_link'] = resolved_link
            
            # Always write to cleaned file
            f_cleaned.write(json.dumps(model) + '\n')
            processed_count += 1
            
            # Write to unresolved file if still ambiguous
            if resolved_name is None or str(resolved_name).lower() in flag_licenses:
                f_unresolved.write(json.dumps(model) + '\n')
                unresolved_count += 1
            
            # Progress indicator for long files
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} models...")

    print("\n" + "="*30)
    print(f"Processing Complete!")
    print(f"Total Processed: {processed_count}")
    print(f"Still Unresolved: {unresolved_count}")
    print(f"Cleaned file: {cleaned_file}")
    print(f"Audit file: {unresolved_file}")
    print("="*30)

if __name__ == "__main__":
    # --- Configuration ---
    INPUT_FILE = 'raw_data/hf_models_09_21_26.jsonl'
    CLEANED_FILE = 'clean_license_data/clean_licenses.jsonl'
    UNRESOLVED_FILE = 'clean_license_data/unresolved_licenses.jsonl'
    HF_TOKEN = os.getenv("HF_TOKEN") 
    # ---------------------

    process_full_dataset(
        input_file=INPUT_FILE,
        cleaned_file=CLEANED_FILE,
        unresolved_file=UNRESOLVED_FILE,
        hf_token=HF_TOKEN
    )