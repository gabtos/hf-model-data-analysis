import json
import os
from huggingface_hub import list_repo_files, hf_hub_download

def deep_resolve_license(model_id, token=None):
    """
    Scans the repo for LICENSE files or a licenses/ folder.
    Returns a string if one is found, a list if multiple are found, or None.
    """
    try:
        # 1. Get a list of all files in the repository
        all_files = list_repo_files(repo_id=model_id, repo_type="model", token=token)
        
        found_licenses = []

        for file_path in all_files:
            # Case 1: File is inside a 'licenses/' folder
            if "licenses/" in file_path.lower():
                # Extract just the filename without the folder path and extension
                filename = os.path.basename(file_path).split('.')[0]
                found_licenses.append(filename)
            
            # Case 2: File is named LICENSE, LICENSE.txt, LICENSE.md, etc. (at root)
            elif os.path.basename(file_path).upper().startswith("LICENSE"):
                filename = os.path.basename(file_path)
                found_licenses.append(filename)

        # Remove duplicates
        found_licenses = list(set(found_licenses))

        if not found_licenses:
            return None
        if len(found_licenses) == 1:
            return found_licenses[0]
        return found_licenses  # Return the list if multiple found

    except Exception as e:
        print(f"Error scanning {model_id}: {e}")
        return None

def process_unresolved_deep(input_file, final_file, hf_token=None):
    """
    Processes the unresolved file and attempts deep resolution via file scanning.
    """
    processed_count = 0
    resolved_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(final_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            model = json.loads(line)
            
            model_id = model.get('id')
            if model_id:
                # Attempt deep resolution
                deep_license = deep_resolve_license(model_id, hf_token)
                
                if deep_license:
                    model['license'] = deep_license
                    resolved_count += 1
            
            f_out.write(json.dumps(model) + '\n')
            processed_count += 1

    print(f"Deep Resolution Complete!")
    print(f"Processed: {processed_count}")
    print(f"Successfully resolved via files: {resolved_count}")
    print(f"Final data saved to: {final_file}")

if __name__ == "__main__":
    # Configuration
    UNRESOLVED_FILE = 'clean_license_data/unresolved_licenses.jsonl'
    FINAL_FILE = 'clean_license_data/final_resolved_licenses.jsonl'
    HF_TOKEN = os.getenv("HF_TOKEN")

    process_unresolved_deep(
        input_file=UNRESOLVED_FILE,
        final_file=FINAL_FILE,
        hf_token=HF_TOKEN
    )