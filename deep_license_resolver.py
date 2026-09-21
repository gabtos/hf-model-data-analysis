import json
import os
import re
import logging
from time import time
from huggingface_hub import list_repo_files
import requests

def setup_logging(log_file):
    """Sets up logging to both console and a file."""
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create format for the logs
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 1. File Handler: Writes everything to the log file
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Console Handler: Prints everything to the screen
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def is_actual_license(license_str):
    if not license_str:
        return False
    noise_pattern = r"^(readme(_\w+)?(\.\w+)?|license(\.\w+)?)$"
    if re.match(noise_pattern, license_str, re.IGNORECASE):
        return False
    return True

def deep_resolve_license(model_id, token=None):
    """
    Scans the repo for LICENSE files or a licenses/ folder.
    """
    try:
        all_files = list_repo_files(repo_id=model_id, repo_type="model", token=token)
        
        found_licenses = []
        logging.info(f"Scanning {model_id}. Found files: {all_files}")

        for file_path in all_files:
            if "licenses/" in file_path.lower():
                filename = os.path.basename(file_path).split('.')[0]
                found_licenses.append(filename)
            elif os.path.basename(file_path).upper().startswith("LICENSE"):
                filename = os.path.basename(file_path)
                found_licenses.append(filename)

        found_licenses = list(set(found_licenses))
        logging.info(f"Model {model_id} potential licenses: {found_licenses}")
        
        cleaned_licenses = list(filter(is_actual_license, found_licenses))
        logging.info(f"Model {model_id} cleaned licenses: {cleaned_licenses}")

        if not cleaned_licenses:
            return None
        if len(cleaned_licenses) == 1:
            return cleaned_licenses[0]
        return cleaned_licenses 
    except Exception as e:
        logging.error(f"Error scanning {model_id}: {e}")
        return None

def process_unresolved_deep(input_file, final_file, hf_token=None):
    processed_count = 0
    resolved_count = 0
    
    logging.info(f"Starting Deep Resolution. Input: {input_file}")

    # OPTIMIZATION: Load existing IDs into a set
    existing_ids = set()
    if os.path.exists(final_file):
        with open(final_file, 'r', encoding='utf-8') as f_check:
            for line in f_check:
                try:
                    # Handle both 'id' and 'model_id'
                    m = json.loads(line)
                    existing_ids.add(m.get('id') or m.get('model_id'))
                except: continue
    logging.info(f"Found {len(existing_ids)} already processed models. Skipping them.")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(final_file, 'a', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            model = json.loads(line)
            model_id = model.get('id') or model.get('model_id')
            
            if not model_id or model_id in existing_ids:
                continue

            while True:
                try:
                    deep_license = deep_resolve_license(model_id, hf_token)
                    
                    if deep_license:
                        model['license'] = deep_license
                        resolved_count += 1
                        f_out.write(json.dumps(model) + '\n')
                        f_out.flush() 
                    
                    break # Success! Exit retry loop

                except requests.exceptions.HTTPError as e:
                    # Check if it's a Rate Limit (429)
                    if e.response is not None and e.response.status_code == 429:
                        # 1. Try Retry-After header
                        wait_time = e.response.headers.get("Retry-After")
                        
                        # 2. Fallback: Parse JSON body
                        if not wait_time:
                            try:
                                error_data = e.response.json()
                                # Search for digits in the error message if it's a string
                                import re
                                msg = error_data.get('error', '')
                                match = re.search(r'(\d+)', msg)
                                wait_time = match.group(1) if match else 60
                            except:
                                wait_time = 60

                        try:
                            seconds = int(wait_time)
                        except (ValueError, TypeError):
                            seconds = 60
                        
                        logging.warning(f"Rate limit hit for {model_id}. Waiting {seconds}s...")
                        time.sleep(seconds)
                        continue # Retry the request

                    else:
                        logging.error(f"HTTP Error {e.response.status_code if e.response else 'Unknown'} for {model_id}")
                        break

                except Exception as e:
                    logging.error(f"Unexpected error for {model_id}: {e}")
                    break
            
            processed_count += 1

    logging.info("="*30)
    logging.info("Deep Resolution Complete!")
    logging.info(f"Processed: {processed_count}")
    logging.info(f"Successfully resolved via files: {resolved_count}")
    logging.info(f"Unresolved: {processed_count - resolved_count}")
    logging.info(f"Final data saved to: {final_file}")
    logging.info("="*30)

if __name__ == "__main__":
    # --- Configuration ---
    DATA_DIR = 'clean_license_data'
    UNRESOLVED_FILE = os.path.join(DATA_DIR, 'unresolved_licenses.jsonl')
    FINAL_FILE = os.path.join(DATA_DIR, 'deep_resolved_licenses.jsonl')
    LOG_FILE = os.path.join(DATA_DIR, 'deep_resolution.log')
    HF_TOKEN = os.getenv("HF_TOKEN")
    # ---------------------

    # 1. Initialize the logging system
    os.makedirs(DATA_DIR, exist_ok=True)
    setup_logging(LOG_FILE)

    # 2. Run the process
    process_unresolved_deep(
        input_file=UNRESOLVED_FILE,
        final_file=FINAL_FILE,
        hf_token=HF_TOKEN
    )