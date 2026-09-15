import os
import re
from huggingface_hub import hf_hub_download

def test_license_fetch(model_id):
    token = os.getenv("HF_TOKEN")
    
    try:
        # 1. Download the README
        readme_path = hf_hub_download(
            repo_id=model_id, 
            filename="README.md", 
            repo_type="model", 
            token=token
        )
        
        with open(readme_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 2. Extract License Name (Prioritize license_name over license)
        name_match = re.search(r'(?i)license_name:\s*(.+)', text)
        if not name_match:
            name_match = re.search(r'(?i)license:\s*(.+)', text)
        
        license_name = name_match.group(1).strip().strip('"\'') if name_match else "Unknown"
        
        # 3. Extract License Link
        link_match = re.search(r'(?i)license_link:\s*(.+)', text)
        license_link = link_match.group(1).strip().strip('"\'') if link_match else "No link found"

        # Return both as a dictionary
        return {
            "license_name": license_name,
            "license_link": license_link
        }

    except Exception as e:
        return {"error": str(e)}

# Test with your specific model
model = "Lightricks/LTX-2.5"
result = test_license_fetch(model)

print(f"Model: {model}")
print(f"Name: {result.get('license_name')}")
print(f"Link: {result.get('license_link')}")