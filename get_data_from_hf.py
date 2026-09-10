import json
import os
import time

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

'''
This script downloads metadata for Hugging Face models and saves it to a JSONL file.
TODO
- Check for duplicates in the output file and skip them
- Date the output files with the current date so we know when the data came in
- Think about how to filter out for things that are more relevant, 
for example by having a minimum number of downloads.

TODO later
- Look at the datasets related to a subset of these models
- Find the papers related to a subset of these models
- Identify models for case studies, emblematic of some part of the issues/practices we analyze
'''

'''
|---------|
|--Ryan---|
|---------|
Make sure you are setting up an environment variable that has your HF_TOKEN in it. You can do this by running the following command in your terminal:
export HF_TOKEN="your_hugging_face_token_here"

Alternatively, ask your favorite AI to write a script that will set a permanent environment variable for you. 
You will have to restart VS code after setting the variable for it to take effect. 
'''



# %% Configuration

# Search
QUERY = None                 # None = all models; e.g. "open" to search by string
LIMIT = 5                 # None = no limit

# Output
OUT_JSONL = "raw_data/hf_models.jsonl"

# Authentication
HF_TOKEN = os.getenv("HF_TOKEN")
print(HF_TOKEN)

# Error handling
MAX_RETRIES = 3
RETRY_WAIT = 10              # seconds between retries

# Progress
PRINT_EVERY = 1_000          # print status every N models


# %% Helpers

def iso_or_none(value):
    """Convert datetime values to ISO strings."""
    return value.isoformat() if value else None


def card_data_to_dict(card_data):
    """Convert Hugging Face cardData to a regular dictionary."""
    if card_data is None:
        return {}

    if isinstance(card_data, dict):
        return card_data

    if hasattr(card_data, "to_dict"):
        try:
            return card_data.to_dict()
        except Exception:
            pass

    try:
        return dict(card_data)
    except Exception:
        return {}


def resolve_license(card_data, tags):
    """
    Resolve license from model card metadata first,
    then fall back to the license:<name> tag.
    """
    license_name = card_data.get("license")

    if license_name:
        return license_name

    for tag in tags or []:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]

    return None


# %% Setup

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is not set. "
        "Set it in your environment before running this script."
    )

api = HfApi(token=HF_TOKEN)

EXPAND_FIELDS = [
    "author",
    "createdAt",
    "lastModified",
    "downloadsAllTime",
    "tags",
    "cardData",
]


# %% Download

def download_models():
    """Download Hugging Face model metadata to JSONL."""

    seen = set()
    count = 0
    retry = 0

    print("Starting Hugging Face model metadata download")
    print(f"Query: {QUERY!r}")
    print(f"Limit: {LIMIT}")
    print(f"Output: {OUT_JSONL}")
    print()

    with open(OUT_JSONL, "w", encoding="utf-8") as f:

        while retry <= MAX_RETRIES:
            try:
                kwargs = {
                    "expand": EXPAND_FIELDS,
                    "limit": LIMIT,
                }

                if QUERY:
                    kwargs["search"] = QUERY

                models = api.list_models(**kwargs)

                for model in models:
                    model_id = model.id

                    # Prevent duplicates if a retry restarts the iterator
                    if model_id in seen:
                        continue

                    tags = model.tags or []
                    card_data = card_data_to_dict(model.card_data)

                    record = {
                        "id": model_id,
                        "author": model.author,
                        "created_at": iso_or_none(model.created_at),
                        "last_modified": iso_or_none(model.last_modified),
                        "downloads_all_time": model.downloads_all_time,
                        "license": resolve_license(card_data, tags),
                        "tags": tags,
                        "card_data": card_data,
                    }

                    f.write(
                        json.dumps(record, ensure_ascii=False)
                        + "\n"
                    )
                    f.flush()

                    seen.add(model_id)
                    count += 1

                    if count % PRINT_EVERY == 0:
                        print(f"Collected {count:,} models...")

                # Iterator completed successfully
                break

            except HfHubHTTPError as e:
                retry += 1

                print(
                    f"\nHugging Face API error "
                    f"(attempt {retry}/{MAX_RETRIES}): {e}"
                )

                if retry > MAX_RETRIES:
                    print("Maximum retries reached.")
                    raise

                print(f"Retrying in {RETRY_WAIT} seconds...")
                time.sleep(RETRY_WAIT)

            except KeyboardInterrupt:
                print(
                    f"\nStopped by user. "
                    f"{count:,} models have already been saved."
                )
                return count

            except Exception as e:
                print(
                    f"\nUnexpected error after "
                    f"{count:,} models: {e}"
                )
                raise

    print()
    print("Finished.")
    print(f"Collected: {count:,} models")
    print(f"Saved to: {OUT_JSONL}")

    return count


if __name__ == "__main__":
    download_models()