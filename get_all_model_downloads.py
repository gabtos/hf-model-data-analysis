import csv
import os
import re
import time
from datetime import datetime

from huggingface_hub.utils import build_hf_headers, get_session

# %% Configuration

START_URL = "https://huggingface.co/api/models?expand[]=downloadsAllTime&sort=createdAt&limit=1000"
OUT_CSV = f"raw_data/hf_model_downloads_{datetime.now():%m_%d_%y}.csv"
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_RETRIES = 8
PRINT_EVERY = 100_000


# %% Helpers

def rate_limit_wait(headers):
    match = re.search(r"r=(\d+);t=(\d+)", headers.get("RateLimit", ""))
    if not match:
        return None
    remaining, reset = int(match.group(1)), int(match.group(2))
    return reset + 1 if remaining == 0 else None


def next_page(headers):
    match = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
    return match.group(1) if match else None


def fetch(session, url, headers):
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, headers=headers, timeout=60)
        except Exception as e:
            wait = min(60, 2 ** attempt)
            print(f"  request error ({e}); retrying in {wait}s", flush=True)
            time.sleep(wait)
            continue
        if response.status_code == 429:
            wait = rate_limit_wait(response.headers) or int(response.headers.get("Retry-After", 60)) + 1
            print(f"  rate limited; waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        if response.status_code >= 500:
            wait = min(60, 2 ** attempt)
            print(f"  server error {response.status_code}; retrying in {wait}s", flush=True)
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError(f"giving up on {url} after {MAX_RETRIES} attempts")


# %% Download

def download_all():
    session = get_session()
    headers = build_hf_headers(token=HF_TOKEN)
    print(f"Fetching id and all-time downloads for every model -> {OUT_CSV}", flush=True)
    print(f"Authenticated: {'Authorization' in headers}", flush=True)

    seen, missing, url, t0 = set(), 0, START_URL, time.time()
    partial = OUT_CSV + ".partial"
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(partial, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "downloads_all_time"])
        while url:
            response = fetch(session, url, headers)
            for model in response.json():
                if model["id"] in seen:
                    continue
                seen.add(model["id"])
                downloads = model.get("downloadsAllTime")
                if downloads is None:
                    missing += 1
                writer.writerow([model["id"], "" if downloads is None else downloads])
                if len(seen) % PRINT_EVERY == 0:
                    print(f"  {len(seen):,} models ({time.time() - t0:.0f}s)", flush=True)
            url = next_page(response.headers)
            wait = rate_limit_wait(response.headers)
            if url and wait:
                print(f"  {len(seen):,} models; request window used up, waiting {wait}s", flush=True)
                time.sleep(wait)

    os.replace(partial, OUT_CSV)
    print(f"Done: {len(seen):,} models in {time.time() - t0:.0f}s; {missing:,} without a download count", flush=True)
    print(f"Saved to {OUT_CSV}", flush=True)


if __name__ == "__main__":
    download_all()
