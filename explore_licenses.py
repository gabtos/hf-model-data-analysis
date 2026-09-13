"""
EXPLORATORY ONLY — not part of the production pipeline (get_data_from_hf.py).

Purpose: characterize the Hugging Face Hub's license landscape *before* we
decide anything about filtering thresholds or "open-ish" criteria. This
script makes NO filtering decisions. It just pulls a broad, non-trending
sample and reports:

  1. The distribution of `downloads_all_time` (to inform, not set, a future
     long-tail cutoff).
  2. The raw `license` field value counts (resolved the same way Gabe's
     pipeline does: cardData.license, falling back to a `license:<x>` tag).
  3. How many models have NO resolvable license at all.
  4. A rough cross-tab of license values against two external reference
     lists (see below) — again for characterization, not filtering.

Reference frameworks used for categorization (not decision-making):
  - OSI Open Source Definition (OSD) v1.9 — the 10-point checklist for what
    counts as an open source *license* (free redistribution, source
    availability, no field-of-use or persons/groups discrimination, etc).
  - Open Source AI Definition (OSAID) v1.0 — extends this to AI systems
    specifically: requires not just an open license on the code/weights but
    published Data Information (what the training data was, how it was
    filtered) and Code (full training pipeline) under OSI-approved terms.
    A model can have an OSI-approved license on its *weights* and still
    fail OSAID if the training data/code aren't disclosed — that can't be
    determined from the `license` field alone, so we flag it as a caveat,
    not something this script measures.

IMPORTANT SAMPLING NOTE:
    HF's default `list_models()` order (no `sort` arg) is trending_score,
    which massively overrepresents currently-popular models and would bias
    any license characterization toward whatever's hot right now. We use
    sort="created_at" instead for a much less popularity-biased cross
    section of the Hub's history.

Rate limits observed empirically (2026-09-13, authenticated): HF enforces
a fixed-window quota of 1000 requests / 300s on the `api` route. Each
`list_models()` page appears to cover many records (thousands per HTTP
request in testing), so a pull in the tens of thousands of records is a
small fraction of that budget. A prior *unbounded* full-Hub enumeration
attempt did trip the 429 after several hundred thousand records — so this
script caps itself with SAMPLE_SIZE rather than trying to walk the entire
Hub in one go.
"""

import json
import os
import time
from collections import Counter

from huggingface_hub import HfApi

# %% Configuration

SAMPLE_SIZE = 50_000  # exploratory cross-section, not the full Hub
SORT = "created_at"   # avoid the default trending-score bias
OUT_PATH = "raw_data/license_exploration_sample.jsonl"
EXPAND_FIELDS = ["downloadsAllTime", "tags", "cardData", "createdAt"]

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set.")

api = HfApi(token=HF_TOKEN)

# OSI-approved SPDX identifiers that show up on the Hub. Not exhaustive of
# the full OSI list -- limited to ones actually observed in HF license
# tags/cardData, so this stays a live, testable claim rather than a
# blanket assumption.
OSI_APPROVED = {
    "apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "bsd-3-clause-clear",
    "gpl-3.0", "gpl-2.0", "lgpl-3.0", "lgpl-2.1", "agpl-3.0", "mpl-2.0",
    "cc0-1.0", "unlicense", "isc", "wtfpl", "zlib", "afl-3.0", "artistic-2.0",
    "bsl-1.0", "ecl-2.0", "epl-1.0", "epl-2.0", "eupl-1.1", "eupl-1.2",
    "ncsa", "osl-3.0", "postgresql", "python-2.0", "0bsd",
}

# The subset of OSI_APPROVED that is not just OSI-listed but genuinely
# well-known and institutionally backed -- maintained/stewarded by a named
# org (ASF, FSF, Google/OSI dual-stewarded MIT/BSD forms, Mozilla, academic
# consortia for the EU-origin ones) and seen constantly in the wild outside
# of HF too. This is the "easy to distinguish" bucket: if the SPDX id is in
# here, near-zero ambiguity about what it grants. Anything not in this set
# -- including rarer OSI-approved ids, "other", missing, or one-off custom
# strings -- gets marked CUSTOM below and should get a human look before
# being treated as equivalent.
WELL_KNOWN_LICENSES = {
    "apache-2.0",   # Apache Software Foundation
    "mit",          # MIT
    "bsd-3-clause", # BSD/Regents of UC
    "bsd-2-clause",
    "gpl-3.0",      # FSF
    "gpl-2.0",
    "lgpl-3.0",
    "lgpl-2.1",
    "agpl-3.0",
    "mpl-2.0",      # Mozilla
    "cc0-1.0",      # Creative Commons (public-domain dedication, not a
                     # software copyleft/permissive license, but widely
                     # recognized and unambiguous)
    "unlicense",
    "isc",
}


def is_well_known_license(license_name):
    """True only for the small, unambiguous set above. Everything else --
    rarer OSI-approved ids, HF's literal 'other', missing licenses, and
    AI-specific custom licenses -- is treated as CUSTOM and needs a human
    look rather than being auto-trusted."""
    if license_name is None:
        return False
    return str(license_name).strip().lower() in WELL_KNOWN_LICENSES

# Common HF/AI-specific license tags that carry field-of-use, non-commercial,
# or redistribution restrictions and therefore fail OSD #5/#6 (no
# discrimination against persons or fields of endeavor) even though HF
# lists them as a "license". This is a known/observed-tag list, not a legal
# determination -- worth a human read of the actual license text before
# this becomes a filtering rule.
KNOWN_RESTRICTIVE_AI_LICENSES = {
    "llama2", "llama3", "llama3.1", "llama3.2", "llama3.3", "llama4",
    "gemma", "cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0",
    "cc-by-nc-2.0", "cc-by-nc-3.0", "openrail", "openrail++",
    "bigscience-openrail-m", "bigscience-bloom-rail-1.0", "creativeml-openrail-m",
    "deepfloyd-if-license", "mrl", "cogvideox", "cc-by-nd-4.0",
}

CC_OPEN_ISH = {"cc-by-4.0", "cc-by-3.0", "cc-by-2.0", "cc-by-sa-4.0", "cc-by-sa-3.0"}
# CC-BY / CC-BY-SA are permissive on redistribution/modification, but OSD
# doesn't actually certify content licenses the same way it certifies
# software licenses -- flagged separately as "permissive but not an
# OSI-approved *software* license" rather than lumped into OSI_APPROVED.


def card_data_to_dict(card_data):
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
    """Same resolution order as get_data_from_hf.py: cardData first, then
    the license:<x> tag. Kept identical so exploration numbers match what
    the production pipeline will actually see.

    Some model cards declare multiple licenses as a list (e.g. dual-licensed
    under two SPDX ids) -- normalize that to a single "+"-joined string so
    it stays hashable/comparable everywhere else in this script. This is an
    exploration-only normalization; the production pipeline should decide
    deliberately whether to keep multi-license models as a list, first-only,
    or split into separate rows.
    """
    license_name = card_data.get("license")
    if isinstance(license_name, list):
        license_name = "+".join(sorted(str(x) for x in license_name)) if license_name else None
    if license_name:
        return license_name
    for tag in tags or []:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def classify_license(license_name):
    if license_name is None:
        return "MISSING"
    key = str(license_name).strip().lower()
    if key in OSI_APPROVED:
        return "OSI_APPROVED"
    if key in KNOWN_RESTRICTIVE_AI_LICENSES:
        return "RESTRICTIVE_AI_LICENSE"
    if key in CC_OPEN_ISH:
        return "PERMISSIVE_CONTENT_LICENSE(non-OSI)"
    if key == "other":
        return "OTHER(unspecified)"
    return "UNCLASSIFIED"


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return sorted_vals[idx]


def main():
    print(f"Pulling {SAMPLE_SIZE:,} models, sort={SORT!r} (exploration only)")
    t0 = time.time()

    records = []
    license_counter = Counter()
    class_counter = Counter()
    well_known_counter = Counter()
    downloads = []

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i, model in enumerate(
            api.list_models(sort=SORT, expand=EXPAND_FIELDS, limit=SAMPLE_SIZE)
        ):
            tags = model.tags or []
            card_data = card_data_to_dict(model.card_data)
            lic = resolve_license(card_data, tags)
            dl = model.downloads_all_time or 0
            well_known = is_well_known_license(lic)

            license_counter[lic] += 1
            class_counter[classify_license(lic)] += 1
            well_known_counter["WELL_KNOWN" if well_known else "CUSTOM_OR_UNRECOGNIZED"] += 1
            downloads.append(dl)

            f.write(json.dumps({
                "id": model.id,
                "license": lic,
                "is_well_known_license": well_known,
                "downloads_all_time": dl,
                "created_at": model.created_at.isoformat() if model.created_at else None,
            }, ensure_ascii=False) + "\n")

            if (i + 1) % 10_000 == 0:
                print(f"  ...{i+1:,} pulled ({time.time()-t0:.0f}s)")

    dt = time.time() - t0
    n = len(downloads)
    downloads.sort()

    print()
    print(f"Pulled {n:,} records in {dt:.1f}s")
    print(f"Saved raw sample to {OUT_PATH}")
    print()

    print("=" * 60)
    print("DOWNLOAD COUNT DISTRIBUTION (downloads_all_time)")
    print("=" * 60)
    for p in (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0):
        print(f"  p{p*100:>5.1f}: {percentile(downloads, p):>12,}")
    zero_dl = sum(1 for d in downloads if d == 0)
    print(f"\n  Models with 0 downloads_all_time: {zero_dl:,} ({zero_dl/n:.1%})")
    for floor in (0, 1, 10, 100, 1_000, 10_000):
        kept = sum(1 for d in downloads if d >= floor)
        print(f"  If floor >= {floor:>7,}: keeps {kept:>7,} / {n:,} ({kept/n:.1%})")

    print()
    print("=" * 60)
    print(f"LICENSE FIELD: {len(license_counter):,} distinct values")
    print("=" * 60)
    missing = license_counter.get(None, 0)
    print(f"  No resolvable license (cardData + tag both empty): {missing:,} ({missing/n:.1%})")
    print()
    print("  Top 30 license values by count:")
    for lic, cnt in license_counter.most_common(30):
        label = lic if lic is not None else "<NONE>"
        print(f"    {label:<35} {cnt:>7,}  ({cnt/n:.1%})")

    print()
    print("=" * 60)
    print("WELL-KNOWN vs. CUSTOM/UNRECOGNIZED (simple two-way split)")
    print("=" * 60)
    print(f"  Well-known set: {sorted(WELL_KNOWN_LICENSES)}")
    for cls, cnt in well_known_counter.most_common():
        print(f"    {cls:<28} {cnt:>7,}  ({cnt/n:.1%})")

    print()
    print("=" * 60)
    print("ROUGH OSD / OSAID-ADJACENT CLASSIFICATION (see script docstring for caveats)")
    print("=" * 60)
    for cls, cnt in class_counter.most_common():
        print(f"    {cls:<38} {cnt:>7,}  ({cnt/n:.1%})")

    print()
    print("NOTE: OSI_APPROVED here means the SPDX id is on OSI's approved list for")
    print("SOFTWARE licenses. It does NOT mean the model satisfies the Open Source")
    print("AI Definition (OSAID) -- that additionally requires published Data")
    print("Information and training Code under OSI-approved terms, which the")
    print("`license` field cannot tell us. That would require parsing model card")
    print("bodies / linked repos, out of scope for this pass.")


if __name__ == "__main__":
    main()
