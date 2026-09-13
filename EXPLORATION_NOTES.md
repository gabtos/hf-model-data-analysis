# License/download exploration — technical notes

`explore_licenses.py`. Exploratory only; produces no filtering decision,
no output for the production pipeline. Run with `python3 explore_licenses.py`
(requires `HF_TOKEN` env var, `pip install huggingface_hub`).

## What it does

1. Pulls `SAMPLE_SIZE` (default 50,000) model records via
   `HfApi.list_models(sort="created_at", expand=[...])`.
2. Resolves each model's license: `cardData.license` first, else the
   `license:<x>` tag — identical precedence to `get_data_from_hf.py`, so
   exploration numbers transfer to the production script.
3. Classifies each resolved license on three independent axes and writes
   all of them, plus `id`/`downloads_all_time`/`created_at`, to
   `raw_data/license_exploration_sample.jsonl` (gitignored, regenerated
   per run — not committed, not a deliverable):
   - `is_well_known_license` (bool) — membership in a 13-entry hardcoded
     set of institutionally-backed, unambiguous SPDX ids (Apache-2.0, MIT,
     BSD-2/3, GPL-2/3, LGPL-2.1/3, AGPL-3, MPL-2, CC0-1, Unlicense, ISC).
     Everything else — rarer OSI ids, HF's literal `"other"`, missing,
     AI-specific custom licenses — is `CUSTOM_OR_UNRECOGNIZED`.
   - `permissive_or_copyleft` — `PERMISSIVE` / `COPYLEFT_WEAK` (LGPL, MPL,
     EPL, EUPL, CC-BY-SA) / `COPYLEFT_STRONG` (GPL, AGPL, OSL) /
     `NOT_APPLICABLE` (no classic-OSS license present, so the axis doesn't
     apply — not lumped into either bucket).
   - an OSD/OSAID-adjacent bucket (printed, not written per-record):
     `MISSING` / `OSI_APPROVED` (full ~28-id OSI SPDX list observed on the
     Hub) / `RESTRICTIVE_AI_LICENSE` (Llama/Gemma/OpenRAIL/CC-NC family —
     fails OSD §5/§6, field-of-use or persons/groups discrimination) /
     `PERMISSIVE_CONTENT_LICENSE(non-OSI)` (CC-BY family) / `OTHER` /
     `UNCLASSIFIED`.
4. Prints percentile table of `downloads_all_time` and keep-rate at
   candidate floors (0/1/10/100/1k/10k), for later floor discussion —
   no floor is applied to the output.

## Sampling method and why

`list_models()` with no `sort` arg orders by `trendingScore` (verified by
decoding the pagination cursor), which would bias any license census
toward currently-popular models. `sort="created_at"` avoids that
popularity bias; it still isn't a uniform random sample of the whole Hub
(recency-weighted by construction), which is a known limitation worth
flagging before treating percentages as Hub-wide truth.

## Throughput / rate limits (measured 2026-09-13, authenticated token)

- `list_models()` paginates server-side at ~5,000 records/request in
  testing — 50,000 records = ~8-10s wall clock, single-digit HTTP requests.
- HF enforces `1000 requests / 300s` (fixed window) on the `api` route
  (from the `RateLimit-Policy` response header). An unbounded, unpaced
  full-Hub enumeration (millions of models, unknown page count) tripped a
  429 in testing after a few hundred thousand records. 50k is a small
  fraction of the budget; anything approaching the full Hub needs
  explicit backoff/pacing, which this script does not implement (it caps
  at `SAMPLE_SIZE` instead).

## Known limitations (not fixed here, deliberately — exploration stage)

- License resolution only reads `cardData`/tags; does not parse model
  card markdown bodies. Models stating a license only in free text won't
  resolve.
- `OSI_APPROVED`/well-known classification is a hardcoded SPDX-id lookup
  against sets built from *observed* Hub values, not the full OSI list —
  it will under-count any real OSI-approved license not yet seen in
  samples.
- OSAID compliance (Data Information + Code disclosure, per the Open
  Source AI Definition v1.0 preamble) cannot be assessed from the
  `license` field at all — an `OSI_APPROVED` license on the weights says
  nothing about whether training data/code were disclosed. Not measured
  here; would require parsing card bodies / linked repos.
- Multi-license models (`cardData.license` as a list) are normalized to a
  single `"+"`-joined string for hashability. The production pipeline
  should decide deliberately how to represent these (keep as list, first
  license only, or split into separate rows) rather than inheriting this
  exploration-only normalization.
