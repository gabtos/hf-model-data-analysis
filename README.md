# hf-model-data-analysis
A pipeline for downloading and analyzing model matada from Hugging Face
The current version only looks at models with more than 100 downloads.



## Model hierarchy analysis
Which models are base models vs. derivatives (fine-tunes, quantizations, adapters, merges), and whose base models the Hub's derivatives trace back to.
Run `python3 model_lineage_analysis.py` (outputs go to `model_lineage/output/`), or open `model_hierarchy_analysis.ipynb` for the definitions, method, results and caveats. Hand labels and their sources live in `model_lineage/`.
# TODO
How do we want to address projects with multiple licenses?
    See https://huggingface.co/TaichuAI/ZDTaichu5.0-9B/tree/main/LICENSES

We need to address repos where the license is under a license file but is not in the readme or other metadata
    Example https://huggingface.co/facebook/sam3/blob/main/LICENSE

Update: I'm trying to tackle these cases via the `deep_license_resolve.py` with mixed success. 