# hf-model-data-analysis
A pipeline for downloading and analyzing model matada from Hugging Face


# TODO
How do we want to address projects with multiple licenses?
    See https://huggingface.co/TaichuAI/ZDTaichu5.0-9B/tree/main/LICENSES

We need to address repos where the license is under a license file but is not in the readme or other metadata
    Example https://huggingface.co/facebook/sam3/blob/main/LICENSE

Update: I'm trying to tackle these cases via the `deep_license_resolve.py` with mixed success. 