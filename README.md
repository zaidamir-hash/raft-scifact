# RAFT: SciFact Retriever Fine-Tuning

This repository studies retrieval quality for scientific claim verification
with BEIR SciFact. The generator remains frozen; only the E5-small-v2
retriever is trained in later stages.

## Stage 1: prepare SciFact

The Stage 1 configuration is recorded in
`configs/stage1_data_prep.json`. To download and prepare data locally:

```bash
python src/prepare_data.py
```

For Google Colab, open `notebooks/01_environment_and_data_prep.ipynb` after
cloning this repository into `/content/RAFT`. The notebook mounts Google
Drive and writes both the cached source archive and reusable processed files
under `/content/drive/MyDrive/RAFT/stage1_scifact`.

The processed directory contains one stable-ID corpus, separate query files
for train/validation/test, matching qrels files, a config snapshot, a split
manifest, and machine-readable sanity statistics. Generated data is ignored
by Git and must not be committed.
