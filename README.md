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

## Stage 2: evaluate untouched retrieval baselines

After Stage 1 artifacts exist, run both BM25 and pretrained E5-small-v2 on
the fixed validation and test splits:

```bash
python src/evaluate_retrieval.py
```

The evaluator writes `results/stage2_baselines.csv` and its run metadata JSON.
Normalized pretrained corpus embeddings, corpus-ID order, and the exact FAISS
index are cached under `embeddings/pretrained_e5_small_v2/` and ignored by Git.
The Colab wrapper at `notebooks/02_retrieval_baselines.ipynb` stores those
large reusable artifacts on Google Drive.
