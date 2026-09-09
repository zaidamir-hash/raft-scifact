# AGENTS.md — persistent project instructions

Codex reads this file automatically in every session in this repo. This
file is the single source of truth for this project. If anything in a
task prompt conflicts with this file, this file wins. Do not infer,
assume, or "improve" scope beyond what is written here. If something is
ambiguous, ask before generating code.

## One-sentence description
Fine-tune a small dense retriever (E5-small-v2) on the SciFact scientific
claim-verification dataset, compare full fine-tuning vs LoRA vs baselines,
and measure whether better retrieval actually produces better final answers
in a fixed RAG pipeline. The generator/LLM is NEVER fine-tuned — only the
retriever is trained. This isolates retrieval quality as the only variable.

## Fixed facts (do not change without explicit instruction)
- Base model: intfloat/e5-small-v2 (~33M params, 512 token max, correct
  query/passage prefixes must be applied)
- Dataset: SciFact (via BEIR) — corpus ~5,183 passages, ~1,109 queries
- Domain: scientific claim verification. No other domain, no Pakistan-
  specific data, no multilingual/Urdu component. English only, SciFact only.
- Compute: free Google Colab only. No paid APIs, no paid vector DB, no
  local GPU assumed. Every notebook must be independently restartable and
  checkpoint to Google Drive (Colab sessions can disconnect at any time).
- Generator/LLM stays frozen across ALL experiments. Never fine-tune it.
  Only the retriever is trained. This is non-negotiable — if a task seems
  to require touching the generator, stop and flag it instead of doing it.
- Split rule: Preserve SciFact's official test set UNCHANGED. Create the
  validation split only from the official training set, grouping claims
  that share relevant source documents so those groups cannot cross the
  train/validation boundary (claims linked to multiple documents are
  assigned as one group). Never randomly split query-passage pairs. This
  protects against leakage while staying comparable to published
  SciFact/BEIR results.
- Dataset annotations: use the BEIR SciFact representation (corpus +
  queries + qrels) for retrieval evaluation. Original SciFact annotations
  (SUPPORTS/REFUTES verdicts, evidence sentences/rationales) may also be
  used for verdict labels and evidence rationales in Stage 7 — this is
  still the same SciFact dataset, not an additional one.
- Frozen generator: Qwen2.5-1.5B-Instruct (fp16, no quantization needed).
  Record the exact model revision/commit hash used at the time Stage 7
  begins, and never change it mid-experiment. Fallback if it proves
  unreliable at producing valid structured JSON output: Llama-3.2-3B-
  Instruct (fp16). Only switch to the fallback after documenting why the
  primary failed — never silently swap models.
- Generator output contract (must be fixed before Stage 7): the generator
  must return structured output:
  {"verdict": "SUPPORTS | REFUTES | INSUFFICIENT_EVIDENCE",
   "explanation": "...", "citations": ["document_id"]}
- Hybrid retrieval method: use Reciprocal Rank Fusion (RRF) to combine
  BM25 and dense rankings, with a fixed, documented fusion constant
  selected before final test evaluation (not tuned on test data).
- "Independently restartable" means: each notebook must be restartable
  from saved input artifacts and checkpoints in Google Drive. A notebook
  may depend on outputs from an earlier stage, but must NOT depend on an
  earlier Colab runtime remaining active.

## The 8 pipeline stages (in order — do not skip or merge stages)
1. Environment + data prep (SciFact download, corpus/query/qrels structure,
   leakage-safe train/val/test split)
2. Baselines: BM25 (rank_bm25) + pretrained E5-small-v2, no training
3. Full fine-tuning with in-batch negatives (MultipleNegativesRankingLoss)
4. Full fine-tuning with mined hard negatives (BM25-mined and/or dense-
   mined, filtered for false negatives)
5. LoRA fine-tuning (via PEFT + Sentence Transformers), same data as
   stage 3/4's best config, for a fair comparison
6. Data-efficiency experiments: repeat best training config at 10%, 25%,
   50%, 100% of training data
7. RAG integration + evaluation: same frozen generator, same prompt, same
   chunking, same top-k, swap only the retriever (pretrained / fine-tuned
   / LoRA / hybrid BM25+dense). Include an oracle condition (gold passages
   given directly to the generator, no retrieval) to separate retrieval
   failure from generation failure.
8. Failure analysis + demo + writeup prep (failure taxonomy, plots, a
   small Gradio demo, results tables for the report/paper)

## Mandatory baselines/comparisons (every evaluation must include all of these)
- BM25 (lexical)
- Pretrained E5-small-v2 (untouched)
- Fully fine-tuned E5 (in-batch negatives)
- Fully fine-tuned E5 (hard negatives)
- LoRA fine-tuned E5
- BM25 + best dense model (hybrid)
- Oracle/gold passages (stage 7 only)

## Experimental procedure (mandatory before any training stage)
Before every training stage, create and save a config file recording:
random seed, batch size, effective batch size, learning rate, number of
epochs, warm-up ratio, max sequence length, optimizer, evaluation
frequency, early-stopping rule, best-checkpoint metric, LoRA rank/alpha/
dropout/target modules (if applicable), and hard-negative count + mining
method (if applicable).

Development must follow this order and not skip steps:
smoke test -> one-seed pilot -> select settings using VALIDATION data only
-> freeze the configuration -> run the final 3-seed run -> evaluate on
test exactly once. Do not repeatedly inspect test performance while still
adjusting the model or hyperparameters.

Scope/compute note: the full design (in-batch x3 seeds, hard-negative x3
seeds, LoRA x3 seeds, 4 data fractions x3 seeds) is at least ~21 training
runs. Use single-seed pilot runs during development, and only run 3 seeds
once a configuration is frozen. Cache hard negatives, tokenized datasets,
and evaluation embeddings to avoid recomputing them across runs.

## Metrics
- Retrieval: Recall@5, Recall@10, MRR@10, nDCG@10, precision@k
- RAG/end-to-end: answer correctness, citation precision/recall,
  faithfulness to retrieved context, unsupported-claim rate
- Efficiency: trainable parameters, peak GPU memory, training time,
  checkpoint/adapter size
- Report confidence intervals via bootstrap resampling; important
  variants should run with 3 random seeds

## Tech stack (use only these unless a stage explicitly requires more)
PyTorch, Hugging Face Transformers + Datasets, Sentence Transformers,
PEFT (LoRA), FAISS (CPU/GPU), rank_bm25, MTEB or Sentence Transformers'
InformationRetrievalEvaluator, pandas/seaborn for analysis, Gradio for
the final demo only. No paid services of any kind.

## Repository structure (create this scaffold in Stage 1, if not present)
README.md
AGENTS.md
paper/
data/
  datasheet.md
src/
  prepare_data.py
  mine_negatives.py
  train.py
  evaluate_retrieval.py
  evaluate_rag.py
  analyze_failures.py
configs/
notebooks/
demo/
results/
model_card.md
requirements.txt

## Git/GitHub workflow
- Commit at the end of each completed stage, not mid-stage. Commit message
  format: "Stage N: <short description>" (e.g. "Stage 1: data prep and
  leakage-safe split").
- Never commit raw model checkpoints, embeddings, or large data files to
  GitHub — those go to Google Drive. Use .gitignore for checkpoints/,
  data/raw/, and any file over a few MB.
- Update the "Current status" section below at the end of every stage,
  in the same commit as that stage's work.
- requirements.txt must be kept current whenever a new dependency is added.

## What NOT to do
- Do not fine-tune the generator/LLM at any stage
- Do not introduce a different base retriever model without being told to
- Do not use a different dataset or add Pakistan-specific data
- Do not skip the leakage-safe split
- Do not merge two pipeline stages into one task
- Do not silently change hyperparameters, loss functions, or eval metrics
  from what's specified — flag it and ask instead
- Do not assume paid compute/APIs are available
- Do not commit large binaries (checkpoints, embeddings, raw data) to git

## Current status
Stages 1-4 complete: BEIR SciFact data preparation and the leakage-safe split
are verified; the BM25 and untouched pretrained E5-small-v2 baselines are
evaluated; and the Stage 3 in-batch and Stage 4 hard-negative full-fine-tuning
configurations are frozen. Stage 4 used one BM25-mined and one Stage 3
seed-42-dense-mined negative per positive, full-corpus candidate search,
qrels/duplicate exclusion, a fixed 0.05 cosine-margin filter, fresh pretrained
initialization, learning rate 5e-6, evaluation every 5 steps, and patience-2
early stopping. Its final validation-only run used seeds 42, 43, and 44. Mean
retrieval metrics with 95% percentile-bootstrap confidence intervals over
seeds were: Recall@5 0.754270 [0.751515, 0.759780], Recall@10 0.795592
[0.795592, 0.795592], Precision@5 0.165840 [0.165289, 0.166942], Precision@10
0.087603 [0.087603, 0.087603], MRR@10 0.689059 [0.688508, 0.689886], and
nDCG@10 0.701914 [0.701469, 0.702551]. The checksum audit verified distinct
checkpoints, loss histories, and embedding caches with no resumed runs. At
this data scale, mined hard negatives were statistically equivalent to
Stage 3 in-batch negatives: confidence intervals overlapped and differences
were small and mixed across metrics. The held-out test split was not loaded or
evaluated during Stage 4 final validation. Stage 5 has not started.
