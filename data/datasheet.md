# SciFact data sheet

## Source

- Dataset: SciFact, using the BEIR corpus/query/qrels representation
- Download: the public BEIR `scifact.zip` archive recorded in the Stage 1 config
- Language/domain: English scientific claim verification
- Corpus IDs and query IDs: preserved as strings from BEIR

## Stage 1 split

The official BEIR test split is copied in full and is never sampled. The
official training queries are divided with a validation fraction of 15% and
fixed split seed 42. These split settings are permanent data-preparation
settings, not training-run seeds.

For leakage control, training queries are represented as a bipartite graph
with their positively relevant source documents. Queries connected directly
or transitively by shared documents form indivisible components. Whole
components are assigned to validation until the query count is as close as
possible to 15%; individual query-document pairs are never randomly split.

Observed preparation statistics:

- Corpus documents: 5,183
- Official training queries before splitting: 809
- Final train/validation/test queries: 688 / 121 / 300
- Train/validation shared relevant source documents: 0
- Average passage length: 214.63 whitespace-delimited tokens over title + text

The generated `split_manifest.json` records the source archive SHA-256 and
the exact configuration, while `sanity_stats.json` records complete counts.
Generated raw and processed data stay out of Git and should be stored on
Google Drive for Colab runs.
