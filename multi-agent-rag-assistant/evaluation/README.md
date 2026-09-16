# RAG Evaluation

This folder contains the ground-truth dataset used to evaluate retrieval independently from LLM generation.

## Why evaluate retrieval separately?

A RAG answer can fail for two different reasons:

1. **Retrieval failure**: the relevant evidence is not present in the retrieved Top-K chunks.
2. **Generation failure**: the evidence is present, but the LLM ignores, misinterprets, or contradicts it.

We therefore measure retrieval before evaluating the final answer.

## Dataset format

Each example in `rag_eval_dataset.json` contains:

- `id`: stable identifier.
- `question`: realistic user question.
- `expected_answer`: short reference answer used for human inspection and later generation evaluation.
- `relevant_evidence`: one or more pieces of text that a retrieved chunk should contain. These should be copied from the source documents rather than invented.
- `source`: expected source metadata when known (filename/page).

The initial file contains templates only. Replace them with 20-30 questions grounded in the PDFs in `data/documents/` before reporting metrics.

## Retrieval metrics

For a query with a labelled set of relevant evidence:

- **Hit@K**: 1 if at least one relevant item appears in Top-K, otherwise 0.
- **Precision@K**: relevant retrieved items / K.
- **Recall@K**: relevant retrieved items / total labelled relevant items.
- **Reciprocal Rank**: `1 / rank` of the first relevant result. MRR is its mean across queries.

Also record retrieval latency. Do not select hyperparameters using Recall@K alone: increasing K can improve recall while adding irrelevant context and degrading final generation.

## Experiment plan

Start from the current baseline (`chunk_size=512`, `chunk_overlap=64`, `k=4`, MMR, `all-MiniLM-L6-v2`). Change one dimension at a time before running a small focused grid. Candidate values:

- chunk size: 256, 512, 1024
- overlap: 32, 64, 128
- k: 3, 5, 10
- retrieval: similarity vs MMR

Only compare embedding models after the evaluation harness is stable, because changing the embedding model requires rebuilding the FAISS index.
