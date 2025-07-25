# QAFD: Code and Reproducibility Structure

This repository provides code for running and reproducing experiments on Query-Aware Flow Diffusion (QAFD) for:
- Question Answering with Retrieval-Augmented Generation (RAG)
- Text-to-SQL (natural language to SQL)

---

## Directory Structure

```
QAFD/
├── QAFD-RAG/                    # Question answering (RAG) code and experiments
├── QAFD_text2sql/
│   ├── QAFD_snow/              # Text-to-SQL code and experiments
│   ├── baselines/              # Baseline methods for QA and T2SQL
│   └── ...
├── README.md
└── .gitignore
```

---

## Usage

To **run or reproduce experiments**, refer to the detailed `README.md` provided in each subfolder:

- [`QAFD-RAG/README.md`](./QAFD-RAG/README.md):  
  Instructions for QAFD-RAG question answering, setup, and experiment reproduction.

- [`QAFD_text2sql/QAFD_snow/README.md`](./QAFD_text2sql/QAFD_snow/README.md):  
  Instructions for QAFD-based text-to-SQL experiments, schema extraction, pipeline usage, and SQL generation.

- [`QAFD_text2sql/baselines/README.md`](./QAFD_text2sql/baselines/README.md):  
  Running and evaluating baselines for text2sql tasks.

---
