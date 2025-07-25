# QAFD_snow

This directory contains the main **Query-Aware Flow Diffusion (QAFD)** pipeline for text-to-SQL generation on Snowflake-like databases.

## Directory Structure

```
QAFD_snow/
├── GraphReasoning/
│   ├── CoFD_snowflake_d_v1.py    # Main pipeline: db summary, graph, path generation
│   ├── ... (supporting modules)
│   └── results_snow_d_v1/        # Output: summaries, graphs, paths
└── spider-agent-snow/
    ├── path_agent.py             # SQL Agent runner
    └── examples/                 # Test queries (JSONL)
```

## Pipeline Overview

**1. GraphReasoning**
- **CoFD_snowflake_d_v1.py** - Extracts Snowflake db summary (table/column/PK-FK) - Builds schema graph - Optionally enhances graph via LLM - Decomposes queries and uses flow diffusion to extract relevant schema subgraphs - Saves all intermediate results
- **Outputs:**
  - `results_snow_d_v1/[db_name]_db_summary.json`
  - `results_snow_d_v1/[db_name]_init_graph.json`
  - `results_snow_d_v1/[db_name]_[llm]_enhanced_graph.json`
  - `results_snow_d_v1/[instance_id]_[llm]_paths.json`

**2. spider-agent-snow**
- **path_agent.py**: Loads generated schema paths/subgraphs and runs LLM-based SQL generation.
- **examples/**: Contains test queries as JSONL with fields: `instance_id`, `db_id`, `instruction`.

---

## Step-by-Step Usage

1. **Prepare your test queries**
   - Place your test queries as JSONL files in `spider-agent-snow/examples/`, e.g.:
   ```json
   {"instance_id": "sf_local038", "db_id": "SNOWFLAKE_DB", "instruction": "Find all users with..."}
   ```

2. **Run the graph reasoning pipeline**
   ```bash
   cd QAFD_snow/GraphReasoning
   python CoFD_snowflake_d_v1.py --output ./results_snow_d_v1 --base_dir ../spider-agent-snow/examples --model gpt-4o --graph_type enhanced --with_paths True
   ```

3. **Run the SQL Agent**
   ```bash
   cd ../spider-agent-snow
   python path_agent.py gpt-4o ADD_SCHEMA_PATH
   ```

---

## Requirements

- Python 3.8+
- Required packages and dependencies as specified in requirements.txt

---

## Output Files

After running the complete pipeline, you'll find:
- Database summaries and schema graphs in `GraphReasoning/results_snow_d_v1/`
- Generated SQL queries and evaluation results in `spider-agent-snow/results/`
