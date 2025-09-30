# QAFD-RAG: Usage Guide

## 1. Installation

Make sure you have Python 3.8+ and install the required dependencies using the provided requirements file:

```
pip install -r requirements.txt
```

## 2. Basic Usage

### 2.1. Select LLM and Set API Key

Before using QAFD-RAG, you need to select your LLM provider and set the corresponding API key. For example, to use OpenAI:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
```

### 2.2. Import and Initialization

```python
from QAFD_RAG import QAFD_RAG, QueryParam

# Initialize the RAG system (choose your storage backends as needed)
rag = QAFD_RAG(
    working_dir="./QAFDRAG_cache",  # Directory for cache and storage files
    kv_storage="JsonKVStorage",     # Key-value storage backend
    vector_storage="NanoVectorDBStorage",  # Vector DB backend
    graph_storage="NetworkXStorage",       # Graph storage backend
    # You can customize other parameters if needed
)
```

### 2.3. Ingest Documents

You can insert a single document or a list of documents (strings):

```python
rag.insert("Your document content here.")

# or

rag.insert([
    "First document content.",
    "Second document content."
])
```

This will:
- Chunk the documents by token size
- Extract entities and relationships
- Build/update the knowledge graph and vector DB

### 2.4. Querying

To ask a question, use the `query` method. You can control the retrieval mode and other options via `QueryParam`:

```python
param = QueryParam(mode="hybrid")  # Options: "local", "global", "hybrid", "combined"
answer = rag.query("What is the relationship between X and Y?", param)
print(answer)
```

**Modes:**
- `"local"`: Focuses on fine-grained, entity-level context
- `"global"`: Focuses on high-level, thematic context
- `"hybrid"`: Combines both local and global

You can also adjust other parameters in `QueryParam` (see code for all options).

## 3. Example End-to-End Script

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

from QAFD_RAG import QAFD_RAG, QueryParam

rag = QAFD_RAG(working_dir="./QAFDRAG_cache")
rag.insert("The Eiffel Tower is located in Paris and is a famous landmark.")
answer = rag.query("Where is the Eiffel Tower?", QueryParam(mode="local"))
print(answer)
```

## 4. Illustration

![Graph-Based RAG Comparison](synthetic/graph-based%20rag%20comp.png)

![QAFD-RAG Illustration](question_illustration.png)

Figure 2: Two-stage architecture of the QAFD-RAG framework. The indexing stage constructs a domain-specific knowledge graph by extracting entities, relations, and document-level structure from raw corpus data. The query stage processes an incoming user query in several steps: (1) keyword extraction identifies query-relevant dual-level keywords; (2) a query-aware flow diffusion algorithm propagates selected seed nodes over the graph based on semantic and structural signals; (3) clusters are collected for each seed node, and each cluster is summarized into natural language; and (4) cluster summaries, along with the original query, are passed to a language model for final response generation.

## 5. Dataset

We evaluate using domain splits from the UltraDomain QA dataset, including Agriculture, Biology, Cooking, History, Legal, etc.. Dataset source: [UltraDomain on Hugging Face](https://huggingface.co/datasets/TommyChien/UltraDomain).

## 6. Numerical Results

Comparison of QAFD-RAG and baseline methods across five evaluation dimensions: Comprehensiveness, Diversity, Logicality, Relevance, and Coherence. Each score (ranging from 0 to 100) is the average of five independent evaluations conducted using GPT-4o. The best score in each row is highlighted in bold.

| Dataset | Metric | GraphRAG | LightRAG | RAPTOR | HippoRAG | QAFD-RAG |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Agriculture | Comprehensiveness | 87.30 (±4.46) | 83.65 (±5.97) | 83.32 (±8.67) | 82.51 (±5.14) | **89.93 (±3.36)** |
| Agriculture | Diversity         | 82.85 (±4.73) | 77.71 (±7.34) | 76.65 (±12.08) | 76.26 (±9.23) | **84.95 (±4.26)** |
| Agriculture | Logicality        | 90.80 (±5.84) | 88.85 (±3.76) | 89.54 (±3.63) | 88.84 (±3.26) | **92.10 (±2.53)** |
| Agriculture | Relevance         | 94.01 (±6.62) | 93.55 (±4.16) | 94.56 (±3.41) | 94.09 (±2.48) | **95.67 (±3.28)** |
| Agriculture | Coherence         | 90.08 (±3.23) | 88.67 (±2.57) | 89.47 (±3.01) | 88.79 (±2.73) | **92.00 (±1.62)** |
| Biology     | Comprehensiveness | 85.76 (±10.80) | 83.92 (±4.13) | 83.57 (±6.20) | 83.07 (±4.15) | **89.44 (±3.92)** |
| Biology     | Diversity         | 81.05 (±10.39) | 78.28 (±6.46) | 77.10 (±9.41) | 76.91 (±7.25) | **85.13 (±4.11)** |
| Biology     | Logicality        | 88.94 (±11.70) | 88.40 (±4.24) | 88.33 (±5.62) | 88.07 (±3.69) | **91.19 (±4.20)** |
| Biology     | Relevance         | 93.00 (±12.50) | 93.31 (±5.42) | 93.62 (±6.42) | 93.62 (±3.53) | **95.05 (±4.71)** |
| Biology     | Coherence         | 88.57 (±11.10) | 88.09 (±2.86) | 88.67 (±4.21) | 88.52 (±2.90) | **91.33 (±2.59)** |
| Cooking     | Comprehensiveness | 86.23 (±7.00) | 82.11 (±7.56) | 83.15 (±6.99) | 82.52 (±4.55) | **89.25 (±3.82)** |
| Cooking     | Diversity         | 79.10 (±8.72) | 74.97 (±9.46) | 75.30 (±9.79) | 74.13 (±7.25) | **83.42 (±5.25)** |
| Cooking     | Logicality        | 90.79 (±2.68) | 87.49 (±5.82) | 88.52 (±5.12) | 88.40 (±3.20) | **91.35 (±2.73)** |
| Cooking     | Relevance         | 95.14 (±2.91) | 92.27 (±5.88) | 93.59 (±5.48) | 93.71 (±2.66) | **95.45 (±2.83)** |
| Cooking     | Coherence         | 90.63 (±1.75) | 87.98 (±4.09) | 88.83 (±3.87) | 88.57 (±2.61) | **91.58 (±2.04)** |
| History     | Comprehensiveness | 84.18 (±10.30) | 82.24 (±6.04) | 82.08 (±7.43) | 80.45 (±6.95) | **87.75 (±3.96)** |
| History     | Diversity         | 78.88 (±9.97) | 76.42 (±7.51) | 75.59 (±9.75) | 74.61 (±8.22) | **83.14 (±4.40)** |
| History     | Logicality        | 88.18 (±11.63) | 86.98 (±5.75) | 87.67 (±4.63) | 86.37 (±6.36) | **90.04 (±3.93)** |
| History     | Relevance         | 92.35 (±13.51) | 92.18 (±6.17) | 92.94 (±4.93) | 91.54 (±8.22) | **93.77 (±6.25)** |
| History     | Coherence         | 88.40 (±10.71) | 87.18 (±4.24) | 88.13 (±3.58) | 86.97 (±4.77) | **90.55 (±2.37)** |
| Legal       | Comprehensiveness | 84.96 (±9.72) | 79.63 (±11.25) | 81.43 (±9.67) | 82.23 (±8.85) | **86.19 (±5.86)** |
| Legal       | Diversity         | **78.74 (±10.28)** | 67.63 (±11.86) | 64.97 (±13.91) | 64.28 (±11.31) | 77.14 (±7.42) |
| Legal       | Logicality        | 88.67 (±8.58) | 86.06 (±7.47) | 88.34 (±6.54) | 88.56 (±7.64) | **90.06 (±5.06)** |
| Legal       | Relevance         | 91.01 (±10.90) | 90.77 (±10.17) | 93.29 (±9.18) | 93.60 (±9.45) | **93.30 (±9.66)** |
| Legal       | Coherence         | 88.44 (±5.77) | 86.12 (±6.10) | 87.70 (±6.16) | 87.95 (±6.39) | **89.99 (±3.35)** |
---

**Acknowledgments:** The development of QAFD-RAG for question answering (QA) tasks utilizes techniques from [GraphRAG](https://github.com/microsoft/graphrag), [LightRAG](https://github.com/HKUDS/LightRAG), and notably [PathRAG](https://github.com/BUPT-GAMMA/PathRAG). Please refer to their respective repositories for technical details.
