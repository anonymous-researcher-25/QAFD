# QAFD-RAG: Usage Guide

## 1. Installation

Make sure you have Python 3.8+ and install the required dependencies using the provided requirements file:

```
pip install -r requirements.txt
```

## 2. Basic Usage

### 2.1. Import and Initialization

```python
from QAFDRAG import QAFDRAG, QueryParam

# Initialize the RAG system (choose your storage backends as needed)
rag = QAFDRAG(
    working_dir="./QAFDRAG_cache",  # Directory for cache and storage files
    kv_storage="JsonKVStorage",     # Key-value storage backend
    vector_storage="NanoVectorDBStorage",  # Vector DB backend
    graph_storage="NetworkXStorage",       # Graph storage backend
    # You can customize other parameters if needed
)
```

### 2.2. Ingest Documents

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

### 2.3. Querying

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
from QAFDRAG import QAFDRAG, QueryParam

rag = QAFDRAG(working_dir="./QAFDRAG_cache")
rag.insert("The Eiffel Tower is located in Paris and is a famous landmark.")
answer = rag.query("Where is the Eiffel Tower?", QueryParam(mode="local"))
print(answer)
```

---

**Acknowledgments:** The development of QAFD-RAG for question answering (QA) tasks utilizes techniques from [GraphRAG](https://github.com/microsoft/graphrag), [LightRAG](https://github.com/HKUDS/LightRAG), and notably [PathRAG](https://github.com/BUPT-GAMMA/PathRAG). Please refer to their respective repositories for technical details.
