from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar

import numpy as np

from .utils import EmbeddingFunc

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

T = TypeVar("T")


@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid"] = "hybrid"
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    top_k: int =40
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 3000
    max_token_for_local_context: int = 5000
    max_source_nodes: int = 40
    min_flow_threshold: float = 0.1
    enable_query_aware_flow_diffusion: bool = True  # Enable query-aware flow diffusion
    alpha: float = 50.0  # Alpha parameter for flow diffusion initialization
    weight_func: Literal["multiply", "add", "original"] = "original"
    return_raw_entities: bool = False  # Return raw entity data as JSON instead of LLM summaries
    return_raw_clusters: bool = False  # Return raw cluster data as JSON instead of LLM summaries
    cluster_json_format: Literal["compact", "minimal", "original", "text"] = "text"  # Format for cluster data
    cluster_text_mode: Literal["detailed", "compact", "minimal"] = "detailed"  # Text format mode when using text format
    use_batch_cluster_summarization: bool = True  # Use batch processing for cluster summarization (more efficient)
    batch_cluster_size: int = 10  # Number of clusters to process in each batch (to avoid token limits)


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
       
        pass

    async def query_done_callback(self):
        
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):

        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    embedding_func: EmbeddingFunc

    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc = None

    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError
    
    async def get_pagerank(self,node_id:str) -> float:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError
    
    async def get_node_in_edges(
        self,source_node_id:str
    ) -> Union[list[tuple[str,str]],None]:
        raise NotImplementedError
    async def get_node_out_edges(
        self,source_node_id:str
    ) -> Union[list[tuple[str,str]],None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def delete_node(self, node_id: str):
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in QAFD_RAG.")