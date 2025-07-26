import asyncio
import json
import re
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from collections import Counter, defaultdict
import warnings
import tiktoken
import time
import csv
import networkx as nx
import random
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


class QueryAwareWeightedFlowDiffusion:
    def __init__(self, graph, source_node, target_node, confidence=0.5, epsilon=0.05, 
                 node_embeddings=None, subquery_embedding=None, weight_func=None):
        """
        Initialize the Query-Aware Weighted Flow Diffusion algorithm.
        IDENTICAL to original except for optional embedding parameters.
        """
        self.graph = graph
        self.source = source_node
        self.target = target_node
        self.confidence = max(0.0, min(1.0, confidence))
        self.epsilon = epsilon
        self.mass = defaultdict(float)
        self.x = defaultdict(float)
        self.sink_capacity = defaultdict(float)
        
        # ONLY addition: query-aware components
        self.node_embeddings = node_embeddings or {}
        # Use actual embedding dimension or default to 1536
        embedding_dim = len(subquery_embedding) if subquery_embedding else 1536
        self.subquery_embedding = subquery_embedding or [0.0] * embedding_dim
        self.weight_func = weight_func
        self.edge_weights_cache = {}
    
    def cosine_similarity(self, vec1, vec2):
        """Fast cosine similarity calculation."""
        if not vec1 or not vec2:
            return 0.0
        
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            mag1 = sum(a * a for a in vec1) ** 0.5
            mag2 = sum(b * b for b in vec2) ** 0.5
            
            if mag1 == 0 or mag2 == 0:
                return 0.0
            
            similarity = dot_product / (mag1 * mag2)
            return max(0.0, (similarity + 1.0) / 2.0)  # Normalize to [0, 1]
        except:
            return 0.0
    
    def get_edge_weight(self, node1, node2):
        """
        ONLY DIFFERENCE: Get smart edge weight instead of original weight.
        This is the ONLY query-aware modification.
        """
        # Use cache if available
        cache_key = (node1, node2)
        if cache_key in self.edge_weights_cache:
            return self.edge_weights_cache[cache_key]
        
        # Get original edge weight (IDENTICAL to original)
        edge_data = self.graph[node1][node2]
        original_weight = edge_data.get('weight', 0.0)  # SAME as original: default to 0.0
        
        # If no embeddings available, return original weight (IDENTICAL behavior)
        if not self.node_embeddings or not self.subquery_embedding:
            self.edge_weights_cache[cache_key] = original_weight
            return original_weight
        
        # ONLY NEW PART: Apply query-aware weighting
        if original_weight <= 0:  # If original weight is 0, keep it 0
            self.edge_weights_cache[cache_key] = 0.0
            return 0.0
        
        # Get node embeddings
        node1_emb = self.node_embeddings.get(node1, [0.0] * len(self.subquery_embedding))
        node2_emb = self.node_embeddings.get(node2, [0.0] * len(self.subquery_embedding))
        
        # Calculate query similarity factor
        node1_query_sim = self.cosine_similarity(node1_emb, self.subquery_embedding)
        node2_query_sim = self.cosine_similarity(node2_emb, self.subquery_embedding)
        
        if self.weight_func == "multiply":
            smart_weight = original_weight * node1_query_sim * node2_query_sim
        elif self.weight_func == "add":
            smart_weight = (original_weight + node1_query_sim + node2_query_sim) / 3.0
        else:
            # Apply modest boost: original_weight * (1 + small_boost)
            query_factor = (node1_query_sim + node2_query_sim) / 2.0
            smart_weight = original_weight * (1.0 + query_factor * 0.5)

        # Cache and return
        self.edge_weights_cache[cache_key] = smart_weight
        return smart_weight
        
    def initialize(self, alpha=50, use_node_degree=True):
        """
        IDENTICAL to original initialization.
        No query-aware modifications here for speed.
        """
        # Set sink capacity for all nodes (IDENTICAL)
        for node in self.graph.nodes():
            if use_node_degree:
                # Set sink capacity to node degree (number of connections)
                self.sink_capacity[node] = self.graph.degree(node)
            else:
                # Set sink capacity to 1 (original behavior)
                self.sink_capacity[node] = 1
        
        # Set all masses to 0 initially (IDENTICAL)
        for node in self.graph.nodes():
            self.mass[node] = 0
        
        # Set source mass - boost based on confidence (IDENTICAL)
        try:
            path = nx.shortest_path(self.graph, self.source, self.target)
            total_sink_on_path = sum(self.sink_capacity[node] for node in path)
            confidence_boost = 1.0 + self.confidence
            self.mass[self.source] = alpha * total_sink_on_path * confidence_boost
        except nx.NetworkXNoPath:
            avg_sink = sum(self.sink_capacity.values()) / len(self.sink_capacity)
            confidence_boost = 1.0 + self.confidence
            self.mass[self.source] = alpha * avg_sink * len(self.graph.nodes()) / 10 * confidence_boost
    
    def push(self, node):
        """
        ALMOST IDENTICAL to original push operation.
        ONLY change: use get_edge_weight() instead of direct weight access.
        """
        # Get neighbors (IDENTICAL)
        neighbors = list(self.graph.neighbors(node))
        if not neighbors:
            return False
        
        # Calculate w_i - ONLY CHANGE: use get_edge_weight()
        w_i = 0
        for neighbor in neighbors:
            weight = self.get_edge_weight(node, neighbor)  # ONLY LINE CHANGED
            w_i += weight
        
        if w_i == 0:
            return False
        
        # Calculate excess mass (IDENTICAL)
        excess = self.mass[node] - self.sink_capacity[node]
        if excess <= 0:
            return False
        
        # Update x_i (IDENTICAL)
        self.x[node] += excess / w_i
        
        # Update mass at node i (IDENTICAL)
        self.mass[node] = self.sink_capacity[node]
        
        # Distribute excess mass - ONLY CHANGE: use get_edge_weight()
        for neighbor in neighbors:
            w_ij = self.get_edge_weight(node, neighbor)  # ONLY LINE CHANGED
            if w_ij > 0:
                self.mass[neighbor] += excess * w_ij / w_i
        
        return True
    
    def flow_diffusion(self, max_iterations=500):
        """
        IDENTICAL to original flow diffusion algorithm.
        No modifications at all.
        """
        iterations = 0
        pushes = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Find nodes with excess mass (IDENTICAL)
            excess_nodes = [node for node in self.graph.nodes() 
                           if self.mass[node] > self.sink_capacity[node]]
            
            if not excess_nodes:
                break
            
            # Pick a node uniformly at random (IDENTICAL)
            node = random.choice(excess_nodes)
            
            # Apply push operation (IDENTICAL)
            if self.push(node):
                pushes += 1
            
            # Check for convergence periodically (IDENTICAL)
            if iterations % 100 == 0:
                remaining_excess = sum(max(0, self.mass[node] - self.sink_capacity[node]) 
                                     for node in self.graph.nodes())
                if remaining_excess < self.epsilon:
                    break
        
        # logger.info(f"Flow diffusion completed in {iterations} iterations with {pushes} pushes")
        return {node: val for node, val in self.x.items() if val > 0}
    
    def find_furthest_reachable_node(self, diffused_nodes):
        """
        IDENTICAL to original furthest reachable node finder.
        """
        if self.source not in diffused_nodes:
            return None, None
            
        support_nodes = set(diffused_nodes.keys())
        subgraph = self.graph.subgraph(support_nodes)
        
        from collections import deque
        queue = deque([(self.source, 0, [self.source])])
        visited = {self.source}
        furthest_nodes = []
        max_distance = 0
        paths_to_nodes = {self.source: [self.source]}
        
        while queue:
            current_node, distance, path = queue.popleft()
            
            if distance > max_distance:
                max_distance = distance
                furthest_nodes = [current_node]
            elif distance == max_distance and current_node not in furthest_nodes:
                furthest_nodes.append(current_node)
            
            for neighbor in subgraph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    new_distance = distance + 1
                    queue.append((neighbor, new_distance, new_path))
                    paths_to_nodes[neighbor] = new_path
        
        if furthest_nodes:
            chosen_node = max(furthest_nodes, key=lambda n: diffused_nodes.get(n, 0)) if len(furthest_nodes) > 1 else furthest_nodes[0]
            logger.info(f"Furthest reachable node: {chosen_node} at distance {max_distance} from source")
            return chosen_node, paths_to_nodes[chosen_node]
        
        return None, None
    
    def calculate_path_score(self, path, confidence_weight=0.7, flow_weight=0.3):
        """
        IDENTICAL to original path scoring.
        """
        if not path or len(path) < 2:
            return 0.0
        
        # Calculate flow strength (IDENTICAL)
        flow_values = [self.x[node] for node in path if node in self.x and self.x[node] > 0]
        if not flow_values:
            flow_strength = 0.0
        else:
            avg_flow = sum(flow_values) / len(flow_values)
            flow_strength = avg_flow / (avg_flow + 1.0)
        
        # Calculate weighted combined score (IDENTICAL)
        combined_score = (confidence_weight * self.confidence) + (flow_weight * flow_strength)
        return combined_score
    
    def find_path(self):
        """
        IDENTICAL to original find_path method.
        """
        # Initialize (IDENTICAL)
        self.initialize()
        
        # Run flow diffusion (IDENTICAL)
        diffused_nodes = self.flow_diffusion()
        
        # Check if diffusion reached the target (IDENTICAL)
        if self.target not in diffused_nodes:
            logger.info(f"Flow diffusion did not reach target node {self.target}")
            
            if len(diffused_nodes) > 1:
                furthest_node, furthest_path = self.find_furthest_reachable_node(diffused_nodes)
                if furthest_path:
                    score = self.calculate_path_score(furthest_path)
                    logger.info(f"Path found to furthest reachable node: {' -> '.join(furthest_path)} (score: {score:.3f})")
                    return furthest_path, score
            
            fallback_path = [self.source] if self.source in diffused_nodes else None
            fallback_score = self.calculate_path_score(fallback_path) if fallback_path else 0.0
            return fallback_path, fallback_score
        
        # Create subgraph and find path (IDENTICAL)
        support_nodes = set(diffused_nodes.keys())
        subgraph = self.graph.subgraph(support_nodes)
        
        # Reweight edges based on x values (IDENTICAL)
        weighted_subgraph = nx.Graph()
        for u, v, data in subgraph.edges(data=True):
            new_weight = 1.0 / (self.x[u] + self.x[v] + 1e-10)
            weighted_subgraph.add_edge(u, v, weight=new_weight)
        
        # Find shortest path (IDENTICAL)
        try:
            path = nx.shortest_path(weighted_subgraph, self.source, self.target, weight='weight')
            score = self.calculate_path_score(path)
            logger.info(f"Path found: {' -> '.join(path)} (score: {score:.3f})")
            return path, score
        except nx.NetworkXNoPath:
            logger.info(f"No path found from {self.source} to {self.target} in diffused subgraph")
            return None, 0.0


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens: 
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
   
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
   
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        f"({src_id}, {tgt_id})", description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    time.sleep(20)
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())
  
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["entity_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
  
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = []
    for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in ordered_chunks]),
        total=len(ordered_chunks),
        desc="Extracting entities from chunks",
        unit="chunk",
    ):
        results.append(await result)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)
    logger.info("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        ),
        total=len(maybe_nodes),
        desc="Inserting entities",
        unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relationships into storage...")
    all_relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                _merge_edges_then_upsert(
                    k[0], k[1], v, knowledge_graph_inst, global_config
                )
                for k, v in maybe_edges.items()
            ]
        ),
        total=len(maybe_edges),
        desc="Inserting relationships",
        unit="relationship",
    ):
        all_relationships_data.append(await result)

    if not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any entities and relationships, maybe your LLM is not working"
        )
        return None

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                + dp["src_id"]
                + dp["tgt_id"]
                + dp["description"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst



async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:

    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if cached_response is not None:
        return cached_response

    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    if query_param.mode not in ["local", "global", "hybrid", "combined"]:
        logger.error(f"Unknown mode {query_param.mode} in kg_query")
        return PROMPTS["fail_response"]


    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query, examples=examples, language=language)
    result = await use_model_func(kw_prompt, keyword_extraction=True)
    logger.info("kw_prompt result:")
    print(result)
    try:

        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            result = match.group(0)
            keywords_data = json.loads(result)

            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
        else:
            logger.error("No JSON-like structure found in the result.")
            return PROMPTS["fail_response"]


    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e} {result}")
        return PROMPTS["fail_response"]


    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]
    
    # Validate keywords based on mode
    if query_param.mode == "local" and ll_keywords == []:
        logger.warning("low_level_keywords is empty for local mode")
        return PROMPTS["fail_response"]
    elif query_param.mode == "global" and hl_keywords == []:
        logger.warning("high_level_keywords is empty for global mode")
        return PROMPTS["fail_response"]
    elif query_param.mode == "combined" and ll_keywords == [] and hl_keywords == []:
        logger.warning("Both low_level_keywords and high_level_keywords are empty for combined mode")
        return PROMPTS["fail_response"]
    elif query_param.mode == "hybrid" and ll_keywords == [] and hl_keywords == []:
        logger.warning("Both low_level_keywords and high_level_keywords are empty for hybrid mode")
        return PROMPTS["fail_response"]
    
    # Convert lists to strings
    if ll_keywords:
        ll_keywords = ", ".join(ll_keywords)
    else:
        ll_keywords = ""
    
    if hl_keywords:
        hl_keywords = ", ".join(hl_keywords)
    else:
        hl_keywords = ""


    keywords = [ll_keywords, hl_keywords]
    context= await _build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        global_config,
    )

    

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    if query_param.only_need_prompt:
        return sys_prompt
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )


    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )
    return response


async def _build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    ll_keywords, hl_keywords = query[0], query[1]
    
    # Initialize context variables
    entities_context, relations_context, text_units_context = "", "", ""
    
    if query_param.mode == "local":
        # Local mode: use ll_keywords only
        if ll_keywords == "":
            logger.warning("Low level keywords is empty for local mode")
            return "", "", ""
        
        (
            entities_context,
            relations_context,
            text_units_context,
        ) = await _get_node_data_with_flow_diffusion(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
            global_config,
        )
        
    elif query_param.mode == "global":
        # Global mode: use hl_keywords only
        if hl_keywords == "":
            logger.warning("High level keywords is empty for global mode")
            return "", "", ""
        
        (
            entities_context,
            relations_context,
            text_units_context,
        ) = await _get_node_data_with_flow_diffusion(
            hl_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
            global_config,
        )
        
    elif query_param.mode == "combined":
        # Combined mode: get top entities from both ll_keywords and hl_keywords separately, then combine for flow diffusion
        if ll_keywords == "" and hl_keywords == "":
            logger.warning("Both Low Level and High Level keywords are empty for combined mode")
            return "", "", ""
        
        (
            entities_context,
            relations_context,
            text_units_context,
        ) = await _get_combined_node_data_with_flow_diffusion(
            ll_keywords,
            hl_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
            global_config,
        )
    
    elif query_param.mode == "hybrid":
        # Hybrid mode: use combined keywords (both ll_keywords and hl_keywords)
        if ll_keywords == "" and hl_keywords == "":
            logger.warning("Both Low Level and High Level keywords are empty for hybrid mode")
            return "", "", ""
        
        # Get local information using ll_keywords
        local_entities_context, local_relations_context, local_text_units_context = "", "", ""
        if ll_keywords:
            (
                local_entities_context,
                local_relations_context,
                local_text_units_context,
            ) = await _get_node_data_with_flow_diffusion(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
                global_config,
            )
        
        # Get global information using hl_keywords
        global_entities_context, global_relations_context, global_text_units_context = "", "", ""
        if hl_keywords:
            (
                global_entities_context,
                global_relations_context,
                global_text_units_context,
            ) = await _get_node_data_with_flow_diffusion(
                hl_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
                global_config,
            )

    # Return context based on mode
    if query_param.mode == "local":
        return f"""
-----local-information-----
-----low-level entity information-----
```csv
{entities_context}
```
-----low-level relationship information-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
    elif query_param.mode == "global":
        return f"""
-----global-information-----
-----high-level entity information-----
```csv
{entities_context}
```
-----high-level relationship information-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
    elif query_param.mode == "combined":
        return f"""
-----combined-information-----
-----combined entity information (from low-level and high-level keywords)-----
```csv
{entities_context}
```
-----combined relationship information-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
    elif query_param.mode == "hybrid":
        return f"""
-----hybrid-information-----
-----local information (from low-level keywords)-----
-----local entity information-----
```csv
{local_entities_context}
```
-----local relationship information-----
```csv
{local_relations_context}
```
-----local sources-----
```csv
{local_text_units_context}
```
-----global information (from high-level keywords)-----
-----global entity information-----
```csv
{global_entities_context}
```
-----global relationship information-----
```csv
{global_relations_context}
```
-----global sources-----
```csv
{global_text_units_context}
```
"""
    else:
        return ""

async def _get_node_data_with_flow_diffusion(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """
    Modified _get_node_data function that uses flow diffusion for finding relationships.
    
    Parameters:
    -----------
    query : str
        Query string (can be either ll_keywords or hl_keywords)
    knowledge_graph_inst : BaseGraphStorage
        Knowledge graph storage instance
    entities_vdb : BaseVectorStorage
        Entity vector database
    text_chunks_db : BaseKVStorage[TextChunkSchema]
        Text chunks database
    query_param : QueryParam
        Query parameters
    global_config : dict
        Global configuration
        
    Returns:
    --------
    tuple
        (entities_context, relations_context, text_units_context)
    """
    results = await entities_vdb.query(query, top_k=query_param.max_source_nodes)
    if not len(results):
        return "", "", ""

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  
    
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )

    # Use flow diffusion instead of the original relationship finding method
    use_relations = await _find_flow_diffusion_clusters_and_summarize(
        node_datas, query, query_param, knowledge_graph_inst, global_config
    )

    logger.info(
        f"Flow diffusion query uses {len(node_datas)} entities, {len(use_relations)} cluster summaries, {len(use_text_units)} text units"
    )

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [["id", "cluster_summary"]]
    for i, summary in enumerate(use_relations):
        relations_section_list.append([i, summary])
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    
    return entities_context, relations_context, text_units_context


async def _get_embeddings_for_flow_diffusion(
    graph: nx.Graph,
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    query_param: QueryParam = None,
) -> tuple[dict, list]:
    """
    Get embeddings for nodes and query for query-aware flow diffusion.
    
    Returns:
    --------
    tuple
        (node_embeddings, subquery_embedding)
    """
    node_embeddings = {}
    subquery_embedding = None
    
    # Check if query-aware flow diffusion is enabled
    if query_param and not query_param.enable_query_aware_flow_diffusion:
        logger.info("Query-aware flow diffusion is disabled, skipping embedding calculation")
        return {}, None
    
    # Try to get embeddings from global config if available
    if "embedding_func" in global_config and global_config["embedding_func"]:
        try:
            if hasattr(knowledge_graph_inst, 'get_cached_node_embeddings'):
                logger.info("Attempting to use cached node embeddings...")
                node_embeddings = await knowledge_graph_inst.get_cached_node_embeddings(global_config)
                if node_embeddings:
                    logger.info(f"Successfully using cached embeddings for {len(node_embeddings)} nodes")
                else:
                    logger.info("No cached embeddings found, will compute new ones")
            else:
                logger.info("Storage class does not support cached embeddings, computing new ones...")
                # Get embeddings for all nodes in the graph with batching
                all_nodes = list(graph.nodes())
                if all_nodes:
                    # Get batch size and token limit from config
                    batch_size = global_config.get("embedding_batch_num", 32)
                    max_tokens_per_request = global_config.get("max_embed_tokens", 8192)
                    
                    # Prepare node texts
                    node_texts = []
                    for node in all_nodes:
                        node_info = await knowledge_graph_inst.get_node(node)
                        if node_info and "description" in node_info:
                            node_texts.append(f"{node} {node_info['description']}")
                        else:
                            node_texts.append(node)
                    
                    # Process in batches to avoid token limit
                    logger.info(f"Computing embeddings for {len(node_texts)} node texts in batches of {batch_size}...")
                    
                    # Import tiktoken for token counting
                    import tiktoken
                    tiktoken_model = global_config.get("tiktoken_model_name", "gpt-4o-mini")
                    try:
                        encoding = tiktoken.encoding_for_model(tiktoken_model)
                    except:
                        encoding = tiktoken.get_encoding("cl100k_base")  # fallback
                    
                    # Process batches
                    all_embeddings = []
                    for i in range(0, len(node_texts), batch_size):
                        batch_texts = node_texts[i:i + batch_size]
                        batch_nodes = all_nodes[i:i + batch_size]
                        
                        # Check token count for this batch
                        total_tokens = sum(len(encoding.encode(text)) for text in batch_texts)
                        
                        if total_tokens > max_tokens_per_request:
                            logger.warning(f"Batch {i//batch_size + 1} exceeds token limit ({total_tokens} > {max_tokens_per_request}), reducing batch size...")
                            # Process this batch with smaller chunks
                            sub_batch_size = max(1, batch_size // 2)
                            for j in range(0, len(batch_texts), sub_batch_size):
                                sub_batch_texts = batch_texts[j:j + sub_batch_size]
                                sub_batch_nodes = batch_nodes[j:j + sub_batch_size]
                                
                                # Check sub-batch token count
                                sub_total_tokens = sum(len(encoding.encode(text)) for text in sub_batch_texts)
                                if sub_total_tokens > max_tokens_per_request:
                                    logger.warning(f"Sub-batch still exceeds token limit ({sub_total_tokens} > {max_tokens_per_request}), processing one by one...")
                                    # Process one by one
                                    for k, (text, node) in enumerate(zip(sub_batch_texts, sub_batch_nodes)):
                                        try:
                                            embedding_array = await global_config["embedding_func"]([text])
                                            all_embeddings.append(embedding_array[0])
                                            logger.debug(f"Processed node {k+1}/{len(sub_batch_texts)} in sub-batch")
                                        except Exception as e:
                                            logger.error(f"Failed to process node {node}: {e}")
                                            # Add zero embedding as fallback
                                            embedding_dim = 1536  # default dimension
                                            all_embeddings.append([0.0] * embedding_dim)
                                else:
                                    try:
                                        embedding_array = await global_config["embedding_func"](sub_batch_texts)
                                        all_embeddings.extend(embedding_array)
                                        logger.debug(f"Processed sub-batch {j//sub_batch_size + 1} with {len(sub_batch_texts)} nodes")
                                    except Exception as e:
                                        logger.error(f"Failed to process sub-batch: {e}")
                                        # Add zero embeddings as fallback
                                        embedding_dim = 1536  # default dimension
                                        for _ in sub_batch_texts:
                                            all_embeddings.append([0.0] * embedding_dim)
                        else:
                            try:
                                embedding_array = await global_config["embedding_func"](batch_texts)
                                all_embeddings.extend(embedding_array)
                                logger.debug(f"Processed batch {i//batch_size + 1} with {len(batch_texts)} nodes")
                            except Exception as e:
                                logger.error(f"Failed to process batch: {e}")
                                # Add zero embeddings as fallback
                                embedding_dim = 1536  # default dimension
                                for _ in batch_texts:
                                    all_embeddings.append([0.0] * embedding_dim)
                    
                    # Store embeddings
                    for i, node in enumerate(all_nodes):
                        if i < len(all_embeddings):
                            node_embeddings[node] = all_embeddings[i].tolist()
                        else:
                            logger.warning(f"Missing embedding for node {node}")
                    
                    logger.info(f"Computed embeddings for {len(node_embeddings)} nodes")
            
            # Get query embedding
            if query:
                if hasattr(knowledge_graph_inst, 'get_cached_query_embedding'):
                    logger.info("Attempting to use cached query embedding...")
                    subquery_embedding = await knowledge_graph_inst.get_cached_query_embedding(query, global_config)
                    if subquery_embedding is None:
                        logger.warning("Failed to get cached query embedding, computing new one...")
                        query_embedding_array = await global_config["embedding_func"]([query])
                        subquery_embedding = query_embedding_array[0].tolist()
                        logger.info("Query embedding computed successfully")
                else:
                    logger.info("Storage class does not support cached query embeddings, computing new one...")
                    query_embedding_array = await global_config["embedding_func"]([query])
                    subquery_embedding = query_embedding_array[0].tolist()
                    logger.info("Query embedding computed successfully")
                
        except Exception as e:
            logger.warning(f"Failed to get embeddings for query-aware flow diffusion: {e}")
            # Fallback to non-query-aware mode
            node_embeddings = {}
            subquery_embedding = None
    else:
        logger.warning("No embedding function available in global config")
    
    return node_embeddings, subquery_embedding


async def _find_flow_diffusion_clusters_and_summarize(
    node_datas: list[dict],
    query: str,
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    Apply flow diffusion to find clusters and summarize them using LLM.
    
    Parameters:
    -----------
    node_datas : list[dict]
        List of node data dictionaries
    query : str
        The original query that provides context for the relationship analysis
    query_param : QueryParam
        Query parameters (includes flow diffusion configuration)
    knowledge_graph_inst : BaseGraphStorage
        Knowledge graph storage instance
    global_config : dict
        Global configuration
        
    Returns:
    --------
    list
        List of summarized cluster relationships
    """
    if hasattr(knowledge_graph_inst, 'get_cached_nx_graph'):
        logger.info("Attempting to use cached NetworkX graph...")
        G = await knowledge_graph_inst.get_cached_nx_graph()
        logger.info(f"Successfully obtained NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    else:
        logger.info("Storage class does not support cached graphs, building new one...")
        G = nx.Graph()
        edges = await knowledge_graph_inst.edges()
        nodes = await knowledge_graph_inst.nodes()
        
        # Add edges with weights
        for u, v in edges:
            edge_data = await knowledge_graph_inst.get_edge(u, v)
            if edge_data and 'weight' in edge_data:
                G.add_edge(u, v, weight=edge_data['weight'])
            else:
                G.add_edge(u, v, weight=1.0)  # Default weight: 1.0 if not specified
        
        G.add_nodes_from(nodes)
        logger.info(f"Built new NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Get source nodes from node_datas (limit to configured maximum)
    source_nodes = [dp["entity_name"] for dp in node_datas[:query_param.max_source_nodes]]
    
    # Apply flow diffusion from each source node independently
    all_clusters = []
    use_llm_func = global_config["llm_model_func"]
    
    logger.info(f"Starting flow diffusion from {len(source_nodes)} source nodes independently")
    
    logger.info("Pre-computing embeddings for flow diffusion...")
    node_embeddings, subquery_embedding = await _get_embeddings_for_flow_diffusion(
        G, query, knowledge_graph_inst, global_config, query_param
    )
    logger.info(f"Pre-computed embeddings: {len(node_embeddings)} nodes, query embedding: {'Yes' if subquery_embedding else 'No'}")
    
    # Run flow diffusion from each source node
    for source_node in source_nodes:
        if not G.has_node(source_node):
            continue
            
        # Get source node information
        source_node_info = await knowledge_graph_inst.get_node(source_node)
        if not source_node_info:
            source_node_info = {"entity_type": "UNKNOWN", "description": "No description available"}
        
        # Calculate confidence based on source node's relevance to the query
        confidence = 0.7  # Default confidence
        
        # Apply flow diffusion from this source node
        wfd = QueryAwareWeightedFlowDiffusion(
            G, source_node, source_node, confidence,
            node_embeddings=node_embeddings,
            subquery_embedding=subquery_embedding,
            weight_func=query_param.weight_func
        )
        wfd.initialize(alpha=query_param.alpha)
        diffused_nodes = wfd.flow_diffusion()
        
        if len(diffused_nodes) > 1:  # Only consider clusters with multiple nodes
            # Get cluster nodes and their flow values
            cluster_nodes = list(diffused_nodes.keys())
            cluster_flow_values = list(diffused_nodes.values())
            
            # Only process if cluster has significant flow (using configured threshold)
            if cluster_flow_values:
                max_flow = max(cluster_flow_values)
                if max_flow < query_param.min_flow_threshold:
                    continue
            else:
                continue  # Skip if no flow values
            
            # Get node information for the cluster
            cluster_node_data = []
            for node in cluster_nodes:
                node_info = await knowledge_graph_inst.get_node(node)
                if node_info:
                    cluster_node_data.append({
                        'name': node,
                        'type': node_info.get('entity_type', 'UNKNOWN'),
                        'description': node_info.get('description', 'UNKNOWN'),
                        'flow_value': diffused_nodes.get(node, 0.0)
                    })
            
            # Sort by flow value
            cluster_node_data.sort(key=lambda x: x['flow_value'], reverse=True)
            
            # Create cluster summary using LLM
            cluster_summary = await _summarize_cluster_with_llm(
                cluster_node_data, source_node, use_llm_func, global_config
            )
            
            if cluster_summary:
                all_clusters.append(cluster_summary)
    
    logger.info(f"Flow diffusion completed, {len(all_clusters)} clusters found")
    
    # Limit the number of clusters based on token constraints
    original_cluster_count = len(all_clusters)
    all_clusters = truncate_list_by_token_size(
        all_clusters,
        key=lambda x: x,
        max_token_size=query_param.max_token_for_local_context,
    )
    
    if original_cluster_count != len(all_clusters):
        logger.info(f"Clusters truncated from {original_cluster_count} to {len(all_clusters)} due to token limit")
    
    return all_clusters


async def _summarize_cluster_with_llm(
    cluster_node_data: list[dict],
    source_node: str,
    use_llm_func: callable,
    global_config: dict = None,
) -> str:
    """
    Summarize a cluster of nodes using LLM.
    
    Parameters:
    -----------
    cluster_node_data : list[dict]
        List of node data in the cluster
    source_node : str
        Source node name
    use_llm_func : callable
        LLM function to use for summarization
    global_config : dict
        Global configuration (optional)
        
    Returns:
    --------
    str
        Summarized cluster relationship
    """
    if len(cluster_node_data) < 2:
        return None
    
    # Create prompt for cluster summarization
    nodes_info = []
    for node_data in cluster_node_data:
        nodes_info.append(
            f"- {node_data['name']} ({node_data['type']}): {node_data['description']} "
            f"(flow strength: {node_data['flow_value']:.3f})"
        )
    
    nodes_text = "\n".join(nodes_info)
    
    prompt = f"""Please analyze the following cluster of entities and their relationships, then provide a concise summary of how they are connected and what this cluster represents.

Cluster Information:
Source Node: {source_node}
Cluster Nodes:
{nodes_text}

Please provide a summary that explains:
1. How these entities are related to each other
2. What concept or theme this cluster represents
3. The significance of the connections between these entities
4. How the source node connects to the other entities in this cluster

Keep the summary concise but informative, focusing on the relationships and thematic connections."""

    # Use configuration-based max_tokens if available, otherwise use a reasonable default
    max_tokens = 500  # Default value
    if global_config and "entity_summary_to_max_tokens" in global_config:
        max_tokens = global_config["entity_summary_to_max_tokens"]
    elif global_config and "summary_to_max_tokens" in global_config:
        max_tokens = global_config["summary_to_max_tokens"]

    try:
        summary = await use_llm_func(prompt, max_tokens=max_tokens)
        return summary.strip()
    except Exception as e:
        logger.error(f"Error summarizing cluster: {e}")
        # Fallback to more informative description
        if len(cluster_node_data) > 0:
            top_nodes = cluster_node_data[:3]  # Get top 3 nodes by flow value
            node_names = [node['name'] for node in top_nodes]
            if len(cluster_node_data) > 1:
                return f"Cluster centered around {source_node} connecting to {', '.join(node_names)} and {len(cluster_node_data)-1} other entities with flow values ranging from {cluster_node_data[0]['flow_value']:.3f} to {cluster_node_data[-1]['flow_value']:.3f}"
            else:
                return f"Cluster centered around {source_node} connecting to {', '.join(node_names)}"
        else:
            return f"Cluster connecting {source_node} with related entities"


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    """
    Find the most related text units from entities.
    
    Parameters:
    -----------
    node_datas : list[dict]
        List of node data dictionaries
    query_param : QueryParam
        Query parameters
    text_chunks_db : BaseKVStorage[TextChunkSchema]
        Text chunks database
    knowledge_graph_inst : BaseGraphStorage
        Knowledge graph storage instance
        
    Returns:
    --------
    list
        List of text unit data
    """
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                    "relation_counts": 0,
                }

            if this_edges:
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def evaluate_multiple_answers(
    query: str,
    answers: list[str],
    use_llm_func: callable,
) -> dict:
    """
    Evaluate multiple answers to the same question based on five criteria.
    
    Parameters:
    -----------
    query : str
        The original question
    answers : list[str]
        List of answers to evaluate
    use_llm_func : callable
        LLM function to use for evaluation
        
    Returns:
    --------
    dict
        Evaluation results with scores and rankings for each answer
    """
    if len(answers) < 2:
        logger.warning("Need at least 2 answers for evaluation")
        return {}
    
    # Create evaluation prompt for multiple answers
    answers_text = ""
    for i, answer in enumerate(answers, 1):
        answers_text += f"Answer {i}: {answer}\n\n"
    
    prompt = f"""---Role---
You are an expert tasked with evaluating multiple answers to the same question based on five criteria: Comprehensiveness, Diversity, Logicality, Relevance, and Coherence.

---Goal---
You will evaluate {len(answers)} answers to the same question based on five criteria:
- Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
- Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
- Logicality: How logically does the answer respond to all parts of the question?
- Relevance: How relevant is the answer to the question, staying focused and addressing the intended topic or issue?
- Coherence: How well does the answer maintain internal logical connections between its parts, ensuring a smooth and consistent structure?

Here is the question: {query}

Here are the {len(answers)} answers:
{answers_text}

For each criterion, assign a score from 1 to 10 to each answer, where:
- 1-2: Poor performance
- 3-4: Below average
- 5-6: Average
- 7-8: Good
- 9-10: Excellent

Then provide an overall ranking of the answers from best to worst.

Output your evaluation in the following JSON format:
{{
    "criterion_scores": {{
        "Comprehensiveness": {{
            "Answer 1": [score],
            "Answer 2": [score],
            ...
        }},
        "Diversity": {{
            "Answer 1": [score],
            "Answer 2": [score],
            ...
        }},
        "Logicality": {{
            "Answer 1": [score],
            "Answer 2": [score],
            ...
        }},
        "Relevance": {{
            "Answer 1": [score],
            "Answer 2": [score],
            ...
        }},
        "Coherence": {{
            "Answer 1": [score],
            "Answer 2": [score],
            ...
        }}
    }},
    "overall_scores": {{
        "Answer 1": [total_score],
        "Answer 2": [total_score],
        ...
    }},
    "ranking": ["Answer X", "Answer Y", ...],
    "best_answer": "Answer X",
    "explanations": {{
        "Answer 1": "Brief explanation of strengths and weaknesses",
        "Answer 2": "Brief explanation of strengths and weaknesses",
        ...
    }}
}}"""

    try:
        response = await use_llm_func(prompt, max_tokens=1000)
        
        # Extract JSON from response
        import re
        import json
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                evaluation_result = json.loads(json_match.group())
                return evaluation_result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM response: {e}")
                logger.error(f"Response: {response}")
                return {}
        else:
            logger.error(f"No JSON found in LLM response: {response}")
            return {}
            
    except Exception as e:
        logger.error(f"Error during answer evaluation: {e}")
        return {}


async def compare_two_answers(
    query: str,
    answer1: str,
    answer2: str,
    use_llm_func: callable,
) -> dict:
    """
    Compare two answers to the same question based on five criteria.
    
    Parameters:
    -----------
    query : str
        The original question
    answer1 : str
        First answer to evaluate
    answer2 : str
        Second answer to evaluate
    use_llm_func : callable
        LLM function to use for evaluation
        
    Returns:
    --------
    dict
        Comparison results with winner for each criterion and overall winner
    """
    prompt = f"""---Role---
You are an expert tasked with evaluating two answers to the same question based on five criteria: Comprehensiveness, Diversity, Logicality, Relevance, and Coherence.

---Goal---
You will evaluate two answers to the same question based on five criteria:
- Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
- Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
- Logicality: How logically does the answer respond to all parts of the question?
- Relevance: How relevant is the answer to the question, staying focused and addressing the intended topic or issue?
- Coherence: How well does the answer maintain internal logical connections between its parts, ensuring a smooth and consistent structure?

Here is the question: {query}

Here are the two answers:
Answer 1: {answer1}

Answer 2: {answer2}

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these five criteria.

Output your evaluation in the following JSON format:
{{
    "Comprehensiveness": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
    "Diversity": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
    "Logicality": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
    "Relevance": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
    "Coherence": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
    "Overall Winner": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Summarize why this answer is the overall winner based on the five criteria]" }}
}}"""

    try:
        response = await use_llm_func(prompt, max_tokens=800)
        
        # Extract JSON from response
        import re
        import json
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                comparison_result = json.loads(json_match.group())
                return comparison_result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM response: {e}")
                logger.error(f"Response: {response}")
                return {}
        else:
            logger.error(f"No JSON found in LLM response: {response}")
            return {}
            
    except Exception as e:
        logger.error(f"Error during answer comparison: {e}")
        return {}


def calculate_evaluation_metrics(evaluation_result: dict) -> dict:
    """
    Calculate additional metrics from evaluation results.
    
    Parameters:
    -----------
    evaluation_result : dict
        Result from evaluate_multiple_answers function
        
    Returns:
    --------
    dict
        Additional metrics including average scores, standard deviations, etc.
    """
    if not evaluation_result or "criterion_scores" not in evaluation_result:
        return {}
    
    metrics = {}
    criterion_scores = evaluation_result["criterion_scores"]
    
    # Calculate average scores for each criterion
    for criterion, scores in criterion_scores.items():
        if isinstance(scores, dict):
            values = [v for v in scores.values() if isinstance(v, (int, float))]
            if values:
                metrics[f"{criterion}_average"] = sum(values) / len(values)
                metrics[f"{criterion}_max"] = max(values)
                metrics[f"{criterion}_min"] = min(values)
    
    # Calculate overall statistics
    if "overall_scores" in evaluation_result:
        overall_scores = evaluation_result["overall_scores"]
        if isinstance(overall_scores, dict):
            values = [v for v in overall_scores.values() if isinstance(v, (int, float))]
            if values:
                metrics["overall_average"] = sum(values) / len(values)
                metrics["overall_max"] = max(values)
                metrics["overall_min"] = min(values)
                metrics["score_range"] = max(values) - min(values)
    
    return metrics

async def _get_combined_node_data_with_flow_diffusion(
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """
    Get combined node data from both low-level and high-level keywords for flow diffusion.
    
    Parameters:
    -----------
    ll_keywords : str
        Low-level keywords
    hl_keywords : str
        High-level keywords
    knowledge_graph_inst : BaseGraphStorage
        Knowledge graph storage instance
    entities_vdb : BaseVectorStorage
        Entity vector database
    text_chunks_db : BaseKVStorage[TextChunkSchema]
        Text chunks database
    query_param : QueryParam
        Query parameters
    global_config : dict
        Global configuration
        
    Returns:
    --------
    tuple
        (entities_context, relations_context, text_units_context)
    """
    # Calculate how many source nodes to allocate to each keyword type
    # Reserve half for each type, but ensure we don't exceed max_source_nodes
    max_source_nodes = query_param.max_source_nodes
    nodes_per_type = max(1, max_source_nodes // 2)  # At least 1 node per type
    
    # Get entities from low-level keywords
    ll_results = []
    if ll_keywords:
        ll_results = await entities_vdb.query(ll_keywords, top_k=max_source_nodes)
    
    # Get entities from high-level keywords
    hl_results = []
    if hl_keywords:
        hl_results = await entities_vdb.query(hl_keywords, top_k=max_source_nodes)
    
    # Combine and deduplicate results with priority to low-level keywords
    all_results = []
    seen_entities = set()
    
    # Add low-level entities first (up to nodes_per_type)
    for result in ll_results[:nodes_per_type]:
        if result["entity_name"] not in seen_entities:
            all_results.append(result)
            seen_entities.add(result["entity_name"])
    
    # Add high-level entities (up to nodes_per_type)
    for result in hl_results[:nodes_per_type]:
        if result["entity_name"] not in seen_entities:
            all_results.append(result)
            seen_entities.add(result["entity_name"])
    
    # Fill remaining slots with additional entities from both types
    remaining_slots = max_source_nodes - len(all_results)
    if remaining_slots > 0:
        # Add remaining low-level entities
        for result in ll_results[nodes_per_type:]:
            if result["entity_name"] not in seen_entities and len(all_results) < max_source_nodes:
                all_results.append(result)
                seen_entities.add(result["entity_name"])
        
        # Add remaining high-level entities
        for result in hl_results[nodes_per_type:]:
            if result["entity_name"] not in seen_entities and len(all_results) < max_source_nodes:
                all_results.append(result)
                seen_entities.add(result["entity_name"])
    
    if not len(all_results):
        return "", "", ""

    # Get node data for all entities
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in all_results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in all_results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(all_results, node_datas, node_degrees)
        if n is not None
    ]  
    
    # Get text units from all entities
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )

    # Use flow diffusion with combined entities
    # Create a combined query string for flow diffusion context
    combined_query = ""
    if ll_keywords and hl_keywords:
        combined_query = f"{ll_keywords}, {hl_keywords}"
    elif ll_keywords:
        combined_query = ll_keywords
    elif hl_keywords:
        combined_query = hl_keywords
    
    use_relations = await _find_flow_diffusion_clusters_and_summarize(
        node_datas, combined_query, query_param, knowledge_graph_inst, global_config
    )

    # Count entities from different sources (handle overlaps)
    ll_entities = set()
    hl_entities = set()
    
    # Collect entities from each source
    for r in ll_results:
        ll_entities.add(r['entity_name'])
    for r in hl_results:
        hl_entities.add(r['entity_name'])
    
    # Count entities that are actually used in node_datas
    ll_only_count = len([n for n in node_datas if n['entity_name'] in ll_entities and n['entity_name'] not in hl_entities])
    hl_only_count = len([n for n in node_datas if n['entity_name'] in hl_entities and n['entity_name'] not in ll_entities])
    overlap_count = len([n for n in node_datas if n['entity_name'] in ll_entities and n['entity_name'] in hl_entities])
    
    logger.info(
        f"Combined flow diffusion query uses {len(node_datas)} entities "
        f"({ll_only_count} low-level only, {hl_only_count} high-level only, {overlap_count} overlap), "
        f"{len(use_relations)} cluster summaries, {len(use_text_units)} text units"
    )

    # Create entities section
    entites_section_list = [["id", "entity", "type", "description", "rank", "source"]]
    for i, n in enumerate(node_datas):
        # Determine source (low-level or high-level)
        source = "unknown"
        if n["entity_name"] in ll_entities:
            source = "low-level"
        if n["entity_name"] in hl_entities:
            if source == "low-level":
                source = "both"
            else:
                source = "high-level"
        
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
                source,
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    # Create relations section
    relations_section_list = [["id", "cluster_summary"]]
    for i, summary in enumerate(use_relations):
        relations_section_list.append([i, summary])
    relations_context = list_of_list_to_csv(relations_section_list)

    # Create text units section
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    
    return entities_context, relations_context, text_units_context
