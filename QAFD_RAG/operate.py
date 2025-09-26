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
    csv_string_to_list,
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
            # smart_weight = original_weight * (node1_query_sim**2) * (node2_query_sim**2)

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
   
    entity_name = clean_str(record_attributes[1].lower())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].lower())
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
   
    source = clean_str(record_attributes[1].lower())
    target = clean_str(record_attributes[2].lower())
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
    
    # Check if we should use database schema entity extraction
    use_database_schema_prompt = global_config["addon_params"].get("use_database_schema_prompt", False)
    
    # Unified approach: use database_schema_entity_extraction for all database-related content
    # This prompt can handle JSON schema, metadata, or both
    if use_database_schema_prompt:
        # Use unified database schema entity extraction prompt
        entity_extract_prompt = PROMPTS["database_schema_entity_extraction"]
        examples_key = "database_schema_entity_extraction_examples"
    else:
        # Use regular entity extraction prompt for non-database content
        entity_extract_prompt = PROMPTS["entity_extraction"]
        examples_key = "entity_extraction_examples"
    
    example_number = global_config["addon_params"].get("example_number", None)
    example_index = global_config["addon_params"].get("example_index", None)
    
    if example_index is not None and example_index < len(PROMPTS[examples_key]):
        # Use specific example by index
        examples = PROMPTS[examples_key][example_index]
    elif example_number and example_number < len(PROMPTS[examples_key]):
        # Use first N examples
        examples = "\n".join(
            PROMPTS[examples_key][: int(example_number)]
        )
    else:
        # Use all examples
        examples = "\n".join(PROMPTS[examples_key])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
        metadata="metadata",
    )
  
    examples = examples.format(**example_context_base)
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
        metadata="metadata",
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
        # Use a safer approach: first format the prompt template with a placeholder for input_text, then manually replace it
        # This avoids conflicts with JSON content containing curly braces
        context_with_placeholder = context_base.copy()
        context_with_placeholder["input_text"] = "{input_text}"
        formatted_prompt = entity_extract_prompt.format(**context_with_placeholder)
        hint_prompt = formatted_prompt.replace("{input_text}", content)

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

    if query_param.mode not in ["local", "global", "hybrid"]:
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
            # print("low level entities: ", local_entities_context)
        
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
            # print("high level entities: ", global_entities_context)
    
    # Return context based on mode
    if query_param.mode == "local":
        # if query_param.return_raw_entities:
        #     return entities_context

        # Handle different relation context formats
        if query_param.return_raw_clusters and query_param.cluster_json_format == "text":
            relations_section = f"""
-----low-level relationship information-----
{relations_context}
"""
        else:
            relations_section = f"""
-----low-level relationship information-----
```csv
{relations_context}
```
"""

        # When returning raw clusters, skip entities_context to avoid duplication
        if query_param.return_raw_clusters:
            return f"""
-----local-information-----
{relations_section}
-----Sources-----
```csv
{text_units_context}
```
"""
        else:
            return f"""
-----local-information-----
-----low-level entity information-----
```csv
{entities_context}
```
{relations_section}
-----Sources-----
```csv
{text_units_context}
```
"""
    elif query_param.mode == "global":
        # if query_param.return_raw_entities:
        #     return entities_context
        
        # Handle different relation context formats
        if query_param.return_raw_clusters and query_param.cluster_json_format == "text":
            relations_section = f"""
-----high-level relationship information-----
{relations_context}
"""
        else:
            relations_section = f"""
-----high-level relationship information-----
```csv
{relations_context}
```
"""

        # When returning raw clusters, skip entities_context to avoid duplication
        if query_param.return_raw_clusters:
            return f"""
-----global-information-----
{relations_section}
-----Sources-----
```csv
{text_units_context}
```
"""
        else:
            return f"""
-----global-information-----
-----high-level entity information-----
```csv
{entities_context}
```
{relations_section}
-----Sources-----
```csv
{text_units_context}
```
"""
    elif query_param.mode == "hybrid":
        # if query_param.return_raw_entities:
        #     # Merge local + global entities CSV into one CSV and reindex id
        #     merged_rows = []
        #     if local_entities_context:
        #         merged_rows += csv_string_to_list(local_entities_context)[1:]
        #     if global_entities_context:
        #         merged_rows += csv_string_to_list(global_entities_context)[1:]
        #     for idx, row in enumerate(merged_rows):
        #         if row:
        #             row[0] = str(idx)
        #     merged_entities_csv = list_of_list_to_csv(
        #         [["id", "entity", "type", "description", "rank"]] + merged_rows
        #     )
        #     return merged_entities_csv
        
        # Combine local and global text units contexts to avoid duplication
        combined_text_units_context = process_combine_contexts(global_text_units_context, local_text_units_context)
        
        # Handle different relation context formats for local
        if query_param.return_raw_clusters and query_param.cluster_json_format == "text":
            local_relations_section = f"""
-----local relationship information-----
{local_relations_context}
"""
        else:
            local_relations_section = f"""
-----local relationship information-----
```csv
{local_relations_context}
```
"""

        # Handle different relation context formats for global
        if query_param.return_raw_clusters and query_param.cluster_json_format == "text":
            global_relations_section = f"""
-----global relationship information-----
{global_relations_context}
"""
        else:
            global_relations_section = f"""
-----global relationship information-----
```csv
{global_relations_context}
```
"""
        
        # When returning raw clusters, skip entities_context to avoid duplication
        if query_param.return_raw_clusters:
            return f"""
-----hybrid-information-----
-----local information (from low-level keywords)-----
{local_relations_section}
-----global information (from high-level keywords)-----
{global_relations_section}
-----combined sources (from both local and global keywords)-----
```csv
{combined_text_units_context}
```
"""
        else:
            return f"""
-----hybrid-information-----
-----local information (from low-level keywords)-----
-----local entity information-----
```csv
{local_entities_context}
```
{local_relations_section}
-----local sources-----
```csv
{local_text_units_context}
```
-----global information (from high-level keywords)-----
-----global entity information-----
```csv
{global_entities_context}
```
{global_relations_section}
-----global sources-----
```csv
{global_text_units_context}
```
-----combined sources (from both local and global keywords)-----
```csv
{combined_text_units_context}
```
"""

#         return f"""
# -----hybrid-information-----
# -----local information (from low-level keywords)-----
# -----local entity information-----
# ```csv
# {local_entities_context}
# ```
# -----local relationship information-----
# ```csv
# {local_relations_context}
# ```
# -----local sources-----
# ```csv
# {local_text_units_context}
# ```
# -----global information (from high-level keywords)-----
# -----global entity information-----
# ```csv
# {global_entities_context}
# ```
# -----global relationship information-----
# ```csv
# {global_relations_context}
# ```
# -----global sources-----
# ```csv
# {global_text_units_context}
# ```
# """
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

    # logger.info(
    #     f"Flow diffusion query uses {len(node_datas)} entities, {len(use_relations)} cluster summaries, {len(use_text_units)} text units"
    # )

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

    # Relations context: return raw clusters when requested; otherwise CSV
    if query_param.return_raw_clusters:
        if query_param.cluster_json_format == "text":
            # For text format, join all cluster texts with separators
            relations_context = "\n\n" + "="*50 + "\n\n".join(use_relations)
        else:
            # For JSON formats, return as-is (list of dicts)
            relations_context = use_relations
    else:
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
            # Prioritize using cached node embeddings
            if hasattr(knowledge_graph_inst, 'get_cached_node_embeddings'):
                # logger.info("Attempting to use cached node embeddings...")
                node_embeddings = await knowledge_graph_inst.get_cached_node_embeddings(global_config)
                # if node_embeddings:
                #     logger.info(f"Successfully using cached embeddings for {len(node_embeddings)} nodes")
                # else:
                #     logger.info("No cached embeddings found, will compute new ones")
            else:
                # logger.info("Storage class does not support cached embeddings, computing new ones...")
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
                    # logger.info(f"Computing embeddings for {len(node_texts)} node texts in batches of {batch_size}...")
                    
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
                    
                    # logger.info(f"Computed embeddings for {len(node_embeddings)} nodes")
            
            # Get query embedding
            if query:
                # Prioritize using cached query embedding
                if hasattr(knowledge_graph_inst, 'get_cached_query_embedding'):
                    # logger.info("Attempting to use cached query embedding...")
                    subquery_embedding = await knowledge_graph_inst.get_cached_query_embedding(query, global_config)
                    if subquery_embedding is None:
                        logger.warning("Failed to get cached query embedding, computing new one...")
                        query_embedding_array = await global_config["embedding_func"]([query])
                        subquery_embedding = query_embedding_array[0].tolist()
                        # logger.info("Query embedding computed successfully")
                else:
                    # logger.info("Storage class does not support cached query embeddings, computing new one...")
                    query_embedding_array = await global_config["embedding_func"]([query])
                    subquery_embedding = query_embedding_array[0].tolist()
                    # logger.info("Query embedding computed successfully")
                
        except Exception as e:
            logger.warning(f"Failed to get embeddings for query-aware flow diffusion: {e}")
            # Fallback to non-query-aware mode
            node_embeddings = {}
            subquery_embedding = None
    else:
        logger.warning("No embedding function available in global config")
    
    return node_embeddings, subquery_embedding


def _convert_subgraph_to_text(G: nx.Graph, cluster_nodes: list, diffused_nodes: dict, source_node: str, text_mode: str = "detailed") -> str:
    """
    Convert a subgraph to compact text format with nodes and edges information.
    
    Parameters:
    -----------
    G : nx.Graph
        The original graph
    cluster_nodes : list
        List of nodes in the cluster
    diffused_nodes : dict
        Dictionary mapping nodes to their flow values
    source_node : str
        The source node for this cluster
    text_mode : str
        Text format mode: "detailed", "compact", or "minimal"
        
    Returns:
    --------
    str
        Text representation of the subgraph
    """
    # Create subgraph from cluster nodes
    support_nodes = set(cluster_nodes)
    subgraph = G.subgraph(support_nodes)
    
    # Build node index map
    node_index_map = {node: idx for idx, node in enumerate(cluster_nodes)}
    
    if text_mode == "minimal":
        # Ultra-compact format: just node names and connections
        nodes_text = [f"{i}:{node}" for i, node in enumerate(cluster_nodes)]
        edges_text = []
        for u, v, data in subgraph.edges(data=True):
            weight = data.get('weight', 1.0)
            u_idx = node_index_map.get(u)
            v_idx = node_index_map.get(v)
            if weight == 1.0:
                edges_text.append(f"{u_idx}-{v_idx}")
            else:
                edges_text.append(f"{u_idx}-{v_idx}({weight:.2f})")
        
        max_flow = max(diffused_nodes.values()) if diffused_nodes else 0.0
        return f"CLUSTER:{source_node}({len(cluster_nodes)},{max_flow:.2f}) | NODES:{','.join(nodes_text)} | EDGES:{','.join(edges_text)}"
    
    elif text_mode == "compact":
        # Compact format: structured but concise
        nodes_text = []
        for i, node in enumerate(cluster_nodes):
            node_attrs = G.nodes[node] if node in G.nodes else {}
            node_type = node_attrs.get("entity_type", "UNK")
            flow_value = diffused_nodes.get(node, 0.0)
            nodes_text.append(f"{i}:{node}({node_type},{flow_value:.2f})")
        
        edges_text = []
        for u, v, data in subgraph.edges(data=True):
            weight = data.get('weight', 1.0)
            u_idx = node_index_map.get(u)
            v_idx = node_index_map.get(v)
            if weight == 1.0:
                edges_text.append(f"{u_idx}->{v_idx}")
            else:
                edges_text.append(f"{u_idx}->{v_idx}({weight:.2f})")
        
        max_flow = max(diffused_nodes.values()) if diffused_nodes else 0.0
        return f"CLUSTER: {source_node} (size:{len(cluster_nodes)}, flow:{max_flow:.2f})\nNODES: {', '.join(nodes_text)}\nEDGES: {', '.join(edges_text)}"
    
    else:  # detailed mode
        # Detailed format: full information
        nodes_text = []
        for i, node in enumerate(cluster_nodes):
            node_attrs = G.nodes[node] if node in G.nodes else {}
            node_degree = subgraph.degree(node) if node in subgraph else 0
            flow_value = diffused_nodes.get(node, 0.0)
            
            node_type = node_attrs.get("entity_type", "UNKNOWN")
            description = node_attrs.get("description", "UNKNOWN")
            
            # Compact node format: [idx] name (type) - description [degree, flow]
            node_line = f"[{i}] {node} ({node_type}) - {description} [deg:{node_degree}, flow:{flow_value:.3f}]"
            nodes_text.append(node_line)
        
        # Format edges
        edges_text = []
        for u, v, data in subgraph.edges(data=True):
            weight = data.get('weight', 1.0)
            u_idx = node_index_map.get(u)
            v_idx = node_index_map.get(v)
            
            if weight == 1.0:
                edge_line = f"[{u_idx}]->[{v_idx}]"
            else:
                edge_line = f"[{u_idx}]->[{v_idx}] (w:{weight:.3f})"
            edges_text.append(edge_line)
        
        # Combine into final text
        max_flow = max(diffused_nodes.values()) if diffused_nodes else 0.0
        text_output = f"""CLUSTER: {source_node} (size:{len(cluster_nodes)}, max_flow:{max_flow:.3f})
NODES:
{chr(10).join(nodes_text)}
EDGES:
{chr(10).join(edges_text)}"""
        
        return text_output


def _convert_subgraph_to_json(G: nx.Graph, cluster_nodes: list, diffused_nodes: dict, source_node: str, compact_mode: str = "compact") -> dict:
    """
    Convert a subgraph to compact JSON format with nodes and edges information.
    
    Parameters:
    -----------
    G : nx.Graph
        The original graph
    cluster_nodes : list
        List of nodes in the cluster
    diffused_nodes : dict
        Dictionary mapping nodes to their flow values
    source_node : str
        The source node for this cluster
    compact_mode : str
        Format mode: "compact" (default), "minimal", or "original"
        
    Returns:
    --------
    dict
        JSON representation of the subgraph
    """
    # Create subgraph from cluster nodes
    support_nodes = set(cluster_nodes)
    subgraph = G.subgraph(support_nodes)
    
    # Build node index map
    node_index_map = {node: idx for idx, node in enumerate(cluster_nodes)}
    
    if compact_mode == "minimal":
        # Ultra-compact format: only essential data
        nodes_minimal = [node for node in cluster_nodes]  # Just node names
        edges_minimal = []
        for u, v, data in subgraph.edges(data=True):
            weight = data.get('weight', 1.0)
            if weight != 1.0:  # Only store non-default weights
                edges_minimal.append([node_index_map.get(u), node_index_map.get(v), round(weight, 2)])
            else:
                edges_minimal.append([node_index_map.get(u), node_index_map.get(v)])
        
        return {
            "s": source_node,  # source
            "n": nodes_minimal,  # nodes (names only)
            "e": edges_minimal,  # edges (indices only)
            "f": round(max(diffused_nodes.values()) if diffused_nodes else 0.0, 2)  # max flow
        }
    
    elif compact_mode == "original":
        # Original verbose format (for backward compatibility)
        nodes_json = []
        for node in cluster_nodes:
            node_attrs = G.nodes[node] if node in G.nodes else {}
            node_degree = subgraph.degree(node) if node in subgraph else 0
            nodes_json.append({
                "id": node_index_map[node],
                "entity": node,
                "type": node_attrs.get("entity_type", "UNKNOWN"),
                "description": node_attrs.get("description", "UNKNOWN"),
                "rank": node_degree,
            })
        
        edges_json = []
        for u, v, data in subgraph.edges(data=True):
            edges_json.append({
                "source": u,
                "target": v,
                "source_id": node_index_map.get(u),
                "target_id": node_index_map.get(v),
                "weight": data.get('weight', 1.0)
            })
        
        return {
            "source_node": source_node,
            "cluster_size": len(cluster_nodes),
            "max_flow_value": max(diffused_nodes.values()) if diffused_nodes else 0.0,
            "nodes": nodes_json,
            "edges": edges_json,
            "total_edges": len(edges_json)
        }
    
    else:  # compact mode (default)
        # Compact format: arrays with short field names
        nodes_compact = []
        for node in cluster_nodes:
            node_attrs = G.nodes[node] if node in G.nodes else {}
            node_degree = subgraph.degree(node) if node in subgraph else 0
            nodes_compact.append([
                node,  # name
                node_attrs.get("entity_type", "UNKNOWN"),  # type
                node_attrs.get("description", "UNKNOWN"),  # description
                node_degree  # degree
            ])
        
        edges_compact = []
        for u, v, data in subgraph.edges(data=True):
            edges_compact.append([
                node_index_map.get(u),  # source index
                node_index_map.get(v),  # target index
                round(data.get('weight', 1.0), 3)  # weight (rounded to 3 decimals)
            ])
        
        return {
            "src": source_node,  # source node
            "sz": len(cluster_nodes),  # cluster size
            "max_flow": round(max(diffused_nodes.values()) if diffused_nodes else 0.0, 3),  # max flow value
            "n": nodes_compact,  # nodes array
            "e": edges_compact,  # edges array
            "te": len(edges_compact)  # total edges
        }


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
        If return_raw_clusters=True: List of cluster JSON objects with subgraph data
        If return_raw_clusters=False: List of summarized cluster relationships
    """
    # Use cached NetworkX graph to avoid repeated construction
    if hasattr(knowledge_graph_inst, 'get_cached_nx_graph'):
        # logger.info("Attempting to use cached NetworkX graph...")
        G = await knowledge_graph_inst.get_cached_nx_graph()
        # logger.info(f"Successfully obtained NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    else:
        # logger.info("Storage class does not support cached graphs, building new one...")
        # Compatibility handling: if no cache method, use original approach
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
        # logger.info(f"Built new NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Get source nodes from node_datas (limit to configured maximum)
    source_nodes = [dp["entity_name"] for dp in node_datas[:query_param.max_source_nodes]]
    
    # Apply flow diffusion from each source node independently
    all_clusters = []
    use_llm_func = global_config["llm_model_func"]
    
    # logger.info(f"Starting flow diffusion from {len(source_nodes)} source nodes independently")
    
    # Pre-compute node embeddings and query embeddings to avoid repeated computation in loops
    # logger.info("Pre-computing embeddings for flow diffusion...")
    node_embeddings, subquery_embedding = await _get_embeddings_for_flow_diffusion(
        G, query, knowledge_graph_inst, global_config, query_param
    )
    # logger.info(f"Pre-computed embeddings: {len(node_embeddings)} nodes, query embedding: {'Yes' if subquery_embedding else 'No'}")
    
    # Initialize cluster processing based on configuration
    all_clusters = []
    
    # if query_param.use_batch_cluster_summarization:
    #     logger.info("Using batch cluster summarization mode (more efficient)")
    # else:
    #     logger.info("Using individual cluster summarization mode (original approach)")
    
    if query_param.use_batch_cluster_summarization:
        # Collect all clusters first for batch processing
        clusters_to_summarize = []
        raw_clusters = []
        
        # Run flow diffusion from each source node
        for source_node in source_nodes:

            if not G.has_node(source_node):
                continue
                
            if G.nodes[source_node].get("entity_type", "").lower() == "complete_table":
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
                
                # Check if we should return raw cluster data instead of LLM summaries
                if query_param.return_raw_clusters:
                    if query_param.cluster_json_format == "text":
                        # Convert subgraph to text format
                        cluster_text = _convert_subgraph_to_text(G, cluster_nodes, diffused_nodes, source_node, query_param.cluster_text_mode)
                        raw_clusters.append(cluster_text)
                    else:
                        # Convert subgraph to JSON format
                        cluster_json = _convert_subgraph_to_json(G, cluster_nodes, diffused_nodes, source_node, query_param.cluster_json_format)
                        # Add node details to the JSON
                        cluster_json["node_details"] = cluster_node_data
                        raw_clusters.append(cluster_json)
                else:
                    # Collect cluster data for batch processing
                    clusters_to_summarize.append((cluster_node_data, source_node))
        
        # Process clusters based on return type
        if query_param.return_raw_clusters:
            all_clusters = raw_clusters
        else:
            # Batch process clusters in chunks to avoid token limits
            if clusters_to_summarize:
                all_clusters = []
                batch_size = query_param.batch_cluster_size
                
                # Process clusters in batches
                for i in range(0, len(clusters_to_summarize), batch_size):
                    batch_clusters = clusters_to_summarize[i:i + batch_size]
                    # logger.info(f"Processing batch {i//batch_size + 1}/{(len(clusters_to_summarize) + batch_size - 1)//batch_size} with {len(batch_clusters)} clusters")
                    
                    cluster_summaries = await _summarize_clusters_batch_with_llm(
                        batch_clusters, use_llm_func, global_config
                    )
                    all_clusters.extend([summary for summary in cluster_summaries if summary])
            else:
                all_clusters = []
    
    else:
        # Process clusters individually (original approach)
        for source_node in source_nodes:

            if not G.has_node(source_node):
                continue
                
            if G.nodes[source_node].get("entity_type", "").lower() == "complete_table":
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
                
                # Check if we should return raw cluster data instead of LLM summaries
                if query_param.return_raw_clusters:
                    if query_param.cluster_json_format == "text":
                        # Convert subgraph to text format
                        cluster_text = _convert_subgraph_to_text(G, cluster_nodes, diffused_nodes, source_node, query_param.cluster_text_mode)
                        all_clusters.append(cluster_text)
                    else:
                        # Convert subgraph to JSON format
                        cluster_json = _convert_subgraph_to_json(G, cluster_nodes, diffused_nodes, source_node, query_param.cluster_json_format)
                        # Add node details to the JSON
                        cluster_json["node_details"] = cluster_node_data
                        all_clusters.append(cluster_json)
                else:
                    # Create cluster summary using LLM (individual processing)
                    cluster_summary = await _summarize_cluster_with_llm(
                        cluster_node_data, source_node, use_llm_func, global_config
                    )
                    
                    if cluster_summary:
                        all_clusters.append(cluster_summary)
    
    logger.info(f"Flow diffusion completed, {len(all_clusters)} clusters found")
    
    # Only apply token constraints if we're returning LLM summaries
    if not query_param.return_raw_clusters:
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


async def _summarize_clusters_batch_with_llm(
    clusters_data: list[tuple[list[dict], str]],
    use_llm_func: callable,
    global_config: dict = None,
) -> list[str]:
    """
    Summarize multiple clusters using a single LLM call.
    
    Parameters:
    -----------
    clusters_data : list[tuple[list[dict], str]]
        List of tuples containing (cluster_node_data, source_node) for each cluster
    use_llm_func : callable
        LLM function to use for summarization
    global_config : dict
        Global configuration (optional)
        
    Returns:
    --------
    list[str]
        List of summarized cluster relationships
    """
    if not clusters_data:
        return []
    
    # Filter out clusters with less than 2 nodes
    valid_clusters = []
    for cluster_node_data, source_node in clusters_data:
        if len(cluster_node_data) >= 2:
            valid_clusters.append((cluster_node_data, source_node))
    
    if not valid_clusters:
        return []
    
    # Create batch prompt for all clusters
    clusters_info = []
    for i, (cluster_node_data, source_node) in enumerate(valid_clusters, 1):
        nodes_info = []
        for node_data in cluster_node_data:
            nodes_info.append(
                f"- {node_data['name']} ({node_data['type']}): {node_data['description']} "
                f"(flow strength: {node_data['flow_value']:.3f})"
            )
        
        nodes_text = "\n".join(nodes_info)
        clusters_info.append(f"""Cluster {i}:
Source Node: {source_node}
Cluster Nodes:
{nodes_text}""")
    
    all_clusters_text = "\n\n".join(clusters_info)
    
    prompt = f"""Please analyze the following clusters of entities and their relationships, then provide a concise summary for each cluster.

Clusters Information:
{all_clusters_text}

For each cluster, please provide a summary that explains:
1. How these entities are related to each other
2. What concept or theme this cluster represents
3. The significance of the connections between these entities
4. How the source node connects to the other entities in this cluster

Please format your response as follows:
Cluster 1: [summary for cluster 1]
Cluster 2: [summary for cluster 2]
...
Cluster N: [summary for cluster N]

Keep each summary concise but informative, focusing on the relationships and thematic connections."""

    # Use configuration-based max_tokens if available, otherwise use a reasonable default
    max_tokens = 2000  # Increased for batch processing
    if global_config and "entity_summary_to_max_tokens" in global_config:
        max_tokens = global_config["entity_summary_to_max_tokens"] * len(valid_clusters)  # Scale with number of clusters
    
    # logger.info(f"Processing batch of {len(valid_clusters)} clusters with max_tokens={max_tokens}")

    try:
        batch_summary = await use_llm_func(prompt, max_tokens=max_tokens)
        batch_summary = batch_summary.strip()
        
        # Parse the response to extract individual cluster summaries
        summaries = []
        lines = batch_summary.split('\n')
        current_summary = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('Cluster ') and ':' in line:
                # Save previous summary if exists
                if current_summary:
                    summaries.append(current_summary.strip())
                # Start new summary
                current_summary = line.split(':', 1)[1].strip()
            elif line and current_summary:
                current_summary += " " + line
        
        # Add the last summary
        if current_summary:
            summaries.append(current_summary.strip())
        
        # Ensure we have the right number of summaries
        while len(summaries) < len(valid_clusters):
            summaries.append("Cluster summary not available")
        
        return summaries[:len(valid_clusters)]
        
    except Exception as e:
        logger.error(f"Error summarizing clusters batch: {e}")
        # Fallback: create individual summaries for each cluster
        fallback_summaries = []
        for cluster_node_data, source_node in valid_clusters:
            if len(cluster_node_data) > 0:
                top_nodes = cluster_node_data[:3]  # Get top 3 nodes by flow value
                node_names = [node['name'] for node in top_nodes]
                if len(cluster_node_data) > 1:
                    fallback_summary = f"Cluster centered around {source_node} connecting to {', '.join(node_names)} and {len(cluster_node_data)-1} other entities with flow values ranging from {cluster_node_data[0]['flow_value']:.3f} to {cluster_node_data[-1]['flow_value']:.3f}"
                else:
                    fallback_summary = f"Cluster centered around {source_node} connecting to {', '.join(node_names)}"
                fallback_summaries.append(fallback_summary)
            else:
                fallback_summaries.append(f"Cluster connecting {source_node} with related entities")
        
        return fallback_summaries


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
    max_tokens = 1000  # Default value
    if global_config and "entity_summary_to_max_tokens" in global_config:
        max_tokens = global_config["entity_summary_to_max_tokens"]

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