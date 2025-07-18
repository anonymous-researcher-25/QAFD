import numpy as np
import re
import json
import networkx as nx
import os
from LLM_service import LLMService
from Init_schema_graph_v3 import SchemaGraphBuilder
from visual_enhnced_graphs import visualize_graph

class SchemaGraphEnhancer:
    """
    Enhances a schema graph with semantic edge weights using LLama LLM and efficient kernel methods.
    Works directly with JSON database summary instead of querying the database.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the SchemaGraphEnhancer.
        
        Args:
            llm_client: Client for LLama LLM API calls
        """
        self.llm_client = llm_client
    
    def enhance_edge_semantics(self, graph, metadata, db_summary_path, use_diffusion=False):
        """
        Enhance graph edge weights using LLM-enhanced features and efficient kernel approach.
        
        Args:
            graph: NetworkX graph of the database schema
            metadata: Dictionary containing schema metadata
            db_summary_path: Path to the JSON database summary file
            use_diffusion: Whether to apply network diffusion (default: False)
            
        Returns:
            Updated graph with semantically enhanced edge weights
        """
        enhanced_graph = graph.copy()
        
        # Load the database summary from JSON
        with open(db_summary_path, 'r', encoding='utf-8') as f:
            db_summary = json.load(f)
        
        # 1. Extract core node features with LLM enhancement
        node_features = self._extract_node_features(enhanced_graph, db_summary)
        
        # 2. Extract edge features with LLM enhancement
        edge_features = self._extract_edge_features(enhanced_graph, metadata)
        
        # 3. Compute weights for combined feature approach
        # (This is more efficient than multiple kernels for smaller schemas)
        node_weights, edge_weights = self._get_feature_weights(enhanced_graph, metadata)
        
        # 4. Apply the weights to edges
        nodes = list(enhanced_graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Compute node similarity matrix once for efficiency
        n = len(nodes)
        node_sim_matrix = np.zeros((n, n))
        
        # Normalize all node feature vectors for consistent scaling
        for i, node_i in enumerate(nodes):
            for j in range(i+1, n):
                node_j = nodes[j]
                
                # Calculate weighted similarity between node feature vectors
                node_sim = 0
                for k in range(len(node_features[node_i])):
                    # Euclidean distance between normalized feature vectors
                    dist = np.linalg.norm(node_features[node_i][k] - node_features[node_j][k])
                    # Convert to similarity with Gaussian kernel
                    sim = np.exp(-0.8 * (dist ** 2))  # 0.8 is a balanced kernel parameter
                    node_sim += node_weights[k] * sim
                
                node_sim_matrix[i, j] = node_sim
                node_sim_matrix[j, i] = node_sim  # Symmetry

        # Apply weights to edges
        for u, v in enhanced_graph.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            # Get node similarity from precomputed matrix
            node_sim = node_sim_matrix[i, j]
            
            # Calculate edge feature similarity
            edge_sim = 0
            for k in range(len(edge_features[(u, v)])):
                edge_sim += edge_weights[k] * np.mean(edge_features[(u, v)][k])
            
            # Combined weight (balance between node similarity and edge features)
            combined_weight = 0.4 * node_sim + 0.6 * edge_sim
            
            # Store weights
            enhanced_graph[u][v]['node_similarity'] = node_sim
            enhanced_graph[u][v]['edge_similarity'] = edge_sim
            enhanced_graph[u][v]['semantic_weight'] = combined_weight
            enhanced_graph[u][v]['weight'] = combined_weight * graph[u][v]['weight']  # Preserve original weight if exists

        # 5. Apply network diffusion if requested
        if use_diffusion:
            enhanced_graph = self._apply_network_diffusion(enhanced_graph, nodes, node_to_idx)
        
        return enhanced_graph

    def _apply_network_diffusion(self, graph, nodes=None, node_to_idx=None):
        """
        Apply network diffusion to enhance the graph weights.
        
        Args:
            graph: NetworkX graph to enhance
            nodes: List of nodes (if None, will be computed)
            node_to_idx: Dictionary mapping nodes to indices (if None, will be computed)
            
        Returns:
            Enhanced graph with diffused weights
        """
        diffused_graph = graph.copy()
        
        # If nodes and node_to_idx are not provided, compute them
        if nodes is None:
            nodes = list(diffused_graph.nodes())
        
        if node_to_idx is None:
            node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        n = len(nodes)
        
        # Create adjacency matrix with edge weights
        A = np.zeros((n, n))
        for u, v, data in diffused_graph.edges(data=True):
            i, j = node_to_idx[u], node_to_idx[v]
            A[i, j] = data.get('weight', 1.0)
            A[j, i] = data.get('weight', 1.0)  # Ensure symmetry
        
        # Apply simplified network diffusion
        k = min(10, n-1)  # Neighborhood size parameter
        A = A - np.diag(np.diag(A))  # Remove self-loops
        
        # Find k nearest neighbors for each node
        sorted_indices = np.argsort(-A, axis=1)
        P = np.zeros_like(A)
        
        for i in range(n):
            neighbors = sorted_indices[i, :k]
            P[i, neighbors] = A[i, neighbors]
        
        # Make P symmetric and normalize
        P = (P + P.T) / 2
        P = P + np.eye(n)  # Add self-loops
        D_inv = np.diag(1.0 / np.maximum(np.sum(P, axis=1), 1e-10))
        P = D_inv @ P
        
        # Apply diffusion with simplified approach (skip eigendecomposition for efficiency)
        alpha = 0.8
        W = np.linalg.matrix_power(P, 3)  # Diffuse 3 steps
        
        # Update graph with diffused weights
        for i in range(n):
            for j in range(i+1, n):
                if W[i, j] > 0:
                    u, v = nodes[i], nodes[j]
                    if diffused_graph.has_edge(u, v):
                        diffused_graph[u][v]['diffused_weight'] = W[i, j]
                        diffused_graph[u][v]['weight'] = W[i, j]
                    elif diffused_graph.has_edge(v, u):
                        diffused_graph[v][u]['diffused_weight'] = W[i, j]
                        diffused_graph[v][u]['weight'] = W[i, j]
        
        return diffused_graph

    def _extract_node_features(self, graph, db_summary):
        """
        Extract essential feature vectors for nodes using JSON database summary
        and LLM-enhanced semantic information.
        
        Args:
            graph: NetworkX graph
            db_summary: JSON database summary dictionary
            
        Returns:
            Dictionary mapping node names to lists of feature vectors
        """
        node_features = {}
        node_stats = self._get_stats_from_json(db_summary)
        
        # Get LLM-enhanced semantic features for nodes
        semantic_features = self._get_semantic_features(graph, node_stats)
        
        # Create streamlined feature vectors for each node
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_type = node_data.get('type', '')
            
            # Get statistics and semantics for this node
            stats = node_stats.get(node, {})
            semantics = semantic_features.get(node, {})
            
            # 1. Semantic Features (enhanced by LLM)
            semantic_feature = np.zeros(5)
            semantic_feature[0] = semantics.get('purpose_score', 0.5)
            semantic_feature[1] = semantics.get('domain_relevance', 0.5)
            semantic_feature[2] = semantics.get('relationship_score', 0.5)
            semantic_feature[3] = semantics.get('data_richness', 0.5)
            semantic_feature[4] = semantics.get('centrality', 0.5)
            
            # 2. Structural Features
            structural_feature = np.zeros(5)
            neighbors = list(graph.neighbors(node))
            structural_feature[0] = min(1.0, len(neighbors) / 20)  # Normalized degree
            structural_feature[1] = 1.0 if node_type == 'table' else 0.0
            structural_feature[2] = 1.0 if node_type == 'column' else 0.0
            structural_feature[3] = float(stats.get('is_primary_key', False))
            structural_feature[4] = float(stats.get('is_foreign_key', False))
            
            # 3. Type and Statistical Features
            type_stat_feature = np.zeros(5)
            data_type = stats.get('data_type', '').lower()
            type_stat_feature[0] = float('int' in data_type or 'number' in data_type)
            type_stat_feature[1] = float('varchar' in data_type or 'text' in data_type)
            type_stat_feature[2] = float('date' in data_type or 'time' in data_type)
            type_stat_feature[3] = stats.get('cardinality', 0.0)
            type_stat_feature[4] = min(1.0, stats.get('nullability', 0.0))
            
            # Store features for this node
            node_features[node] = [
                semantic_feature,
                structural_feature,
                type_stat_feature
            ]
        
        return node_features

    def _get_stats_from_json(self, db_summary):
        """
        Extract database statistics from JSON database summary.
        
        Args:
            graph: NetworkX graph
            db_summary: JSON database summary dictionary
            
        Returns:
            Dictionary of node statistics
        """
        node_stats = {}
        tables_info = db_summary.get("tables", {})
        
        # Process each table and its columns from the JSON summary
        for table_name, table_info in tables_info.items():
            # Store table statistics
            node_stats[table_name] = {
                'row_count': table_info.get('row_count', 0),
                'col_count': table_info.get('column_count', 0),
                'columns': {}
            }
            
            # Process each column
            for col_info in table_info.get('columns', []):
                col_name = col_info.get('name', '')
                col_key = f"{table_name}.{col_name}"
                
                # Store column statistics
                node_stats[col_key] = {
                    'data_type': col_info.get('type', ''),
                    'not_null': col_info.get('not_null', False),
                    'is_primary_key': col_info.get('primary_key', False),
                    'is_foreign_key': col_info.get('is_foreign_key', False),
                    'references_table': col_info.get('references_table'),
                    'distinct_count': col_info.get('distinct_count', 0),
                    'null_count': col_info.get('null_count', 0),
                    'nullability': col_info.get('nullability', 0),
                    'cardinality': col_info.get('cardinality', 0),
                }
        
        return node_stats

    def _extract_edge_features(self, graph, metadata):
        """
        Extract streamlined feature vectors for edges with LLM enhancement.
        
        Args:
            graph: NetworkX graph
            metadata: Schema metadata
            
        Returns:
            Dictionary mapping edge tuples to lists of feature vectors
        """
        # First get LLM-enhanced semantic understanding of relationships
        semantic_rels = self._get_relationship_semantics(graph, metadata)
        
        edge_features = {}
        
        for u, v, data in graph.edges(data=True):
            edge_key = (u, v)
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            u_type = u_data.get('type', '')
            v_type = v_data.get('type', '')
            u_name = u if '.' not in u else u.split('.')[1]
            v_name = v if '.' not in v else v.split('.')[1]
            rel_type = data.get('relationship_type', '')
            
            # 1. Structural Features
            structural_feature = np.zeros(5)
            structural_feature[0] = float(rel_type == 'pk_fk_column')
            structural_feature[1] = float(rel_type == 'table_column')
            structural_feature[2] = float(rel_type == 'same_table')
            structural_feature[3] = float(u_data.get('is_primary_key', False) and v_data.get('is_foreign_key', False))
            structural_feature[4] = float(u_data.get('is_foreign_key', False) and v_data.get('is_primary_key', False))
            
            # 2. Semantic Similarity Features (with LLM enhancement)
            semantic_feature = np.zeros(5)
            # Name similarity (simple Jaccard-like metric)
            u_tokens = set(u_name.lower().split('_'))
            v_tokens = set(v_name.lower().split('_'))
            if u_tokens and v_tokens:
                semantic_feature[0] = len(u_tokens.intersection(v_tokens)) / len(u_tokens.union(v_tokens))
            
            # Get LLM-identified semantic score if available
            edge_semantic_info = semantic_rels.get((u, v), {})
            semantic_feature[1] = edge_semantic_info.get('semantic_relevance', 0.5)
            semantic_feature[2] = edge_semantic_info.get('business_importance', 0.5)
            semantic_feature[3] = edge_semantic_info.get('query_frequency', 0.5)
            semantic_feature[4] = edge_semantic_info.get('data_relationship', 0.5)
            
            # 3. Compatibility Features
            compatibility_feature = np.zeros(5)
            u_type_str = u_data.get('data_type', '').lower()
            v_type_str = v_data.get('data_type', '').lower()
            
            # Type compatibility
            if u_type_str == v_type_str:
                compatibility_feature[0] = 1.0
            elif ('int' in u_type_str and 'int' in v_type_str) or \
                ('char' in u_type_str and 'char' in v_type_str) or \
                ('date' in u_type_str and 'date' in v_type_str):
                compatibility_feature[0] = 0.5
                
            # Column relationship in same table
            if u_type == 'column' and v_type == 'column':
                u_table = u.split('.')[0] if '.' in u else ''
                v_table = v.split('.')[0] if '.' in v else ''
                compatibility_feature[1] = float(u_table == v_table and u_table != '')
            
            # Store features for this edge
            edge_features[edge_key] = [
                structural_feature,
                semantic_feature,
                compatibility_feature
            ]
        
        return edge_features

    def _get_semantic_features(self, graph, node_stats):
        """
        Use LLama LLM to extract semantic features for nodes.
        
        Args:
            graph: NetworkX graph
            node_stats: Dictionary of node statistics
            
        Returns:
            Dictionary of semantic features for nodes
        """
        semantic_features = {}
        
        # Group nodes by chunks to avoid too large prompts
        chunk_size = 20
        all_nodes = list(graph.nodes())
        
        for i in range(0, len(all_nodes), chunk_size):
            node_chunk = all_nodes[i:i+chunk_size]
            
            # Prepare the semantic analysis prompt
            prompt = f"""
            I need semantic analysis of database schema elements to create feature vectors.
            
            For each table or column listed below, provide semantic features across these dimensions:
            1. Purpose: What is the likely purpose/role of this element?
            2. Domain: What business domain concepts does it relate to?
            3. Relationships: What likely relationships does it have with other elements?
            4. Data characteristics: What can we infer about the data?
            5. Importance: How central is this element to the schema?
            
            Database elements to analyze:
            """
            
            for node in node_chunk:
                node_type = graph.nodes[node].get('type', '')
                if node_type == 'table':
                    stats = node_stats.get(node, {})
                    prompt += f"\nTABLE: {node} (rows: {stats.get('row_count', '?')}, columns: {stats.get('col_count', '?')})"
                elif node_type == 'column':
                    table = node.split('.')[0] if '.' in node else ''
                    column = node.split('.')[1] if '.' in node else node
                    stats = node_stats.get(node, {})
                    prompt += f"\nCOLUMN: {node} (type: {stats.get('data_type', '?')}, " \
                            f"pk: {stats.get('is_primary_key', False)}, " \
                            f"fk: {stats.get('is_foreign_key', False)})"
            
            prompt += """
            
            Format your response as:
            <NODE>
            name: node_name
            purpose_score: 0-1 (e.g., 0.8 for primary identifier, 0.6 for descriptive field)
            domain_relevance: 0-1 (e.g., 0.9 for core business concept, 0.3 for auxiliary data)
            relationship_score: 0-1 (e.g., 0.7 for highly connected, 0.2 for isolated)
            data_richness: 0-1 (e.g., 0.8 for varied valuable data, 0.4 for simple data)
            centrality: 0-1 (e.g., 0.9 for schema centerpiece, 0.2 for peripheral)
            tags: comma-separated list of relevant concepts (e.g., "user, authentication, security")
            </NODE>
            """

            # Call LLama LLM
            llm_response = self._call_llama(prompt, max_tokens=1500, temperature=0.1)
            
            # Parse LLM response
            node_matches = re.findall(r'<NODE>(.*?)</NODE>', llm_response, re.DOTALL)
            
            for node_match in node_matches:
                try:
                    # Extract node name
                    name_match = re.search(r'name:\s*(.*?)$', node_match, re.MULTILINE)
                    if not name_match:
                        continue
                        
                    node_name = name_match.group(1).strip()
                    
                    # Find the corresponding graph node
                    matching_node = None
                    for n in node_chunk:
                        if node_name == n or node_name in n:
                            matching_node = n
                            break
                    
                    if not matching_node:
                        continue
                    
                    # Extract features
                    purpose_score = float(re.search(r'purpose_score:\s*([\d\.]+)', node_match).group(1))
                    domain_relevance = float(re.search(r'domain_relevance:\s*([\d\.]+)', node_match).group(1))
                    relationship_score = float(re.search(r'relationship_score:\s*([\d\.]+)', node_match).group(1))
                    data_richness = float(re.search(r'data_richness:\s*([\d\.]+)', node_match).group(1))
                    centrality = float(re.search(r'centrality:\s*([\d\.]+)', node_match).group(1))
                    
                    tags_match = re.search(r'tags:\s*(.*?)$', node_match, re.MULTILINE)
                    tags = tags_match.group(1).strip() if tags_match else ""
                    
                    semantic_features[matching_node] = {
                        'purpose_score': purpose_score,
                        'domain_relevance': domain_relevance,
                        'relationship_score': relationship_score,
                        'data_richness': data_richness,
                        'centrality': centrality,
                        'tags': tags
                    }
                    
                except Exception as e:
                    print(f"Error parsing node semantic features: {e}")
        
        return semantic_features

    def _get_relationship_semantics(self, graph, metadata):
        """
        Use LLama LLM to enhance understanding of edge relationships.
        
        Args:
            graph: NetworkX graph
            metadata: Schema metadata
            
        Returns:
            Dictionary mapping edge tuples to semantic information
        """
        # Format schema info
        schema_text = self._format_schema_for_prompt(metadata)
        
        # Identify important column-to-column edges for LLM analysis
        edge_semantics = {}
        column_edges = []
        
        for u, v, data in graph.edges(data=True):
            if graph.nodes[u].get('type') == 'column' and graph.nodes[v].get('type') == 'column':
                rel_type = data.get('relationship_type', '')
                column_edges.append((u, v, rel_type))
        
        # Sample edges if too many
        if len(column_edges) > 100:
            import random
            sampled_edges = random.sample(column_edges, 100)
        else:
            sampled_edges = column_edges
        
        # Create prompt for LLM
        prompt = f"""
        Analyze the semantic relationships between database columns.
        
        {schema_text}
        
        For each column pair, evaluate:
        1. Semantic relevance: How semantically related are these columns? (0-1)
        2. Business importance: How important is this relationship to business logic? (0-1)
        3. Query frequency: How likely are these columns to be queried together? (0-1)
        4. Data relationship: How strong is the data dependency between these columns? (0-1)
        
        Column pairs to analyze:
        """
        
        for u, v, rel_type in sampled_edges:
            prompt += f"\nPAIR: {u} <-> {v} (relationship: {rel_type})"
        
        prompt += """
        
        Format your response as:
        <RELATIONSHIP>
        column1: name_of_column1
        column2: name_of_column2
        semantic_relevance: 0.X
        business_importance: 0.X
        query_frequency: 0.X
        data_relationship: 0.X
        </RELATIONSHIP>
        """
        
        # Call LLama LLM
        llm_response = self._call_llama(prompt, max_tokens=1500, temperature=0.1)
        
        # Parse the response
        rel_matches = re.findall(r'<RELATIONSHIP>(.*?)</RELATIONSHIP>', llm_response, re.DOTALL)
        
        for rel_match in rel_matches:
            try:
                col1_match = re.search(r'column1:\s*(.*?)$', rel_match, re.MULTILINE)
                col2_match = re.search(r'column2:\s*(.*?)$', rel_match, re.MULTILINE)
                
                if not col1_match or not col2_match:
                    continue
                    
                col1 = col1_match.group(1).strip()
                col2 = col2_match.group(1).strip()
                
                # Find the corresponding edge
                edge_pair = None
                for u, v, _ in sampled_edges:
                    if (col1 in u and col2 in v) or (col1.split('.')[-1] == u.split('.')[-1] and col2.split('.')[-1] == v.split('.')[-1]):
                        edge_pair = (u, v)
                        break
                    elif (col2 in u and col1 in v) or (col2.split('.')[-1] == u.split('.')[-1] and col1.split('.')[-1] == v.split('.')[-1]):
                        edge_pair = (u, v)
                        break
                
                if not edge_pair:
                    continue
                
                # Extract relationship features
                semantic_relevance = float(re.search(r'semantic_relevance:\s*([\d\.]+)', rel_match).group(1))
                business_importance = float(re.search(r'business_importance:\s*([\d\.]+)', rel_match).group(1))
                query_frequency = float(re.search(r'query_frequency:\s*([\d\.]+)', rel_match).group(1))
                data_relationship = float(re.search(r'data_relationship:\s*([\d\.]+)', rel_match).group(1))
                
                edge_semantics[edge_pair] = {
                    'semantic_relevance': semantic_relevance,
                    'business_importance': business_importance,
                    'query_frequency': query_frequency,
                    'data_relationship': data_relationship
                }
                
            except Exception as e:
                print(f"Error parsing relationship semantics: {e}")
        
        return edge_semantics

    def _get_feature_weights(self, graph, metadata):
        """
        Get optimal weights for node and edge features using LLama LLM.
        
        Args:
            graph: NetworkX graph
            metadata: Schema metadata
            
        Returns:
            Tuple of (node_weights, edge_weights)
        """
        # Format schema for prompt
        schema_text = self._format_schema_for_prompt(metadata)
        
        # Create prompt
        prompt = f"""
        Your task is to determine the importance of different feature types when assessing relationships in this database schema.
        
        {schema_text}
        
        For node features, determine weights for:
        1. Semantic Features (LLM-derived meaning, purpose, roles)
        2. Structural Features (position in schema, connections)
        3. Type & Statistical Features (data types, cardinality, nullability)
        
        For edge features, determine weights for:
        1. Structural Features (foreign keys, table-column relationships)
        2. Semantic Features (LLM-derived semantic relationships) 
        3. Compatibility Features (type compatibility, same-table relationships)
        
        Weights should sum to 1.0 for each category.
        
        Respond in this exact format:
        <NODE_WEIGHTS>
        semantic: 0.X
        structural: 0.X
        type_statistical: 0.X
        </NODE_WEIGHTS>
        
        <EDGE_WEIGHTS>
        structural: 0.X
        semantic: 0.X
        compatibility: 0.X
        </EDGE_WEIGHTS>
        """
        
        # Call LLama LLM
        llm_response = self._call_llama(prompt, max_tokens=500, temperature=0.2)
  
        
        # Default weights
        node_weights = np.array([0.4, 0.3, 0.3])  # Balanced default
        edge_weights = np.array([0.4, 0.4, 0.2])  # Balanced default
        
        # Extract node weights
        node_match = re.search(r'<NODE_WEIGHTS>(.*?)</NODE_WEIGHTS>', llm_response, re.DOTALL)
        
        if node_match:
            node_text = node_match.group(1).strip()
            try:
                weights = []
                for line in node_text.split('\n'):
                    if ':' in line:
                        weight = float(re.search(r'0\.\d+', line).group(0))
                        weights.append(weight)
                if len(weights) == len(node_weights) and abs(sum(weights) - 1.0) < 0.1:
                    node_weights = np.array(weights)
                    node_weights = node_weights / sum(node_weights)  # Normalize
            except:
                pass  # Keep default weights if parsing fails
            
        edge_match = re.search(r'<EDGE_WEIGHTS>(.*?)</EDGE_WEIGHTS>', llm_response, re.DOTALL)
        if edge_match:
            node_text = edge_match.group(1).strip()
            try:
                weights = []
                for line in node_text.split('\n'):
                    if ':' in line:
                        weight = float(re.search(r'0\.\d+', line).group(0))
                        weights.append(weight)
                if len(weights) == len(edge_weights) and abs(sum(weights) - 1.0) < 0.1:
                    edge_weights = np.array(weights)
                    edge_weights = edge_weights / sum(edge_weights)  # Normalize
            except:
                pass  # Keep default weights if parsing fails
        
        return node_weights, edge_weights

    def _format_schema_for_prompt(self, schema_details):
        """
        Format schema details for inclusion in LLM prompts.
        
        Args:
            schema_details: List of dictionaries containing schema information
            
        Returns:
            String representation of the schema
        """
        schema_text = "DATABASE SCHEMA:\n"
        
        for table in schema_details:
            schema_text += f"Table: {table['table_name']}\n"
            schema_text += "Columns:\n"
            
            for col in table['columns']:
                pk_marker = " (Primary Key)" if col['is_pk'] else ""
                null_marker = " NOT NULL" if col['not_null'] else ""
                schema_text += f"  - {col['name']} ({col['type']}){pk_marker}{null_marker}\n"
            
            if table['foreign_keys']:
                schema_text += "Foreign Keys:\n"
                for fk in table['foreign_keys']:
                    schema_text += f"  - {fk['column']} -> {fk['ref_table']}.{fk['ref_column']}\n"
            
            schema_text += "\n"
        
        return schema_text

    def _call_llama(self, prompt, max_tokens=1000, temperature=0.2):
        """
        Call LLama LLM API with the given prompt.
        
        Args:
            prompt: Text prompt to send to LLM
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature
            
        Returns:
            LLM response as string

        """
        # If LLM client is provided, use it
        if self.llm_client:
            try:
                response = self.llm_client.call_llm(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response
            except Exception as e:
                print(f"Error calling LLama LLM: {e}")
                return ""
        else:
            # Mock response for testing when no LLM client is available
            print("No LLM client provided. Using mock response.")
            # This would be replaced with actual LLM call in production
            return f"""
            <NODE>
            name: users
            purpose_score: 0.9
            domain_relevance: 0.8
            relationship_score: 0.7
            data_richness: 0.6
            centrality: 0.9
            tags: user, authentication, core
            </NODE>
            
            <RELATIONSHIP>
            column1: users.id
            column2: orders.user_id
            semantic_relevance: 0.9
            business_importance: 0.8
            query_frequency: 0.7
            data_relationship: 0.9
            </RELATIONSHIP>
            
            <NODE_WEIGHTS>
            semantic: 0.5
            structural: 0.3
            type_statistical: 0.2
            </NODE_WEIGHTS>
            
            <EDGE_WEIGHTS>
            structural: 0.4
            semantic: 0.4
            compatibility: 0.2
            </EDGE_WEIGHTS>
            """



def save_graph(graph, output_path = "schema_graph.json"):
    """
    Save the graph to a file for later use.
    
    Args:
        graph: NetworkX graph
        output_file: Path to output file
    """
    # Convert graph to a serializable format
    serializable_graph = {
        'nodes': [],
        'edges': []
    }
    
    for node, attrs in graph.nodes(data=True):
        node_data = {'id': node}
        node_data.update(attrs)
        serializable_graph['nodes'].append(node_data)
    
    for source, target, attrs in graph.edges(data=True):
        edge_data = {'source': source, 'target': target}
        edge_data.update(attrs)
        serializable_graph['edges'].append(edge_data)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(serializable_graph, f, indent=2)
    
    print(f"Graph saved to {output_path}")



def compress_schema_to_txt_string(full_schema_path):
    with open(full_schema_path, "r") as f:
        schema = json.load(f)

    txt_lines = []

    # Process nodes
    txt_lines.append("NODES:")
    for node in schema.get("nodes", []):
        if node["type"] == "table":
            txt_lines.append(f'  - [TABLE] id: {node["id"]}')
        elif node["type"] == "column":
            txt_lines.append(
                f'  - [COLUMN] id: {node["id"]}, table: {node.get("table")},'
                f'dtype: {node.get("data_type")}, PK: {node.get("is_primary_key")}, '
                f'FK: {node.get("is_foreign_key")}'
            )

    # Process edges
    txt_lines.append("\nEDGES:")
    for edge in schema.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        relation = edge.get("relationship_type")
        weight = round(edge.get("weight", 0), 3)
        txt_lines.append(f'  - {source} --[{relation}, weight={weight}]--> {target}')

    return "\n".join(txt_lines)



def main():
    # Load configuration from JSON file
    config_file = "./configs/config_local004.json"    
    with open(config_file, 'r') as f:
        config = json.load(f)

    
    db_name = os.path.basename(config['db_file'])
    db_name = os.path.splitext(db_name)[0]
    graph_json = 'init_graph_' + db_name + '.json'

    print('..... db_name:', db_name, ', ..... graph_json:', graph_json)

    
    # Initialize LLM service
    llm_service = LLMService(
        provider=config['provider'],
        api_url=config['api_url'],
        api_key=config['api_key'],
        model=config.get('model')  # optional
    )
    
    # Example LLM call to check the api call
    response = llm_service.call_llm("Write a very short message that you are up and running!")
    print(response)

    # load json summary file & initial graph
    graph = SchemaGraphBuilder.load_graph( os.path.join("./results_init_graph", graph_json))
    print('init graph file:', os.path.join("./results_init_graph", graph_json))

    json_summary_file = "db_summary_" + db_name +  ".json"
    db_summary_path = os.path.join("./results_init_graph", json_summary_file)
    print('json_summary_file:', json_summary_file)

    # Create and use the enhancer with the JSON summary
    schema_details = SchemaGraphBuilder.extract_schema_details(graph)
    enhancer = SchemaGraphEnhancer(llm_service)
    enhanced_graph = enhancer.enhance_edge_semantics(graph, schema_details, db_summary_path)

    # Visualize and save the enhanced graph
    output_path = os.path.join("./results_init_graph", "enhanced_graph_" + db_name +".json" )
    save_graph(enhanced_graph, output_path=output_path)

    output_file =  os.path.join("./results_init_graph", "graph_enhanced_" + db_name + ".png")
    visualize_graph(graph, output_file=output_file)




if __name__ == "__main__":
    main()
