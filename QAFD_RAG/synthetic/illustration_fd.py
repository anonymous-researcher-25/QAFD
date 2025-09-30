import asyncio
import json
import re
from typing import Union
from collections import Counter, defaultdict
import warnings
import time
import csv
import networkx as nx
import random
import logging
import urllib.request
import urllib.error



class WeightedFlowDiffusion:
    def __init__(self, graph, source_node, target_node, confidence=0.5, epsilon=0.05):
        """
        Initialize the Weighted Flow Diffusion algorithm.
        
        Parameters:
        -----------
        graph : networkx.Graph
            Graph data containing nodes and edges
        source_node : str
            Source node ID
        target_node : str
            Target node ID
        confidence : float
            Confidence score for this source-target pair (0.0 to 1.0)
        epsilon : float
            Convergence threshold for the algorithm
        """
        self.graph = graph
        self.source = source_node
        self.target = target_node
        self.confidence = max(0.0, min(1.0, confidence))  # Ensure valid range
        self.epsilon = epsilon
        self.mass = defaultdict(float)  # Current mass at each node
        self.x = defaultdict(float)     # Solution vector x
        self.sink_capacity = defaultdict(float)  # Sink capacity for each node
        
    def initialize(self, alpha=10, use_node_degree=True):
        """
        Initialize source mass and sink capacities.
        
        Parameters:
        -----------
        alpha : float
            Multiplier for source mass (should be > 1)
        use_node_degree : bool
            If True, set sink capacity to node degree, otherwise set to 1
        """
        # Set sink capacity for all nodes based on their degree
        for node in self.graph.nodes():
            if use_node_degree:
                # Set sink capacity to node degree (number of connections)
                self.sink_capacity[node] = self.graph.degree(node)
            else:
                # Set sink capacity to 1 (original behavior)
                self.sink_capacity[node] = 1
        
        # Set all masses to 0 initially
        for node in self.graph.nodes():
            self.mass[node] = 0
        
        # Set source mass - boost based on confidence
        try:
            path = nx.shortest_path(self.graph, self.source, self.target)
            total_sink_on_path = sum(self.sink_capacity[node] for node in path)
            # Boost source mass based on confidence (higher confidence = more initial mass)
            confidence_boost = 1.0 + self.confidence
            self.mass[self.source] = alpha * total_sink_on_path * confidence_boost
        except nx.NetworkXNoPath:
            avg_sink = sum(self.sink_capacity.values()) / len(self.sink_capacity)
            confidence_boost = 1.0 + self.confidence
            self.mass[self.source] = alpha * avg_sink * len(self.graph.nodes()) / 10 * confidence_boost
    
    def push(self, node):
        """
        Push operation as described in Algorithm 1 in the paper.
        Diffuses excess mass from a node to its neighbors.
        """
        # Calculate w_i (sum of weights of edges connected to node i)
        w_i = sum(self.graph[node][neighbor]['weight'] for neighbor in self.graph.neighbors(node))
        
        if w_i == 0:
            return False  # No neighbors to push to
        
        # Calculate excess mass (m_i - T_i)
        excess = self.mass[node] - self.sink_capacity[node]
        
        if excess <= 0:
            return False  # No excess mass to push
        
        # Store the excess mass before updating
        excess_to_distribute = excess
        
        # Update x_i by adding (m_i - T_i)/w_i
        self.x[node] += excess / w_i
        
        # Update mass at node i to its capacity (m_i = T_i)
        self.mass[node] = self.sink_capacity[node]
        
        # Distribute excess mass to neighbors: m_j += (m_i - T_i)w_ij/w_i
        for neighbor in self.graph.neighbors(node):
            w_ij = self.graph[node][neighbor]['weight']
            self.mass[neighbor] += excess_to_distribute * w_ij / w_i
        
        return True
    
    def flow_diffusion(self, max_iterations=5000):
        """
        Run the flow diffusion algorithm (Algorithm 1 in the paper).
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations
            
        Returns:
        --------
        dict
            Nodes with positive x values and their values
        """
        iterations = 0
        pushes = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Find nodes with excess mass {j : m_j > T_j}
            excess_nodes = [node for node in self.graph.nodes() 
                           if self.mass[node] > self.sink_capacity[node]]
            
            if not excess_nodes:
                break  # No more excess mass
            
            # Pick a node uniformly at random from those with excess mass
            node = random.choice(excess_nodes)
            
            # Apply push operation
            if self.push(node):
                pushes += 1
            
            # Check for convergence periodically
            if iterations % 100 == 0:
                remaining_excess = sum(max(0, self.mass[node] - self.sink_capacity[node]) 
                                     for node in self.graph.nodes())
                if remaining_excess < self.epsilon:
                    break
        
        # logger.info(f"Flow diffusion completed in {iterations} iterations with {pushes} pushes")
        return {node: val for node, val in self.x.items() if val > 0}

    def find_furthest_reachable_node(self, diffused_nodes):
        """
        Find the furthest reachable node from source within the diffused subgraph.
        """
        if self.source not in diffused_nodes:
            return None, None
            
        # Create subgraph of diffused nodes
        support_nodes = set(diffused_nodes.keys())
        subgraph = self.graph.subgraph(support_nodes)
        
        # Find furthest node using BFS
        from collections import deque
        queue = deque([(self.source, 0, [self.source])])
        visited = {self.source}
        furthest_nodes = []
        max_distance = 0
        paths_to_nodes = {self.source: [self.source]}
        
        while queue:
            current_node, distance, path = queue.popleft()
            
            # Update furthest nodes if we found a longer path
            if distance > max_distance:
                max_distance = distance
                furthest_nodes = [current_node]
            elif distance == max_distance and current_node not in furthest_nodes:
                furthest_nodes.append(current_node)
            
            # Explore neighbors
            for neighbor in subgraph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    new_distance = distance + 1
                    queue.append((neighbor, new_distance, new_path))
                    paths_to_nodes[neighbor] = new_path
        
        # Choose the one with highest flow diffusion value among furthest nodes
        if furthest_nodes:
            chosen_node = max(furthest_nodes, key=lambda n: diffused_nodes.get(n, 0)) if len(furthest_nodes) > 1 else furthest_nodes[0]
            logger.info(f"Furthest reachable node: {chosen_node} at distance {max_distance} from source")
            return chosen_node, paths_to_nodes[chosen_node]
        
        return None, None

    def calculate_path_score(self, path, confidence_weight=0.7, flow_weight=0.3):
        """
        Calculate combined score for a path using confidence and flow strength.
        
        Parameters:
        -----------
        path : list
            Path as a list of node IDs
        confidence_weight : float
            Weight for confidence score (default 0.7)
        flow_weight : float
            Weight for flow strength (default 0.3)
            
        Returns:
        --------
        float
            Combined score (0.0 to 1.0, higher is better)
        """
        if not path or len(path) < 2:
            return 0.0
        
        # Calculate flow strength: average x values along the path
        flow_values = [self.x[node] for node in path if node in self.x and self.x[node] > 0]
        if not flow_values:
            flow_strength = 0.0
        else:
            # Normalize flow strength
            avg_flow = sum(flow_values) / len(flow_values)
            # Normalize to 0-1 range using sigmoid-like function
            flow_strength = avg_flow / (avg_flow + 1.0)
        
        # Calculate weighted combined score
        combined_score = (confidence_weight * self.confidence) + (flow_weight * flow_strength)
        
        return combined_score

    def find_path(self):
        """
        Find a path from source to target using weighted flow diffusion.
        
        Returns:
        --------
        tuple
            (path, score) where path is a list of node IDs and score is the combined score
        """
        # Initialize source mass and sink capacities
        self.initialize()
        
        # Run flow diffusion
        diffused_nodes = self.flow_diffusion()
        
        # Check if diffusion reached the target
        if self.target not in diffused_nodes:
            logger.info(f"Flow diffusion did not reach target node {self.target}")
            
            # Find furthest reachable node instead
            if len(diffused_nodes) > 1:
                furthest_node, furthest_path = self.find_furthest_reachable_node(diffused_nodes)
                if furthest_path:
                    score = self.calculate_path_score(furthest_path)
                    logger.info(f"Path found to furthest reachable node: {' -> '.join(furthest_path)} (score: {score:.3f})")
                    return furthest_path, score
            
            # If only source has flow or no furthest node found
            fallback_path = [self.source] if self.source in diffused_nodes else None
            fallback_score = self.calculate_path_score(fallback_path) if fallback_path else 0.0
            return fallback_path, fallback_score
        
        # Create a subgraph of nodes where mass diffused to
        support_nodes = set(diffused_nodes.keys())
        subgraph = self.graph.subgraph(support_nodes)
        
        # Reweight edges based on x values
        weighted_subgraph = nx.Graph()
        for u, v, data in subgraph.edges(data=True):
            # Higher x values indicate more flow, so we want to prioritize these paths
            # Use inverse of x values as edge weight for shortest path
            new_weight = 1.0 / (self.x[u] + self.x[v] + 1e-10)
            weighted_subgraph.add_edge(u, v, weight=new_weight)
        
        # Find shortest path in reweighted graph
        try:
            path = nx.shortest_path(weighted_subgraph, self.source, self.target, weight='weight')
            score = self.calculate_path_score(path)
            logger.info(f"Path found: {' -> '.join(path)} (score: {score:.3f})")
            return path, score
        except nx.NetworkXNoPath:
            logger.info(f"No path found from {self.source} to {self.target} in diffused subgraph")
            return None, 0.0


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
            # smart_weight = original_weight * (1.0 + query_factor * 0.5)
            smart_weight = original_weight * (node1_query_sim**2) * (node2_query_sim**2)
            print(f'node1: {node1}')
            print(f'node2: {node2}')
            print(f"query_factor: {query_factor}")
            print(f"smart_weight: {smart_weight}")

        # No edge description blending; keep algorithm unchanged

        # Cache and return
        self.edge_weights_cache[cache_key] = smart_weight
        return smart_weight
        
    def initialize(self, alpha=10, use_node_degree=True):
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
    
    def flow_diffusion(self, max_iterations=5000):
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


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

# Basic logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# Step 1: Node Information
# ========================
nodes_info = {
    "Apple": {
        "description": "Merged entity of Apple Inc. (technology company) and Apple (fruit).",
        "types": ["Company", "Fruit"],
        "source": "Apple Inc. + Apple (fruit)"
    },
    "Amazon": {
        "description": "Merged entity of Amazon.com, Inc. (tech company) and Amazon River.",
        "types": ["Company", "River"],
        "source": "Amazon.com, Inc. + Amazon River"
    },
    "Tim Cook": {"description": "CEO of Apple", "types": ["Person"], "source": "Apple Inc."},
    "Steve Jobs": {"description": "Co-founder of Apple", "types": ["Person"], "source": "Apple Inc."},
    "Jeff Bezos": {"description": "Founder of Amazon", "types": ["Person"], "source": "Amazon.com, Inc."},
    "Andy Jassy": {"description": "CEO of Amazon", "types": ["Person"], "source": "Amazon.com, Inc."},
    "iPhone": {"description": "Apple's flagship smartphone", "types": ["Product"], "source": "Apple Inc."},
    "iPad": {"description": "Apple's tablet device", "types": ["Product"], "source": "Apple Inc."},
    "Mac": {"description": "Apple's line of personal computers", "types": ["Product"], "source": "Apple Inc."},
    "Apple Watch": {"description": "Apple's smartwatch", "types": ["Product"], "source": "Apple Inc."},
    "iOS": {"description": "Apple's mobile operating system", "types": ["Software"], "source": "Apple Inc."},
    "macOS": {"description": "Apple's desktop operating system", "types": ["Software"], "source": "Apple Inc."},
    "App Store": {"description": "Apple's digital distribution platform", "types": ["Service"], "source": "Apple Inc."},
    "Apple Music": {"description": "Apple's music streaming service", "types": ["Service"], "source": "Apple Inc."},
    "iCloud": {"description": "Apple's cloud storage service", "types": ["Service"], "source": "Apple Inc."},
    "Apple TV+": {"description": "Apple's streaming service", "types": ["Service"], "source": "Apple Inc."},
    "Apple Pay": {"description": "Apple's mobile payment service", "types": ["Service"], "source": "Apple Inc."},
    "Apple Card": {"description": "Apple's credit card", "types": ["Service"], "source": "Apple Inc."},
    "Whole Foods Market": {"description": "Grocery chain owned by Amazon", "types": ["Company"], "source": "Amazon.com, Inc."},
    "Twitch": {"description": "Streaming platform owned by Amazon", "types": ["Service"], "source": "Amazon.com, Inc."},
    "Kindle": {"description": "Amazon's e-reader", "types": ["Product"], "source": "Amazon.com, Inc."},
    "Echo": {"description": "Amazon's smart speaker", "types": ["Product"], "source": "Amazon.com, Inc."},
    "Fire TV": {"description": "Amazon's streaming media player", "types": ["Product"], "source": "Amazon.com, Inc."},
    "Zoox": {"description": "Autonomous vehicle company acquired by Amazon", "types": ["Company"], "source": "Amazon.com, Inc."},
    "Malus domestica": {"description": "Scientific name for the domesticated apple", "types": ["Species"], "source": "Apple (fruit)"},
    "Malus": {"description": "Genus of apples", "types": ["Genus"], "source": "Apple (fruit)"},
    "Rosaceae": {"description": "Rose family, includes apples", "types": ["Family"], "source": "Apple (fruit)"},
    "Central Asia": {"description": "Region where Malus sieversii is native", "types": ["Place"], "source": "Apple (fruit)"},
    "Malus sieversii": {"description": "Wild apple species native to Central Asia", "types": ["Species"], "source": "Apple (fruit)"},
    "Fuji": {"description": "Popular apple cultivar from Japan", "types": ["Cultivar"], "source": "Apple (fruit)"},
    "Gala": {"description": "Sweet apple cultivar", "types": ["Cultivar"], "source": "Apple (fruit)"},
    "Golden Delicious": {"description": "Yellow apple cultivar", "types": ["Cultivar"], "source": "Apple (fruit)"},
    "Granny Smith": {"description": "Green apple cultivar from Australia", "types": ["Cultivar"], "source": "Apple (fruit)"},
    "Red Delicious": {"description": "Red apple cultivar", "types": ["Cultivar"], "source": "Apple (fruit)"},
    "Vitamin C": {"description": "Essential nutrient abundant in apples", "types": ["Nutrient"], "source": "Apple (fruit)"},
    "Dietary Fiber": {"description": "Important component of apple nutrition", "types": ["Nutrient"], "source": "Apple (fruit)"},
    "Antioxidants": {"description": "Health-promoting compounds in apples", "types": ["Compound"], "source": "Apple (fruit)"},
    "Apple Juice": {"description": "Beverage made from pressed apples", "types": ["Product"], "source": "Apple (fruit)"},
    "Apple Pie": {"description": "Traditional dessert made with apples", "types": ["Food"], "source": "Apple (fruit)"},
    "Cider": {"description": "Alcoholic beverage made from fermented apple juice", "types": ["Beverage"], "source": "Apple (fruit)"},
    "Pollination": {"description": "Process required for apple fruit production", "types": ["Process"], "source": "Apple (fruit)"},
    "Honeybees": {"description": "Primary pollinators of apple trees", "types": ["Animal"], "source": "Apple (fruit)"},
    "Amazon basin": {"description": "Region drained by the Amazon River", "types": ["Place"], "source": "Amazon River"},
    "Brazil": {"description": "Country in South America, part of Amazon basin", "types": ["Place"], "source": "Amazon River"},
    "Peru": {"description": "Country in South America, part of Amazon basin", "types": ["Place"], "source": "Amazon River"},
    "Andes": {"description": "Mountain range in South America near Amazon basin", "types": ["Place"], "source": "Amazon River"},
    "Iquitos": {"description": "City in the Peruvian Amazon", "types": ["Place"], "source": "Amazon River"},
    "Atlantic Ocean": {"description": "Ocean receiving the Amazon River", "types": ["Ocean"], "source": "Amazon River"},
    "Marañón River": {"description": "Major tributary of the Amazon River", "types": ["River"], "source": "Amazon River"},
    "Ucayali River": {"description": "Major tributary of the Amazon River", "types": ["River"], "source": "Amazon River"},
    "Negro River": {"description": "Major tributary of the Amazon River", "types": ["River"], "source": "Amazon River"},
    "Madeira River": {"description": "Major tributary of the Amazon River", "types": ["River"], "source": "Amazon River"},
    "Manaus": {"description": "Major city in the Brazilian Amazon", "types": ["Place"], "source": "Amazon River"},
    "Belém": {"description": "City at the mouth of the Amazon River", "types": ["Place"], "source": "Amazon River"},
    "Amazon Rainforest": {"description": "Tropical rainforest in the Amazon basin", "types": ["Ecosystem"], "source": "Amazon River"},
    "Biodiversity": {"description": "Extremely high species diversity in Amazon region", "types": ["Concept"], "source": "Amazon River"},
    "Piranha": {"description": "Carnivorous fish found in Amazon River", "types": ["Animal"], "source": "Amazon River"},
    "Anaconda": {"description": "Large snake species in Amazon region", "types": ["Animal"], "source": "Amazon River"},
    "Amazon River Dolphin": {"description": "Freshwater dolphin species in Amazon River", "types": ["Animal"], "source": "Amazon River"},
    "6437 km": {"description": "Length of the Amazon River", "types": ["Measurement"], "source": "Amazon River"},
    "7.05 million km²": {"description": "Drainage basin area of the Amazon River", "types": ["Measurement"], "source": "Amazon River"},
    "Cupertino, California": {"description": "Apple headquarters location", "types": ["Place"], "source": "Apple Inc."},
    "AWS": {"description": "Amazon Web Services, cloud computing platform", "types": ["Service"], "source": "Amazon.com, Inc."},
    "Steve Wozniak": {"description": "Co-founder of Apple, engineered early Apple computers", "types": ["Person"], "source": "Apple Inc."},
    "Amazon Prime": {"description": "Amazon’s paid membership bundling shipping, media, and other benefits", "types": ["Service"], "source": "Amazon.com, Inc."},
    "Amazon Prime Video": {"description": "Amazon’s subscription video streaming service", "types": ["Service"], "source": "Amazon.com, Inc."}
}

# =========================
# Step 2: Relationship Info
# =========================
edges_info = [
    ("Apple", "Tim Cook", 0.95, "Tim Cook is CEO of Apple", ["leadership"]),
    ("Apple", "Steve Jobs", 0.95, "Steve Jobs co-founded Apple", ["founder"]),
    ("Apple", "Steve Wozniak", 0.90, "Steve Wozniak co-founded Apple and engineered early Apple computers", ["founder"]),
    ("Apple", "iPhone", 0.98, "iPhone is Apple’s flagship product", ["product"]),
    ("Apple", "iPad", 0.93, "iPad is a tablet by Apple", ["product"]),
    ("Apple", "Mac", 0.93, "Mac is Apple's personal computer line", ["product"]),
    ("Apple", "Apple Watch", 0.92, "Apple Watch is Apple's smartwatch", ["product"]),
    ("Apple", "iOS", 0.94, "iOS is Apple's mobile OS", ["software"]),
    ("Apple", "macOS", 0.94, "macOS is Apple's desktop OS", ["software"]),
    ("Apple", "App Store", 0.90, "App Store is Apple's digital distribution platform", ["service"]),
    ("Apple", "Apple Music", 0.90, "Apple Music is Apple's music streaming service", ["service"]),
    ("Apple", "iCloud", 0.90, "iCloud is Apple's cloud storage service", ["service"]),
    ("Apple", "Apple TV+", 0.88, "Apple TV+ is Apple's streaming service", ["service"]),
    ("Apple", "Apple Pay", 0.89, "Apple Pay is Apple's mobile payment service", ["service"]),
    ("Apple", "Apple Card", 0.87, "Apple Card is Apple's credit card", ["service"]),
    ("Apple", "Cupertino, California", 0.95, "Apple headquarters in Cupertino, California", ["location"]),
    ("Amazon", "Jeff Bezos", 0.95, "Jeff Bezos founded Amazon", ["founder"]),
    ("Amazon", "Andy Jassy", 0.95, "Andy Jassy is CEO of Amazon", ["leadership"]),
    ("Amazon", "Amazon Prime", 0.90, "Amazon Prime is Amazon’s paid membership program", ["subscription"]),
    ("Amazon", "Amazon Prime Video", 0.90, "Prime Video is Amazon’s streaming service", ["digital_media_service"]),
    ("Amazon", "AWS", 0.97, "AWS is Amazon’s cloud division", ["cloud"]),
    ("Amazon", "Whole Foods Market", 0.90, "Whole Foods is owned by Amazon", ["acquisition"]),
    ("Amazon", "Twitch", 0.90, "Twitch is owned by Amazon", ["acquisition"]),
    ("Amazon", "Kindle", 0.92, "Kindle is Amazon's e-reader", ["product"]),
    ("Amazon", "Echo", 0.91, "Echo is Amazon's smart speaker", ["product"]),
    ("Amazon", "Fire TV", 0.90, "Fire TV is Amazon's streaming device", ["product"]),
    ("Amazon", "Zoox", 0.85, "Zoox is an autonomous vehicle company acquired by Amazon", ["acquisition"]),
    ("Amazon", "Atlantic Ocean", 0.95, "Amazon River flows into Atlantic Ocean", ["geography"]),
    ("Amazon", "Brazil", 0.90, "A large portion of the Amazon’s course and basin lies in Brazil", ["geography"]),
    ("Amazon", "Peru", 0.85, "Upper tributaries meet near Iquitos, Peru, forming much of the main stem", ["geography"]),
    ("Amazon", "Andes", 0.85, "Headwaters arise in the Andes via tributaries", ["source_region"]),
    ("Amazon", "Iquitos", 0.75, "Iquitos is a major city near the confluence of Ucayali and Marañón", ["riverine_settlement"]),
    ("Amazon basin", "Brazil", 0.90, "Brazil is part of Amazon basin", ["geography"]),
    ("Amazon basin", "Peru", 0.90, "Peru is part of Amazon basin", ["geography"]),
    ("Amazon basin", "Andes", 0.88, "Andes are near Amazon basin", ["geography"]),
    ("Amazon basin", "Iquitos", 0.87, "Iquitos is a city in the Amazon basin", ["geography"]),
    ("Apple", "Amazon", 0.95, "Both are Big Tech entities", ["big tech"]),
    ("Malus domestica", "Malus", 0.95, "Malus domestica belongs to genus Malus", ["taxonomy"]),
    ("Malus", "Rosaceae", 0.90, "Malus is part of the Rosaceae family", ["taxonomy"]),
    ("Malus sieversii", "Malus", 0.92, "Malus sieversii is a species of Malus", ["taxonomy"]),
    ("Malus sieversii", "Central Asia", 0.85, "Malus sieversii is native to Central Asia", ["geography"]),
    ("Malus domestica", "Apple", 0.95, "Malus domestica is the scientific name for domesticated apple", ["taxonomy"]),
    ("Malus domestica", "Malus sieversii", 0.90, "The wild ancestor of domesticated apple is Malus sieversii", ["phylogeny", "ancestry"]),
    ("Apple", "Central Asia", 0.85, "Domesticated apples originated in Central Asia", ["biogeography"]),
    ("Mac", "macOS", 0.92, "macOS runs on Apple's Mac computers", ["software_hardware"]),
    ("iPhone", "iOS", 0.94, "iOS powers Apple's iPhone devices", ["software_hardware"]),
    ("iOS", "App Store", 0.90, "App Store distributes iOS apps", ["ecosystem"]),
    # Enhanced Apple (fruit) relations
    ("Apple", "Fuji", 0.88, "Fuji is a popular apple cultivar", ["cultivar"]),
    ("Apple", "Gala", 0.88, "Gala is a sweet apple cultivar", ["cultivar"]),
    ("Apple", "Golden Delicious", 0.88, "Golden Delicious is a yellow apple cultivar", ["cultivar"]),
    ("Apple", "Granny Smith", 0.88, "Granny Smith is a green apple cultivar", ["cultivar"]),
    ("Apple", "Red Delicious", 0.88, "Red Delicious is a red apple cultivar", ["cultivar"]),
    ("Apple", "Vitamin C", 0.90, "Apples are rich in Vitamin C", ["nutrition"]),
    ("Apple", "Dietary Fiber", 0.90, "Apples contain dietary fiber", ["nutrition"]),
    ("Apple", "Antioxidants", 0.85, "Apples contain health-promoting antioxidants", ["nutrition"]),
    ("Apple", "Apple Juice", 0.92, "Apple juice is made from pressed apples", ["product"]),
    ("Apple", "Apple Pie", 0.90, "Apple pie is a traditional dessert made with apples", ["culinary"]),
    ("Apple", "Cider", 0.88, "Cider is made from fermented apple juice", ["beverage"]),
    ("Apple", "Pollination", 0.95, "Apple trees require pollination to produce fruit", ["cultivation"]),
    ("Apple", "Honeybees", 0.90, "Honeybees are primary pollinators of apple trees", ["pollination"]),
    ("Fuji", "Malus domestica", 0.95, "Fuji is a cultivar of Malus domestica", ["taxonomy"]),
    ("Gala", "Malus domestica", 0.95, "Gala is a cultivar of Malus domestica", ["taxonomy"]),
    ("Golden Delicious", "Malus domestica", 0.95, "Golden Delicious is a cultivar of Malus domestica", ["taxonomy"]),
    ("Granny Smith", "Malus domestica", 0.95, "Granny Smith is a cultivar of Malus domestica", ["taxonomy"]),
    ("Red Delicious", "Malus domestica", 0.95, "Red Delicious is a cultivar of Malus domestica", ["taxonomy"]),
    # Enhanced Amazon River relations
    ("Amazon", "Marañón River", 0.90, "Marañón River is a major tributary of the Amazon", ["tributary"]),
    ("Amazon", "Ucayali River", 0.90, "Ucayali River is a major tributary of the Amazon", ["tributary"]),
    ("Amazon", "Negro River", 0.90, "Negro River is a major tributary of the Amazon", ["tributary"]),
    ("Amazon", "Madeira River", 0.90, "Madeira River is a major tributary of the Amazon", ["tributary"]),
    ("Amazon", "Manaus", 0.85, "Manaus is a major city on the Amazon River", ["settlement"]),
    ("Amazon", "Belém", 0.90, "Belém is located at the mouth of the Amazon River", ["settlement"]),
    ("Amazon", "Amazon Rainforest", 0.95, "Amazon Rainforest is located in the Amazon basin", ["ecosystem"]),
    ("Amazon", "Biodiversity", 0.95, "Amazon region has extremely high biodiversity", ["ecology"]),
    ("Amazon", "Piranha", 0.80, "Piranha fish are found in the Amazon River", ["fauna"]),
    ("Amazon", "Anaconda", 0.80, "Anaconda snakes are found in the Amazon region", ["fauna"]),
    ("Amazon", "Amazon River Dolphin", 0.85, "Amazon River Dolphin lives in the Amazon River", ["fauna"]),
    ("Amazon", "6437 km", 0.95, "Amazon River is 6437 km long", ["measurement"]),
    ("Amazon", "7.05 million km²", 0.95, "Amazon River has a drainage basin of 7.05 million km²", ["measurement"]),
    ("Marañón River", "Andes", 0.90, "Marañón River originates in the Andes", ["source"]),
    ("Ucayali River", "Andes", 0.90, "Ucayali River originates in the Andes", ["source"]),
    ("Marañón River", "Ucayali River", 0.85, "Marañón and Ucayali rivers meet to form the Amazon", ["confluence"]),
    ("Amazon Rainforest", "Biodiversity", 0.95, "Amazon Rainforest contains extremely high biodiversity", ["ecology"]),
    ("Amazon Rainforest", "Brazil", 0.90, "Amazon Rainforest is primarily located in Brazil", ["geography"]),
    ("Amazon Rainforest", "Peru", 0.85, "Amazon Rainforest extends into Peru", ["geography"]),
]

# =========================
# Step 3: Filter and Build Graph
# =========================

def filter_clusters(nodes_info, edges_info, max_entities_per_cluster=7):
    # Group nodes by source
    source_groups = {}
    for node, attrs in nodes_info.items():
        source = attrs["source"]
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(node)
    
    # Calculate relevance scores
    node_scores = {}
    for src, tgt, strength, desc, keywords in edges_info:
        node_scores[src] = node_scores.get(src, 0) + strength
        node_scores[tgt] = node_scores.get(tgt, 0) + strength
    
    # Filter each cluster
    filtered_nodes = set()
    for source, nodes in source_groups.items():
        filtered_nodes_in_cluster = [node for node in nodes if node not in ["Apple", "Amazon"]]
        filtered_nodes_in_cluster.sort(key=lambda x: node_scores.get(x, 0), reverse=True)
        top_nodes = filtered_nodes_in_cluster[:max_entities_per_cluster]
        
        # Always include Apple and Amazon
        for node in nodes:
            if node in ["Apple", "Amazon"]:
                top_nodes.append(node)
        
        filtered_nodes.update(top_nodes)
    
    # Filter data
    filtered_nodes_info = {node: attrs for node, attrs in nodes_info.items() if node in filtered_nodes}
    filtered_edges_info = [(src, tgt, strength, desc, keywords) for src, tgt, strength, desc, keywords in edges_info 
                          if src in filtered_nodes and tgt in filtered_nodes]
    
    return filtered_nodes_info, filtered_edges_info

# Apply filtering
filtered_nodes_info, filtered_edges_info = filter_clusters(nodes_info, edges_info)

G = nx.Graph()

# Add filtered nodes
for node, attrs in filtered_nodes_info.items():
    G.add_node(node, **attrs)

# Add filtered edges
for src, tgt, strength, desc, keywords in filtered_edges_info:
    G.add_edge(src, tgt, weight=strength, description=desc, keywords=keywords)




############################################
# Embedding utilities and query-run helpers
############################################

OPENAI_API_KEY = "Your OpenAI API Key"

def embed_texts_openai(texts, model="text-embedding-3-small", api_key=OPENAI_API_KEY):
    """Return list of embeddings for the provided texts using OpenAI embeddings API."""
    if not texts:
        return []
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({"input": texts, "model": model}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        resp_data = resp.read()
    data = json.loads(resp_data.decode("utf-8"))
    # The API returns in the same order as inputs
    return [item["embedding"] for item in data.get("data", [])]

def build_node_text(node, attrs):
    parts = [str(node)]
    desc = attrs.get("description")
    if desc:
        parts.append(str(desc))
    types = attrs.get("types")
    if types:
        parts.append("Types: " + ", ".join(map(str, types)))
    source = attrs.get("source")
    if source:
        parts.append(f"Source: {source}")
    return " | ".join(parts)

def build_edge_text(u, v, data):
    parts = [f"{u} <-> {v}"]
    desc = data.get("description")
    if desc:
        parts.append(str(desc))
    keywords = data.get("keywords")
    if keywords:
        parts.append("Keywords: " + ", ".join(map(str, keywords)))
    return " | ".join(parts)

def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0
    try:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        if mag1 == 0 or mag2 == 0:
            return 0.0
        sim = dot / (mag1 * mag2)
        return max(0.0, (sim + 1.0) / 2.0)
    except Exception:
        return 0.0

def run_diffusions_with_query(seed_node="Apple", query_text="Describe the Apple company."):
    # 1) Prepare texts
    node_texts = []
    node_order = []
    for n, attrs in G.nodes(data=True):
        node_order.append(n)
        node_texts.append(build_node_text(n, attrs))

    edge_texts = []
    edge_order = []
    for u, v, data in G.edges(data=True):
        edge_order.append((u, v))
        edge_texts.append(build_edge_text(u, v, data))

    # 2) Embed query, nodes, edges
    embeddings = embed_texts_openai([query_text] + node_texts + edge_texts)
    subquery_embedding = embeddings[0]
    node_embeddings_list = embeddings[1:1+len(node_texts)]
    edge_embeddings_list = embeddings[1+len(node_texts):]

    node_embeddings = {node_order[i]: node_embeddings_list[i] for i in range(len(node_order))}
    edge_embeddings = {edge_order[i]: edge_embeddings_list[i] for i in range(len(edge_order))}

    # 3) Edge embeddings computed but not used to modify FD algorithms

    # 4) Run plain WeightedFD (seed as both source/target)
    wfd = WeightedFlowDiffusion(G, seed_node, seed_node, confidence=0.9, epsilon=0.001)
    wfd.initialize(alpha=2)
    wfd_nodes = wfd.flow_diffusion()

    # 5) Run QueryAwareWFD (unchanged algorithm; only node/query embeddings used by weight_func)
    qawfd = QueryAwareWeightedFlowDiffusion(
        G,
        seed_node,
        seed_node,
        confidence=0.9,
        epsilon=0.001,
        node_embeddings=node_embeddings,
        subquery_embedding=subquery_embedding,
        weight_func=None,
    )
    qawfd.initialize(alpha=1.5)
    qawfd_nodes = qawfd.flow_diffusion()

    # 6) Prepare clusters (support sets) and sorted flow values
    def to_sorted_list(flow_dict):
        return sorted(flow_dict.items(), key=lambda kv: kv[1], reverse=True)

    wfd_sorted = to_sorted_list(wfd_nodes)
    qawfd_sorted = to_sorted_list(qawfd_nodes)

    print("\n=== Weighted Flow Diffusion (seed: Apple) ===")
    for n, val in wfd_sorted:
        print(f"{n}\t{val:.6f}")

    print("\n=== Query-Aware Weighted Flow Diffusion (seed: Apple) ===")
    for n, val in qawfd_sorted:
        print(f"{n}\t{val:.6f}")

    return wfd_nodes, qawfd_nodes

if __name__ == "__main__":
    try:
        run_diffusions_with_query("Apple", "Introduce Steve Jobs's products in Apple.")
    except Exception as e:
        logger.exception("Failed to run diffusions: %s", e)
