import networkx as nx
import random
from collections import defaultdict


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
        
    def initialize(self, alpha=50, use_node_degree=True):
        """
        Initialize source mass and sink capacities.
        
        Parameters:
        -----------
        alpha : float
            Multiplier for source mass (should be > 1)
        use_node_degree : bool
            If True, set sink capacity to node degree, otherwise set to 1
        """
        # Set sink capacity for all nodes
        for node in self.graph.nodes():
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
        Push operation:
        Diffuses excess mass from a node to its neighbors.
        """
        # Get neighbors
        neighbors = list(self.graph.neighbors(node))
        if not neighbors:
            return False  # No neighbors to push to
        
        # Calculate w_i (sum of weights of edges connected to node i)
        w_i = 0
        for neighbor in neighbors:
            # Handle missing weight attribute - missing weight means 0 (no flow)
            edge_data = self.graph[node][neighbor]
            weight = edge_data.get('weight', 0.0)  # Default to 0.0 if no weight
            w_i += weight
        
        if w_i == 0:
            return False  # No positive weights, no flow possible
        
        # Calculate excess mass (m_i - T_i)
        excess = self.mass[node] - self.sink_capacity[node]
        
        if excess <= 0:
            return False  # No excess mass to push
        
        # Update x_i by adding (m_i - T_i)/w_i
        self.x[node] += excess / w_i
        
        # Update mass at node i to its capacity (m_i = T_i)
        self.mass[node] = self.sink_capacity[node]
        
        # Distribute excess mass to neighbors: m_j += (m_i - T_i) * w_ij/w_i
        for neighbor in neighbors:
            edge_data = self.graph[node][neighbor]
            w_ij = edge_data.get('weight', 0.0)  # Default to 0.0 if no weight
            if w_ij > 0:  # Only distribute to edges with positive weight
                self.mass[neighbor] += excess * w_ij / w_i
        
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
        
        print(f"Flow diffusion completed in {iterations} iterations with {pushes} pushes")
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
            print(f"Furthest reachable node: {chosen_node} at distance {max_distance} from source")
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
            print(f"Flow diffusion did not reach target node {self.target}")
            
            # Find furthest reachable node instead
            if len(diffused_nodes) > 1:
                furthest_node, furthest_path = self.find_furthest_reachable_node(diffused_nodes)
                if furthest_path:
                    score = self.calculate_path_score(furthest_path)
                    print(f"Path found to furthest reachable node: {' -> '.join(furthest_path)} (score: {score:.3f})")
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
            print(f"Path found: {' -> '.join(path)} (score: {score:.3f})")
            return path, score
        except nx.NetworkXNoPath:
            print(f"No path found from {self.source} to {self.target} in diffused subgraph")
            return None, 0.0


def find_path_with_wfd(graph_data, source_node, target_node, confidence=0.5):
    """
    Find a path from source to target using Weighted Flow Diffusion with confidence scoring.
    
    Parameters:
    -----------
    graph_data : networkx.Graph
        Graph data containing nodes and edges
    source_node : str
        Source node ID
    target_node : str
        Target node ID
    confidence : float
        Confidence score for this source-target pair (0.0 to 1.0)
        
    Returns:
    --------
    tuple
        (path, score) where path is a list of node IDs and score is the combined score
    """
    wfd = WeightedFlowDiffusion(graph_data, source_node, target_node, confidence)
    return wfd.find_path()


def get_readable_paths(all_paths, delimiter=" -> "):
    """
    Convert a list of paths into readable strings.
    Now handles both simple paths and (path, score) tuples.
    """
    readable_paths = []
    for item in all_paths:
        if isinstance(item, tuple) and len(item) == 2:
            path, score = item
            readable_paths.append(f"{delimiter.join(path)} (score: {score:.3f})")
        else:
            # Assume it's just a path
            readable_paths.append(delimiter.join(item))
    return readable_paths