import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

"""
visualization.py - Module for visualizing database schema graphs with different relationship types
"""

def visualize_schema_graph_communities(graph, output_file="schema_communities.png"):
    """
    Visualize the database schema with community detection to highlight related tables.
    
    Args:
        graph: NetworkX graph of the schema
        output_file: Path to save the visualization
    """
    # Create a simplified graph with just tables for community detection
    table_graph = nx.Graph()
    
    # Get all table nodes
    table_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get('type') == 'table']
    
    # Add table nodes
    for node in table_nodes:
        table_graph.add_node(node)
    
    # Add edges between tables that have relationships
    for u, v, data in graph.edges(data=True):
        rel_type = data.get('relationship_type', '')
        if u in table_nodes and v in table_nodes and 'pk_fk_table' in rel_type:
            # Add weighted edge based on relationship type
            weight = 1.0
            if 'inferred' in rel_type:
                weight = 0.7  # Lower weight for inferred relationships
            
            # Add or update edge weight
            if table_graph.has_edge(u, v):
                table_graph[u][v]['weight'] += weight
            else:
                table_graph.add_edge(u, v, weight=weight)
    
    # Apply community detection algorithm
    try:
        from community import best_partition
        partition = best_partition(table_graph)
    except ImportError:
        try:
            # Attempt to use NetworkX's community detection as fallback
            from networkx.algorithms import community
            # Use Louvain method if available
            communities = community.louvain_communities(table_graph)
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
        except ImportError:
            print("Community detection requires python-louvain package.")
            print("Install with: pip install python-louvain")
            # Fallback to a simpler approach
            partition = {}
            for i, node in enumerate(table_graph.nodes()):
                partition[node] = i % 8  # Use 8 colors
    
    # Draw the community graph
    plt.figure(figsize=(16, 14))
    
    # Calculate positions
    pos = nx.spring_layout(table_graph, k=0.3, iterations=50, seed=42)
    
    # Get number of communities
    community_count = len(set(partition.values()))
    
    # Generate a color map
    cmap = plt.cm.get_cmap('tab20', community_count)
    
    # Draw nodes colored by community
    for community_id in set(partition.values()):
        nodes = [node for node in table_graph.nodes() if partition[node] == community_id]
        nx.draw_networkx_nodes(table_graph, pos,
                              nodelist=nodes,
                              node_size=1500,
                              node_color=[cmap(community_id)],
                              edgecolors='black',
                              alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(table_graph, pos,
                          width=1.5,
                          edge_color='gray',
                          alpha=0.6)
    
    # Add labels
    nx.draw_networkx_labels(table_graph, pos,
                           font_size=12,
                           font_weight='bold')
    
    # Get relationships count for each pair of communities
    community_links = defaultdict(int)
    for u, v in table_graph.edges():
        if partition[u] != partition[v]:
            key = tuple(sorted([partition[u], partition[v]]))
            community_links[key] += 1
    
    # Add annotations for community relationships
    for (comm1, comm2), count in community_links.items():
        if count > 1:  # Only annotate significant relationships
            # Find the centroid of each community
            comm1_nodes = [node for node in table_graph.nodes() if partition[node] == comm1]
            comm2_nodes = [node for node in table_graph.nodes() if partition[node] == comm2]
            
            centroid1 = np.mean([pos[node] for node in comm1_nodes], axis=0)
            centroid2 = np.mean([pos[node] for node in comm2_nodes], axis=0)
            
            # Calculate midpoint between centroids
            midpoint = (centroid1 + centroid2) / 2
            
            # Add a text annotation
            plt.text(midpoint[0], midpoint[1], f"{count} links",
                    fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title("Database Schema Communities", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Schema communities visualization saved to '{output_file}'")
    plt.close()

    """
    Create a matrix visualization showing relationships between tables.
    
    Args:
        graph: NetworkX graph of the schema
        output_file: Path to save the visualization
    """
    # Get all table nodes
    table_nodes = sorted([node for node, attrs in graph.nodes(data=True) if attrs.get('type') == 'table'])
    
    if not table_nodes:
        print("No tables found in the graph.")
        return
    
    # Create matrix to represent relationships
    num_tables = len(table_nodes)
    relationship_matrix = np.zeros((num_tables, num_tables))
    
    # Create a mapping from table names to matrix indices
    table_to_idx = {table: i for i, table in enumerate(table_nodes)}
    
    # Fill the matrix with relationship types
    # 0: No relationship
    # 1: Explicit relationship
    # 0.5: Inferred relationship
    for i, table1 in enumerate(table_nodes):
        for j, table2 in enumerate(table_nodes):
            if i == j:
                # Self-relationships are not interesting
                continue
                
            # Check direct table-to-table relationships
            has_explicit = False
            has_inferred = False
            
            for _, _, data in graph.edges([table1, table2], data=True):
                rel_type = data.get('relationship_type', '')
                if 'pk_fk_table' in rel_type:
                    if 'inferred' in rel_type:
                        has_inferred = True
                    else:
                        has_explicit = True
            
            # Set relationship value
            if has_explicit:
                relationship_matrix[i, j] = 1.0
            elif has_inferred:
                relationship_matrix[i, j] = 0.5
    
    # Create the matrix visualization
    plt.figure(figsize=(12, 10))
    
    # Use a colormap to represent relationship types
    cmap = plt.cm.Blues
    plt.imshow(relationship_matrix, cmap=cmap, interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['No Relationship', 'Inferred Relationship', 'Explicit Relationship'])
    
    # Add grid lines
    plt.grid(False)
    
    # Add table labels
    plt.xticks(range(num_tables), table_nodes, rotation=90, fontsize=8)
    plt.yticks(range(num_tables), table_nodes, fontsize=8)
    
    plt.title("Table Relationship Matrix", fontsize=14)
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Relationship matrix visualization saved to '{output_file}'")
    plt.close()


def visualize_graph(graph, title="Enhanced Schema Graph", figsize=(16, 14), output_file=None):
    """
    Visualize the enhanced schema graph with edge weights.
    Only edges with weight > 0 will be displayed.
    
    Args:
        graph: NetworkX graph with existing edge weights
        title: Title for the visualization
        figsize: Size of the figure
        output_file: Path to save the visualization (optional)
        
    Returns:
        matplotlib figure and axes
    """
    # Create a clean copy for visualization
    viz_graph = graph.copy()
    
    # Check for and remove problematic nodes
    nodes_to_remove = []
    for node, attrs in viz_graph.nodes(data=True):
        if node is None or (isinstance(node, str) and not node.strip()) or 'type' not in attrs:
            print(f"Found problematic node: {repr(node)}")
            nodes_to_remove.append(node)
    
    for node in nodes_to_remove:
        print(f"Removing node: {repr(node)}")
        viz_graph.remove_node(node)
    
    # Remove edges with weight 0
    edges_to_remove = []
    for u, v, data in viz_graph.edges(data=True):
        weight = data.get('weight', 0)
        if weight == 0:
            edges_to_remove.append((u, v))
    
    for u, v in edges_to_remove:
        print(f"Removing zero-weight edge: {u} -> {v}")
        viz_graph.remove_edge(u, v)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get valid edge weights for visualization scaling
    edge_weights = []
    for _, _, data in viz_graph.edges(data=True):
        weight = data.get('weight', 0)
        if isinstance(weight, (int, float)) and weight > 0 and weight != float('inf'):
            edge_weights.append(weight)
    
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        
        # Scale edge widths for visualization (0.5 to 4.0)
        for u, v, data in viz_graph.edges(data=True):
            weight = data.get('weight', 0)
            if isinstance(weight, (int, float)) and weight > 0 and weight != float('inf'):
                # Add a 'width' attribute for visualization
                viz_graph[u][v]['width'] = 0.5 + 3.5 * (weight - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 1.0
            else:
                # This should not happen since we removed zero-weight edges
                viz_graph[u][v]['width'] = 0.5
    
    # Use a force-directed layout that accounts for edge weights
    try:
        # Edge weights influence layout - higher weights pull nodes closer
        pos = nx.spring_layout(viz_graph, k=0.5, iterations=100, seed=42, weight='weight')
    except Exception as e:
        print(f"Error in spring layout: {e}")
        # Fall back to simpler layout
        pos = nx.shell_layout(viz_graph)
    
    # Identify node types
    table_nodes = [node for node, attrs in viz_graph.nodes(data=True) if attrs.get('type') == 'table']
    column_nodes = [node for node, attrs in viz_graph.nodes(data=True) if attrs.get('type') == 'column']
    
    # Further categorize columns
    pk_columns = [
        node for node, attrs in viz_graph.nodes(data=True) 
        if attrs.get('type') == 'column' and attrs.get('is_primary_key', False)
    ]
    
    fk_columns = [
        node for node, attrs in viz_graph.nodes(data=True) 
        if attrs.get('type') == 'column' and attrs.get('is_foreign_key', False) and node not in pk_columns
    ]
    
    # Columns that are both primary keys and foreign keys
    pk_fk_columns = [
        node for node, attrs in viz_graph.nodes(data=True) 
        if attrs.get('type') == 'column' and attrs.get('is_primary_key', False) 
        and attrs.get('is_foreign_key', False)
    ]
    
    # Remove pk_fk_columns from both pk_columns and fk_columns
    pk_columns = [node for node in pk_columns if node not in pk_fk_columns]
    fk_columns = [node for node in fk_columns if node not in pk_fk_columns]
    
    # Regular columns (neither PK nor FK)
    regular_columns = [
        node for node in column_nodes 
        if node not in pk_columns and node not in fk_columns and node not in pk_fk_columns
    ]
    
    # Draw nodes with different colors and sizes
    nx.draw_networkx_nodes(viz_graph, pos, 
                          nodelist=table_nodes,
                          node_size=2000, 
                          node_color="lightblue",
                          edgecolors='black',
                          alpha=0.9,
                          ax=ax)
    
    if pk_columns:
        nx.draw_networkx_nodes(viz_graph, pos, 
                              nodelist=pk_columns,
                              node_size=800, 
                              node_color="lightgreen",
                              edgecolors='black',
                              alpha=0.8,
                              ax=ax)
    
    if fk_columns:
        nx.draw_networkx_nodes(viz_graph, pos, 
                              nodelist=fk_columns,
                              node_size=800, 
                              node_color="orange",
                              edgecolors='black',
                              alpha=0.8,
                              ax=ax)
    
    if pk_fk_columns:
        nx.draw_networkx_nodes(viz_graph, pos, 
                              nodelist=pk_fk_columns,
                              node_size=900, 
                              node_color="mediumpurple",
                              edgecolors='black',
                              alpha=0.8,
                              ax=ax)
    
    if regular_columns:
        nx.draw_networkx_nodes(viz_graph, pos, 
                              nodelist=regular_columns,
                              node_size=600, 
                              node_color="lightyellow",
                              edgecolors='black',
                              alpha=0.7,
                              ax=ax)
    
    # Group edges by relationship type
    relationship_edges = defaultdict(list)
    
    for u, v, data in viz_graph.edges(data=True):
        rel_type = data.get('relationship_type', 'other')
        relationship_edges[rel_type].append((u, v, data.get('width', 1.0)))
    
    # Define visual styles for each relationship type
    relationship_styles = {
        # Explicit relationships
        'table_column': {'color': 'black', 'style': 'dotted', 'alpha': 0.5},
        'column_table': {'color': 'black', 'style': 'dotted', 'alpha': 0.5},
        'same_table': {'color': 'gray', 'style': 'solid', 'alpha': 0.2},
        
        # Primary key to foreign key relationships
        'pk_fk_column': {'color': 'red', 'style': 'solid', 'alpha': 0.9},
        'pk_fk_table': {'color': 'purple', 'style': 'solid', 'alpha': 0.9},
        'fk_table': {'color': 'blue', 'style': 'dashed', 'alpha': 0.7},
        'table_fk': {'color': 'blue', 'style': 'dashed', 'alpha': 0.7},
        'pk_table': {'color': 'green', 'style': 'solid', 'alpha': 0.7},
        'table_pk': {'color': 'green', 'style': 'solid', 'alpha': 0.7},
        
        # Inferred relationships
        'inferred_pk_fk_column': {'color': 'salmon', 'style': 'dashed', 'alpha': 0.7},
        'inferred_pk_fk_table': {'color': 'mediumpurple', 'style': 'dashed', 'alpha': 0.7},
        'inferred_fk_table': {'color': 'lightskyblue', 'style': 'dashed', 'alpha': 0.6},
        'inferred_table_fk': {'color': 'lightskyblue', 'style': 'dashed', 'alpha': 0.6},
        'inferred_pk_table': {'color': 'lightgreen', 'style': 'dashed', 'alpha': 0.6},
        'inferred_table_pk': {'color': 'lightgreen', 'style': 'dashed', 'alpha': 0.6},
        
        # Default for other types
        'other': {'color': 'gray', 'style': 'solid', 'alpha': 0.5}
    }
    
    # Draw edges for each relationship type with width based on weights
    for rel_type, edge_data in relationship_edges.items():
        if not edge_data:
            continue
            
        style = relationship_styles.get(rel_type, relationship_styles['other'])
        
        # Sample edges for common types to reduce clutter
        if rel_type in ['same_table', 'table_column', 'column_table'] and len(edge_data) > 50:
            # Sample 50 edges
            sample_indices = np.random.choice(len(edge_data), min(50, len(edge_data)), replace=False)
            edge_data = [edge_data[i] for i in sample_indices]
        
        # Extract edges and widths
        edges = [(u, v) for u, v, w in edge_data]
        widths = [w for _, _, w in edge_data]
        
        nx.draw_networkx_edges(viz_graph, pos,
                              edgelist=edges,
                              width=widths,  # Use scaled weights for edge width
                              edge_color=style['color'],
                              style=style['style'],
                              alpha=style['alpha'],
                              arrows=True,
                              arrowsize=10 if 'inferred' in rel_type else 15,
                              ax=ax)
    
    # Add labels
    table_labels = {node: str(node) for node in table_nodes}
    pk_labels = {node: str(node).split('.')[-1] for node in pk_columns}
    fk_labels = {node: str(node).split('.')[-1] for node in fk_columns}
    pk_fk_labels = {node: str(node).split('.')[-1] for node in pk_fk_columns}
    column_labels = {node: str(node).split('.')[-1] for node in regular_columns}
    
    # Draw labels
    if table_labels:
        nx.draw_networkx_labels(viz_graph, pos, 
                               labels=table_labels,
                               font_size=12, 
                               font_weight='bold',
                               ax=ax)
    
    if pk_labels:
        nx.draw_networkx_labels(viz_graph, pos, 
                               labels=pk_labels,
                               font_size=9,
                               font_weight='bold',
                               ax=ax)
    
    if fk_labels:
        nx.draw_networkx_labels(viz_graph, pos, 
                               labels=fk_labels,
                               font_size=9,
                               font_weight='bold',
                               ax=ax)
    
    if pk_fk_labels:
        nx.draw_networkx_labels(viz_graph, pos, 
                               labels=pk_fk_labels,
                               font_size=10,
                               font_weight='bold',
                               ax=ax)
    
    if column_labels:
        nx.draw_networkx_labels(viz_graph, pos, 
                               labels=column_labels,
                               font_size=8,
                               ax=ax)
    
    # Create legend for nodes and edges
    handles = []
    labels = []
    
    # Node legend items
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                             markersize=15, label='Tables'))
    labels.append('Tables')
    
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                             markersize=10, label='Primary Key Columns'))
    labels.append('Primary Key Columns')
    
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                             markersize=10, label='Foreign Key Columns'))
    labels.append('Foreign Key Columns')
    
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumpurple', 
                             markersize=11, label='PK & FK Columns'))
    labels.append('PK & FK Columns')
    
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightyellow', 
                             markersize=8, label='Regular Columns'))
    labels.append('Regular Columns')
    
    # Edge weight legend
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        
        if min_weight != max_weight:
            handles.append(plt.Line2D([0], [0], color='black', linewidth=1, label=f'Weight: {min_weight:.2f}'))
            labels.append(f'Weight: {min_weight:.2f}')
            
            mid_weight = (min_weight + max_weight) / 2
            handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label=f'Weight: {mid_weight:.2f}'))
            labels.append(f'Weight: {mid_weight:.2f}')
            
            handles.append(plt.Line2D([0], [0], color='black', linewidth=3, label=f'Weight: {max_weight:.2f}'))
            labels.append(f'Weight: {max_weight:.2f}')
        else:
            handles.append(plt.Line2D([0], [0], color='black', linewidth=2, label=f'Weight: {min_weight:.2f}'))
            labels.append(f'Weight: {min_weight:.2f}')
    
    # Key relationship types
    key_relationships = [
        'pk_fk_table', 'pk_fk_column', 'fk_table', 'pk_table'
    ]
    
    for rel_type in key_relationships:
        if rel_type in relationship_edges and relationship_edges[rel_type]:
            style = relationship_styles[rel_type]
            handles.append(plt.Line2D([0], [0], 
                                     linestyle=style['style'], 
                                     color=style['color'], 
                                     linewidth=2.0, 
                                     label=rel_type.replace('_', '-')))
            labels.append(rel_type.replace('_', '-'))
    
    # Add the legend
    legend = ax.legend(handles, labels, loc='best', fontsize=10, title="Schema Elements")
    
    # Add title and remove axis
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    
    # Add weight information
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        
        weight_info = f"Edge weights range: {min_weight:.2f} to {max_weight:.2f}"
        ax.text(0.02, 0.02, weight_info, transform=ax.transAxes, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.7))
    else:
        ax.text(0.02, 0.02, "No weighted edges found", transform=ax.transAxes, fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save visualization if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Enhanced graph visualization saved to {output_file}")
    
    return fig, ax



if __name__ == "__main__":

    from Init_schema_graph_v3 import SchemaGraphBuilder

    graph_json =    "enhanced_graph_Db-IMDB.json"  #  "init_graph_Db-IMDB.json" # 
    
    if "init" in graph_json:
        without_quotes = graph_json.replace("'", "")
        db_name = without_quotes.replace("init_graph_", "").replace(".json", "")
        output_file =  os.path.join("./results_init_graph", "graph_init_" + db_name + ".png")
    else:
        without_quotes = graph_json.replace("'", "")
        db_name = without_quotes.replace("enhanced_graph_", "").replace(".json", "")
        output_file =  os.path.join("./results_init_graph", "graph_enhanced_" + db_name + ".png")
    # load the graph
    graph = SchemaGraphBuilder.load_graph( os.path.join("./results_init_graph", graph_json))
    # visualize the graph
    visualize_graph(graph, output_file=output_file)

