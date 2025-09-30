import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

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
    #"Marañón River": {"description": "Major tributary of the Amazon River", "types": ["River"], "source": "Amazon River"},
    #"Ucayali River": {"description": "Major tributary of the Amazon River", "types": ["River"], "source": "Amazon River"},
    "Negro River": {"description": "Major tributary of the Amazon River", "types": ["River"], "source": "Amazon River"},
    #"Madeira River": {"description": "Major tributary of the Amazon River", "types": ["River"], "source": "Amazon River"},
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

# =========================
# Step 4: Visualization
# =========================

plt.figure(figsize=(8, 8))
plt.rcParams['font.family'] = 'serif'

# Create clustered layout by source
def create_clustered_layout(G):
    # Group nodes by source
    source_groups = {}
    for node in G.nodes():
        source = G.nodes[node]["source"]
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(node)
    
    
    # Create positions for each cluster
    pos = {}
    cluster_centers = {}
    
    # Separate merged entities from others
    merged_sources = []
    other_sources = []
    
    for source in source_groups.keys():
        if "Apple Inc. + Apple (fruit)" in source or "Amazon.com, Inc. + Amazon River" in source:
            merged_sources.append(source)
        else:
            other_sources.append(source)
    
    # Position merged entities in a more balanced way
    for i, source in enumerate(merged_sources):
        if len(merged_sources) == 1:
            # Single merged entity - position slightly off-center for visual interest
            center_x, center_y = 0.3, 0.2
        else:
            # Two merged entities - position them very close together
            if "Apple" in source:
                center_x, center_y = -0.2, 0.2  # Apple very close to center
            else:  # Amazon
                center_x, center_y = 0.2, -0.2  # Amazon very close to center
        cluster_centers[source] = (center_x, center_y)
        
        # Position nodes within merged cluster
        nodes_in_cluster = source_groups[source]
        n_nodes = len(nodes_in_cluster)
        
        for j, node in enumerate(nodes_in_cluster):
            if n_nodes == 1:
                pos[node] = (center_x, center_y)
            else:
                node_angle = 2 * np.pi * j / n_nodes
                radius = 0.6  # Smaller radius for center clusters
                node_x = center_x + radius * np.cos(node_angle)
                node_y = center_y + radius * np.sin(node_angle)
                pos[node] = (node_x, node_y)
    
    # Position other clusters to complement the merged entities
    n_other_clusters = len(other_sources)
    if n_other_clusters > 0:
        # Create a more organic, balanced arrangement
        cluster_positions = []
        
        # Define positions that complement the closer Apple/Amazon layout for narrow canvas
        if n_other_clusters == 1:
            cluster_positions = [(1.2, 0)]
        elif n_other_clusters == 2:
            cluster_positions = [(1.0, 1.2), (1.0, -1.2)]
        elif n_other_clusters == 3:
            cluster_positions = [(0.8, 1.5), (0.8, -1.5), (1.5, 0)]
        elif n_other_clusters == 4:
            # Arrange around the closer Apple/Amazon for narrow layout
            cluster_positions = [
                (1.2, 1.0),   # Upper right
                (1.2, -1.0),  # Lower right  
                (-1.2, 1.0),  # Upper left
                (-1.2, -1.0)  # Lower left
            ]
        else:
            # For more clusters, use a gentle arc adapted for narrow layout
            for i, source in enumerate(other_sources):
                angle = 2 * np.pi * i / n_other_clusters
                center_x = 1.2 * np.cos(angle)
                center_y = 1.2 * np.sin(angle)
                cluster_centers[source] = (center_x, center_y)
        
        # Apply positions to clusters
        for i, source in enumerate(other_sources):
            if i < len(cluster_positions):
                center_x, center_y = cluster_positions[i]
                cluster_centers[source] = (center_x, center_y)
        
        # Position nodes within each cluster
        for source in other_sources:
            center_x, center_y = cluster_centers[source]
            nodes_in_cluster = source_groups[source]
            n_nodes = len(nodes_in_cluster)
            
            for j, node in enumerate(nodes_in_cluster):
                if n_nodes == 1:
                    pos[node] = (center_x, center_y)
                else:
                    node_angle = 2 * np.pi * j / n_nodes
                    radius = 0.6  # Smaller cluster radius for compactness
                    node_x = center_x + radius * np.cos(node_angle)
                    node_y = center_y + radius * np.sin(node_angle)
                    pos[node] = (node_x, node_y)
    
    return pos

# Use custom clustered layout
pos = create_clustered_layout(G)

# Node values for color mapping
node_values = {
    "Apple": 11.984435,
    "Steve Jobs": 8.210441,
    "Tim Cook": 7.754995,
    "Mac": 3.629086,
    "macOS": 3.024318,
    "iPhone": 2.353143,
    "App Store": 1.120938,
    "iOS": 0.527458
}

# Viridis color mapping based on values
def get_node_color(node_name):
    """Get color based on node value using viridis color scheme"""
    value = node_values.get(node_name, None)
    
    # Only apply color mapping if the node has a value
    if value is not None:
        if value > 10:
            return "#35B779"  # Green
        elif value > 3:
            return "#6DCD59"  # Light green
        elif value > 2:
            return "#B4DE2C"  # Light yellow-green
        elif value > 0:
            return "#FDE725"  # Yellow
        else:
            return "#FDE725"  # Yellow (viridis end)
    else:
        # Return None to indicate no color change needed
        return None

# Create node colors using value-based mapping
node_colors = []
for n in G.nodes():
    if n == "Apple":
        node_colors.append(get_node_color("Apple"))  # Use value-based color
    elif n == "Amazon":
        node_colors.append("#2F5597")  # Keep Amazon as dark blue (no value provided)
    else:
        color = get_node_color(n)
        if color is not None:
            node_colors.append(color)  # Use value-based color
        else:
            node_colors.append('white')  # White for nodes without values

# Node sizing and edge styling
degrees = [G.degree(n) for n in G.nodes()]
min_degree, max_degree = min(degrees), max(degrees)
node_sizes = [200 + ((G.degree(n) - min_degree) / (max_degree - min_degree)) * 100 for n in G.nodes()]

weights = [G[u][v]['weight'] for u, v in G.edges()]
min_weight, max_weight = min(weights), max(weights)
edge_widths = [1 + ((G[u][v]['weight'] - min_weight) / (max_weight - min_weight)) * 1 for u, v in G.edges()]

# Draw network (this section is now handled by the specific node drawing sections below)

# Draw edges
# =========================
# Query-Aware Edge Styling
# =========================

query_relevant = {"iOS", "macOS", "App Store", "Mac", "iPhone"}
query_focus = {"Steve Jobs"}
special_cluster = {"Tim Cook", "Steve Wozniak"}
query_irrelevant = {"Amazon", "Fuji", "Malus domestica","Gala",
                    "Golden Delicious", "Granny Smith", "Red Delicious"}

edges = G.edges()
edge_colors = []
edge_widths = []

for u, v in edges:
    color, width = "#cccccc", 0.5  # default faint

    # Case 1: Steve Jobs ↔ Apple (strongest)
    if (u == "Steve Jobs" and v == "Apple") or (v == "Steve Jobs" and u == "Apple"):
        color, width = "#111111", 5.5
        # Case 3: Relevant ↔ Relevant edges
    elif u in query_relevant and v in query_relevant:
        color, width = "#444444", 2.5
        
    # Case 2: Apple ↔ Relevant Products
    elif (u == "Apple" and v in query_relevant) or (v == "Apple" and u in query_relevant):
        color, width = "#222222", 4.0



    # Case 4: Apple ↔ Special Cluster
    elif (u == "Apple" and v in special_cluster) or (v == "Apple" and u in special_cluster):
        color, width = "#333333", 2.5

    # Case 5: Apple ↔ Irrelevant (only 1-hop neighbors)
    elif (u == "Apple" and v in query_irrelevant) or (v == "Apple" and u in query_irrelevant):
        color, width = "#e0e0e0", 0.2  # weaken only direct Apple–irrelevant
    elif (u == "macOS" and v == "Mac") or (v == "Mac" and u == "macOS"):
        color, width = "#00AAFF", 2   # bright blue and thick

    # Case 6: All other edges untouched
    else:
        w = G[u][v]['weight']
        width = 0.5 + (w - min_weight) / (max_weight - min_weight) * 1.5
        color = "#aaaaaa"

    edge_colors.append(color)
    edge_widths.append(width)



# Draw all edges except Mac–macOS
special_edge = [("Mac", "macOS")]
other_edges = [e for e in G.edges() if e not in special_edge and (e[::-1] not in special_edge)]

nx.draw_networkx_edges(G, pos, edgelist=other_edges,
                       width=edge_widths, edge_color=edge_colors, alpha=0.8)

# Draw Mac–macOS edge separately with emphasis
nx.draw_networkx_edges(G, pos, edgelist=special_edge,
                       width=4.5, edge_color="#111111", alpha=0.8)


#nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='#666666')

# Define seed node and get 1-hop neighbors
seed_node = "Apple"
if seed_node in G.nodes():
    # Get 1-hop neighbors of Apple
    neighbors = list(G.neighbors(seed_node))
    all_selected_nodes = [seed_node] + neighbors
else:
    all_selected_nodes = []

# Create viridis color mapping
import matplotlib.pyplot as plt
viridis = plt.get_cmap('viridis')

# Draw other nodes (not Apple, Amazon, or its neighbors) with value-based colors and black borders
other_nodes = [node for node in G.nodes() if node not in [seed_node, "Amazon"] + neighbors]
if other_nodes:
    other_sizes = [200 + ((G.degree(n) - min_degree) / (max_degree - min_degree)) * 100 for n in other_nodes]
    other_colors = []
    for n in other_nodes:
        color = get_node_color(n)
        if color is not None:
            other_colors.append(color)
        else:
            other_colors.append('white')  # White for nodes without values
    
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color=other_colors, 
                          node_size=other_sizes, alpha=0.8, edgecolors='black', linewidths=1)

# Draw Apple (seed node) with value-based color, no border - using Circle patch like in your code
if seed_node in G.nodes():
    x, y = pos[seed_node]
    size = (150 + ((G.degree(seed_node) - min_degree) / (max_degree - min_degree)) * 100) * 0.5
    radius = np.sqrt(size / np.pi) / 50
    
    circle = plt.Circle((x, y), radius, facecolor=get_node_color(seed_node), 
                      edgecolor='none', linewidth=0, alpha=0.8)
    plt.gca().add_patch(circle)

# Draw Amazon node with Circle patch like in your code
if "Amazon" in G.nodes():
    x, y = pos["Amazon"]
    size = (150 + ((G.degree("Amazon") - min_degree) / (max_degree - min_degree)) * 100) * 0.5
    radius = np.sqrt(size / np.pi) / 50
    
    circle = plt.Circle((x, y), radius, facecolor="white", 
                      edgecolor='black', linewidth=1, alpha=0.8)
    plt.gca().add_patch(circle)

# Draw 1-hop neighbors with value-based colors and appropriate borders (excluding Amazon which is drawn separately)
neighbors_except_amazon = [n for n in neighbors if n != "Amazon"]
if neighbors_except_amazon:
    neighbor_sizes = [200 + ((G.degree(n) - min_degree) / (max_degree - min_degree)) * 100 for n in neighbors_except_amazon]
    neighbor_colors = []
    neighbor_edgecolors = []
    neighbor_linewidths = []
    
    for n in neighbors_except_amazon:
        color = get_node_color(n)
        if color is not None:
            neighbor_colors.append(color)
            neighbor_edgecolors.append('none')  # No border for colored nodes
            neighbor_linewidths.append(0)
        else:
            neighbor_colors.append('white')  # White for nodes without values
            neighbor_edgecolors.append('black')  # Black border for white nodes
            neighbor_linewidths.append(1)
    
    nx.draw_networkx_nodes(G, pos, nodelist=neighbors_except_amazon, node_color=neighbor_colors, 
                          node_size=neighbor_sizes, alpha=0.8, edgecolors=neighbor_edgecolors, linewidths=neighbor_linewidths)

# Draw labels
labels_to_show = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels_to_show, font_size=14, 
                       font_color="black", font_weight="bold", font_family="serif")

# Cluster labels removed as requested

# Finalize plot
plt.margins(0.2)  # Adds space around the graph

plt.axis('off')
plt.axis('equal')

plt.savefig('QAFD-RAG.png', dpi=300, bbox_inches='tight', facecolor='white')

plt.show()