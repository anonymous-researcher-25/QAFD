import json

def create_ranked_paths_json(results, db_id=None):
    """
    Create a ranked JSON structure with most confident path as rank 1 
    and other paths ranked by their WFD scores.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from the subquery processing
    db_id : str, optional
        Database ID to prefix to all table.column references
        
    Returns:
    --------
    dict
        JSON structure with ranked paths for each subquery
    """
    def add_db_prefix_to_path(path_str, db_id):
        """Add db_id prefix to each table.column in the path for Snowflake format with quotes"""
        if not path_str or not db_id:
            return path_str
        
        # Split by " -> " to get individual nodes
        nodes = path_str.split(" -> ")
        # Add db_id.db_id prefix to each node with quotes (database.schema.table.column)
        prefixed_nodes = []
        for node in nodes:
            if "." in node:
                table, column = node.split(".", 1)
                prefixed_nodes.append(f'"{db_id}"."{db_id}"."{table}"."{column}"')
            else:
                # Fallback if no dot found
                prefixed_nodes.append(f'"{db_id}"."{db_id}"."{node}"')
        # Join back with " -> "
        return " -> ".join(prefixed_nodes)
    
    ranked_output = {}
    
    # Find the maximum number of total paths across all subqueries to determine ranks needed
    max_paths = 0
    for subquery_idx, result in results.items():
        total_paths = 1  # most_confident_path
        if 'paths_with_scores' in result:
            total_paths += len(result['paths_with_scores'])
        max_paths = max(max_paths, total_paths)
    
    # Create rank entries
    for rank in range(1, max_paths + 1):
        rank_key = f"rank{rank}"
        ranked_output[rank_key] = {"subqueries": []}
        
        # Process each subquery
        for subquery_idx in sorted(results.keys()):
            result = results[subquery_idx]
            subquery_text = result['q']
            most_confident_path = result.get('most_confident_path', '')
            paths_with_scores = result.get('paths_with_scores', [])
            
            # Determine what path goes in this rank for this subquery
            path_entry = None
            
            if rank == 1:
                # Rank 1: Always use most_confident_path
                if most_confident_path:
                    path_entry = {
                        "division": subquery_text,
                        "paths": [
                            {
                                "path": add_db_prefix_to_path(most_confident_path, db_id),
                                "reward": 1.0,  # Most confident gets highest score
                            }
                        ]
                    }
            else:
                # Rank 2+: Use WFD paths sorted by score
                wfd_rank = rank - 2  # Adjust for 0-based indexing (rank 2 = index 0)
                if wfd_rank < len(paths_with_scores):
                    path, score = paths_with_scores[wfd_rank]
                    path_str = " -> ".join(path)
                    path_entry = {
                        "division": subquery_text,
                        "paths": [
                            {
                                "path": add_db_prefix_to_path(path_str, db_id),
                                "reward": round(score, 3),
                            }
                        ]
                    }
            
            # Add to this rank if we have a path
            if path_entry:
                ranked_output[rank_key]["subqueries"].append(path_entry)
        
        # Remove rank if no subqueries have paths for this rank
        if not ranked_output[rank_key]["subqueries"]:
            del ranked_output[rank_key]
    
    return ranked_output

def create_comprehensive_ranked_paths_json(results, db_id=None):
    """
    Create a more comprehensive ranking that includes multiple paths per subquery per rank.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from the subquery processing
    db_id : str, optional
        Database ID to prefix to all table.column references
        
    Returns:
    --------
    dict
        JSON structure with ranked paths
    """
    def add_db_prefix_to_path(path_str, db_id):
        """Add db_id prefix to each table.column in the path for Snowflake format with quotes"""
        if not path_str or not db_id:
            return path_str
        
        # Split by " -> " to get individual nodes
        nodes = path_str.split(" -> ")
        # Add db_id.db_id prefix to each node with quotes (database.schema.table.column)
        prefixed_nodes = []
        for node in nodes:
            if "." in node:
                table, column = node.split(".", 1)
                prefixed_nodes.append(f'"{db_id}"."{db_id}"."{table}"."{column}"')
            else:
                # Fallback if no dot found
                prefixed_nodes.append(f'"{db_id}"."{db_id}"."{node}"')
        # Join back with " -> "
        return " -> ".join(prefixed_nodes)
    
    ranked_output = {}
    
    # Collect all unique paths across all subqueries with their scores
    all_ranked_paths = []
    
    for subquery_idx in sorted(results.keys()):
        result = results[subquery_idx]
        subquery_text = result['q']
        most_confident_path = result.get('most_confident_path', '')
        paths_with_scores = result.get('paths_with_scores', [])
        
        # Add most confident path as rank 1
        if most_confident_path:
            all_ranked_paths.append({
                'subquery_idx': subquery_idx,
                'subquery_text': subquery_text,
                'path': add_db_prefix_to_path(most_confident_path, db_id),
                'score': 1.0,
                'rank_within_subquery': 1
            })
        
        # Add WFD paths as subsequent ranks
        for i, (path, score) in enumerate(paths_with_scores):
            path_str = " -> ".join(path)
            all_ranked_paths.append({
                'subquery_idx': subquery_idx,
                'subquery_text': subquery_text,
                'path': add_db_prefix_to_path(path_str, db_id),
                'score': score,
                'rank_within_subquery': i + 2  # Start from rank 2
            })
    
    # Group by rank within subquery
    max_rank = max([p['rank_within_subquery'] for p in all_ranked_paths]) if all_ranked_paths else 1
    
    for rank in range(1, max_rank + 1):
        rank_key = f"rank{rank}"
        ranked_output[rank_key] = {"subqueries": []}
        
        # Get all paths for this rank
        rank_paths = [p for p in all_ranked_paths if p['rank_within_subquery'] == rank]
        
        # Group by subquery
        subquery_groups = {}
        for path_info in rank_paths:
            subquery_idx = path_info['subquery_idx']
            if subquery_idx not in subquery_groups:
                subquery_groups[subquery_idx] = {
                    'division': path_info['subquery_text'],
                    'paths': []
                }
            subquery_groups[subquery_idx]['paths'].append({
                'path': path_info['path'],
                'reward': round(path_info['score'], 3),
            })
        
        # Add to ranked output
        for subquery_idx in sorted(subquery_groups.keys()):
            ranked_output[rank_key]["subqueries"].append(subquery_groups[subquery_idx])
    
    return ranked_output

def save_ranked_paths_json(results, db_id=None, filename="ranked_paths.json", comprehensive=False):
    """
    Save ranked paths to a JSON file.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from subquery processing
    db_id : str, optional
        Database ID to prefix to all table.column references
    filename : str
        Output filename
    comprehensive : bool
        If True, use comprehensive ranking; if False, use simple ranking
    """
    if comprehensive:
        ranked_data = create_comprehensive_ranked_paths_json(results, db_id)
    else:
        ranked_data = create_ranked_paths_json(results, db_id)
    
    with open(filename, 'w') as f:
        json.dump(ranked_data, f, indent=2)
    
    print(f"Ranked paths saved to {filename}")
    return ranked_data

# Usage in your processing code:
def integrate_ranking_with_processing(results, db_id=None):
    """
    Integration function to add to your existing processing code.
    """
    # Create and save ranked paths
    ranked_paths = create_ranked_paths_json(results, db_id)
    # Print summary
    print("\n=== RANKED PATHS SUMMARY ===")
    for rank_key, rank_data in ranked_paths.items():
        print(f"\n{rank_key.upper()}:")
        for subquery in rank_data["subqueries"]:
            print(f"  Subquery: {subquery['division']}")
            for path_info in subquery['paths']:
                print(f"    Path: {path_info['path']}")
                print(f"    Score: {path_info['reward']}")
    
    return ranked_paths