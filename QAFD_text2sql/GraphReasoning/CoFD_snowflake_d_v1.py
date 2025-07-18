import os
import json
import time
import shutil

# Set API keys as environment variables
os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"  # Replace with your actual
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"  # Replace with your actual OpenAI API key
os.environ["LLAMA_API_KEY"] = "EMPTY"

# LLM configurations
LLM_CONFIGS = {
    "claude": {
        "api_url": "",
        "model": "claude-3-7-sonnet-20250219",
        "api_key": " YOUR_ANTHROPIC_API_KEY",
        "provider": "anthropic",
        "model_name": "claude"
    },
    "llama": {
        "api_url": "http://xxx/v1/chat/completions",
        "model": "llama70b",
        "api_key": "EMPTY",
        "provider": "openai",
        "model_name": "llama"
    },
    "gpt-4o": {
        "api_url": "",
        "model": "gpt-4o",
        "api_key": os.environ["OPENAI_API_KEY"],
        "provider": "openai",
        "model_name": "gpt-4o"
    }
}

# Snowflake connection configuration
SNOWFLAKE_CONFIG = {
    "user": "",
    "password": "",
    "account": "",
}

TARGET_INSTANCE_IDS  = [ 'sf_local038']


def load_test_data(test_path):
    """Load test data from JSONL file and filter by target instance IDs"""
    instances = []
    with open(test_path, 'r') as f:
        for line in f:
            instance = json.loads(line)
            if instance.get("instance_id") in TARGET_INSTANCE_IDS:
                instances.append(instance)
    return instances


def prepare_initial_data(instance_id, database_name, output_dir, sample_limit=10):
    """Extract database summary and build initial graph for Snowflake"""
    print(f"\n=== Preparing initial data for instance: {instance_id} ===")
    print(f"Using Snowflake database: {database_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define filenames for summary and initial graph
    summary_filename = f"{database_name}_db_summary.json"
    init_graph_name = f"{database_name}_init_graph.json"
    
    summary_path = os.path.join(output_dir, summary_filename)
    init_graph_path = os.path.join(output_dir, init_graph_name)
    
    # Extract database summary if it doesn't exist
    if not os.path.exists(summary_path):
        print(f"Extracting Snowflake database summary for {database_name}")
        
        from snowflake_extract_db_summary_d_v1 import extract_snowflake_db_summary, save_snowflake_db_summary
        db_summary = extract_snowflake_db_summary(
            connection_params=SNOWFLAKE_CONFIG,
            database_name=database_name,
            schema_name=None,  # Analyze all schemas
            sample_limit=sample_limit,
            include_samples=True,
            include_row_count=True,
            include_distinct_count=True,
            include_null_count=True,
            include_cardinality=True,
            include_nullability=True,
            include_min_max=False,
            include_average=False,
            include_avg_length=False,
            include_common_values=False,
            include_date_range=False,
            include_not_null_constraint=False,
            include_default_values=False,
            include_db_metadata=True,
            include_table_metadata=True,
            include_extraction_timestamp=True,
            include_table_relationships=True,
            include_schema_summary=True
        )
        save_snowflake_db_summary(db_summary, summary_path)
    else:
        print(f"Using existing db summary: {summary_filename}")
    
    # Build initial graph if it doesn't exist
    if not os.path.exists(init_graph_path):
        print(f"Building initial graph for {database_name}")
        from  init_schema_graph_snow_d_v1 import SchemaGraphBuilder
        from visual_enhnced_graphs import visualize_graph
        
        graph = SchemaGraphBuilder.build_from_json_summary(summary_path)
        
        # Save and visualize initial graph
        SchemaGraphBuilder.save_graph(graph, output_file=init_graph_path)
        visualize_graph(graph, output_file=os.path.join(output_dir, f"{database_name}_init_graph"))
    else:
        print(f"Using existing initial graph: {init_graph_name}")
    
    return summary_path, init_graph_path, database_name


def enhance_graph_with_llm(instance_id, database_name, summary_path, init_graph_path, llm_config, output_dir):
    """Enhance graph with specified LLM configuration"""
    from LLM_service import LLMService
    from init_schema_graph_snow_d_v1 import SchemaGraphBuilder
    from Enhanced_graph_v2 import SchemaGraphEnhancer, save_graph
    from visual_enhnced_graphs import visualize_graph

    llm_name = llm_config["model_name"]
    print(f"\n=== Enhancing graph with {llm_name} for instance: {instance_id} ===")
    
    # Initialize LLM service with the provided configuration
    llm_service = LLMService(
        provider=llm_config["provider"],
        api_url=llm_config["api_url"],
        api_key=llm_config["api_key"],
        model=llm_config["model"]
    )
    
    # Define filename for enhanced graph
    enhanced_graph_name = f"{database_name}_{llm_name}_enhanced_graph.json"
    enhanced_graph_path = os.path.join(output_dir, enhanced_graph_name)
    
    # Enhance graph if it doesn't exist
    if not os.path.exists(enhanced_graph_path):
        print(f"Loading initial graph")
        
        # Enhance graph
        print(f"Enhancing graph with {llm_name}")
        graph = SchemaGraphBuilder.load_graph(init_graph_path)
        schema_details = SchemaGraphBuilder.extract_schema_details(graph)
        enhancer = SchemaGraphEnhancer(llm_service)
        enhanced_graph = enhancer.enhance_edge_semantics(graph, schema_details, summary_path)
        
        # Save and visualize enhanced graph
        save_graph(enhanced_graph, output_path=enhanced_graph_path)
        visualize_graph(enhanced_graph, output_file=os.path.join(output_dir, f"{database_name}_{llm_name}_enhanced_graph"))
        
        print(f"Successfully enhanced graph with {llm_name}")
    else:
        print(f"Enhanced graph already exists: {enhanced_graph_name}")
    
    return enhanced_graph_path


def generate_paths(query, db_id, graph, llm_config, output_dir=None, instance_id=None, enk_path=None):    
    """Generate paths for given query using specified LLM configuration"""
    from query_decompose_seed_nodes_snow import (
        seeds_subqueries_from_graph_source_target_without_EKN,
        seeds_subqueries_from_graph_source_target_with_EKN
    )
    from flow_diffusion import find_path_with_wfd, get_readable_paths
    from init_schema_graph_snow_d_v1 import SchemaGraphBuilder
    from LLM_service import LLMService
    from ranked_paths_snow import create_ranked_paths_json,save_ranked_paths_json

    llm_name = llm_config["model_name"]

    print(f"Generating new paths for {instance_id} with {llm_name}")
    
    # Initialize LLM service with the provided configuration
    llm_service = LLMService(
        provider=llm_config["provider"],
        api_url=llm_config["api_url"],
        api_key=llm_config["api_key"],
        model=llm_config["model"]
    )

    schema_details = SchemaGraphBuilder.extract_schema_details(graph)
    schema_text = SchemaGraphBuilder.format_schema_for_prompt(schema_details)
    
    # Generate decompositions
    if enk_path:
        with open(enk_path, 'r', encoding='utf-8') as file: external_knowledge = file.read()
        print(f"Using external knowledge for {instance_id}")
        decompositions = seeds_subqueries_from_graph_source_target_with_EKN(query, schema_text, external_knowledge, llm_service)
    else:
        decompositions = seeds_subqueries_from_graph_source_target_without_EKN(query, schema_text, llm_service)
    subqueries = decompositions.get("subqueries", [])
    if not subqueries:
        print(f"No subqueries found for query: {query}")
        return {}, {}
    
    results = {}
    path_results = {}
    
    # Process each subquery
    for e, subquery in enumerate(subqueries):
        print(f"  - subquery {e}: {subquery}")
        
        # Extract reasoning confidence data for this subquery
        source_target_confidence = decompositions['source_target_confidence'][e]
        
        # Extract the most confident path from decompositions
        most_confident_path = decompositions['most_confident_paths'][e]
        
        # Get all unique source and target nodes from source_target_confidence
        source_nodes = list(set([item[0] for item in source_target_confidence]))
        target_nodes = list(set([item[1] for item in source_target_confidence]))
        
        print(f"Source nodes: {source_nodes}, Target nodes: {target_nodes}")
        print(f"source_target_confidence: {source_target_confidence}")
        print(f"Most confident path: {most_confident_path}")
        
        # Find paths between all source and target node pairs with their confidence scores
        all_paths_with_scores = []
        
        for source_conf_item in source_target_confidence:
            source, target, confidence = source_conf_item
            if source != target:  # Skip self-loops
                print(f"Source: {source}, Target: {target}, Confidence: {confidence}")
                
                # Find path with confidence-aware WFD
                path, score = find_path_with_wfd(graph, source, target, confidence)
                
                if path:
                    print(f"Path found: {' -> '.join(path)} (score: {score:.3f})")
                    all_paths_with_scores.append((path, score))
                else:
                    print("No path found.")
        
        # Sort paths by score (best first)
        all_paths_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just the paths for compatibility
        all_paths = [path for path, score in all_paths_with_scores]
        
        # Display top scored paths
        print(f"Top scored paths for subquery {e}:")
        for i, (path, score) in enumerate(all_paths_with_scores[:3]):  # Show top 3
            print(f"  Rank {i+1}: {' -> '.join(path)} (score: {score:.3f})")
        
        # Store results
        results[e] = {
            'q': subquery, 
            'source_nodes': source_nodes, 
            'target_nodes': target_nodes, 
            'source_target_confidence': source_target_confidence,
            'most_confident_path': most_confident_path,  # Add most confident path
            'paths': all_paths,  # Sorted by score
            'paths_with_scores': all_paths_with_scores,  # Include scores
        }
        path_results[subquery] = get_readable_paths(all_paths_with_scores)

    # After processing all subqueries, create ranked paths JSON
    print("\n" + "="*50)
    print("CREATING RANKED PATHS JSON")
    print("="*50)
 
    ranked_paths = create_ranked_paths_json(results, db_id)  # Fixed: Added db_id parameter

    # Save to file
    paths_output = os.path.join(output_dir, f"{instance_id}_{llm_name}_paths.json")
    save_ranked_paths_json(results, db_id, paths_output)  # Add db_id parameter

    # Print summary
    print("\n=== RANKED PATHS SUMMARY ===")
    for rank_key, rank_data in ranked_paths.items():
        print(f"\n{rank_key.upper()}:")
        for subquery in rank_data["subqueries"]:
            print(f"  Subquery: {subquery['division']}")
            for path_info in subquery['paths']:
                print(f"    Path: {path_info['path']}")
                print(f"    Score: {path_info['reward']}")

    return results, ranked_paths  # Fixed: Return both results


def process_instance(instance_id, database_name, output_dir, models, sample_limit=10, graph_type='enhanced'):
    """Process a single instance with selected models"""
    from  init_schema_graph_snow_d_v1 import SchemaGraphBuilder
    
    print(f"\n=== Processing instance: {instance_id} ===")
    
    # Prepare initial data (summary and init graph)
    summary_path, init_graph_path, db_name = prepare_initial_data(
        instance_id, database_name, output_dir, sample_limit
    )
    
    # Define the result structure
    final_graphs = {}

    # For each model
    for model_name in models:
        if model_name in LLM_CONFIGS:
            llm_config = LLM_CONFIGS[model_name]
            llm_name = llm_config["model_name"]
            
            # Always create the enhanced graph if graph_type is 'enhanced'
            if graph_type == 'enhanced':
                # Determine the path for the enhanced graph
                enhanced_graph_name = f"{db_name}_{llm_name}_enhanced_graph.json"
                enhanced_graph_path = os.path.join(output_dir, enhanced_graph_name)
                
                # Create enhanced graph if it doesn't exist
                if not os.path.exists(enhanced_graph_path):
                    enhanced_graph_path = enhance_graph_with_llm(
                        instance_id, database_name, summary_path, 
                        init_graph_path, llm_config, output_dir
                    )
                else:
                    print(f"Enhanced graph already exists: {enhanced_graph_name}")
                
                final_graph_name = enhanced_graph_name
                final_output_path = enhanced_graph_path
            else:
                # Use initial graph as final output
                final_graph_name = f"{db_name}_init_graph.json"
                final_output_path = os.path.join(output_dir, final_graph_name)
            
            print(f"Using final output graph: {final_graph_name}")

            # Store the result
            final_graphs[model_name] = {
                "path": final_output_path,
                "type": final_graph_name,
                "database_name": db_name
            }
            
            # Add a small delay between model runs
            time.sleep(2)
    
    return final_graphs


def main():
    """Main function to process specified instances with selected models"""
    import argparse
    from  init_schema_graph_snow_d_v1 import SchemaGraphBuilder
    
    parser = argparse.ArgumentParser(description='Process Snowflake database schemas with multiple LLMs')
    parser.add_argument('--output', help='Path to save the output', default='./results_snow_d_v1')
    parser.add_argument('--sample_limit', type=int, default=10, help='Number of sample rows to extract per table')
    parser.add_argument('--model', choices=['llama', 'claude', 'gpt4o', 'all'], default='gpt4o', help='Which model to use')
    parser.add_argument('--base_dir', default='home/Spider2/spider2-snow', help='Base directory for test files')
    parser.add_argument('--graph_type', choices=['init', 'enhanced'], default='init', 
                       help='Type of graph to use as final output: init=initial graph, enhanced=enhanced graph')
    parser.add_argument('--with_paths', default='True', help='Generate paths for queries')
    
    args = parser.parse_args()
    
    # Determine which models to use
    if args.model == 'all':
        models = ['llama', 'claude', 'gpt4o']
    else:
        models = [args.model]
    
    print(f"Snowflake Database Schema Processor")
    print(f"Base directory: {args.base_dir}")
    print(f"Using models: {models}")
    
    # Path to Snowflake test file - FIXED PATH
    test_path = os.path.join(args.base_dir, "spider2-snow.jsonl")
    print(f"Using test file: {test_path}")
    
    # Check if test file exists
    if not os.path.exists(test_path):
        print(f"Error: Test file not found: {test_path}")
        print(f"Please ensure spider2-snow.jsonl exists in {args.base_dir}/")
        return
    
    # Load and filter test data
    instances = load_test_data(test_path)
    print(f"Loaded {len(instances)} target instances from test file")
    
    # Process each instance
    results = {}
    for instance in instances:
        instance_id = instance["instance_id"]
        
        # Get Snowflake database name - FIXED DATABASE EXTRACTION
        database_name = instance.get("db_id")
        print(f"Extracted database name: {database_name} for instance: {instance_id}")
            
        if not database_name:
            print(f"Warning: Could not find database name for instance {instance_id}, skipping")
            continue
            
        print(f"Processing {instance_id} with Snowflake database: {database_name}")
            
        # Process instance to get graphs
        instance_results = process_instance(
            instance_id, database_name, args.output, models, 
            args.sample_limit, args.graph_type
        )
        results[instance_id] = instance_results

        # Generate paths for each model if requested
        if args.with_paths == 'True':
            query = instance["instruction"]  # FIXED: Use "instruction" instead of "question"
            path_results = {}
            
            for model_name in models:
                if model_name in LLM_CONFIGS:
                    llm_config = LLM_CONFIGS[model_name]
                    # Check if paths already exist
                    paths_output = os.path.join(args.output, f"{instance_id}_{llm_config['model_name']}_paths.json")
                    if os.path.exists(paths_output):
                        print(f"Paths already exist for {instance_id} with {model_name}, loading from: {paths_output}")
                        with open(paths_output, 'r') as f:
                            paths_json = json.load(f)
                    else:
                        # Load the graph
                        graph_path = instance_results[model_name]["path"]
                        graph = SchemaGraphBuilder.load_graph(graph_path)

                        # Generate paths
                        print(f"\n=== Generating paths with {model_name} for instance: {instance_id} ===")
                        _, paths_json = generate_paths(query, database_name, graph, llm_config, args.output, instance_id)  # Add database_name
                    
                    # Store path results
                    path_results[model_name] = paths_json
                    
            # Add path results to instance results
            results[instance_id]["paths"] = path_results
    
    # Save overall results
    # #results_path = os.path.join(args.output, "processing_results.json")
    # with open(results_path, 'w') as f:
    #     json.dump(results, f, indent=2)
    
    # print(f"\n=== All processing completed. Results saved to {results_path} ===")
    # print(f"Configuration used:")
    # print(f"  Database type: Snowflake")
    # print(f"  Base directory: {args.base_dir}")
    # print(f"  Test file: {test_path}")
    # print(f"  Models: {models}")

if __name__ == "__main__":
    main()