import os
import time
import sys
import shutil
import logging
import json

# Snow-specific imports
sys.path.append(os.path.join(os.path.expanduser("~"),'GraphReasoning'))
from init_schema_graph_snow_d_v1 import SchemaGraphBuilder
from CoFD_snowflake_d_v1 import generate_paths

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('schema_path_operations.log')]
)

# Set API keys
os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"  # Replace with your actual
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"  # Replace with your actual OpenAI API key
os.environ["LLAMA_API_KEY"] = "EMPTY"

# Snow structure paths
HOME_DIR = os.path.expanduser("~")
BASE_DIR = os.path.join(HOME_DIR, "QAFD_text2sql/spider-agent-snow")
EKN_PATHS = os.path.join(HOME_DIR, "Spider2/spider2-snow/resource/documents")
RESULTS_FOLDER = os.path.join(HOME_DIR, "QAFD_text2sql/GraphReasoning/results_snow_d_v1")
EXAMPLE_FOLDER = os.path.join(BASE_DIR, "examples")
DEFAULT_MODEL = "gpt-4o"

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

INSTANCE_IDS = ['sf_local038']
# Operation flags
ADD_SCHEMA_PATH = "ADD_SCHEMA_PATH"
REMOVE_SCHEMA_PATH = "REMOVE_SCHEMA_PATH"
NO_OPERATION = "NO_OPERATION"

def load_data():
    """
    Load test data and create modified run script.
    
    Returns:
        tuple: (instances_dict, run_script_path)
    """
    # Load test instances
    instances_dict = {}
    test_path = os.path.join(BASE_DIR, "examples", "spider2-snow.jsonl")
    
    try:
        with open(test_path, 'r') as f:
            for line in f:
                instance = json.loads(line)
                instance_id = instance.get("instance_id")
                if instance_id in INSTANCE_IDS:
                    instances_dict[instance_id] = instance
        
        print(f"Loaded {len(instances_dict)} instances from spider2-snow.jsonl")
        
        # Create modified run script
        run_script = os.path.join(BASE_DIR, "run_modified.py")
        with open(os.path.join(BASE_DIR, "run.py"), "r") as original:
            content = original.read()
            modified_content = content.replace(
                'task_configs = [task for task in task_configs if args.example_name in task["id"]]',
                'task_configs = [task for task in task_configs if args.example_name == task["instance_id"]]'
            )
        
        with open(run_script, "w") as modified:
            modified.write(modified_content)
        
        print(f"Created modified script at {run_script}")
        return instances_dict, run_script
        
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        return {}, None

def handle_path(instance_id, instance, model_name, operation=ADD_SCHEMA_PATH):
    """
    Handle schema path operations (add or remove).
    
    Args:
        instance_id: Instance identifier
        instance: Instance data
        model_name: Model to use
        operation: ADD_SCHEMA_PATH or REMOVE_SCHEMA_PATH
        
    Returns:
        bool: Success status
    """
    try:
        # Debug: Print instance keys to understand the data structure
        print(f"Instance keys for {instance_id}: {list(instance.keys())}")
        
        # Setup paths
        source_path = os.path.join(RESULTS_FOLDER, f"{instance_id}_{model_name}_paths.json")
        dest_dir = os.path.join(EXAMPLE_FOLDER, instance_id)
        dest_path = os.path.join(dest_dir, "schema.json")
        
        # Handle based on operation
        if operation == ADD_SCHEMA_PATH:
            # Always generate paths (should_generate=True)
            if os.path.exists(source_path):
                print(f"Regenerating paths for {instance_id} with {model_name} (overwriting existing)...")
            else:
                print(f"Generating paths for {instance_id} with {model_name}...")
            
            # Get instance data using correct keys
            external_knowledge = instance.get("external_knowledge", None)
            query = instance["instruction"]
            db_name = instance["db_id"]
            
            print(f"Query: {query[:100]}...")
            print(f"Database: {db_name}")
            if external_knowledge:
                print(f"External knowledge: {external_knowledge}")
            
            graph_path = os.path.join(RESULTS_FOLDER, f"{db_name}_init_graph.json")
            llm_config = LLM_CONFIGS[model_name]
            
            # Load graph and generate paths
            graph = SchemaGraphBuilder.load_graph(graph_path)
            enk_path = None
            if external_knowledge:
                enk_path = os.path.join(EKN_PATHS, external_knowledge)
                
            generate_paths(query, db_name, graph, llm_config, RESULTS_FOLDER, instance_id, enk_path)  # Fixed: Changed db_id to db_name
            print(f"Successfully generated paths for {instance_id}")
            
            # Copy path to example folder
            if os.path.exists(source_path):
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(source_path, dest_path)
                print(f"Added schema path to {dest_path}")
            else:
                print(f"Warning: Source path {source_path} does not exist")
                return False
            
        elif operation == REMOVE_SCHEMA_PATH:
            # Remove the schema file if it exists
            if os.path.exists(dest_path):
                os.remove(dest_path)
                print(f"Removed schema path file: {dest_path}")
            else:
                print(f"No schema path file to remove at: {dest_path}")
                
        else:
            print(f"NO operation: {operation}")
            
        return True
        
    except Exception as e:
        print(f"Error in handle_path for {instance_id}: {str(e)}")
        return False

def run_test(instance_id, model_name, run_script_path, operation=ADD_SCHEMA_PATH):
    """
    Run test for a single instance.
    
    Args:
        instance_id: Instance identifier
        model_name: Model to use
        run_script_path: Path to the modified run script
        operation: ADD_SCHEMA_PATH or REMOVE_SCHEMA_PATH
        
    Returns:
        bool: Success status
    """
    try:
        # Get model parameter from config
        model_param = LLM_CONFIGS[model_name]["model"]
        test_path = os.path.join(BASE_DIR, "examples", "spider2-snow.jsonl")
        
        # Build command based on operation
        if operation == ADD_SCHEMA_PATH:
            suffix = f"path-{instance_id}"
            overwriting_flag = "--overwriting"
        else:
            suffix = instance_id
            overwriting_flag = ""

        cmd = (
            f"python {run_script_path} --test_path {test_path} "
            f"--example_name {instance_id} --suffix {suffix} "
            f"--model {model_param} --temperature 0.5 --top_p 0.9 "
            f"--max_tokens 3500  {overwriting_flag}"
        )
            
        # cmd = (
        #     f"python {run_script_path} --test_path {test_path} "
        # )

        
        print(f"Executing: {cmd}")
        exit_code = os.system(cmd)

        if exit_code == 0:
            print(f"Successfully ran test for {instance_id}")
            return True
        else:
            print(f"Error running test for {instance_id}, exit code: {exit_code}")
            return False
    except Exception as e:
        print(f"Error in run_test for {instance_id}: {str(e)}")
        return False

if __name__ == "__main__":
    # Print current configuration
    print(f"\n=== Spider Agent Snow Test Runner ===")
    print(f"Base Directory: {BASE_DIR}")
    print(f"EKN Paths: {EKN_PATHS}")
    print(f"Results Folder: {RESULTS_FOLDER}")
    
    # Parse command-line arguments
    args = sys.argv[1:]
    
    # Default values
    model = DEFAULT_MODEL
    operation = ADD_SCHEMA_PATH
    
    # Process arguments
    if args:
        if args[0] in ["-h", "--help"]:
            print("Usage: python script.py [MODEL] [OPERATION]")
            print(f"  MODEL: Model name (default: {DEFAULT_MODEL})")
            print(f"  OPERATION: One of the following (default: {ADD_SCHEMA_PATH}):")
            print(f"    {ADD_SCHEMA_PATH} - Add schema path (always regenerates)")
            print(f"    {REMOVE_SCHEMA_PATH} - Remove schema path")
            print("\nExamples:")
            print(f"  python script.py claude {ADD_SCHEMA_PATH}")
            print(f"  python script.py gpt-4o {ADD_SCHEMA_PATH}")
            print(f"  python script.py llama {REMOVE_SCHEMA_PATH}")
            sys.exit(0)
        
        # First arg is model
        if args[0] in LLM_CONFIGS:
            model = args[0]
        
        # Second arg is operation if provided
        if len(args) > 1 and args[1] in [ADD_SCHEMA_PATH, REMOVE_SCHEMA_PATH]:
            operation = args[1]
    
    print(f"\n=== Running tests with model: {model}, operation: {operation} ===")
    print(f"Instance IDs: {INSTANCE_IDS}")
    
    # Step 1: Load data and prepare environment
    instances_dict, run_script = load_data()
    if not instances_dict or not run_script:
        print("Failed to load data or create run script. Exiting.")
        sys.exit(1)
    
    # Keep track of results
    results = {}
    
    # Process each instance
    for instance_id in INSTANCE_IDS:
        if instance_id not in instances_dict:
            print(f"Warning: Instance {instance_id} not found in test data, skipping.")
            results[instance_id] = "Failed: Not found in test data"
            continue
        
        instance = instances_dict[instance_id]
        print(f"\n=== Processing instance: {instance_id} ===")
        
        # Step 2: Handle path (add or remove)
        if handle_path(instance_id, instance, model, operation):
            # Step 3: Run test
            success = run_test(instance_id, model, run_script, operation)
            results[instance_id] = "Success" if success else "Failed: Test execution error"
        else:
            results[instance_id] = "Failed: Path handling error"
    
    # Print summary
    print("\n=== Test Summary ===")
    for instance_id, result in results.items():
        print(f"{instance_id}: {result}")
    
    # Final success count
    success_count = sum(1 for result in results.values() if result.startswith("Success"))
    print(f"\nSuccessful: {success_count}/{len(results)}")