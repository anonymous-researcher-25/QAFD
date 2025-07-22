import argparse
import yaml
import json
import os
from datetime import datetime
from typing import Any, Dict, List
import time

from runner.run_manager import RunManager

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the pipeline with the specified configuration.")
    parser.add_argument('--data_mode', type=str, required=True, help="Mode of the data to be processed.")
    parser.add_argument('--data_type', type=str, required=True, help="Bird or Spider")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data file.")
    parser.add_argument('--doc_path', type=str, required=True, help="Path to the documentation folder.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--num_workers', type=int, default=5, help="Number of workers to use.")
    parser.add_argument('--log_level', type=str, default='warning', help="Logging level.")
    parser.add_argument('--pick_final_sql', type=bool, default=False, help="Pick the final SQL from the generated SQLs.")
    parser.add_argument('--snowflake_credentials', type=str, default=False, help="Path to snowflake credential file")
    args = parser.parse_args()

    args.run_start_time = datetime.now().isoformat()
    with open(args.config, 'r') as file:
        args.config=yaml.safe_load(file)
    
    return args

def read_jsonl_with_json(file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line.strip()}: {e}")
                    continue
        return data

def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        List[Dict[str, Any]]: The loaded dataset.
    """
    try:
        with open(data_path, 'r') as file:
            dataset = json.load(file)
    except:
        dataset = read_jsonl_with_json(data_path)
    return dataset

def main():
    """
    Main function to run the pipeline with the specified configuration.
    """
   
    args = parse_arguments()
    dataset = load_dataset(args.data_path)
    run_manager = RunManager(args)
    run_manager.initialize_tasks(dataset)
    run_manager.run_tasks()
    run_manager.generate_sql_files()

if __name__ == '__main__':
    main()
