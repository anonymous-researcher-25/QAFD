"""
Module for decomposing the query and finding the seed nodes interacting with LLM APIs (like Llama)
"""

import json
import re


def parse_llm_response_json(llm_response):
    """
    Parse LLM response containing JSON data for subqueries.
    
    Args:
        llm_response: String response from LLM
            
    Returns:
        Dictionary with parsed subquery data
    """
    # Extract the JSON part from the response
    json_pattern = r'\[\s*\{.*\}\s*\]'
    json_match = re.search(json_pattern, llm_response, re.DOTALL)

    if not json_match:
        raise ValueError("LLM response did not contain the expected JSON output")

    json_text = json_match.group(0).strip()

    try:
        # Parse the JSON
        subqueries_data = json.loads(json_text)
        
        # Extract each component separately
        subqueries = [item["subquery"] for item in subqueries_data]
        query_seeds = [item["query_seeds"] for item in subqueries_data]
        most_confident_paths = [item.get("most_confident_path", "") for item in subqueries_data]
        reasoning_confidence = [item.get("reasoning_confidence", []) for item in subqueries_data]
        
        return {
            "subqueries": subqueries,
            "query_seeds": query_seeds,
            "most_confident_paths": most_confident_paths,
            "source_target_confidence": reasoning_confidence,
        }
    except json.JSONDecodeError as e:
        # If standard parsing fails, try more aggressive fixing
        try:
            # Clean up potential formatting issues
            # Replace single quotes with double quotes
            json_text = json_text.replace("'", '"')
            
            # Fix potential array formatting issues
            json_text = re.sub(r'(\["[^"]+"), (\["[^"]+)', r'\1, \2', json_text)
            json_text = re.sub(r'\]\s*,\s*\[', ', ', json_text)
            
            subqueries_data = json.loads(json_text)
            
            # Extract each component separately
            subqueries = [item["subquery"] for item in subqueries_data]
            query_seeds = [item["query_seeds"] for item in subqueries_data]
            most_confident_paths = [item.get("most_confident_path", "") for item in subqueries_data]
            reasoning_confidence = [item.get("reasoning_confidence", []) for item in subqueries_data]
            
            return {
                "subqueries": subqueries,
                "query_seeds": query_seeds,
                "most_confident_paths": most_confident_paths,
                "source_target_confidence": reasoning_confidence,
            }
        except Exception as e2:
            raise ValueError(f"Failed to parse JSON from LLM response: {e}\nSecond attempt failed with: {e2}\nJSON text: {json_text}")


def generate_sqls_from_subquery(subquery, schema_text, llm_service, seed_nodes=None, clusters=None):
    """
    Generate multiple SQL statements from a single subquery using seed nodes and cluster information.
    
    Args:
        subquery: The subquery text
        schema_json_path: Path to the JSON file containing the schema information
        llm_service: Service to call the language model
        seed_nodes: List of important seed nodes for this subquery
        clusters: List of related column/table clusters for this subquery
        
    Returns:
        List of SQL statements
    """
    # Load and format the schema from JSON file
    # try:
    #     with open(schema_json_path, 'r') as file:
    #         schema_data = json.load(file)
        
    #     # Format schema for the prompt
    #     schema_text = format_schema_from_json(schema_data)
    # except Exception as e:
    #     raise Exception(f"Error reading or processing schema JSON file: {str(e)}")
    
    # Format seed nodes and clusters for the prompt
    seed_nodes_text = ""
    if seed_nodes:
        seed_nodes_text = "Key columns and values to focus on:\n"
        seed_nodes_text += ", ".join([str(node) for node in seed_nodes])
    
    clusters_text = ""
    if clusters:
        clusters_text = "Related column and table groups that may be useful:\n"
        for i, cluster in enumerate(clusters):
            if isinstance(cluster, (list, tuple)):
                clusters_text += f"Group {i+1}: " + ", ".join([str(item) for item in cluster]) + "\n"
            else:
                clusters_text += f"Group {i+1}: {str(cluster)}\n"
    
    # Create prompt for LLM
    prompt = f"""
    You are an expert SQL developer. Your task is to generate SQL statements that answer a database query.
    
    DATABASE SCHEMA:
    {schema_text}
    
    SUBQUERY: {subquery}
    
    {seed_nodes_text}
    
    {clusters_text}
    
    Based on the subquery and schema information, generate 3 different, valid SQL statements that correctly solve this subquery.
    Each SQL should be complete and executable. Ensure variety in the solutions:
    
    1. First SQL: Simple, direct approach focusing on the most relevant tables and columns
    2. Second SQL: More comprehensive approach that captures all potential edge cases
    3. Third SQL: Alternative approach using different join patterns or query structure
    
    Pay special attention to:
    - Using the seed nodes as key filtering or selection columns
    - Leveraging the column/table clusters to determine appropriate joins
    - Ensuring proper handling of NULL values
    - Using appropriate aliases for readability
    
    Format your response exactly as:
    <SQL_1>
    Your first SQL code here
    </SQL_1>
    
    <SQL_2>
    Your second SQL code here
    </SQL_2>
    
    <SQL_3>
    Your third SQL code here
    </SQL_3>
    
    Only include the SQL statements within the tags - no explanations or other text.
    """
    
    # Call LLM
    llm_response = llm_service.call_llm(prompt, max_tokens=1500, temperature=0.4)
    
    sqls = []
    
    # Extract all SQL statements
    for i in range(1, 4):
        # Extract SQL from the tags
        sql_match = re.search(rf'<SQL_{i}>(.*?)</SQL_{i}>', llm_response, re.DOTALL)
        if sql_match:
            sql = sql_match.group(1).strip()
            sqls.append(sql)
    
    return sqls



def seeds_subqueries_from_graph_source_target_with_EKN(query, schema_db_text, knowledge_documnet, llm_service, 
                                                prompt_file_path="./prompts/source_target_confidence_with_EKN_snow.txt"):
    """
    Use LLM to decompose a complex query into simpler subqueries with source and target nodes for min-cost flow. 
    
    Args:
        query: The original natural language question
        schema_db_sampled: Database schema information in txt
        llm_service: Service for calling the LLM
        prompt_file_path: Path to the text file containing the prompt template
    
    Returns:
        Dictionary containing subqueries, query seeds, source nodes, target nodes, and full structured data
    """
    # Load the prompt template from the file
    try:
        with open(prompt_file_path, 'r') as file:
            prompt_template = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found at: {prompt_file_path}")

    prompt = prompt_template.replace("{SCHEMA_SUMMARY}", schema_db_text)
    prompt = prompt.replace("{QUERY}", query)
    prompt = prompt.replace("{KNOWLEDGE_DOCUMENT}", knowledge_documnet)

    # print('prompt:', prompt)
    # print()


    # Call LLM with the constructed prompt
    llm_response = llm_service.call_llm(prompt)
    print(llm_response)

    try:
        json_pattern = r'\[\s*\{.*\}\s*\]'
        json_match = re.search(json_pattern, llm_response, re.DOTALL)
        json_text = json_match.group(0).strip()
    except:
        json_text = llm_response

    return  parse_llm_response_json(json_text)



def seeds_subqueries_from_graph_source_target_without_EKN(query, schema_db_text, llm_service,
                            prompt_file_path="./prompts/source_target_confidence_without_EKN_snow.txt"):
    """
    Use LLM to decompose a complex query into simpler subqueries with source and target nodes for min-cost flow. 
    
    Args:
        query: The original natural language question
        schema_db_sampled: Database schema information in txt
        llm_service: Service for calling the LLM
        prompt_file_path: Path to the text file containing the prompt template
    
    Returns:
        Dictionary containing subqueries, query seeds, source nodes, target nodes, and full structured data
    """
    # Load the prompt template from the file
    try:
        with open(prompt_file_path, 'r') as file:
            prompt_template = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found at: {prompt_file_path}")

    prompt = prompt_template.replace("{SCHEMA_SUMMARY}", schema_db_text)
    prompt = prompt.replace("{QUERY}", query)

    # print('prompt:', prompt)
    # print()


    # Call LLM with the constructed prompt
    llm_response = llm_service.call_llm(prompt)


    try:
        json_pattern = r'\[\s*\{.*\}\s*\]'
        json_match = re.search(json_pattern, llm_response, re.DOTALL)
        json_text = json_match.group(0).strip()
    except:
        print('failed to extract json from llm response')
        json_text = llm_response

    return  parse_llm_response_json(json_text)



def subpath_combiner(query, paths_json, schema_json, llm_service, prompt_file_path="./GraphReasoning/prompts/subpath_combiner_prompt.txt"):
    """
    Use LLM to get the path and reward for spider 2 agent 
    Args:
        query (str): The query to be processed
        json_file (str): Path to the JSON file containing the database schema
        llm_service (object): An instance of the LLM service to call
    """
    # Load the prompt template from the file
    try:
        with open(prompt_file_path, 'r') as file:
            prompt_template = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found at: {prompt_file_path}")
    

    # Replace placeholders in the prompt template
    prompt = prompt_template.replace("{MAIN_QUERY}", query)
    prompt = prompt.replace("{SCHEMA_GRAPH}", schema_json)
    prompt = prompt.replace("{SUBQUERY_SUBPATHS}", paths_json)
    #prompt = prompt.replace("{SEMANTIC_PATH}", paths_json)
   
    # Call LLM with the constructed prompt
    llm_response = llm_service.call_llm(prompt)
    # Extract JSON from the LLM response
    json_pattern = r'\{.*\}'
    json_match = re.search(json_pattern, llm_response, re.DOTALL)

    if not json_match:
        raise ValueError("LLM response did not contain the expected JSON output")
    
    json_text = json_match.group(0).strip()

    # Validate JSON (optional but recommended)
    try:
        parsed_json = json.loads(json_text)
    except:
        print("Error parsing JSON from LLM response")
        return json_text

    return parsed_json

