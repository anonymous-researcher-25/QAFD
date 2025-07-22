import logging
from typing import List, Dict

from database_utils.execution import execute_sql, execute_sql_snow

def get_db_all_tables(db_path: str) -> List[str]:
    """
    Retrieves all table names from the database.
    
    Args:
        db_path (str): The path to the database file.
        
    Returns:
        List[str]: A list of table names.
    """
    try:
        raw_table_names = execute_sql(db_path, "SELECT name FROM sqlite_master WHERE type='table';")
        return [table[0].replace('\"', '').replace('`', '') for table in raw_table_names if table[0] != "sqlite_sequence"]
    except Exception as e:
        logging.error(f"Error in get_db_all_tables: {e}")
        raise e

def get_table_all_columns(db_path: str, table_name: str) -> List[str]:
    """
    Retrieves all column names for a given table.
    
    Args:
        db_path (str): The path to the database file.
        table_name (str): The name of the table.
        
    Returns:
        List[str]: A list of column names.
    """
    try:
        table_info_rows = execute_sql(
            db_path, 
            f"PRAGMA table_info(`{table_name}`);")
        
        return [row[1].replace('\"', '').replace('`', '') for row in table_info_rows]
    except Exception as e:
        logging.error(f"Error in get_table_all_columns: {e}\nTable: {table_name}")
        raise e

def get_db_schema(db_path: str) -> Dict[str, List[str]]:
    """
    Retrieves the schema of the database.
    
    Args:
        db_path (str): The path to the database file.
        
    Returns:
        Dict[str, List[str]]: A dictionary mapping table names to lists of column names.
    """
    try:
        table_names = get_db_all_tables(db_path)
        return {table_name: get_table_all_columns(db_path, table_name) for table_name in table_names}
    except Exception as e:
        logging.error(f"Error in get_db_schema: {e}")
        raise e
    

# ----------------------- Snowflake functions ----------------------

def get_db_all_tables_snow(creds: Dict, db_name: str) -> List[str]:

    sql = f"SELECT table_name FROM tables WHERE table_type = 'BASE TABLE';"
    dbs = [item[0] for item in execute_sql_snow(creds, sql, database=db_name, schema='information_schema')]
    return dbs

def get_table_all_columns_snow(creds: Dict, db_name: str, table_name: str) -> List[str]:
    
    sql = f"SELECT column_name FROM columns WHERE table_name = '{table_name}';"
    dbs = [item[0] for item in execute_sql_snow(creds, sql, database=db_name, schema='information_schema')] 
    return dbs

def get_db_schema_snow(creds: Dict, db_name: str) -> Dict[str, List[str]]:    
    
    print(f'Getting schema for snowflake database {db_name}')

    res = {}
    # for schema_name in schema_names:
    table_names = get_db_all_tables_snow(creds, db_name)
    for table_name in table_names:
        res[table_name] = get_table_all_columns_snow(creds, db_name, table_name)

    return res
    
def get_db_schema_names(creds: Dict, db_name: str):

    sql = 'SHOW SCHEMAS;'
    schemas = execute_sql_snow(creds, sql, database=db_name)
    return [r[1] for r in schemas if r[1] not in ['INFORMATION_SCHEMA', 'PUBLIC']]

def get_schema_table_mapping(creds, db_name):
    
    mapping = {}
    res = {}
    schema_names = get_db_schema_names(creds, db_name)
    for schema_name in schema_names:
        table_names = get_db_all_tables_snow(creds, db_name)
        for table_name in table_names:
            res[table_name] = get_table_all_columns_snow(creds, db_name, table_name)
            mapping[table_name] = schema_name
    return res, mapping