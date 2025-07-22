from database_manager import DatabaseManagerSnowflake

# List of functions to be added to the class
functions_to_add = [
    subprocess_sql_executor,
    execute_sql, 
    compare_sqls,
    validate_sql_query,
    aggregate_sqls,
    get_db_all_tables,
    get_table_all_columns,
    get_db_schema,
    get_sql_tables,
    get_sql_columns_dict,
    get_sql_condition_literals,
    get_execution_status
]

from database_utils.execution import execute_sql_snow
from database_utils.db_info import get_db_schema_snow, get_db_all_tables_snow, get_table_all_columns_snow 

functions_to_add_snowflake = [
    subprocess_sql_executor,
    execute_sql_snow, 
    compare_sqls,
    validate_sql_query,
    aggregate_sqls,
    get_db_all_tables_snow,
    get_table_all_columns,
    get_db_schema_snow,
    get_sql_tables,
    get_sql_columns_dict,
    get_sql_condition_literals,
    get_execution_status
]

# Adding methods to the class
DatabaseManager.add_methods_to_class(functions_to_add)
DatabaseManagerSnowflake.add_methods_to_class(functions_to_add)
