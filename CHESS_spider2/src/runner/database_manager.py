import os
import socket
import pickle
from threading import Lock
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from typing import Callable, Dict, List, Any
import time
import json

from database_utils.schema import DatabaseSchema
from database_utils.schema_generator import DatabaseSchemaGenerator, SnowflakeDatabaseSchemaGenerator
from database_utils.execution import execute_sql, compare_sqls, validate_sql_query, aggregate_sqls, get_execution_status, subprocess_sql_executor
from database_utils.execution import execute_sql_snow, compare_sqls_snow
from database_utils.db_info import get_db_all_tables, get_table_all_columns, get_db_schema
from database_utils.db_info import get_db_all_tables_snow, get_table_all_columns_snow, get_db_schema_snow, get_schema_table_mapping
from database_utils.sql_parser import get_sql_tables, get_sql_columns_dict, get_sql_condition_literals
from database_utils.db_values.search import query_lsh
from database_utils.db_catalog.search import query_vector_db
from database_utils.db_catalog.preprocess import EMBEDDING_FUNCTION
from database_utils.db_catalog.csv_utils import load_tables_description

# load_dotenv(override=True)
# To fix path
DB_ROOT_PATH = Path(os.getenv("DB_ROOT_PATH"))

# INDEX_SERVER_HOST = os.getenv("INDEX_SERVER_HOST")
# INDEX_SERVER_PORT = int(os.getenv("INDEX_SERVER_PORT"))

class DatabaseManager:
    """
    A singleton class to manage database operations including schema generation, 
    querying LSH and vector databases, and managing column profiles.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls, db_mode=None, db_id=None, db_type=None, credentials=None, db_schema_file=None):
        if (db_mode is not None) and (db_id is not None):
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
                    cls.add_methods_snow(functions_to_add_snowflake)
                    cls._instance._init(db_mode, db_id, db_type, credentials, db_schema_file)
                    
                elif cls._instance.db_id != db_id:
                    cls._instance._init(db_mode, db_id, db_type, credentials, db_schema_file)
                return cls._instance
        else:
            if cls._instance is None:
                raise ValueError("DatabaseManager instance has not been initialized yet.")
            return cls._instance

    def _init(self, 
              db_mode: str, 
              db_id: str, 
              db_type: str, 
              credentials: Dict = None,
              db_schema_file = None):
        """
        Initializes the DatabaseManager instance.

        Args:
            db_mode (str): The mode of the database (e.g., 'train', 'test').
            db_id (str): The database identifier.
        """
      
        self.db_mode = db_mode
        self.db_id = db_id
        self.db_type = db_type

        if credentials:
            with open(credentials) as f:
                self.creds = json.load(f)
        else:
            self.creds = None

        self._set_paths()
        self.lsh = None
        self.minhashes = None
        self.vector_db = None

        if db_type == 'snowflake':
            if db_schema_file and os.path.exists(db_schema_file):
                with open(db_schema_file) as f:
                    self.db_schema = json.load(f)
                mapping_file = db_schema_file.replace('db_schemas', 'schema_table_mapping')
                with open(mapping_file) as f:
                    self.schema_table_mapping = json.load(f)
            else:
                self.set_schema_table_mapping()

        elif db_type == 'sqlite':
            if db_schema_file and os.path.exists(db_schema_file):
                with open(db_schema_file) as f:
                    self.db_schema = json.load(f)
            else:
                self.db_schema = get_db_schema(self.db_path)
            self.add_methods_sqlite(functions_to_add_sqlite)

    def _set_paths(self):
        """Sets the paths for the database files and directories."""
        self.db_path = os.path.join(DB_ROOT_PATH, self.db_id, f"{self.db_id}.sqlite")
        self.db_directory_path = os.path.join(DB_ROOT_PATH, self.db_id)

    def set_lsh(self) -> str:
        """Sets the LSH and minhashes attributes by loading from pickle files."""
       
        with self._lock:
            if self.lsh is None:
                try:
                    
                    start_time = time.time()
 
                    path_tmp = os.path.join(self.db_directory_path , 'preprocessed', f"{self.db_id}_lsh.pkl")
                    with open(path_tmp, 'rb') as file: self.lsh = pickle.load(file)
                    print(' ........    loaded the lsh')

                    path_tmp = os.path.join(self.db_directory_path , "preprocessed" , f"{self.db_id}_minhashes.pkl")
                    with open(path_tmp, 'rb') as file: self.minhashes = pickle.load(file)

                    print(' ........    loaded the minhashes')
                    # with (self.db_directory_path / "preprocessed" / f"{self.db_id}_lsh.pkl").open("rb") as file:
                    # self.lsh = pickle.load(file)
                    # after_lsh_time = time.time()
                    # with (self.db_directory_path / "preprocessed" / f"{self.db_id}_minhashes.pkl").open("rb") as file:
                    #     self.minhashes = pickle.load(file)
                    return "success"
                except Exception as e:
                    self.lsh = "error"
                    self.minhashes = "error"
                    print(f"Error loading LSH for {self.db_id}: {e}")
                    return "error"
            elif self.lsh == "error":
                return "error"
            else:
                return "success"

    def set_vector_db(self) -> str:
        """Sets the vector_db attribute by loading from the context vector database."""
        
        if self.vector_db is None:
            try:
                # vector_db_path = self.db_directory_path / "context_vector_db"
                vector_db_path = os.path.join(self.db_directory_path , "context_vector_db_OpenAIEmbeddings")
                
                self.vector_db = Chroma(
                    collection_name=self.db_id,
                    persist_directory=str(vector_db_path), 
                    embedding_function=EMBEDDING_FUNCTION)
                return "success"
            except Exception as e:
                self.vector_db = "error"
                print(f"Error loading Vector DB for {self.db_id}: {e}")
                return "error"
        elif self.vector_db == "error":
            return "error"
        else:
            return "success"

    def set_schema_table_mapping(self):
        
        schema, st_mapping = self.get_schema_table_mapping()
        self.schema_table_mapping = st_mapping
        self.db_schema = schema

        schema_mapping_path = Path(os.getenv("SCHEMA_MAPPING_PATH"))
        db_schema_path = Path(os.getenv("DB_SCHEMA_PATH"))

        with open(os.path.join(db_schema_path, f'/{self.db_id}.json'), 'w') as f:
            json.dump(schema, f, indent=2)

        with open(os.path.join(schema_mapping_path, f'/{self.db_id}.json'), 'w') as f:
            json.dump(st_mapping, f, indent=2)

    def query_lsh(self, keyword: str, signature_size: int = 100, n_gram: int = 3, top_n: int = 10) -> Dict[str, List[str]]:
        """
        Queries the LSH for similar values to the given keyword.

        Args:
            keyword (str): The keyword to search for.
            signature_size (int, optional): The size of the MinHash signature. Defaults to 20.
            n_gram (int, optional): The n-gram size for the MinHash. Defaults to 3.
            top_n (int, optional): The number of top results to return. Defaults to 10.

        Returns:
            Dict[str, List[str]]: The dictionary of similar values.
        """
        
        # try:
        #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #         s.connect((INDEX_SERVER_HOST, INDEX_SERVER_PORT))
        #         s.sendall(pickle.dumps({
        #             "db_directory_path": str(self.db_directory_path),
        #             "db_id": self.db_id,
        #             "type": "query_lsh",
        #             "keyword": keyword
        #         }))
        #         return receive_data_in_chunks(s)
        # except ConnectionRefusedError:
        #     print(f"Connection refused for {self.db_id}")
        
        lsh_status = self.set_lsh()
        if lsh_status == "success":
            return query_lsh(self.lsh, self.minhashes, keyword, signature_size, n_gram, top_n)
        else:
            raise Exception(f"Error loading LSH for {self.db_id}")
        # except Exception as e:
        #     print(f"Error querying LSH for {self.db_id}: {e}")
        #     raise Exception(f"Error querying LSH for {self.db_id}: {e}")

    def query_vector_db(self, keyword: str, top_k: int) -> Dict[str, Any]:
        """
        Queries the vector database for similar values to the given keyword.

        Args:
            keyword (str): The keyword to search for.
            top_k (int): The number of top results to return.

        Returns:
            Dict[str, Any]: The dictionary of similar values.
        """
        
        # try:
        #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #         s.connect((INDEX_SERVER_HOST, INDEX_SERVER_PORT))
        #         s.sendall(pickle.dumps({
        #             "db_directory_path": str(self.db_directory_path),
        #             "db_id": self.db_id,
        #             "type": "query_vector_db",
        #             "keyword": keyword,
        #             "top_k": top_k
        #         }))
        #         return receive_data_in_chunks(s)
        # except ConnectionRefusedError:
        vector_db_status = self.set_vector_db()
        if vector_db_status == "success":
            return query_vector_db(self.vector_db, keyword, top_k)
        else:
            raise Exception(f"Error loading Vector DB for {self.db_id}")
        # except Exception as e:
        #     raise Exception(f"Error querying Vector DB for {self.db_id}: {e}")

    def get_column_profiles(self, 
                            schema_with_examples: Dict[str, Dict[str, List[str]]],
                            use_value_description: bool, 
                            with_keys: bool, 
                            with_references: bool,
                            tentative_schema: Dict[str, List[str]] = None) -> Dict[str, Dict[str, str]]:
        """
        Generates column profiles for the schema.

        Args:
            schema_with_examples (Dict[str, List[str]]): Schema with example values.
            use_value_description (bool): Whether to use value descriptions.
            with_keys (bool): Whether to include keys.
            with_references (bool): Whether to include references.

        Returns:
            Dict[str, Dict[str, str]]: The dictionary of column profiles.
        """
    
        schema_with_descriptions = load_tables_description(self.db_directory_path, use_value_description)
        
        if self.db_type == 'snowflake':
            print('\n\nCreating SF DB Schema Generator ')
            
            database_schema_generator = SnowflakeDatabaseSchemaGenerator(
                tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema if tentative_schema else self.db_schema),
                schema_with_examples=DatabaseSchema.from_schema_dict_with_examples(schema_with_examples),
                schema_with_descriptions=DatabaseSchema.from_schema_dict_with_descriptions(schema_with_descriptions),
                db_id=self.db_id,
                db_schema = self.db_schema,
                schema_table_mapping=self.schema_table_mapping,
                creds=self.creds,
                add_examples=True
            )
            print('\n\nSnowflake DB Schema Generator created.')
        else:
            database_schema_generator = DatabaseSchemaGenerator(
                tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema if tentative_schema else self.get_db_schema()),
                schema_with_examples=DatabaseSchema.from_schema_dict_with_examples(schema_with_examples),
                schema_with_descriptions=DatabaseSchema.from_schema_dict_with_descriptions(schema_with_descriptions),
                db_id=self.db_id,
                db_path=self.db_path,
                add_examples=True,
            )
            print('\n\nSQLite DB Schema Generator created.')
        
        column_profiles = database_schema_generator.get_column_profiles(with_keys, with_references)
        return column_profiles

    def get_database_schema_string(self, tentative_schema: Dict[str, List[str]], 
                                   schema_with_examples: Dict[str, List[str]], 
                                   schema_with_descriptions: Dict[str, Dict[str, Dict[str, Any]]], 
                                   include_value_description: bool) -> str:
        """
        Generates a schema string for the database.

        Args:
            tentative_schema (Dict[str, List[str]]): The tentative schema.
            schema_with_examples (Dict[str, List[str]]): Schema with example values.
            schema_with_descriptions (Dict[str, Dict[str, Dict[str, Any]]]): Schema with descriptions.
            include_value_description (bool): Whether to include value descriptions.

        Returns:
            str: The generated schema string.
        """
        if self.db_type == 'snowflake':
            schema_generator = SnowflakeDatabaseSchemaGenerator(
                tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema),
                schema_with_examples=DatabaseSchema.from_schema_dict_with_examples(schema_with_examples) if schema_with_examples else None,
                schema_with_descriptions=DatabaseSchema.from_schema_dict_with_descriptions(schema_with_descriptions) if schema_with_descriptions else None,
                db_id=self.db_id,
                db_schema=self.db_schema,
                schema_table_mapping=self.schema_table_mapping,
                creds=self.creds,
            )

            schema_names = []
            for table_name in tentative_schema:
                schema_names.append(self.schema_table_mapping[table_name])
            schema_names = list(set(schema_names))

            table_names = {}
            for schema_name in schema_names: 
                table_names[schema_name] = []
                for table_name in tentative_schema:
                    if self.schema_table_mapping[table_name] == schema_name:
                        table_names[schema_name].append(table_name)

            schema_strings = []
            for schema_name in schema_names:
                schema_strings.append(f'\nCREATE SCHEMA {schema_name};\n')

                # Get schema string for each snowflake schema separately 
                schema_string = schema_generator.generate_schema_string(
                    include_value_description=include_value_description,
                    table_subset=table_names[schema_name])
                
                # Combine
                schema_strings.append(schema_string)

            full_schema_string = '\n'.join(schema_strings)

        else:
            schema_generator = DatabaseSchemaGenerator(
                tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema),
                schema_with_examples=DatabaseSchema.from_schema_dict_with_examples(schema_with_examples) if schema_with_examples else None,
                schema_with_descriptions=DatabaseSchema.from_schema_dict_with_descriptions(schema_with_descriptions) if schema_with_descriptions else None,
                db_id=self.db_id,
                db_path=self.db_path,
            )
            full_schema_string = schema_generator.generate_schema_string(include_value_description=include_value_description)
        return full_schema_string
    
    def add_connections_to_tentative_schema(self, tentative_schema: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Adds connections to the tentative schema.

        Args:
            tentative_schema (Dict[str, List[str]]): The tentative schema.

        Returns:
            Dict[str, List[str]]: The updated schema with connections.
        """
        if self.db_type == 'snowflake':
            schema_generator = SnowflakeDatabaseSchemaGenerator(
                tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema),
                db_id=self.db_id,
                db_schema=self.db_schema,
                schema_table_mapping=self.schema_table_mapping,
                creds=self.creds,
            )

        else:
            schema_generator = DatabaseSchemaGenerator(
                tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema),
                db_id=self.db_id,
                db_path=self.db_path,
            )
        tentative_schema = schema_generator.get_schema_with_connections()

    def get_union_schema_dict(self, schema_dict_list: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        """
        Unions a list of schemas.

        Args:
            schema_dict_list (List[Dict[str, List[str]]): The list of schemas.

        Returns:
            Dict[str, List[str]]: The unioned schema.
        """
        full_schema = DatabaseSchema.from_schema_dict(self.get_db_schema())
        actual_name_schemas = []
        for schema in schema_dict_list:
            subselect_schema = full_schema.subselect_schema(DatabaseSchema.from_schema_dict(schema))
            schema_dict = subselect_schema.to_dict()
            actual_name_schemas.append(schema_dict)
        # actual_name_schemas = [(full_schema.subselect_schema(DatabaseSchema.from_schema_dict(schema))).to_dict() for schema in schema_dict_list]
        union_schema = {}
        for schema in actual_name_schemas:
            for table, columns in schema.items():
                if table not in union_schema:
                    union_schema[table] = columns
                else:
                    union_schema[table] = list(set(union_schema[table] + columns))
        return union_schema

    @staticmethod
    def with_db_path(func: Callable):
        """
        Decorator to inject db_path as the first argument to the function.
        """
        def wrapper(self, *args, **kwargs):
            return func(self.db_path, self.creds, *args, **kwargs)
        return wrapper

    @classmethod
    def add_methods_sqlite(cls, funcs: List[Callable]):
        """
        Adds methods to the class with db_path automatically provided.

        Args:
            funcs (List[Callable]): List of functions to be added as methods.
        """
        for func in funcs:
            method = cls.with_db_path(func)
            setattr(cls, func.__name__, method)

    @staticmethod
    def with_credentials(func: Callable):
        """
        Decorator to inject credentials as the first argument to the function.
        """
        def wrapper(self, *args, **kwargs):
            return func(self.creds, self.db_id, *args, **kwargs)
        return wrapper
    
    @classmethod
    def add_methods_snow(cls, funcs: List[Callable]):
        """
        Adds methods to the class with credentials automatically provided.

        Args:
            funcs (List[Callable]): List of functions to be added as methods.
        """
        for func in funcs:
            method = cls.with_credentials(func)
            setattr(cls, func.__name__.replace('_snow', ''), method)

functions_to_add_sqlite = [
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

functions_to_add_snowflake = [
    subprocess_sql_executor,
    execute_sql_snow, 
    compare_sqls_snow,
    validate_sql_query,
    aggregate_sqls,
    get_db_all_tables_snow,
    get_table_all_columns_snow,
    get_db_schema_snow,
    get_sql_tables,
    get_sql_columns_dict,
    get_sql_condition_literals,
    get_execution_status,
    get_schema_table_mapping
]

# Auxiliary function for interacting with the index server
def receive_data_in_chunks(conn, chunk_size=1024):
            length_bytes = conn.recv(4)
            if not length_bytes:
                return None
            data_length = int.from_bytes(length_bytes, byteorder='big')
            chunks = []
            bytes_received = 0
            while bytes_received < data_length:
                chunk = conn.recv(min(data_length - bytes_received, chunk_size))
                if not chunk:
                    raise ConnectionError("Connection lost")
                chunks.append(chunk)
                bytes_received += len(chunk)
            return pickle.loads(b''.join(chunks))