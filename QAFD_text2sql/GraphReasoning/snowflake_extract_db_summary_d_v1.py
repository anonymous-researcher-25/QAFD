import snowflake.connector
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import os
import datetime
import statistics
import re
from collections import defaultdict
import pandas as pd

# Database-specific table-specific columns to skip
# Format: {'database_name': {'schema_name': {'table_name': ['column1', 'column2', ...]}}}
DB_SCHEMA_TABLE_SPECIFIC_SKIP_COLUMNS = {
    # Add Snowflake-specific skip rules as needed
    # 'MY_DATABASE': {
    #     'PUBLIC': {
    #         'MY_TABLE': ['METADATA_COLUMN', 'SYSTEM_COLUMN']
    #     }
    # }
}

def quote_identifier(identifier):
    """Properly quote Snowflake identifiers that need it"""
    if not identifier:
        return '""'
    
    # Snowflake identifiers are case-insensitive unless quoted
    # Quote if contains special characters, spaces, or reserved words
    special_chars = ['-', '+', '*', '/', '(', ')', '[', ']', ' ', '.', '&', '|', '!', '@', '#', '$', '^', '~', '`', '=', '<', '>', '?', ',', ';', ':', "'", '"', '%']
    
    # Common Snowflake reserved words
    reserved_words = [
        'account', 'all', 'alter', 'and', 'any', 'as', 'asc', 'between', 'by', 'case', 'cast',
        'check', 'column', 'connect', 'connection', 'constraint', 'create', 'cross', 'current',
        'database', 'delete', 'desc', 'distinct', 'drop', 'else', 'end', 'except', 'exists',
        'false', 'following', 'for', 'from', 'full', 'grant', 'group', 'having', 'ilike',
        'in', 'increment', 'inner', 'insert', 'intersect', 'into', 'is', 'issue', 'join',
        'lateral', 'left', 'like', 'localtime', 'localtimestamp', 'minus', 'natural', 'not',
        'null', 'of', 'on', 'or', 'order', 'organization', 'qualify', 'regexp', 'revoke',
        'right', 'rlike', 'row', 'rows', 'sample', 'schema', 'select', 'set', 'some', 'start',
        'table', 'tablesample', 'then', 'to', 'trigger', 'true', 'try_cast', 'union', 'unique',
        'update', 'using', 'values', 'view', 'when', 'whenever', 'where', 'with'
    ]
    
    # Check if quoting is needed
    needs_quoting = (
        any(char in identifier for char in special_chars) or
        identifier.lower() in reserved_words or
        identifier.isdigit() or
        (identifier and identifier[0].isdigit()) or
        identifier != identifier.upper()  # Snowflake convention for case-sensitive names
    )
    
    if needs_quoting:
        # Escape any existing double quotes by doubling them
        escaped_name = identifier.replace('"', '""')
        return f'"{escaped_name}"'
    return identifier

class SnowflakeKeyFinder:
    """
    A tool to identify potential primary and foreign keys in Snowflake databases
    where they are not explicitly defined.
    """
    
    def __init__(self, connection_params, database_name=None, schema_name=None):
        """Initialize with Snowflake connection parameters."""
        self.connection_params = connection_params
        self.database_name = database_name
        self.schema_name = schema_name or 'PUBLIC'
        self.conn = None
        self.cursor = None
        self.databases = []
        self.schemas = {}
        self.tables = {}
        self.table_columns = {}
        self.primary_keys = {}
        self.foreign_keys = defaultdict(list)
        
        self._connect()
        
    def _connect(self):
        """Establish connection to Snowflake."""
        try:
            self.conn = snowflake.connector.connect(**self.connection_params)
            self.cursor = self.conn.cursor()
            print("Connected to Snowflake successfully")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Snowflake: {e}")
            
    def should_skip_column(self, database_name, schema_name, table_name, column_name):
        """
        Determine if a column should be skipped based on database-specific rules.
        """
        if (database_name in DB_SCHEMA_TABLE_SPECIFIC_SKIP_COLUMNS and 
            schema_name in DB_SCHEMA_TABLE_SPECIFIC_SKIP_COLUMNS[database_name] and
            table_name in DB_SCHEMA_TABLE_SPECIFIC_SKIP_COLUMNS[database_name][schema_name] and
            column_name.upper() in [col.upper() for col in DB_SCHEMA_TABLE_SPECIFIC_SKIP_COLUMNS[database_name][schema_name][table_name]]):
            return True
        return False

    def _get_databases(self):
        """Get all databases accessible to the user."""
        self.cursor.execute("SHOW DATABASES")
        self.databases = [row[1] for row in self.cursor.fetchall()]  # Database name is in column 1
        return self.databases
    
    def _get_schemas(self, database_name=None):
        """Get all schemas for a database."""
        if database_name:
            quoted_db = quote_identifier(database_name)
            self.cursor.execute(f"SHOW SCHEMAS IN DATABASE {quoted_db}")
            schemas = [row[1] for row in self.cursor.fetchall()]  # Schema name is in column 1
            self.schemas[database_name] = schemas
        else:
            # Get schemas for all databases
            for db in self.databases:
                try:
                    quoted_db = quote_identifier(db)
                    self.cursor.execute(f"SHOW SCHEMAS IN DATABASE {quoted_db}")
                    schemas = [row[1] for row in self.cursor.fetchall()]
                    self.schemas[db] = schemas
                except Exception as e:
                    print(f"Error getting schemas for database {db}: {e}")
                    self.schemas[db] = []
        return self.schemas
    
    def _get_tables(self, database_name=None, schema_name=None):
        """Get all tables for database/schema combinations."""
        if database_name and schema_name:
            key = f"{database_name}.{schema_name}"
            quoted_db = quote_identifier(database_name)
            quoted_schema = quote_identifier(schema_name)
            
            try:
                self.cursor.execute(f"SHOW TABLES IN SCHEMA {quoted_db}.{quoted_schema}")
                tables = [row[1] for row in self.cursor.fetchall()]  # Table name is in column 1
                self.tables[key] = tables
            except Exception as e:
                print(f"Error getting tables for {key}: {e}")
                self.tables[key] = []
        else:
            # Get tables for all database/schema combinations
            for db in self.databases:
                if db not in self.schemas:
                    continue
                for schema in self.schemas[db]:
                    key = f"{db}.{schema}"
                    try:
                        quoted_db = quote_identifier(db)
                        quoted_schema = quote_identifier(schema)
                        self.cursor.execute(f"SHOW TABLES IN SCHEMA {quoted_db}.{quoted_schema}")
                        tables = [row[1] for row in self.cursor.fetchall()]
                        self.tables[key] = tables
                    except Exception as e:
                        print(f"Error getting tables for {key}: {e}")
                        self.tables[key] = []
        return self.tables
    
    def _get_table_columns(self, database_name, schema_name, table_name):
        """Get all columns for a specific table."""
        quoted_db = quote_identifier(database_name)
        quoted_schema = quote_identifier(schema_name)
        quoted_table = quote_identifier(table_name)
        
        full_table_name = f"{database_name}.{schema_name}.{table_name}"
        
        try:
            # Use INFORMATION_SCHEMA to get column details
            query = f"""
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
                FROM {quoted_db}.INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = '{schema_name}' 
                AND TABLE_NAME = '{table_name}'
                ORDER BY ORDINAL_POSITION
            """
            self.cursor.execute(query)
            columns = []
            for row in self.cursor.fetchall():
                col_name, data_type, is_nullable, col_default = row
                columns.append((col_name, data_type, is_nullable, col_default))
            self.table_columns[full_table_name] = columns
        except Exception as e:
            print(f"Error getting columns for {full_table_name}: {e}")
            self.table_columns[full_table_name] = []
            
        return self.table_columns.get(full_table_name, [])
    
    def _check_defined_keys(self, database_name, schema_name):
        """Check for already defined primary and foreign keys."""
        defined_pk = {}
        defined_fk = defaultdict(list)
        
        quoted_db = quote_identifier(database_name)
        
        try:
            # Check primary keys using INFORMATION_SCHEMA
            pk_query = f"""
                SELECT 
                    tc.TABLE_NAME,
                    kcu.COLUMN_NAME
                FROM {quoted_db}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN {quoted_db}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu 
                    ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                    AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
                    AND tc.TABLE_NAME = kcu.TABLE_NAME
                WHERE tc.TABLE_SCHEMA = '{schema_name}'
                    AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                ORDER BY tc.TABLE_NAME, kcu.ORDINAL_POSITION
            """
            self.cursor.execute(pk_query)
            
            pk_data = defaultdict(list)
            for row in self.cursor.fetchall():
                table_name, column_name = row
                pk_data[table_name].append(column_name)
            
            for table_name, columns in pk_data.items():
                full_table_name = f"{database_name}.{schema_name}.{table_name}"
                defined_pk[full_table_name] = {'columns': columns, 'origin': 'db'}
            
            # Check foreign keys
            fk_query = f"""
                SELECT 
                    tc.TABLE_NAME,
                    kcu.COLUMN_NAME,
                    ccu.TABLE_NAME AS FOREIGN_TABLE_NAME,
                    ccu.COLUMN_NAME AS FOREIGN_COLUMN_NAME
                FROM {quoted_db}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN {quoted_db}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu 
                    ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                    AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
                    AND tc.TABLE_NAME = kcu.TABLE_NAME
                JOIN {quoted_db}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu 
                    ON tc.CONSTRAINT_NAME = ccu.CONSTRAINT_NAME
                WHERE tc.TABLE_SCHEMA = '{schema_name}'
                    AND tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
            """
            self.cursor.execute(fk_query)
            
            for row in self.cursor.fetchall():
                table_name, column_name, ref_table_name, ref_column_name = row
                full_table_name = f"{database_name}.{schema_name}.{table_name}"
                defined_fk[full_table_name].append({
                    'from': column_name,
                    'to_table': f"{database_name}.{schema_name}.{ref_table_name}",
                    'to_column': ref_column_name,
                    'origin': 'db'
                })
                
        except Exception as e:
            print(f"Error checking defined keys for {database_name}.{schema_name}: {e}")
                
        return defined_pk, defined_fk

    def find_potential_primary_keys(self, database_name, schema_name):
        """Identify columns that are likely to be primary keys."""
        # First check for explicitly defined keys
        defined_pk, _ = self._check_defined_keys(database_name, schema_name)
        if defined_pk:
            print(f"Found defined primary keys in {database_name}.{schema_name}:", defined_pk)
            self.primary_keys.update(defined_pk)
        
        schema_key = f"{database_name}.{schema_name}"
        if schema_key not in self.tables:
            return self.primary_keys
            
        for table_name in self.tables[schema_key]:
            full_table_name = f"{database_name}.{schema_name}.{table_name}"
            
            if full_table_name in self.primary_keys:
                continue
            
            quoted_db = quote_identifier(database_name)
            quoted_schema = quote_identifier(schema_name)
            quoted_table = quote_identifier(table_name)
            full_table_ref = f"{quoted_db}.{quoted_schema}.{quoted_table}"
            
            # Get row count
            try:
                self.cursor.execute(f"SELECT COUNT(*) FROM {full_table_ref}")
                total_rows = self.cursor.fetchone()[0]
            except Exception as e:
                print(f"Error getting row count for {full_table_name}: {e}")
                continue
            
            if total_rows == 0:
                continue
            
            # Get columns for this table
            columns = self._get_table_columns(database_name, schema_name, table_name)
            pk_candidates = {}
            
            for col_name, data_type, is_nullable, col_default in columns:
                if self.should_skip_column(database_name, schema_name, table_name, col_name):
                    continue
                
                try:
                    pk_score = 0
                    quoted_col_name = quote_identifier(col_name)
                    
                    # Check uniqueness
                    self.cursor.execute(f"SELECT COUNT(DISTINCT {quoted_col_name}) FROM {full_table_ref}")
                    distinct_values = self.cursor.fetchone()[0]
                    uniqueness_ratio = distinct_values / max(1, total_rows)
                    
                    if uniqueness_ratio < 0.9:
                        continue
                    
                    # Check for NULL values
                    self.cursor.execute(f"SELECT COUNT(*) FROM {full_table_ref} WHERE {quoted_col_name} IS NULL")
                    null_count = self.cursor.fetchone()[0]
                    null_ratio = null_count / max(1, total_rows)
                    
                    if null_ratio > 0.1:
                        continue
                    
                    # Score uniqueness
                    if uniqueness_ratio == 1.0:
                        pk_score += 30
                    elif uniqueness_ratio > 0.98:
                        pk_score += 20
                    
                    # No nulls is good for PKs
                    if null_count == 0:
                        pk_score += 20
                    
                    # Data type scoring
                    if any(int_type in data_type.upper() for int_type in ['NUMBER', 'INTEGER', 'BIGINT', 'SMALLINT']):
                        pk_score += 15
                    elif any(text_type in data_type.upper() for text_type in ['VARCHAR', 'CHAR', 'STRING', 'TEXT']):
                        pk_score += 5
                    
                    # Naming patterns (same as SQLite version but adapted)
                    name_patterns = [
                        (r'^ID$', 15),
                        (r'^{}_ID$'.format(table_name), 15),
                        (r'^{}_KEY$'.format(table_name), 15),
                        (r'^PK_', 15),
                        (r'^KEY$', 10),
                        (r'^CODE$', 8),
                        (r'^UUID$', 15),
                        (r'^GUID$', 15),
                        (r'^SERIAL$', 15),
                        (r'^SEQ', 10),
                        (r'ID$', 5),
                        (r'UUID$', 10),
                        (r'CODE$', 5),
                        (r'NUM$', 5),
                        (r'NO$', 5),
                        (r'^RECORD', 8),
                        (r'^PID$', 15),
                        (r'^MID$', 15),
                        (r'^UID$', 15),
                        (r'^EID$', 15),
                        (r'^[A-Z]+ID$', 10),
                        (r'^[A-Z]+_ID$', 10),
                    ]
                    
                    for pattern, score in name_patterns:
                        if re.search(pattern, col_name.upper(), re.IGNORECASE):
                            pk_score += score
                            break
                    
                    # Check for auto-increment indication (Snowflake sequences)
                    if col_default and 'NEXTVAL' in str(col_default).upper():
                        pk_score += 15
                    
                    # Add to candidates if score is high enough
                    if pk_score >= 25:
                        pk_candidates[col_name] = {
                            'score': pk_score,
                            'data_type': data_type,
                            'uniqueness': uniqueness_ratio,
                            'null_ratio': null_ratio
                        }
                        
                except Exception as e:
                    print(f"Error analyzing column {full_table_name}.{col_name}: {e}")
                    continue
            
            # Select the best primary key candidate(s)
            if pk_candidates:
                sorted_candidates = sorted(
                    pk_candidates.items(), 
                    key=lambda x: x[1]['score'], 
                    reverse=True
                )
                
                pk_columns = []
                threshold_score = sorted_candidates[0][1]['score'] * 0.8
                
                for col_name, info in sorted_candidates:
                    if info['score'] >= threshold_score:
                        pk_columns.append(col_name)
                
                self.primary_keys[full_table_name] = {
                    'columns': pk_columns,
                    'origin': 'potential'
                }
                    
        return self.primary_keys

    def find_potential_foreign_keys(self, database_name, schema_name):
        """Identify columns that are likely to be foreign keys."""
        # First check for explicitly defined foreign keys
        _, defined_fk = self._check_defined_keys(database_name, schema_name)
        if defined_fk:
            print(f"Found defined foreign keys in {database_name}.{schema_name}:", dict(defined_fk))
            self.foreign_keys.update(defined_fk)
        
        schema_key = f"{database_name}.{schema_name}"
        if schema_key not in self.tables:
            return dict(self.foreign_keys)
            
        for src_table in self.tables[schema_key]:
            full_src_table = f"{database_name}.{schema_name}.{src_table}"
            defined_fks_for_table = self.foreign_keys.get(full_src_table, [])
            defined_fk_cols = set(fk['from'] for fk in defined_fks_for_table)
            
            processed_relationships = set()
            for fk in defined_fks_for_table:
                processed_relationships.add((fk['from'], fk['to_table'], fk['to_column']))
            
            quoted_db = quote_identifier(database_name)
            quoted_schema = quote_identifier(schema_name)
            quoted_src_table = quote_identifier(src_table)
            full_src_table_ref = f"{quoted_db}.{quoted_schema}.{quoted_src_table}"
            
            src_columns = self._get_table_columns(database_name, schema_name, src_table)
            
            for src_col_name, src_data_type, _, _ in src_columns:
                if self.should_skip_column(database_name, schema_name, src_table, src_col_name):
                    continue
                    
                if src_col_name in defined_fk_cols:
                    continue
                
                quoted_src_col_name = quote_identifier(src_col_name)
                
                # Check against other tables in the same schema for potential FK relationships
                for ref_table in self.tables[schema_key]:
                    if src_table == ref_table:
                        continue
                    
                    full_ref_table = f"{database_name}.{schema_name}.{ref_table}"
                    if full_ref_table not in self.primary_keys:
                        continue
                    
                    ref_col_list = self.primary_keys[full_ref_table].get('columns', [])
                    if not ref_col_list:
                        continue
                    
                    quoted_ref_table = quote_identifier(ref_table)
                    full_ref_table_ref = f"{quoted_db}.{quoted_schema}.{quoted_ref_table}"
                    
                    for ref_col in ref_col_list:
                        if self.should_skip_column(database_name, schema_name, ref_table, ref_col):
                            continue
                            
                        quoted_ref_col = quote_identifier(ref_col)
                        
                        if (src_col_name, full_ref_table, ref_col) in processed_relationships:
                            continue
                        
                        # Define naming patterns for foreign keys
                        fk_patterns = [
                            r'^{}_{}$'.format(ref_table, ref_col),
                            r'^{}{}$'.format(ref_table, ref_col.capitalize()),
                            r'^{}_ID$'.format(ref_table),
                            r'^{}ID$'.format(ref_table),
                            r'^{}_KEY$'.format(ref_table),
                            r'^FK_{}_'.format(ref_table),
                            r'^{}$'.format(ref_col)
                        ]
                        
                        name_pattern_match = False
                        for pattern in fk_patterns:
                            if re.match(pattern, src_col_name.upper(), re.IGNORECASE):
                                name_pattern_match = True
                                break
                        
                        # Get reference column data type
                        ref_columns = self._get_table_columns(database_name, schema_name, ref_table)
                        ref_col_type = None
                        for ref_col_info in ref_columns:
                            if ref_col_info[0] == ref_col:
                                ref_col_type = ref_col_info[1]
                                break
                        
                        if name_pattern_match:
                            if ref_col_type and src_data_type and ref_col_type.upper() != src_data_type.upper():
                                confidence = "low"
                            else:
                                confidence = "medium"
                            
                            self.foreign_keys[full_src_table].append({
                                'from': src_col_name,
                                'to_table': full_ref_table,
                                'to_column': ref_col,
                                'origin': 'potential',
                                'confidence': confidence
                            })
                            
                            processed_relationships.add((src_col_name, full_ref_table, ref_col))
                        
                        # Check data value matching (limited for performance)
                        else:
                            try:
                                # Get row counts to avoid expensive operations on large tables
                                self.cursor.execute(f"SELECT COUNT(*) FROM {full_src_table_ref}")
                                src_rows = self.cursor.fetchone()[0]
                                self.cursor.execute(f"SELECT COUNT(*) FROM {full_ref_table_ref}")
                                ref_rows = self.cursor.fetchone()[0]
                                
                                if src_rows > 10000 or ref_rows > 10000:
                                    continue
                                
                                # Check if all non-null values in source exist in reference
                                self.cursor.execute(f"""
                                    SELECT COUNT(*) FROM {full_src_table_ref} 
                                    WHERE {quoted_src_col_name} IS NOT NULL
                                    AND {quoted_src_col_name} NOT IN (
                                        SELECT {quoted_ref_col} FROM {full_ref_table_ref}
                                        WHERE {quoted_ref_col} IS NOT NULL
                                    )
                                """)
                                invalid_refs = self.cursor.fetchone()[0]
                                
                                if invalid_refs == 0:
                                    self.cursor.execute(f"""
                                        SELECT COUNT(DISTINCT {quoted_src_col_name}) 
                                        FROM {full_src_table_ref}
                                        WHERE {quoted_src_col_name} IS NOT NULL
                                    """)
                                    distinct_values = self.cursor.fetchone()[0]
                                    
                                    self.cursor.execute(f"""
                                        SELECT COUNT(DISTINCT {quoted_ref_col}) 
                                        FROM {full_ref_table_ref}
                                        WHERE {quoted_ref_col} IS NOT NULL
                                    """)
                                    ref_distinct = self.cursor.fetchone()[0]
                                    
                                    coverage_ratio = distinct_values / max(1, ref_distinct)
                                    
                                    if (coverage_ratio > 0.01 and 
                                        (not ref_col_type or not src_data_type or 
                                         ref_col_type.upper() == src_data_type.upper())):
                                        
                                        self.foreign_keys[full_src_table].append({
                                            'from': src_col_name,
                                            'to_table': full_ref_table,
                                            'to_column': ref_col,
                                            'origin': 'potential',
                                            'confidence': 'high' if coverage_ratio > 0.3 else 'medium',
                                            'evidence': 'data_match'
                                        })
                                        
                                        processed_relationships.add((src_col_name, full_ref_table, ref_col))
                            except Exception as e:
                                print(f"Error checking FK relationship {full_src_table}.{src_col_name} -> {full_ref_table}.{ref_col}: {e}")
                                continue
                    
        return dict(self.foreign_keys)

    def get_row_count(self, database_name, schema_name, table_name):
        """Get the number of rows in a table."""
        quoted_db = quote_identifier(database_name)
        quoted_schema = quote_identifier(schema_name)
        quoted_table = quote_identifier(table_name)
        
        try:
            self.cursor.execute(f"SELECT COUNT(*) FROM {quoted_db}.{quoted_schema}.{quoted_table}")
            return self.cursor.fetchone()[0]
        except Exception as e:
            print(f"Error getting row count for {database_name}.{schema_name}.{table_name}: {e}")
            return 0
    
    def analyze(self, database_name=None, schema_name=None):
        """Run the full analysis to find potential primary and foreign keys."""
        print(f"Analyzing Snowflake database...")
        
        # Get available databases
        self._get_databases()
        print(f"Found {len(self.databases)} databases: {', '.join(self.databases)}")
        
        # Determine scope of analysis
        if database_name:
            if database_name not in self.databases:
                raise ValueError(f"Database {database_name} not found")
            databases_to_analyze = [database_name]
        else:
            databases_to_analyze = self.databases
        
        total_tables = 0
        
        for db in databases_to_analyze:
            # Get schemas for this database
            self._get_schemas(db)
            
            if schema_name:
                if schema_name not in self.schemas.get(db, []):
                    print(f"Schema {schema_name} not found in database {db}")
                    continue
                schemas_to_analyze = [schema_name]
            else:
                schemas_to_analyze = self.schemas.get(db, [])
            
            for schema in schemas_to_analyze:
                print(f"\nAnalyzing {db}.{schema}...")
                
                # Get tables for this schema
                self._get_tables(db, schema)
                schema_key = f"{db}.{schema}"
                tables = self.tables.get(schema_key, [])
                total_tables += len(tables)
                
                print(f"Found {len(tables)} tables in {db}.{schema}")
                
                if tables:
                    print("Finding potential primary keys...")
                    self.find_potential_primary_keys(db, schema)
                    
                    print("Finding potential foreign keys...")
                    self.find_potential_foreign_keys(db, schema)
        
        print(f"\nAnalysis complete. Processed {total_tables} tables total.")
        
        return {
            'databases': self.databases,
            'schemas': self.schemas,
            'tables': self.tables,
            'columns': self.table_columns,
            'primary_keys': self.primary_keys,
            'foreign_keys': self.foreign_keys
        }
    
    def close(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


def extract_snowflake_db_summary(connection_params: Dict[str, str],
                                database_name: str = None,
                                schema_name: str = None,
                                sample_limit: int = 10,
                                include_samples: bool = True,
                                include_column_names: bool = True,
                                include_data_types: bool = True,
                                detect_primary_keys: bool = True,
                                detect_foreign_keys: bool = True,
                                include_key_confidence: bool = True,
                                include_row_count: bool = False,
                                include_column_count: bool = True,
                                include_distinct_count: bool = False,
                                include_null_count: bool = False,
                                include_cardinality: bool = False,
                                include_nullability: bool = False,
                                include_min_max: bool = False,
                                include_average: bool = False,
                                include_median: bool = False,
                                include_stddev: bool = False,
                                include_avg_length: bool = False,
                                include_common_values: bool = False,
                                common_values_limit: int = 5,
                                common_values_threshold: int = 100,
                                include_date_range: bool = False,
                                include_date_range_days: bool = False,
                                include_not_null_constraint: bool = False,
                                include_default_values: bool = False,
                                include_indexes: bool = False,
                                max_rows_for_expensive_stats: int = 10000,
                                max_string_display_length: int = 100,
                                max_binary_display_bytes: int = 50,
                                include_db_metadata: bool = False,
                                include_table_metadata: bool = False,
                                include_extraction_timestamp: bool = False,
                                skip_empty_tables: bool = False,
                                include_table_relationships: bool = True,
                                include_schema_summary: bool = False):
    """
    Extract comprehensive Snowflake database summary including schema, statistics, and sample data.
    
    Args:
        connection_params: Snowflake connection parameters (user, password, account, etc.)
        database_name: Specific database to analyze (None for all accessible)
        schema_name: Specific schema to analyze (None for all in database)
        sample_limit: Number of sample rows per table
        include_samples: Include sample data
        include_column_names: Include column names
        include_data_types: Include data types
        detect_primary_keys: Detect primary keys
        detect_foreign_keys: Detect foreign keys
        include_key_confidence: Include key detection confidence
        include_row_count: Include row counts
        include_column_count: Include column counts
        include_distinct_count: Include distinct value counts
        include_null_count: Include NULL value counts
        include_cardinality: Include uniqueness ratio
        include_nullability: Include NULL ratio
        include_min_max: Include min/max values
        include_average: Include average values
        include_median: Include median (expensive)
        include_stddev: Include standard deviation (expensive)
        include_avg_length: Include average text length
        include_common_values: Include most frequent values
        common_values_limit: How many common values to show
        common_values_threshold: Only for columns with <= N distinct values
        include_date_range: Include min/max dates
        include_date_range_days: Include date range in days
        include_not_null_constraint: Include NOT NULL constraints
        include_default_values: Include default values
        include_indexes: Include index information (limited in Snowflake)
        max_rows_for_expensive_stats: Limit for median/stddev calculation
        max_string_display_length: String truncation length
        max_binary_display_bytes: Show binary data info up to N bytes
        include_db_metadata: Include database metadata
        include_table_metadata: Include table metadata
        include_extraction_timestamp: Include timestamp
        skip_empty_tables: Skip empty tables
        include_table_relationships: Include relationship summary
        include_schema_summary: Include overall schema statistics
        
    Returns:
        Dictionary containing database summary
    """
    
    # Initialize key finder
    key_finder = None
    
    # Use SnowflakeKeyFinder for enhanced PK/FK detection if requested
    if detect_primary_keys or detect_foreign_keys:
        key_finder = SnowflakeKeyFinder(connection_params, database_name, schema_name)
        key_analysis = key_finder.analyze(database_name, schema_name)
    else:
        key_analysis = {'databases': [], 'schemas': {}, 'tables': {}, 'columns': {}, 'primary_keys': {}, 'foreign_keys': {}}
        # Still need to get basic info
        conn = snowflake.connector.connect(**connection_params)
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        key_analysis['databases'] = [row[1] for row in cursor.fetchall()]
        conn.close()
    
    # Connect to Snowflake for additional info extraction
    conn = snowflake.connector.connect(**connection_params)
    cursor = conn.cursor()

    # Create summary structure
    db_summary = {}
    
    # Add metadata if requested
    if include_db_metadata:
        db_summary["metadata"] = {
            "connection_account": connection_params.get('account'),
            "analyzed_databases": list(key_analysis['databases']) if database_name is None else [database_name],
            "total_databases": len(key_analysis['databases'])
        }
        
        if include_extraction_timestamp:
            db_summary["metadata"]["extracted_at"] = datetime.datetime.now().isoformat()
    
    # Initialize schema summary if requested
    schema_summary = {}
    if include_schema_summary:
        schema_summary = {
            "total_databases": len(key_analysis['databases']),
            "total_schemas": sum(len(schemas) for schemas in key_analysis['schemas'].values()),
            "total_tables": sum(len(tables) for tables in key_analysis['tables'].values()),
            "total_columns": 0,
            "tables_with_primary_keys": 0,
            "tables_with_foreign_keys": 0,
            "total_relationships": 0
        }
    
    db_summary["databases"] = {}
    
    # Determine scope of analysis
    if database_name:
        databases_to_process = [database_name] if database_name in key_analysis['databases'] else []
    else:
        databases_to_process = key_analysis['databases']
    
    # Extract information for each database/schema/table
    for db_name in databases_to_process:
        db_summary["databases"][db_name] = {"schemas": {}}
        
        if schema_name:
            schemas_to_process = [schema_name] if schema_name in key_analysis['schemas'].get(db_name, []) else []
        else:
            schemas_to_process = key_analysis['schemas'].get(db_name, [])
        
        for schema_name_current in schemas_to_process:
            db_summary["databases"][db_name]["schemas"][schema_name_current] = {"tables": {}}
            
            schema_key = f"{db_name}.{schema_name_current}"
            tables_to_process = key_analysis['tables'].get(schema_key, [])
            
            for table_name in tables_to_process:
                full_table_name = f"{db_name}.{schema_name_current}.{table_name}"
                
                # Skip empty tables if requested
                quoted_db = quote_identifier(db_name)
                quoted_schema = quote_identifier(schema_name_current)
                quoted_table = quote_identifier(table_name)
                full_table_ref = f"{quoted_db}.{quoted_schema}.{quoted_table}"
                
                if skip_empty_tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {full_table_ref}")
                        if cursor.fetchone()[0] == 0:
                            continue
                    except Exception as e:
                        print(f"Error checking if table {full_table_name} is empty: {e}")
                        continue

                # Table structure
                table_info = {"name": table_name}
                
                # Add table metadata if requested
                if include_table_metadata:
                    table_info["column_count"] = 0  # We'll count only non-skipped columns
                
                # Get row count if requested
                if include_row_count:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {full_table_ref}")
                        table_info["row_count"] = cursor.fetchone()[0]
                    except Exception as e:
                        print(f"Error getting row count for table {full_table_name}: {e}")
                        table_info["row_count"] = -1
                
                # Initialize columns list
                table_info["columns"] = []
                
                # Initialize samples if requested
                if include_samples:
                    table_info["samples"] = {}
                
                # Get column information
                columns_info = key_analysis['columns'].get(full_table_name, [])
                
                # Get enhanced PK detection results
                if detect_primary_keys:
                    pk_info = key_analysis['primary_keys'].get(full_table_name, {})
                    primary_keys = pk_info.get('columns', [])
                    pk_origin = pk_info.get('origin', 'potential')
                else:
                    primary_keys = []
                    pk_origin = 'unknown'
                
                # Track foreign key columns
                fk_columns = set()
                
                # Process columns, skipping those that should be excluded
                for col_name, data_type, is_nullable, col_default in columns_info:
                    # Skip columns that should be excluded
                    if key_finder and key_finder.should_skip_column(db_name, schema_name_current, table_name, col_name):
                        continue
                    
                    # Count this column for metadata
                    if include_table_metadata:
                        table_info["column_count"] += 1
                    
                    # Update schema summary
                    if include_schema_summary:
                        schema_summary["total_columns"] += 1
                    
                    # Basic column information
                    column = {}
                    
                    if include_column_names:
                        column["name"] = col_name
                    
                    if include_data_types:
                        column["type"] = data_type
                    
                    # Primary key information
                    if detect_primary_keys:
                        is_primary_key = col_name in primary_keys
                        column["is_primary_key"] = is_primary_key
                        
                        if is_primary_key and include_key_confidence:
                            column["pk_origin"] = pk_origin
                    
                    # NOT NULL constraint
                    if include_not_null_constraint:
                        column["not_null"] = is_nullable == 'NO'
                    
                    # Default values
                    if include_default_values:
                        column["default"] = col_default
                    
                    # Extract column statistics
                    try:
                        quoted_col_name = quote_identifier(col_name)
                        
                        # Get distinct value count
                        if include_distinct_count:
                            cursor.execute(f"SELECT COUNT(DISTINCT {quoted_col_name}) FROM {full_table_ref}")
                            column["distinct_count"] = cursor.fetchone()[0]
                        
                        # Get null count
                        if include_null_count:
                            cursor.execute(f"SELECT COUNT(*) FROM {full_table_ref} WHERE {quoted_col_name} IS NULL")
                            column["null_count"] = cursor.fetchone()[0]
                        
                        # Calculate derived statistics
                        if include_row_count and table_info.get("row_count", 0) > 0:
                            if include_nullability and include_null_count:
                                column["nullability"] = round(column["null_count"] / table_info["row_count"], 4)
                            
                            if include_cardinality and include_distinct_count:
                                column["cardinality"] = round(column["distinct_count"] / table_info["row_count"], 4)
                        
                        # For numeric columns, get min, max, avg
                        if any(num_type in data_type.upper() for num_type in ["NUMBER", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "BYTEINT", "FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC"]):
                            try:
                                if include_min_max or include_average:
                                    cursor.execute(f"SELECT MIN({quoted_col_name}), MAX({quoted_col_name}), AVG({quoted_col_name}) FROM {full_table_ref} WHERE {quoted_col_name} IS NOT NULL")
                                    min_val, max_val, avg_val = cursor.fetchone()
                                    
                                    if include_min_max:
                                        column["min"] = min_val
                                        column["max"] = max_val
                                    
                                    if include_average:
                                        column["avg"] = round(float(avg_val), 4) if avg_val is not None else None
                                
                                # For small to medium tables, get median and stddev
                                if (include_median or include_stddev) and table_info.get("row_count", 0) <= max_rows_for_expensive_stats:
                                    cursor.execute(f"SELECT {quoted_col_name} FROM {full_table_ref} WHERE {quoted_col_name} IS NOT NULL")
                                    values = [float(row[0]) for row in cursor.fetchall() if row[0] is not None]
                                    
                                    if values:
                                        try:
                                            if include_median:
                                                column["median"] = round(statistics.median(values), 4)
                                            if include_stddev and len(values) > 1:
                                                column["stddev"] = round(statistics.stdev(values), 4)
                                        except:
                                            pass  # Skip if not truly numeric
                            except:
                                pass  # Skip if not truly numeric
                        
                        # For text columns, get average length
                        if any(text_type in data_type.upper() for text_type in ["VARCHAR", "CHAR", "STRING", "TEXT"]):
                            try:
                                if include_avg_length:
                                    cursor.execute(f"SELECT AVG(LENGTH({quoted_col_name})) FROM {full_table_ref} WHERE {quoted_col_name} IS NOT NULL")
                                    avg_length = cursor.fetchone()[0]
                                    column["avg_length"] = round(float(avg_length), 2) if avg_length is not None else None
                                
                                # Get most common values for columns with reasonable cardinality
                                if (include_common_values and 
                                    include_distinct_count and 
                                    column.get("distinct_count", float('inf')) <= common_values_threshold):
                                    cursor.execute(f"SELECT {quoted_col_name}, COUNT(*) as cnt FROM {full_table_ref} WHERE {quoted_col_name} IS NOT NULL GROUP BY {quoted_col_name} ORDER BY cnt DESC LIMIT {common_values_limit}")
                                    common_values = []
                                    for row in cursor.fetchall():
                                        value, count = row
                                        if isinstance(value, str) and len(value) > max_string_display_length:
                                            value = value[:max_string_display_length-3] + "..."
                                        common_values.append({"value": value, "count": count})
                                    column["common_values"] = common_values
                            except:
                                pass
                        
                        # For date/time columns, get min and max dates
                        if any(date_type in data_type.upper() for date_type in ["DATE", "TIME", "TIMESTAMP", "DATETIME", "TIMESTAMP_LTZ", "TIMESTAMP_NTZ", "TIMESTAMP_TZ"]):
                            try:
                                if include_date_range:
                                    cursor.execute(f"SELECT MIN({quoted_col_name}), MAX({quoted_col_name}) FROM {full_table_ref} WHERE {quoted_col_name} IS NOT NULL")
                                    min_date, max_date = cursor.fetchone()
                                    column["min_date"] = str(min_date) if min_date else None
                                    column["max_date"] = str(max_date) if max_date else None
                                
                                # Try to calculate date range in days
                                if include_date_range_days:
                                    try:
                                        cursor.execute(f"SELECT DATEDIFF(DAY, MIN({quoted_col_name}), MAX({quoted_col_name})) FROM {full_table_ref} WHERE {quoted_col_name} IS NOT NULL")
                                        date_range = cursor.fetchone()[0]
                                        if date_range is not None:
                                            column["date_range_days"] = int(date_range)
                                    except:
                                        pass
                            except:
                                pass
                        
                    except Exception as e:
                        print(f"Error getting statistics for column {full_table_name}.{col_name}: {e}")
                    
                    # Add column to table
                    table_info["columns"].append(column)
                    
                    # Prepare for sample data
                    if include_samples:
                        table_info["samples"][col_name] = []
                
                # Get foreign keys using enhanced detection
                if detect_foreign_keys:
                    table_info["foreign_keys"] = []
                    fk_list = key_analysis['foreign_keys'].get(full_table_name, [])
                    
                    for fk in fk_list:
                        # Skip this foreign key if it involves columns that should be excluded
                        if key_finder and (key_finder.should_skip_column(db_name, schema_name_current, table_name, fk['from']) or 
                                          any(key_finder.should_skip_column(ref_db, ref_schema, ref_table, fk['to_column']) 
                                              for ref_db, ref_schema, ref_table in [fk['to_table'].split('.')])):
                            continue
                            
                        # Regular foreign key
                        foreign_key = {
                            "column": fk['from'],
                            "references": {
                                "table": fk['to_table'],
                                "column": fk['to_column']
                            }
                        }
                        
                        if include_key_confidence:
                            foreign_key["fk_origin"] = fk.get('origin', 'potential')
                            foreign_key["confidence"] = fk.get('confidence', 'medium')
                        
                        # Update the column record to mark it as a foreign key
                        for column in table_info["columns"]:
                            if column.get("name") == fk['from']:
                                column["is_foreign_key"] = True
                                if include_key_confidence:
                                    column["fk_origin"] = fk.get('origin', 'potential')
                                    column["references_table"] = fk['to_table']
                                    column["references_column"] = fk['to_column']
                                fk_columns.add(fk['from'])
                        
                        # Add to table foreign keys
                        table_info["foreign_keys"].append(foreign_key)
                
                # Get sample data if requested
                if include_samples and columns_info:
                    try:
                        # Get column names for SELECT, excluding those that should be skipped
                        valid_column_names = []
                        for col_name, _, _, _ in columns_info:
                            if key_finder:
                                if not key_finder.should_skip_column(db_name, schema_name_current, table_name, col_name):
                                    valid_column_names.append(col_name)
                            else:
                                valid_column_names.append(col_name)
                        
                        if valid_column_names:
                            # Convert column names to quoted form for SQL
                            quoted_col_names = [quote_identifier(col) for col in valid_column_names]
                            
                            # Build SELECT statement with only non-skipped columns
                            select_sql = f"SELECT {', '.join(quoted_col_names)} FROM {full_table_ref} SAMPLE ({min(sample_limit * 10, 1000)} ROWS)"
                            
                            # Try to get a sample using Snowflake's SAMPLE
                            cursor.execute(select_sql)
                            sample_rows = cursor.fetchall()
                            
                            # Limit to requested sample size
                            sample_rows = sample_rows[:sample_limit]
                            
                            # Process each sample row
                            for row in sample_rows:
                                for i, col_name in enumerate(valid_column_names):
                                    # Skip columns not in our table_info (already filtered)
                                    if col_name not in table_info["samples"]:
                                        continue
                                        
                                    value = row[i]
                                    
                                    # Handle special data types for JSON serialization
                                    if isinstance(value, bytes):
                                        if len(value) <= max_binary_display_bytes:
                                            value = f"<binary data: {len(value)} bytes>"
                                        else:
                                            value = f"<binary data: {len(value)} bytes (truncated)>"
                                    elif isinstance(value, str) and len(value) > max_string_display_length:
                                        value = value[:max_string_display_length-3] + "..."  # Truncate long strings
                                    elif hasattr(value, 'isoformat'):  # DateTime objects
                                        value = value.isoformat()
                                    
                                    table_info["samples"][col_name].append(value)
                    
                    except Exception as e:
                        print(f"Error getting sample data for table {full_table_name}: {e}")
                
                # Update schema summary
                if include_schema_summary:
                    if detect_primary_keys and full_table_name in key_analysis['primary_keys']:
                        schema_summary["tables_with_primary_keys"] += 1
                    if detect_foreign_keys and full_table_name in key_analysis['foreign_keys']:
                        schema_summary["tables_with_foreign_keys"] += 1
                        schema_summary["total_relationships"] += len(key_analysis['foreign_keys'][full_table_name])
                        
                # Add table to schema
                db_summary["databases"][db_name]["schemas"][schema_name_current]["tables"][table_name] = table_info
    
    # Add schema summary if requested
    if include_schema_summary:
        db_summary["schema_summary"] = schema_summary
    
    # Add table relationships summary if requested
    if include_table_relationships and detect_foreign_keys:
        relationships = []
        for table, fk_list in key_analysis['foreign_keys'].items():
            for fk in fk_list:
                relationships.append({
                    "from_table": table,
                    "from_column": fk['from'],
                    "to_table": fk['to_table'],
                    "to_column": fk['to_column'],
                    "confidence": fk.get('confidence', 'medium'),
                    "origin": fk.get('origin', 'potential')
                })
        db_summary["relationships"] = relationships
    
    # Close database connection
    cursor.close()
    conn.close()
    if key_finder:
        key_finder.close()
    
    return db_summary


def save_snowflake_db_summary(db_summary, output_path, indent=2):
    """
    Save Snowflake database summary to JSON file
    """
    def json_serialize(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if hasattr(obj, 'isoformat'):  # Handle other datetime-like objects
            return obj.isoformat()
        return str(obj)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(db_summary, f, default=json_serialize, indent=indent)
    
    print(f"Snowflake database summary saved to {output_path}")


def main():
    """
    Main function to run the Snowflake schema analysis
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze a Snowflake database and generate schema summary')
    parser.add_argument('--user', help='Snowflake username', default='tarzanaq')
    parser.add_argument('--password', help='Snowflake password', default='Snowsam235711!')
    parser.add_argument('--account', help='Snowflake account identifier', default='RSRSBDK-YDB67606')
    parser.add_argument('--database', help='Specific database to analyze (optional)')
    parser.add_argument('--schema', help='Specific schema to analyze (optional)', default='PUBLIC')
    parser.add_argument('--warehouse', help='Snowflake warehouse to use', default='COMPUTE_WH')
    parser.add_argument('--role', help='Snowflake role to use', default='ACCOUNTADMIN')
    parser.add_argument('--output', help='Path to save the JSON output', default='./snowflake_results')
    
    args = parser.parse_args()
    
    # Setup connection parameters
    connection_params = {
        'user': args.user,
        'password': args.password,
        'account': args.account,
        'warehouse': args.warehouse,
        'role': args.role
    }
    
    # Add database to connection if specified
    if args.database:
        connection_params['database'] = args.database
    
    print(f"Connecting to Snowflake account: {args.account}")
    print(f"Database: {args.database or 'All accessible databases'}")
    print(f"Schema: {args.schema or 'All schemas'}")
    
    try:
        db_summary = extract_snowflake_db_summary(
            connection_params=connection_params,
            database_name=args.database,
            schema_name=args.schema,
            include_samples=True,
            include_row_count=True,
            include_distinct_count=True,
            include_null_count=True,
            include_cardinality=True,
            include_nullability=True,
            include_min_max=True,
            include_average=True,
            include_avg_length=True,
            include_common_values=True,
            include_date_range=True,
            include_not_null_constraint=True,
            include_default_values=True,
            include_db_metadata=True,
            include_table_metadata=True,
            include_extraction_timestamp=True,
            include_table_relationships=True,
            include_schema_summary=True
        )
    except Exception as e:
        print(f"Error extracting Snowflake database summary: {e}")
        return None
    
    # Save the summary
    if args.database:
        filename = f'{args.database}_snowflake_db_summary.json'
    else:
        filename = 'snowflake_db_summary.json'
    output_path = os.path.join(args.output, filename)
    
    try:
        save_snowflake_db_summary(db_summary, output_path)
    except Exception as e:
        print(f"Error saving Snowflake database summary: {e}")
        return None
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    if 'metadata' in db_summary:
        metadata = db_summary['metadata']
        print(f"  Account: {metadata.get('connection_account', 'unknown')}")
        print(f"  Analyzed databases: {', '.join(metadata.get('analyzed_databases', []))}")
        print(f"  Total accessible databases: {metadata.get('total_databases', 'unknown')}")
    
    if 'schema_summary' in db_summary:
        schema = db_summary['schema_summary']
        print(f"  Total databases: {schema.get('total_databases', 'unknown')}")
        print(f"  Total schemas: {schema.get('total_schemas', 'unknown')}")
        print(f"  Total tables: {schema.get('total_tables', 'unknown')}")
        print(f"  Total columns: {schema.get('total_columns', 'unknown')}")
        print(f"  Tables with PKs: {schema.get('tables_with_primary_keys', 'unknown')}")
        print(f"  Tables with FKs: {schema.get('tables_with_foreign_keys', 'unknown')}")
        print(f"  Total relationships: {schema.get('total_relationships', 'unknown')}")
    
    # Calculate output file size
    try:
        output_size = os.path.getsize(output_path)
        output_size_kb = round(output_size / 1024, 2)
        print(f"  Output file size: {output_size:,} bytes ({output_size_kb} KB)")
    except:
        pass
    
    print(f"\nSnowflake database summary extraction completed and saved to {output_path}")
    return db_summary


if __name__ == "__main__":
    main()