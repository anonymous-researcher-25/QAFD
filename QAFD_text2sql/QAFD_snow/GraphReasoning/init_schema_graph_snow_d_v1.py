import json
import networkx as nx
import time
import os

"""Build a schema graph from a JSON summary file with improved foreign key handling"""

class SchemaGraphBuilder:
    """
    Class for building and manipulating schema graphs from various sources.
    """
    
    @staticmethod
    def build_from_json_summary(json_file):
        """
        Build a schema graph from a JSON database summary file instead of direct DB querying.
        Updated to handle both flat and nested Snowflake database structures.
        
        Args:
            json_file: Path to JSON summary file
            
        Returns:
            NetworkX graph representing the schema
        """
        print(f"Building schema graph from JSON summary {json_file}...")
        start_time = time.time()
        
        # Create directed graph
        graph = nx.DiGraph()
        
        # Load the JSON summary
        with open(json_file, 'r', encoding='utf-8') as f:
            db_summary = json.load(f)
        
        # Handle different JSON structures
        tables_data = {}
        
        # Check if it's the old flat structure with tables at root
        if 'tables' in db_summary:
            tables_data = db_summary['tables']
            print("Using flat JSON structure (tables at root level)")
        
        # Check if it's the new nested Snowflake structure
        elif 'databases' in db_summary:
            print("Using nested Snowflake JSON structure")
            # Extract tables from the nested structure
            for db_name, db_info in db_summary['databases'].items():
                if 'schemas' in db_info:
                    for schema_name, schema_info in db_info['schemas'].items():
                        if 'tables' in schema_info and schema_info['tables']:
                            # Add all tables from this schema
                            tables_data.update(schema_info['tables'])
                            print(f"Found {len(schema_info['tables'])} tables in schema {schema_name}")
            
            if not tables_data:
                raise ValueError("No tables found in any schema within the databases structure")
        
        else:
            raise ValueError("Invalid JSON structure: neither 'tables' nor 'databases' key found")
        
        # Get tables from the extracted data
        tables = list(tables_data.keys())
        print(f"Processing {len(tables)} tables: {tables}")

        # First pass: Add table nodes and column nodes with basic properties
        for table in tables:
            table_info = tables_data[table]
            
            # Validate table structure
            if not isinstance(table_info, dict):
                print(f"WARNING: Table {table} has invalid structure, skipping")
                continue
                
            if "columns" not in table_info or not isinstance(table_info["columns"], list):
                print(f"WARNING: Table {table} has no valid columns, skipping")
                continue
            
            # Add table node
            if table:  # Ensure we don't add empty table names
                # Check if this is an archive table
                is_archive = table.endswith('_Archive')
                graph.add_node(table, type='table', is_archive=is_archive)
            else:
                print("WARNING: Attempted to add an empty table name")
                continue
            
            # Add column nodes and edges from table to columns
            for col in table_info.get("columns", []):
                col_name = col["name"]
                # Skip empty column names
                if not col_name:
                    print(f"WARNING: Attempted to add an empty column name in table {table}")
                    continue
                    
                col_id = f"{table}.{col_name}"
                is_pk = col.get("is_primary_key", False)
                is_fk = col.get("is_foreign_key", False)  # Initial FK flag
                
                # Create column node with all available attributes
                node_attrs = {
                    'type': 'column', 
                    'table': table, 
                    'column_name': col_name,
                    'data_type': col.get("type", ""),
                    'is_primary_key': is_pk,
                    'is_foreign_key': is_fk,
                    'not_null': col.get("not_null", False)
                }
                
                # Add additional metadata if available
                if "distinct_count" in col:
                    node_attrs["distinct_count"] = col["distinct_count"]
                if "null_count" in col:
                    node_attrs["null_count"] = col["null_count"]
                if "nullability" in col:
                    node_attrs["nullability"] = col["nullability"]
                if "cardinality" in col:
                    node_attrs["cardinality"] = col["cardinality"]
                
                # Add references info if available
                if "references_table" in col:
                    node_attrs["references_table"] = col["references_table"]
                if "references_column" in col:
                    node_attrs["references_column"] = col["references_column"]
                if "composite_fk" in col:
                    node_attrs["composite_fk"] = col["composite_fk"]
                if "fk_origin" in col:
                    node_attrs["fk_origin"] = col["fk_origin"]
                
                graph.add_node(col_id, **node_attrs)
                
                # Add edge from table to column with weights based on column type
                if is_pk:
                    # Give priority to primary key columns with lower weight
                    graph.add_edge(table, col_id, relationship_type='table_pk', weight=1.8)
                    graph.add_edge(col_id, table, relationship_type='pk_table', weight=1.8)
                else:
                    # Regular column weight
                    graph.add_edge(table, col_id, relationship_type='table_column', weight=1.0)
                    graph.add_edge(col_id, table, relationship_type='column_table', weight=1.0)
        
        # Second pass: Process foreign key information
        for table in tables:
            table_info = tables_data[table]
            fk_list = table_info.get("foreign_keys", [])
            
            # Process all foreign keys
            for fk in fk_list:
                # Handle composite foreign keys
                if "columns" in fk and "references" in fk and "columns" in fk["references"]:
                    # This is a composite foreign key
                    source_columns = fk["columns"]
                    ref_table = fk["references"]["table"]
                    ref_columns = fk["references"]["columns"]
                    
                    # Skip if any part is invalid
                    if not source_columns or not ref_columns or len(source_columns) != len(ref_columns):
                        print(f"WARNING: Invalid composite foreign key in {table}: columns mismatch")
                        continue
                    
                    if not ref_table:
                        print(f"WARNING: Invalid composite foreign key in {table}: no referenced table")
                        continue
                    
                    # Get confidence and origin info if available
                    confidence = fk.get("confidence", "medium")
                    fk_origin = fk.get("fk_origin", "db")
                    fk_type = fk.get("type", "composite") 
                    
                    # Generate a unique group ID for this composite key
                    composite_group = f"{table}_to_{ref_table}_{','.join(source_columns)}"
                    
                    # Process each column in the composite key
                    for i, source_col in enumerate(source_columns):
                        source_col_id = f"{table}.{source_col}"
                        ref_col_id = f"{ref_table}.{ref_columns[i]}"
                        
                        # Update the column node with composite key info
                        if graph.has_node(source_col_id):
                            graph.nodes[source_col_id]['is_foreign_key'] = True
                            graph.nodes[source_col_id]['references_table'] = ref_table
                            graph.nodes[source_col_id]['references_column'] = ref_columns[i]
                            graph.nodes[source_col_id]['composite_fk'] = True
                            graph.nodes[source_col_id]['composite_group'] = composite_group
                            graph.nodes[source_col_id]['fk_origin'] = fk_origin
                            graph.nodes[source_col_id]['fk_confidence'] = confidence
                            
                            # Add edge from table to this foreign key column
                            graph.add_edge(table, source_col_id, relationship_type='table_fk', weight=1.8)
                            graph.add_edge(source_col_id, table, relationship_type='fk_table', weight=1.8)
                            
                            # Add edge to referenced column if it exists
                            if graph.has_node(ref_col_id):
                                graph.add_edge(source_col_id, ref_col_id, 
                                            relationship_type='pk_fk_column', 
                                            weight=1.2, 
                                            composite=True, 
                                            composite_group=composite_group,
                                            confidence=confidence,
                                            fk_origin=fk_origin)
                                graph.add_edge(ref_col_id, source_col_id, 
                                            relationship_type='pk_fk_column', 
                                            weight=1.2, 
                                            composite=True, 
                                            composite_group=composite_group,
                                            confidence=confidence,
                                            fk_origin=fk_origin)
                else:
                    # Process regular single-column foreign key
                    fk_col = fk.get("column")  # Column in current table
                    
                    # Skip empty column names
                    if not fk_col:
                        print(f"WARNING: Empty foreign key column name in table {table}")
                        continue
                        
                    # Check if references field exists and has required structure
                    if "references" not in fk or "table" not in fk["references"] or "column" not in fk["references"]:
                        print(f"WARNING: Invalid foreign key reference structure for {fk_col} in {table}")
                        continue
                        
                    ref_table = fk["references"]["table"]  # Referenced table
                    ref_col = fk["references"]["column"]  # Referenced column
                    
                    # Handle cases where ref_table has schema prefix (e.g., "PAGILA.PAGILA.ACTOR")
                    if "." in ref_table:
                        ref_table = ref_table.split(".")[-1]  # Get just the table name
                    
                    # Skip if referenced table or column is empty
                    if not ref_table or not ref_col:
                        print(f"WARNING: Empty referenced table or column for FK {fk_col} in {table}")
                        continue
                        
                    fk_col_id = f"{table}.{fk_col}"
                    ref_col_id = f"{ref_table}.{ref_col}"
                    
                    # Get confidence and origin info if available
                    confidence = fk.get("confidence", "medium")
                    fk_origin = fk.get("fk_origin", "db")
                    
                    # Update the foreign key attribute for this column
                    if graph.has_node(fk_col_id):
                        graph.nodes[fk_col_id]['is_foreign_key'] = True
                        
                        # Add reference information
                        graph.nodes[fk_col_id]['references_table'] = ref_table
                        graph.nodes[fk_col_id]['references_column'] = ref_col
                        graph.nodes[fk_col_id]['fk_origin'] = fk_origin
                        graph.nodes[fk_col_id]['fk_confidence'] = confidence
                        
                        # Strengthen connection from table to this foreign key column
                        graph.add_edge(table, fk_col_id, relationship_type='table_fk', weight=1.8)
                        graph.add_edge(fk_col_id, table, relationship_type='fk_table', weight=1.8)
                        
                        # Add edge to referenced column if it exists
                        if graph.has_node(ref_col_id):
                            graph.add_edge(fk_col_id, ref_col_id, 
                                        relationship_type='pk_fk_column', 
                                        weight=1.2,
                                        confidence=confidence,
                                        fk_origin=fk_origin)
                            graph.add_edge(ref_col_id, fk_col_id, 
                                        relationship_type='pk_fk_column', 
                                        weight=1.2,
                                        confidence=confidence,
                                        fk_origin=fk_origin)
                        else:
                            print(f"WARNING: Referenced column {ref_col_id} not found in graph")
        
        # Process global relationships if available (from Snowflake structure)
        if 'relationships' in db_summary:
            print(f"Processing {len(db_summary['relationships'])} global relationships...")
            for rel in db_summary['relationships']:
                # Extract table names, removing schema prefixes
                from_table = rel['from_table'].split('.')[-1]
                from_column = rel['from_column']
                to_table = rel['to_table'].split('.')[-1]
                to_column = rel['to_column']
                
                from_node = f"{from_table}.{from_column}"
                to_node = f"{to_table}.{to_column}"
                
                # Only add edge if both nodes exist in graph
                if graph.has_node(from_node) and graph.has_node(to_node):
                    confidence = rel.get('confidence', 'unknown')
                    origin = rel.get('origin', 'potential')
                    
                    # Update FK attributes if not already set
                    if not graph.nodes[from_node].get('is_foreign_key', False):
                        graph.nodes[from_node]['is_foreign_key'] = True
                        graph.nodes[from_node]['references_table'] = to_table
                        graph.nodes[from_node]['references_column'] = to_column
                        graph.nodes[from_node]['fk_confidence'] = confidence
                        graph.nodes[from_node]['fk_origin'] = origin
                    
                    # Add edges if they don't already exist
                    if not graph.has_edge(from_node, to_node):
                        graph.add_edge(from_node, to_node,
                                    relationship_type='foreign_key',
                                    confidence=confidence,
                                    fk_origin=origin,
                                    weight=1.2)
                        graph.add_edge(to_node, from_node,
                                    relationship_type='foreign_key',
                                    confidence=confidence,
                                    fk_origin=origin,
                                    weight=1.2)
        
        # Debug: Check for problematic nodes with no type
        for node, attrs in graph.nodes(data=True):
            if 'type' not in attrs:
                print(f"WARNING: Node without type: {node}, attributes: {attrs}")
        
        # Print statistics
        build_time = time.time() - start_time
        print(f"Schema graph built in {build_time:.2f} seconds with:")
        print(f" - {len([n for n, d in graph.nodes(data=True) if d.get('type') == 'table'])} table nodes")
        print(f" - {len([n for n, d in graph.nodes(data=True) if d.get('type') == 'column'])} column nodes")
        print(f" - {graph.number_of_edges()} total edges")
        
        return graph

    @staticmethod
    def extract_schema_details(graph):
        """
        Extract structured schema details from the graph for use in prompts.
        
        Args:
            graph: NetworkX graph of the database schema
            
        Returns:
            Dictionary with structured schema information
        """
        tables = {}
        
        # First, collect all table nodes
        for node, attrs in graph.nodes(data=True):
            if attrs.get('type') == 'table':
                tables[node] = {
                    'columns': [],
                    'primary_keys': [],
                    'foreign_keys': [],
                    'composite_foreign_keys': [],
                    'is_archive': attrs.get('is_archive', False)
                }
        
        # Then collect column information
        for node, attrs in graph.nodes(data=True):
            if attrs.get('type') == 'column' and '.' in node:
                table, column = node.split('.')
                if table in tables:
                    # Add column details
                    column_info = {
                        'name': column,
                        'type': attrs.get('data_type', 'unknown'),
                        'is_pk': attrs.get('is_primary_key', False),
                        'is_fk': attrs.get('is_foreign_key', False),
                        'not_null': attrs.get('not_null', False)
                    }
                    
                    # Add additional attributes if available
                    if 'distinct_count' in attrs:
                        column_info['distinct_count'] = attrs['distinct_count']
                    if 'null_count' in attrs:
                        column_info['null_count'] = attrs['null_count']
                    
                    tables[table]['columns'].append(column_info)
                    
                    # Add to primary keys if applicable
                    if attrs.get('is_primary_key', False):
                        tables[table]['primary_keys'].append(column)
        
        # Process composite foreign keys
        composite_groups = {}
        
        # First identify all composite key groups
        for node, attrs in graph.nodes(data=True):
            if attrs.get('type') == 'column' and attrs.get('composite_fk', False) and 'composite_group' in attrs:
                group = attrs['composite_group']
                table, column = node.split('.')
                
                if group not in composite_groups:
                    composite_groups[group] = {
                        'source_table': table,
                        'columns': [],
                        'ref_table': attrs.get('references_table', ''),
                        'ref_columns': [],
                        'confidence': attrs.get('fk_confidence', 'medium'),
                        'origin': attrs.get('fk_origin', 'potential')
                    }
                
                # Add to the group
                composite_groups[group]['columns'].append(column)
                composite_groups[group]['ref_columns'].append(attrs.get('references_column', ''))
        
        # Add composite FKs to their tables
        for group_id, group_info in composite_groups.items():
            source_table = group_info['source_table']
            if source_table in tables:
                fk_info = {
                    'columns': group_info['columns'],
                    'ref_table': group_info['ref_table'],
                    'ref_columns': group_info['ref_columns'],
                    'confidence': group_info['confidence'],
                    'origin': group_info['origin']
                }
                tables[source_table]['composite_foreign_keys'].append(fk_info)
        
        # Collect single-column foreign key relationships
        for node, attrs in graph.nodes(data=True):
            if attrs.get('type') == 'column' and attrs.get('is_foreign_key', False) and not attrs.get('composite_fk', False):
                if '.' in node:
                    source_table, source_col = node.split('.')
                    
                    # Get referenced table and column
                    ref_table = attrs.get('references_table')
                    ref_col = attrs.get('references_column')
                    
                    if source_table in tables and ref_table and ref_col:
                        fk_info = {
                            'column': source_col,
                            'ref_table': ref_table,
                            'ref_column': ref_col,
                            'confidence': attrs.get('fk_confidence', 'medium'),
                            'origin': attrs.get('fk_origin', 'db')
                        }
                        
                        # Add to foreign keys list if not duplicate
                        existing = False
                        for existing_fk in tables[source_table]['foreign_keys']:
                            if existing_fk['column'] == source_col and existing_fk['ref_table'] == ref_table:
                                existing = True
                                break
                                
                        if not existing:
                            tables[source_table]['foreign_keys'].append(fk_info)
        
        # Convert to list format for the schema details
        schema_details = []
        for table_name, table_info in tables.items():
            schema_details.append({
                'table_name': table_name,
                'columns': table_info['columns'],
                'primary_keys': table_info['primary_keys'],
                'foreign_keys': table_info['foreign_keys'],
                'composite_foreign_keys': table_info['composite_foreign_keys'],
                'is_archive': table_info['is_archive']
            })
        
        return schema_details
    
    @staticmethod
    def format_schema_for_prompt(schema_details):
        """
        Format schema details for inclusion in LLM prompts.
        
        Args:
            schema_details: List of dictionaries containing schema information
            
        Returns:
            String representation of the schema
        """
        schema_text = "DATABASE SCHEMA:\n"
        
        # First list non-archive tables, then archive tables
        active_tables = [table for table in schema_details if not table['is_archive']]
        archive_tables = [table for table in schema_details if table['is_archive']]
        
        # Format active tables
        schema_text += "\n=== ACTIVE TABLES ===\n\n"
        for table in active_tables:
            schema_text += f"Table: {table['table_name']}\n"
            schema_text += "Columns:\n"
            
            for col in table['columns']:
                pk_marker = " (Primary Key)" if col['is_pk'] else ""
                fk_marker = " (Foreign Key)" if col['is_fk'] else ""
                schema_text += f"  - {col['name']} ({col['type']}){pk_marker}{fk_marker}\n"
            
            if table['primary_keys']:
                schema_text += "Primary Key: " + ", ".join(table['primary_keys']) + "\n"
            
            if table['foreign_keys']:
                schema_text += "Foreign Keys:\n"
                for fk in table['foreign_keys']:
                    confidence = f" (confidence: {fk['confidence']})" if 'confidence' in fk else ""
                    origin = f" (origin: {fk['origin']})" if 'origin' in fk else ""
                    schema_text += f"  - {fk['column']} -> {fk['ref_table']}.{fk['ref_column']}{confidence}{origin}\n"
            
            if table['composite_foreign_keys']:
                schema_text += "Composite Foreign Keys:\n"
                for cfk in table['composite_foreign_keys']:
                    cols = ", ".join(cfk['columns'])
                    ref_cols = ", ".join(cfk['ref_columns'])
                    confidence = f" (confidence: {cfk['confidence']})" if 'confidence' in cfk else ""
                    origin = f" (origin: {cfk['origin']})" if 'origin' in cfk else ""
                    schema_text += f"  - ({cols}) -> {cfk['ref_table']}.({ref_cols}){confidence}{origin}\n"
            
            schema_text += "\n"
        
        # Format archive tables if present
        if archive_tables:
            schema_text += "\n=== ARCHIVE TABLES ===\n\n"
            for table in archive_tables:
                schema_text += f"Table: {table['table_name']}\n"
                schema_text += "Columns:\n"
                
                for col in table['columns']:
                    pk_marker = " (Primary Key)" if col['is_pk'] else ""
                    fk_marker = " (Foreign Key)" if col['is_fk'] else ""
                    schema_text += f"  - {col['name']} ({col['type']}){pk_marker}{fk_marker}\n"
                
                # Add similar information as for active tables
                # ...
                
                schema_text += "\n"
        
        return schema_text
    
    @staticmethod
    def save_graph(graph, output_file = "schema_graph.json"):
        """
        Save the graph to a file for later use.
        
        Args:
            graph: NetworkX graph
            output_file: Path to output file
        """
        # Convert graph to a serializable format
        serializable_graph = {
            'nodes': [],
            'edges': []
        }
        
        for node, attrs in graph.nodes(data=True):
            node_data = {'id': node}
            node_data.update(attrs)
            serializable_graph['nodes'].append(node_data)
        
        for source, target, attrs in graph.edges(data=True):
            edge_data = {'source': source, 'target': target}
            edge_data.update(attrs)
            serializable_graph['edges'].append(edge_data)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(serializable_graph, f, indent=2)
        
        print(f"Graph saved to {output_file}")
    
    @staticmethod
    def load_graph(input_file):
        """
        Load a graph from a file.
        
        Args:
            input_file: Path to input file
            
        Returns:
            NetworkX graph
        """
        # Load from file
        with open(input_file, 'r') as f:
            serializable_graph = json.load(f)
        
        # Create new graph
        graph = nx.DiGraph()
        
        # Add nodes
        for node_data in serializable_graph['nodes']:
            node_id = node_data.pop('id')
            graph.add_node(node_id, **node_data)
        
        # Add edges
        for edge_data in serializable_graph['edges']:
            source = edge_data.pop('source')
            target = edge_data.pop('target')
            graph.add_edge(source, target, **edge_data)
        
        print(f"Graph loaded from {input_file}")
        return graph

