import json
import asyncio
import re
from typing import Dict, List, Any, Optional
from .base import BaseGraphStorage, BaseVectorStorage
from .utils import logger, compute_mdhash_id
from .prompt import PROMPTS
import os


class DatabaseSchemaBuilder:
    """
    Database Schema Builder for QAFD_RAG
    
    This class handles the manual construction of knowledge graphs from database schema JSON files,
    avoiding the chunking issues that can cause LLM errors. It follows the approach used in CoFD.
    """
    
    def __init__(self, 
                 graph_storage: BaseGraphStorage,
                 entities_vdb: BaseVectorStorage,
                 relationships_vdb: BaseVectorStorage,
                 llm_model_func: callable,
                 schema_file_path: Optional[str] = None):
        self.graph_storage = graph_storage
        self.entities_vdb = entities_vdb
        self.relationships_vdb = relationships_vdb
        self.llm_model_func = llm_model_func
        self.schema_file_path = schema_file_path
        self._tables_info_cache = {}  # Cache for tables_info
    
    async def build_from_json_schema(self, 
                                   schema_file_path: str, 
                                   metadata_file_path: Optional[str] = None,
                                   language: str = "English") -> Dict[str, Any]:
        """
        Build knowledge graph from JSON schema file
        
        Args:
            schema_file_path: Path to the JSON schema file
            metadata_file_path: Optional path to metadata file
            language: Output language for descriptions
            
        Returns:
            Dictionary containing build statistics
        """
        logger.info(f"Building knowledge graph from schema: {schema_file_path}")
        
        # Update instance schema file path for validation methods
        self.schema_file_path = schema_file_path
        
        # Load JSON schema
        with open(schema_file_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        
        # Load metadata if provided
        metadata_content = None
        if metadata_file_path:
            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                metadata_content = f.read()
        
        # Step 1: Manually extract tables and columns from JSON schema
        tables_info = self._extract_tables_from_schema(schema_data)
        self._tables_info_cache = tables_info  # Cache for weight enhancement
        
        # Step 2: Insert tables and columns as entities
        entities_added = await self._insert_schema_entities(tables_info)
        
        # Step 3: Create relationships between tables and columns
        relationships_added = await self._create_schema_relationships(tables_info)
        
        # Step 3.5: Clean up any duplicate nodes that might have been created
        duplicates_removed = await self.graph_storage.remove_duplicate_nodes()
        if duplicates_removed > 0:
            logger.info(f"Cleaned up {duplicates_removed} duplicate nodes")
        
        # Step 3.6: Log graph statistics
        graph_stats = await self.graph_storage.get_graph_stats()
        logger.info(f"Graph statistics: {graph_stats['total_nodes']} nodes, {graph_stats['total_edges']} edges")
        logger.info(f"Node types: {graph_stats['node_types']}")
        
        # Step 4: Use LLM to enhance descriptions
        if self.llm_model_func:
            await self._enhance_descriptions_with_llm(tables_info, metadata_content, language)
        
        # Step 4.5: Use LLM to enhance relationship weights (CoFD-style)
        if self.llm_model_func:
            await self.enhance_relationship_weights_with_llm(tables_info, metadata_content, language)
        
        # Step 5: Insert metadata entities and relationships if available
        metadata_entities = 0
        metadata_relationships = 0
        if metadata_content and self.llm_model_func:
            metadata_entities, metadata_relationships = await self._process_metadata(
                metadata_content, tables_info, language
            )
        
        return {
            "tables_added": len(tables_info),
            "entities_added": entities_added,
            "relationships_added": relationships_added,
            "metadata_entities_added": metadata_entities,
            "metadata_relationships_added": metadata_relationships
        }
    
    def _extract_tables_from_schema(self, schema_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract table and column information from JSON schema
        
        Args:
            schema_data: Parsed JSON schema data
            
        Returns:
            Dictionary mapping table names to table information
        """
        tables_info = {}
        
        # Extract metadata
        metadata = schema_data.get("metadata", {})
        
        # Extract tables
        tables = schema_data.get("tables", {})
        
        for table_name, table_data in tables.items():
            table_info = {
                "name": table_name,
                "column_count": table_data.get("column_count", 0),
                "row_count": table_data.get("row_count", 0),
                "columns": []
            }
            
            # Extract columns
            columns = table_data.get("columns", [])
            for col in columns:
                column_info = {
                    "name": col.get("name", ""),
                    "type": col.get("type", ""),
                    "is_primary_key": col.get("is_primary_key", False),
                    "is_foreign_key": col.get("is_foreign_key", False),
                    "not_null": col.get("not_null", False),
                    "default": col.get("default", None),
                    "references_table": col.get("references_table", None),
                    "references_column": col.get("references_column", None)
                }
                table_info["columns"].append(column_info)
            
            tables_info[table_name] = table_info
        
        return tables_info
    
    async def _insert_schema_entities(self, tables_info: Dict[str, Dict[str, Any]]) -> int:
        """
        Insert tables and columns as entities in the graph and vector database
        
        Args:
            tables_info: Dictionary of table information
            
        Returns:
            Number of entities added
        """
        entities_added = 0
        entities_for_vdb = {}
        
        for table_name, table_data in tables_info.items():
            # Add table entity
            table_id = f'"{table_name.lower()}"'
            table_node_data = {
                "entity_type": "complete_table",
                "description": f"Database table '{table_name}' with {table_data['column_count']} columns and {table_data['row_count']} rows",
                "source_id": "schema_extraction",
                "table_name": table_name,
                "column_count": table_data["column_count"],
                "row_count": table_data["row_count"]
            }
            
            await self.graph_storage.upsert_node(table_id, node_data=table_node_data)
            
            # Prepare for vector database
            from .utils import compute_mdhash_id
            table_vdb_id = compute_mdhash_id(table_id, prefix="ent-")
            entities_for_vdb[table_vdb_id] = {
                "content": table_id + " " + table_node_data["description"],
                "entity_name": table_id
            }
            
            entities_added += 1
            
            # Add column entities
            for col in table_data["columns"]:
                col_name = col["name"]
                if not col_name:
                    continue
                    
                col_id = f'"{table_name.lower()}.{col_name.lower()}"'
                # Filter out None values to avoid GraphML writer issues
                col_node_data = {
                    "entity_type": "column",
                    "description": f"Column '{col_name}' of type {col['type']} in table '{table_name}'",
                    "source_id": "schema_extraction",
                    "table_name": table_name,
                    "column_name": col_name,
                    "data_type": col["type"],
                    "is_primary_key": col["is_primary_key"],
                    "is_foreign_key": col["is_foreign_key"],
                    "not_null": col["not_null"]
                }
                
                # Add optional fields only if they are not None
                if col["default"] is not None:
                    col_node_data["default"] = col["default"]
                if col["references_table"] is not None:
                    col_node_data["references_table"] = col["references_table"]
                if col["references_column"] is not None:
                    col_node_data["references_column"] = col["references_column"]
                
                await self.graph_storage.upsert_node(col_id, node_data=col_node_data)
                
                # Prepare for vector database
                col_vdb_id = compute_mdhash_id(col_id, prefix="ent-")
                entities_for_vdb[col_vdb_id] = {
                    "content": col_id + " " + col_node_data["description"],
                    "entity_name": col_id
                }
                
                entities_added += 1
        
        # Insert entities into vector database
        if entities_for_vdb and self.entities_vdb:
            await self.entities_vdb.upsert(entities_for_vdb)
            logger.info(f"Inserted {len(entities_for_vdb)} entities into vector database")
        
        return entities_added
    
    async def _create_schema_relationships(self, tables_info: Dict[str, Dict[str, Any]]) -> int:
        """
        Create relationships between tables and columns
        
        Args:
            tables_info: Dictionary of table information
            
        Returns:
            Number of relationships added
        """
        relationships_added = 0
        relationships_for_vdb = {}
        
        for table_name, table_data in tables_info.items():
            table_id = f'"{table_name.lower()}"'
            
            for col in table_data["columns"]:
                col_name = col["name"]
                if not col_name:
                    continue
                    
                col_id = f'"{table_name.lower()}.{col_name.lower()}"'
                
                # Create table-to-column relationship
                edge_data = {
                    "weight": 10.0,
                    "description": f"Table '{table_name}' contains column '{col_name}'",
                    "keywords": "table_structure, contains_column",
                    "source_id": "schema_extraction"
                }
                
                await self.graph_storage.upsert_edge(table_id, col_id, edge_data=edge_data)
                
                # Prepare for vector database
                from .utils import compute_mdhash_id
                edge_vdb_id = compute_mdhash_id(f"{table_id}->{col_id}", prefix="rel-")
                relationships_for_vdb[edge_vdb_id] = {
                    "src_id": table_id,
                    "tgt_id": col_id,
                    "content": edge_data["keywords"] + " " + table_id + " " + col_id + " " + edge_data["description"]
                }
                
                relationships_added += 1
                
                # Create foreign key relationships if applicable
                if col["is_foreign_key"] and col["references_table"] and col["references_column"]:
                    ref_table_id = f'"{col["references_table"].lower()}"'
                    ref_col_id = f'"{col["references_table"].lower()}.{col["references_column"].lower()}"'
                    
                    fk_edge_data = {
                        "weight": 15.0,
                        "description": f"Foreign key relationship: '{col_name}' in '{table_name}' references '{col['references_column']}' in '{col['references_table']}'",
                        "keywords": "foreign_key, references, data_integrity",
                        "source_id": "schema_extraction"
                    }
                    
                    await self.graph_storage.upsert_edge(col_id, ref_col_id, edge_data=fk_edge_data)
                    
                    # Prepare for vector database
                    fk_edge_vdb_id = compute_mdhash_id(f"{col_id}->{ref_col_id}", prefix="rel-")
                    relationships_for_vdb[fk_edge_vdb_id] = {
                        "src_id": col_id,
                        "tgt_id": ref_col_id,
                        "content": fk_edge_data["keywords"] + " " + col_id + " " + ref_col_id + " " + fk_edge_data["description"]
                    }
                    
                    relationships_added += 1
        
        # Insert relationships into vector database
        if relationships_for_vdb and self.relationships_vdb:
            await self.relationships_vdb.upsert(relationships_for_vdb)
            logger.info(f"Inserted {len(relationships_for_vdb)} relationships into vector database")
        
        return relationships_added
    
    async def _enhance_descriptions_with_llm(self, 
                                           tables_info: Dict[str, Dict[str, Any]], 
                                           metadata_content: Optional[str],
                                           language: str) -> None:
        """
        Use LLM to enhance entity descriptions
        
        Args:
            tables_info: Dictionary of table information
            metadata_content: Optional metadata content
            language: Output language
        """
        logger.info("Enhancing descriptions with LLM...")
        
        # Prepare input for LLM
        schema_text = self._format_schema_for_llm(tables_info)
        
        # Create prompt for description enhancement
        prompt = self._create_description_enhancement_prompt(schema_text, metadata_content, language)
        
        try:
            # Get enhanced descriptions from LLM
            enhanced_result = await self.llm_model_func(prompt)
            
            # Parse and apply enhanced descriptions
            await self._apply_enhanced_descriptions(enhanced_result, tables_info)
            
        except Exception as e:
            logger.error(f"Error enhancing descriptions with LLM: {e}")
    
    async def enhance_relationship_weights_with_llm(self, 
                                                  tables_info: Dict[str, Dict[str, Any]], 
                                                  metadata_content: Optional[str],
                                                  language: str = "English") -> None:
        """
        Use LLM to enhance relationship weights (CoFD-style weight reassignment)
        
        Args:
            tables_info: Dictionary of table information
            metadata_content: Optional metadata content
            language: Output language
        """
        logger.info("Enhancing relationship weights with LLM (CoFD-style)...")
        
        # Prepare input for LLM
        schema_text = self._format_schema_for_llm(tables_info)
        
        # Get all relationships from the graph
        relationships_list = await self._get_all_relationships()
        
        # Create prompt for weight enhancement
        prompt = self._create_weight_enhancement_prompt(schema_text, metadata_content, relationships_list, language)
        
        try:
            # Get enhanced weights from LLM
            enhanced_result = await self.llm_model_func(prompt)
            
            # Parse and apply enhanced weights
            await self._apply_enhanced_weights(enhanced_result)
            
        except Exception as e:
            logger.error(f"Error enhancing relationship weights with LLM: {e}")
    
    async def _get_all_relationships(self) -> str:
        """
        Get all relationships from the graph for LLM processing
        
        Returns:
            Formatted string of all relationships
        """
        relationships = []
        
        try:
            # Get all edges from the actual graph
            all_edges = await self.graph_storage.edges()
            
            if all_edges:
                # Convert EdgeView to list for easier handling
                edges_list = list(all_edges)
                logger.info(f"Found {len(edges_list)} edges in graph")
                
                for source, target in edges_list:
                    # Get edge data to get current weight and description
                    edge_data = await self.graph_storage.get_edge(source, target)
                    if edge_data:
                        current_weight = edge_data.get('weight', 1.0)
                        description = edge_data.get('description', 'unknown')
                        relationship_info = f"- {source} -> {target} (current_weight: {current_weight}, description: {description})"
                        relationships.append(relationship_info)
                    else:
                        logger.warning(f"Could not get edge data for {source} -> {target}")
            else:
                # Fallback to schema-based relationships if graph is empty
                logger.warning("Graph is empty, falling back to schema-based relationships")
                for table_name, table_data in self._tables_info_cache.items():
                    table_id = f'"{table_name.lower()}"'
                    
                    for col in table_data.get("columns", []):
                        col_name = col.get("name", "")
                        if col_name:
                            col_id = f'"{table_name.lower()}.{col_name.lower()}"'
                            
                            # Table-to-column relationship
                            relationship_info = f"- {table_id} -> {col_id} (current_weight: 10.0, description: table_structure)"
                            relationships.append(relationship_info)
                            
                            # Foreign key relationships if applicable
                            if col.get("is_foreign_key") and col.get("references_table") and col.get("references_column"):
                                ref_table_id = f'"{col["references_table"].lower()}"'
                                ref_col_id = f'"{col["references_table"].lower()}.{col["references_column"].lower()}"'
                                
                                fk_relationship_info = f"- {col_id} -> {ref_col_id} (current_weight: 9.0, description: foreign_key)"
                                relationships.append(fk_relationship_info)
        
        except Exception as e:
            logger.error(f"Error getting relationships from graph: {e}")
            # Fallback to empty list
            relationships = []
        
        return "\n".join(relationships)
    
    def _create_weight_enhancement_prompt(self, 
                                        schema_text: str, 
                                        metadata_content: Optional[str],
                                        relationships_list: str,
                                        language: str) -> str:
        """
        Create prompt for LLM weight enhancement
        
        Args:
            schema_text: Formatted schema text
            metadata_content: Optional metadata content
            relationships_list: List of all relationships
            language: Output language
            
        Returns:
            Formatted prompt
        """
        # Load the complete JSON file to include sample data
        try:
            # Use the instance schema file path if available
            if self.schema_file_path and os.path.exists(self.schema_file_path):
                with open(self.schema_file_path, 'r', encoding='utf-8') as f:
                    complete_schema_data = json.load(f)
                complete_schema_text = json.dumps(complete_schema_data, indent=2, ensure_ascii=False)
            else:
                complete_schema_text = schema_text
        except Exception as e:
            logger.warning(f"Could not load complete schema file: {e}")
            complete_schema_text = schema_text
        
        # Use the new weight enhancement prompt template
        prompt_template = PROMPTS["enhanced_graph_weight_assignment"]
        
        # Format the prompt with the complete schema data
        prompt = prompt_template.format(
            language=language,
            schema_text=complete_schema_text,
            metadata_content=metadata_content or "No additional metadata provided.",
            relationships_list=relationships_list
        )
        
        return prompt
    
    async def _apply_enhanced_weights(self, enhanced_result: str) -> None:
        """
        Apply enhanced weights from LLM to graph relationships (CoFD-style multiplication)
        
        Args:
            enhanced_result: LLM response with enhanced weights
        """
        try:
            # Extract JSON from the response with improved parsing
            import re
            
            # Try multiple patterns to find JSON
            json_patterns = [
                r'\{.*\}',  # Basic JSON object
                r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
                r'```\s*(\{.*?\})\s*```',  # JSON in generic code blocks
                r'```\s*(\{.*\})\s*```',  # JSON in code blocks (non-greedy)
            ]
            
            enhanced_data = None
            for pattern in json_patterns:
                json_match = re.search(pattern, enhanced_result, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
                        enhanced_data = json.loads(json_str)
                        break
                    except json.JSONDecodeError:
                        continue
            
            if not enhanced_data:
                logger.warning("No valid JSON found in enhanced result, skipping weight enhancement")
                return
            
            # Apply relationship weights (CoFD-style multiplication)
            relationship_weights = enhanced_data.get("relationship_weights", {})
            weighting_rationale = enhanced_data.get("weighting_rationale", {})
            
            weights_updated = 0
            for relationship_key, llm_score in relationship_weights.items():
                if '->' in relationship_key:
                    source, target = relationship_key.split('->', 1)
                    
                    # Clean up the source and target IDs (remove quotes if present)
                    source = source.strip().strip('"')
                    target = target.strip().strip('"')
                    
                    # Add quotes to match the graph format
                    source_with_quotes = f'"{source}"'
                    target_with_quotes = f'"{target}"'
                    
                    # Try both formats: with and without quotes
                    edge_data = await self.graph_storage.get_edge(source_with_quotes, target_with_quotes)
                    actual_source = source_with_quotes
                    actual_target = target_with_quotes
                    
                    if not edge_data:
                        # Try without quotes as fallback
                        edge_data = await self.graph_storage.get_edge(source, target)
                        actual_source = source
                        actual_target = target
                        
                    if edge_data:
                        # Get original weight (CoFD-style)
                        original_weight = edge_data.get('weight', 1.0)
                        
                        # Multiply original weight with LLM score (CoFD approach)
                        enhanced_weight = original_weight * llm_score
                        
                        # Update edge with new weight
                        edge_data['weight'] = enhanced_weight
                        edge_data['llm_enhanced'] = True
                        edge_data['llm_score'] = llm_score
                        edge_data['original_weight'] = original_weight
                        
                        # Add rationale if available
                        if relationship_key in weighting_rationale:
                            edge_data['weighting_rationale'] = weighting_rationale[relationship_key]
                        
                        await self.graph_storage.upsert_edge(actual_source, actual_target, edge_data=edge_data)
                        weights_updated += 1
                        logger.info(f"Updated weight for {relationship_key}: {original_weight} * {llm_score} = {enhanced_weight}")
                    else:
                        logger.warning(f"Edge not found in graph: {source} -> {target} (tried both quoted and unquoted formats)")
            
            if weights_updated == 0:
                logger.warning("No edges were found to update. This might indicate:")
                logger.warning("1. The graph hasn't been built yet")
                logger.warning("2. The relationship keys don't match the actual edges in the graph")
                logger.warning("3. The LLM response format is incorrect")
                
                # Log some debug information
                all_edges = await self.graph_storage.edges()
                edges_list = list(all_edges) if all_edges else []
                logger.info(f"Total edges in graph: {len(edges_list)}")
                if edges_list:
                    logger.info(f"Sample edges: {edges_list[:5]}")
                logger.info(f"Relationship keys from LLM: {list(relationship_weights.keys())[:5]}")
            
            logger.info(f"Successfully updated {weights_updated} relationship weights with LLM enhancement")
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from enhanced result: {e}")
        except Exception as e:
            logger.error(f"Error applying enhanced weights: {e}")
    
    def _format_schema_for_llm(self, tables_info: Dict[str, Dict[str, Any]]) -> str:
        """
        Format schema information for LLM processing
        
        Args:
            tables_info: Dictionary of table information
            
        Returns:
            Formatted schema text
        """
        schema_text = "JSON Schema:\n"
        schema_text += json.dumps({"tables": tables_info}, indent=2, ensure_ascii=False)
        return schema_text
    

    
    def _create_description_enhancement_prompt(self, 
                                             schema_text: str, 
                                             metadata_content: Optional[str],
                                             language: str) -> str:
        """
        Create prompt for LLM description enhancement
        
        Args:
            schema_text: Formatted schema text
            metadata_content: Optional metadata content
            language: Output language
            
        Returns:
            Formatted prompt
        """
        # Load the complete JSON file to include sample data
        try:
            # Use the instance schema file path if available
            if self.schema_file_path and os.path.exists(self.schema_file_path):
                with open(self.schema_file_path, 'r', encoding='utf-8') as f:
                    complete_schema_data = json.load(f)
                complete_schema_text = json.dumps(complete_schema_data, indent=2, ensure_ascii=False)
            else:
                complete_schema_text = schema_text
        except Exception as e:
            logger.warning(f"Could not load complete schema file: {e}")
            complete_schema_text = schema_text
        
        # Use the new enhanced graph description prompt template
        prompt_template = PROMPTS["enhanced_graph_description"]
        
        # Format the prompt with the complete schema data
        prompt = prompt_template.format(
            language=language,
            schema_text=complete_schema_text,
            metadata_content=metadata_content or "No additional metadata provided."
        )
        
        return prompt
    
    async def _apply_enhanced_descriptions(self, 
                                         enhanced_result: str, 
                                         tables_info: Dict[str, Dict[str, Any]]) -> None:
        """
        Apply enhanced descriptions from LLM to graph entities
        
        Args:
            enhanced_result: LLM response with enhanced descriptions
            tables_info: Dictionary of table information
        """
        try:
            # Extract JSON from the response with improved parsing
            import re
            
            # Try multiple patterns to find JSON
            json_patterns = [
                r'\{.*\}',  # Basic JSON object
                r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
                r'```\s*(\{.*?\})\s*```',  # JSON in generic code blocks
                r'```\s*(\{.*\})\s*```',  # JSON in code blocks (non-greedy)
            ]
            
            enhanced_data = None
            for pattern in json_patterns:
                json_match = re.search(pattern, enhanced_result, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
                        enhanced_data = json.loads(json_str)
                        break
                    except json.JSONDecodeError:
                        continue
            
            if not enhanced_data:
                logger.warning("No valid JSON found in enhanced result, skipping description enhancement")
                return
            
            # Apply table descriptions
            table_descriptions = enhanced_data.get("table_descriptions", {})
            for table_name, description in table_descriptions.items():
                table_id = f'"{table_name.lower()}"'
                if await self.graph_storage.has_node(table_id):
                    # Get existing node data and update description
                    existing_data = await self.graph_storage.get_node(table_id)
                    if existing_data:
                        existing_data["description"] = description
                        await self.graph_storage.upsert_node(table_id, existing_data)
                        logger.info(f"Updated table description for {table_name}")
            
            # Apply column descriptions
            column_descriptions = enhanced_data.get("column_descriptions", {})
            for column_name, description in column_descriptions.items():
                column_id = f'"{column_name.lower()}"'
                if await self.graph_storage.has_node(column_id):
                    # Get existing node data and update description
                    existing_data = await self.graph_storage.get_node(column_id)
                    if existing_data:
                        existing_data["description"] = description
                        await self.graph_storage.upsert_node(column_id, existing_data)
                        logger.info(f"Updated column description for {column_name}")
            
            # Log data insights if available
            data_insights = enhanced_data.get("data_insights", {})
            if data_insights:
                logger.info(f"Data insights extracted: {data_insights}")
                
                # Store insights as metadata entities
                await self._store_data_insights(data_insights, tables_info)
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from enhanced result: {e}")
        except Exception as e:
            logger.error(f"Error applying enhanced descriptions: {e}")
    
    async def _store_data_insights(self, data_insights: Dict[str, Any], tables_info: Dict[str, Dict[str, Any]]) -> None:
        """
        Store data insights as metadata entities in the graph
        
        Args:
            data_insights: Dictionary containing data insights
            tables_info: Dictionary of table information
        """
        try:
            # Skip all data insights - don't insert them into the graph to avoid long node names
            logger.info("Data insights found but not inserted into graph to avoid long node names")
                
        except Exception as e:
            logger.error(f"Error storing data insights: {e}")
    
    async def _process_metadata(self, 
                              metadata_content: str, 
                              tables_info: Dict[str, Dict[str, Any]],
                              language: str) -> tuple[int, int]:
        """
        Process metadata content to extract business concepts and relationships
        
        Args:
            metadata_content: Metadata text content
            tables_info: Dictionary of table information
            language: Output language
            
        Returns:
            Tuple of (entities_added, relationships_added)
        """
        logger.info("Processing metadata for business concepts...")
        
        # Create prompt for metadata processing
        prompt = self._create_metadata_processing_prompt(metadata_content, tables_info, language)
        
        try:
            # Get business concepts from LLM
            metadata_result = await self.llm_model_func(prompt)
            
            # Parse and insert business concepts and relationships
            entities_added, relationships_added = await self._parse_and_insert_metadata_entities(
                metadata_result, tables_info
            )
            
            return entities_added, relationships_added
            
        except Exception as e:
            logger.error(f"Error processing metadata with LLM: {e}")
            return 0, 0
    
    def _create_metadata_processing_prompt(self, 
                                         metadata_content: str, 
                                         tables_info: Dict[str, Dict[str, Any]],
                                         language: str) -> str:
        """
        Create prompt for metadata processing
        
        Args:
            metadata_content: Metadata text content
            tables_info: Dictionary of table information
            language: Output language
            
        Returns:
            Formatted prompt
        """
        schema_text = self._format_schema_for_llm(tables_info)
        
        # Use the enhanced metadata extraction prompt for better entity naming and formula preservation
        prompt_template = PROMPTS["enhanced_metadata_extraction"]
        examples = PROMPTS["enhanced_metadata_extraction_examples"][0]  # Use first example
        
        context_base = {
            "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
            "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            "entity_types": "complete_table, column, business_concept, domain_rule, data_type, calculation_formula, database, schema, primary_key, foreign_key, index, constraint, relationship, metadata_document",
            "examples": examples.format(**{
                "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
                "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
                "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
            }),
            "language": language,
            "input_text": f"{schema_text}\n\nMetadata:\n{metadata_content}"
        }
        
        return prompt_template.format(**context_base)
    
    async def _parse_and_insert_metadata_entities(self, 
                                                 metadata_result: str, 
                                                 tables_info: Dict[str, Dict[str, Any]]) -> tuple[int, int]:
        """
        Parse LLM result and insert metadata entities and relationships
        
        Args:
            metadata_result: LLM response with metadata entities
            tables_info: Dictionary of table information
            
        Returns:
            Tuple of (entities_added, relationships_added)
        """
        entities_added = 0
        relationships_added = 0
        
        try:
            # Parse the result using the same logic as extract_entities
            # This is a simplified implementation - you may want to reuse the parsing logic from operate.py
            
            # Split by record delimiter
            records = metadata_result.split(PROMPTS["DEFAULT_RECORD_DELIMITER"])
            
            for record in records:
                record = record.strip()
                if not record or not record.startswith("("):
                    continue
                    
                # Parse entity or relationship
                if record.startswith('("entity"'):
                    # Parse entity
                    entity_data = self._parse_entity_record(record)
                    if entity_data:
                        await self._insert_metadata_entity(entity_data)
                        entities_added += 1
                        
                elif record.startswith('("relationship"'):
                    # Parse relationship
                    relationship_data = self._parse_relationship_record(record)
                    if relationship_data:
                        await self._insert_metadata_relationship(relationship_data)
                        relationships_added += 1
            
        except Exception as e:
            logger.error(f"Error parsing metadata entities: {e}")
        
        return entities_added, relationships_added
    
    def _parse_entity_record(self, record: str) -> Optional[Dict[str, Any]]:
        """
        Parse entity record from LLM response
        
        Args:
            record: Entity record string
            
        Returns:
            Parsed entity data or None
        """
        try:
            # Remove outer parentheses and split by tuple delimiter
            content = record[1:-1]  # Remove outer parentheses
            parts = content.split(PROMPTS["DEFAULT_TUPLE_DELIMITER"])
            
            if len(parts) >= 4:
                entity_name = parts[1].strip('"')
                entity_type = parts[2].strip('"')
                entity_description = parts[3].strip('"')
                
                return {
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "description": entity_description
                }
        except Exception as e:
            logger.error(f"Error parsing entity record: {e}")
        
        return None
    
    def _parse_relationship_record(self, record: str) -> Optional[Dict[str, Any]]:
        """
        Parse relationship record from LLM response
        
        Args:
            record: Relationship record string
            
        Returns:
            Parsed relationship data or None
        """
        try:
            # Remove outer parentheses and split by tuple delimiter
            content = record[1:-1]  # Remove outer parentheses
            parts = content.split(PROMPTS["DEFAULT_TUPLE_DELIMITER"])
            
            if len(parts) >= 6:
                source_entity = parts[1].strip('"')
                target_entity = parts[2].strip('"')
                relationship_description = parts[3].strip('"')
                relationship_keywords = parts[4].strip('"')
                relationship_strength = float(parts[5])
                
                return {
                    "source_entity": source_entity,
                    "target_entity": target_entity,
                    "description": relationship_description,
                    "keywords": relationship_keywords,
                    "weight": relationship_strength
                }
        except Exception as e:
            logger.error(f"Error parsing relationship record: {e}")
        
        return None
    
    async def _insert_metadata_entity(self, entity_data: Dict[str, Any]) -> None:
        """
        Insert metadata entity into graph and vector database
        
        Args:
            entity_data: Entity data dictionary
        """
        entity_name = entity_data["entity_name"]
        
        # Validate column references - if entity_name contains a table.column pattern,
        # verify that the column actually exists in the schema
        if "." in entity_name and entity_data["entity_type"] == "column":
            parts = entity_name.split(".", 1)
            if len(parts) == 2:
                table_name, column_name = parts
                # Check if this column exists in the schema
                if not self._is_valid_column_reference(table_name, column_name):
                    logger.warning(f"Skipping invalid column reference: {entity_name}")
                    return
        
        # Normalize entity name to avoid overly long names
        normalized_name = self._normalize_entity_name(entity_name, "METADATA_ENTITY")
        entity_id = f'"{normalized_name}"'
        
        node_data = {
            "entity_type": entity_data["entity_type"],
            "description": entity_data["description"],
            "source_id": "metadata_extraction"
        }
        
        await self.graph_storage.upsert_node(entity_id, node_data=node_data)
        
        # Insert into vector database
        if self.entities_vdb:
            from .utils import compute_mdhash_id
            entity_vdb_id = compute_mdhash_id(entity_id, prefix="ent-")
            entity_vdb_data = {
                entity_vdb_id: {
                    "content": entity_id + " " + entity_data["description"],
                    "entity_name": entity_id
                }
            }
            await self.entities_vdb.upsert(entity_vdb_data)
    
    async def _insert_metadata_relationship(self, relationship_data: Dict[str, Any]) -> None:
        """
        Insert metadata relationship into graph and vector database
        
        Args:
            relationship_data: Relationship data dictionary
        """
        source_entity = relationship_data["source_entity"]
        target_entity = relationship_data["target_entity"]
        
        # Validate column references in relationships
        if "." in target_entity:
            parts = target_entity.split(".", 1)
            if len(parts) == 2:
                table_name, column_name = parts
                if not self._is_valid_column_reference(table_name, column_name):
                    logger.warning(f"Skipping relationship with invalid column reference: {target_entity}")
                    return
        
        # Normalize entity names for relationships
        normalized_source = self._normalize_entity_name(source_entity, "SOURCE_ENTITY")
        normalized_target = self._normalize_entity_name(target_entity, "TARGET_ENTITY")
        
        src_id = f'"{normalized_source}"'
        tgt_id = f'"{normalized_target}"'
        
        edge_data = {
            "weight": relationship_data["weight"],
            "description": relationship_data["description"],
            "keywords": relationship_data["keywords"],
            "source_id": "metadata_extraction"
        }
        
        await self.graph_storage.upsert_edge(src_id, tgt_id, edge_data=edge_data)
        
        # Insert into vector database
        if self.relationships_vdb:
            from .utils import compute_mdhash_id
            relationship_vdb_id = compute_mdhash_id(f"{src_id}->{tgt_id}", prefix="rel-")
            relationship_vdb_data = {
                relationship_vdb_id: {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "content": relationship_data["keywords"] + " " + src_id + " " + tgt_id + " " + relationship_data["description"]
                }
            }
            await self.relationships_vdb.upsert(relationship_vdb_data)
    
    def _is_valid_column_reference(self, table_name: str, column_name: str) -> bool:
        """
        Check if a column reference is valid by comparing with actual schema
        
        Args:
            table_name: Name of the table
            column_name: Name of the column
            
        Returns:
            True if the column exists in the schema, False otherwise
        """
        # Load the schema to check column existence
        try:
            if self.schema_file_path and os.path.exists(self.schema_file_path):
                with open(self.schema_file_path, 'r', encoding='utf-8') as f:
                    schema_data = json.load(f)
                
                tables = schema_data.get("tables", {})
                if table_name in tables:
                    columns = tables[table_name].get("columns", [])
                    for col in columns:
                        if col.get("name") == column_name:
                            return True
        except Exception as e:
            logger.warning(f"Error validating column reference: {e}")
        
        return False
    
    def _normalize_entity_name(self, text: str, fallback_name: str = "ENTITY") -> str:
        """
        Normalize entity name to be concise and knowledge graph friendly
        
        Args:
            text: Original text to normalize
            fallback_name: Fallback name if normalization fails
            
        Returns:
            Normalized entity name
        """
        try:
            # If the text already contains a dot (like table.column format), preserve it
            if '.' in text:
                return text.lower()
            
            # Remove special characters and extra spaces
            cleaned = re.sub(r'[^\w\s]', ' ', text)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # Split into words and take first few meaningful words
            words = cleaned.split()
            meaningful_words = []
            
            # Filter out common stop words and take first 3-5 meaningful words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
            
            for word in words:
                if word.lower() not in stop_words and len(word) > 2:
                    meaningful_words.append(word.lower())
                    if len(meaningful_words) >= 4:  # Limit to 4 words max
                        break
            
            # If we have meaningful words, create a name
            if meaningful_words:
                return '_'.join(meaningful_words)
            else:
                # If no meaningful words, use fallback
                return fallback_name
                
        except Exception as e:
            logger.warning(f"Error normalizing entity name '{text}': {e}")
            return fallback_name 