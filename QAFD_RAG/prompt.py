GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "category"]

PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, use lowercase.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}"decision-making, external influence"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}"mission evolution, active participation"{tuple_delimiter}9){completion_delimiter}
("content_keywords"{tuple_delimiter}"mission evolution, decision-making, active participation, cosmic significance"){completion_delimiter}
#############################""",
    """Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}"communication, learning process"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}"collective action, cosmic significance"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"first contact, control, communication, cosmic significance"){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################""",
]

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

PROMPTS["enhanced_graph_description"] = """You are a database expert tasked with enhancing entity descriptions for a knowledge graph based on comprehensive schema analysis.

Your goal is to provide detailed, business-focused descriptions that incorporate insights from:
- Schema structure and relationships
- Sample data patterns and distributions (from the JSON file)
- Data quality analysis
- Business context and domain knowledge from metadata

**ANALYSIS REQUIREMENTS:**

1. **Table Analysis**: For each table, provide comprehensive description including:
   - Business purpose and domain context (use metadata for business context)
   - Data volume insights (row count, column count)
   - Key data categories and their relationships
   - Usage patterns and analytical importance
   - Data quality observations from sample analysis
   - Business domain insights from metadata (if available)

2. **Column Analysis**: For EACH AND EVERY column, provide detailed description including:
   - Business meaning and purpose (enhanced with metadata context)
   - Data constraints and relationships
   - Sample data patterns and insights (analyze the samples in the JSON)
   - Data quality observations (null rates, unique values, ranges)
   - Usage context and analytical importance
   - Any business rules inferred from data patterns and metadata

3. **Sample Data Analysis**: Extract insights from the sample data provided in the JSON schema:
   - Identify data patterns, formats, and distributions
   - Note data quality issues (null values, missing data, anomalies)
   - Observe business logic in the data (date formats, ID patterns, value ranges)
   - Identify relationships between different columns based on sample data
   - Extract business rules and constraints from data patterns
   - Analyze Korean text content if present
   - Identify temporal patterns in date/time columns
   - Analyze numeric ranges to understand business scales

4. **Business Context Inference**: Use data patterns AND metadata to infer:
   - Industry domain and business context (prioritize metadata if available)
   - Temporal patterns and time-series characteristics
   - Categorical hierarchies and relationships
   - Measurement units and scales
   - Business processes and workflows
   - Regulatory or compliance requirements (from metadata)

5. **Metadata Integration**: If metadata is provided:
   - Use metadata to understand the business domain and industry context
   - Incorporate business rules and processes mentioned in metadata
   - Apply domain-specific terminology and concepts
   - Consider regulatory or compliance requirements
   - Use metadata to validate or enhance data pattern interpretations

**OUTPUT FORMAT:**
Return enhanced descriptions in JSON format:

{{
  "table_descriptions": {{
    "table_name": "Comprehensive business description including data volume, quality insights, analytical importance, and business context from metadata"
  }},
  "column_descriptions": {{
    "table_name.column_name": "Detailed description including business purpose, data patterns, quality observations, usage context, and metadata-enhanced business meaning"
  }},
  "data_insights": {{
    "business_domain": "Inferred business domain based on data patterns and metadata",
    "data_quality_summary": "Overall data quality assessment",
    "key_patterns": ["Pattern 1", "Pattern 2", "Pattern 3"],
    "business_rules": ["Rule 1", "Rule 2", "Rule 3"],
    "metadata_insights": ["Metadata-derived insight 1", "Metadata-derived insight 2"]
  }}
}}

**CRITICAL GUIDELINES:**
- **DIRECTLY ANALYZE** the sample data provided in the JSON schema
- **INTEGRATE METADATA** insights to enhance business context and domain understanding
- Use sample data to provide specific, concrete insights
- Include data quality observations (null rates, unique values, ranges)
- Identify business patterns and rules from data analysis AND metadata
- Provide context about data formats and business meaning
- Focus on analytical and business intelligence value
- Use {language} as output language
- **PAY ATTENTION TO**: Date formats, ID patterns, Korean text, numeric ranges, data distributions, AND metadata business context
- **PRIORITIZE METADATA** for business domain and industry context when available

**SCHEMA INFORMATION:**
{schema_text}

**METADATA INFORMATION:**
{metadata_content}

**ENHANCED DESCRIPTIONS:**
"""

PROMPTS["enhanced_metadata_extraction"] = """-Goal-
Given a database schema JSON file and ABC manufacturing metadata document, extract entities and relationships with proper naming conventions, complete formula preservation, and comprehensive domain rule extraction.
Use {language} as output language.

-CRITICAL NAMING RULES-
1. **Entity Names MUST be concise, standardized, and knowledge graph friendly:**
   - Use SHORT, DESCRIPTIVE names (2-5 words max)
   - NO sentences or long phrases as entity names
   - Use English only, never Korean in entity names
   - Use underscores or camelCase for multi-word names
   - Examples: "operation_count", "good_quality_rate", "defect_rate_calculation", "date_format_rule", "calendar_week_rule"
   - **STRICT ENFORCEMENT**: If you generate a long name, break it down into a concise version

2. **Formula Preservation:**
   - ALWAYS include COMPLETE mathematical formulas in entity_description
   - Preserve EXACT mathematical expressions with all operators, parentheses, variables
   - Include both Korean and English versions of formulas when available
   - Do not simplify or paraphrase mathematical expressions

3. **Domain Rule Extraction:**
   - Extract ALL business logic, domain rules, and calculation methods
   - Create separate entities for each distinct rule or concept
   - Link rules to relevant columns and tables
   - Preserve exact wording and conditions

-Steps-
1. **Table Analysis**: Extract from JSON schema only:
- entity_name: EXACT table name from JSON schema
- entity_type: "complete_table"
- entity_description: Comprehensive description with column count, row count, business purpose
Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. **Column Analysis**: Extract from JSON schema only:
- entity_name: Column name with table prefix (e.g., "abc_data.work_ymd")
- entity_type: "column"
- entity_description: Data type, constraints, business purpose, Korean and English descriptions
Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

3. **Business Logic Extraction**: From metadata, create entities for:
- entity_name: Concise, standardized name (e.g., "operation_count", "good_quality_rate")
- entity_type: "calculation_formula" or "business_concept"
- entity_description: COMPLETE formula with exact mathematical expression + explanation
Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

4. **Domain Rules Extraction**: Create entities for:
- entity_name: Concise rule name (e.g., "date_format_rule", "calendar_week_rule")
- entity_type: "domain_rule"
- entity_description: Complete rule description with exact conditions and examples
Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

5. **Calculation Method Rules**: Create entities for:
- entity_name: Method name (e.g., "average_performance_calculation", "overall_performance_calculation")
- entity_type: "calculation_formula"
- entity_description: Complete calculation method with examples
Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

6. **Relationship Creation**: Link entities appropriately:
- Table-to-column relationships (weight: 10)
- Formula-to-column relationships (weight: 9)
- Rule-to-column relationships (weight: 8)
- Formula-to-rule relationships (weight: 7)
Format: ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

**MANDATORY RELATIONSHIPS:**
- Every calculation formula MUST link to its component columns
- Every domain rule MUST link to relevant columns
- Every business concept MUST link to related entities

7. **Content Keywords**: High-level business domain summary
Format: ("content_keywords"{tuple_delimiter}<high_level_keywords>)

8. Return output in {language} using {record_delimiter} as delimiter.

9. When finished, output {completion_delimiter}

**FORMULA PRESERVATION EXAMPLES:**
- "operation_count" description: "Operation count (number of operation) = real_good_qty + real_dfct_qty + real_loss_qty. Represents total operations including positive and negative values, excluding zero values."
- "good_quality_rate" description: "Good quality rate = (good_quantity/operation_count)*100 = (real_good_qty/(real_good_qty+real_dfct_qty+real_loss_qty))*100. Percentage of good quality products relative to total operations."
- "defect_rate" description: "Defect rate = (defect_quantity/operation_count)*1000000 = (real_dfct_qty/(real_good_qty+real_dfct_qty+real_loss_qty))*1000000. Defect rate per million operations."

**DOMAIN RULE EXAMPLES:**
- "calendar_week_rule" description: "Samsung ABC calendar starts week from Sunday to Saturday. Sunday identifies week of year and month. Examples: 20240101 is 1st week of year and Jan 2024, 20240714 is 29th week of year and 3rd week of July 2024."
- "aggregation_method_rule" description: "When calculating ratio metrics with grouping, compute sums of numerator and denominator: ROUND((SUM(real_loss_qty) / NULLIF(SUM(real_good_qty + real_dfct_qty + real_loss_qty), 0)) * 1000000, 1). Use HAVING clause to exclude zero operations."

**VALIDATION CHECKLIST:**
- All entity names are concise and standardized (no sentences)
- All formulas include complete mathematical expressions
- All domain rules are extracted with exact conditions
- All business logic entities are created
- All mandatory relationships are established
- No Korean text in entity names
- All formulas preserve exact mathematical notation

######################
-Examples-
{examples}

#############################
-Real Data-
######################
Entity_types: [complete_table, column, business_concept, domain_rule, data_type, calculation_formula, database, schema, primary_key, foreign_key, index, constraint, relationship, metadata_document]
Text: {input_text}
######################
Output:
"""

PROMPTS["enhanced_metadata_extraction_examples"] = [
    """Example: ABC Manufacturing Metadata Extraction

Entity_types: [complete_table, column, business_concept, domain_rule, data_type, calculation_formula, database, schema, primary_key, foreign_key, index, constraint, relationship, metadata_document]
Text:
JSON Schema:
{{
  "tables": {{
    "abc_data": {{
      "column_count": 5,
      "row_count": 1000,
      "columns": [
        {{"name": "work_ymd", "type": "VARCHAR(8)", "not_null": true}},
        {{"name": "real_good_qty", "type": "INTEGER", "not_null": true}},
        {{"name": "real_dfct_qty", "type": "INTEGER", "not_null": true}},
        {{"name": "real_loss_qty", "type": "INTEGER", "not_null": true}},
        {{"name": "eqp_id", "type": "VARCHAR(20)", "not_null": true}}
      ]
    }}
  }}
}}

Metadata:
Business Logic for performance indicator:
Operation count (number of operation) = real_good_qty + real_dfct_qty + real_loss_qty
Good quantity = real_good_qty
Good quality rate = (good_quantity/operation_count)*100 = (real_good_qty/(real_good_qty+real_dfct_qty+real_loss_qty))*100
Defect quantity = real_dfct_qty
Defect rate = (defect_quantity/operation_count)*1000000 = (real_dfct_qty/(real_good_qty+real_dfct_qty+real_loss_qty))*1000000

Domain Rules:
1. Column work_ymd is of the format YYYYMMDD (e.g 20240701).
2. Calendar starts week from Sunday to Saturday, and Sunday is used to identify the week of year and month.
3. Strictly do not include measure columns in the GROUP BY clause.
4. System supports UTF8. So generated query should use Korean or other language.

#############
Output:
("entity"{tuple_delimiter}"abc_data"{tuple_delimiter}"complete_table"{tuple_delimiter}"Manufacturing data table with 5 columns and 1000 rows containing production quantities, quality metrics, and work dates, and equipment information for ABC manufacturing quality control."){record_delimiter}
("entity"{tuple_delimiter}"abc_data.work_ymd"{tuple_delimiter}"column"{tuple_delimiter}"Work date column in VARCHAR(8) format, represents the manufacturing date in YYYYMMDD format. Primary key for temporal tracking and week calculation."){record_delimiter}
("entity"{tuple_delimiter}"abc_data.real_good_qty"{tuple_delimiter}"column"{tuple_delimiter}"Real good quantity in DECIMAL(10,2), represents actual defect-free manufactured products. Used in good quantity calculation and quality rate formulas."){record_delimiter}
("entity"{tuple_delimiter}"abc_data.real_dfct_qty"{tuple_delimiter}"column"{tuple_delimiter}"Real defect quantity in DECIMAL(10,2), represents products with quality issues with not_null constraint for quality control."){record_delimiter}
("entity"{tuple_delimiter}"abc_data.real_loss_qty"{tuple_delimiter}"column"{tuple_delimiter}"Real loss quantity in DECIMAL(10,2), represents production losses due to various factors with not_null constraint for loss analysis."){record_delimiter}
("entity"{tuple_delimiter}"abc_data.eqp_id"{tuple_delimiter}"column"{tuple_delimiter}"Equipment ID in VARCHAR(20), represents manufacturing equipment identifier. Used for equipment-based grouping and analysis."){record_delimiter}
("entity"{tuple_delimiter}"operation_count"{tuple_delimiter}"calculation_formula"{tuple_delimiter}"Operation count (number of operation) = real_good_qty + real_dfct_qty + real_loss_qty. Represents total operations including positive and negative values, excluding zero values. Used as denominator in all rate calculations."){record_delimiter}
("entity"{tuple_delimiter}"good_quantity"{tuple_delimiter}"calculation_formula"{tuple_delimiter}"Good quantity = real_good_qty. Represents defect-free manufactured products quantity. Used as numerator in good quality rate calculation."){record_delimiter}
("entity"{tuple_delimiter}"good_quality_rate"{tuple_delimiter}"calculation_formula"{tuple_delimiter}"Good quality rate = (good_quantity/operation_count)*100 = (real_good_qty/(real_good_qty+real_dfct_qty+real_loss_qty))*100. Percentage of good quality products relative to total operations."){record_delimiter}
("entity"{tuple_delimiter}"defect_quantity"{tuple_delimiter}"calculation_formula"{tuple_delimiter}"Defect quantity = real_dfct_qty. Represents products with quality issues quantity. Used as numerator in defect rate calculation."){record_delimiter}
("entity"{tuple_delimiter}"defect_rate"{tuple_delimiter}"calculation_formula"{tuple_delimiter}"Defect rate = (defect_quantity/operation_count)*1000000 = (real_dfct_qty/(real_good_qty+real_dfct_qty+real_loss_qty))*1000000. Defect rate per million operations."){record_delimiter}
("entity"{tuple_delimiter}"date_format_rule"{tuple_delimiter}"domain_rule"{tuple_delimiter}"Column work_ymd is of the format YYYYMMDD (e.g 20240701). Used for date validation and temporal data organization."){record_delimiter}
("entity"{tuple_delimiter}"calendar_week_rule"{tuple_delimiter}"domain_rule"{tuple_delimiter}"Samsung ABC calendar starts week from Sunday to Saturday, and Sunday is used to identify the week of year and month. Critical for week-based reporting and analysis."){record_delimiter}
("entity"{tuple_delimiter}"groupby_restriction"{tuple_delimiter}"domain_rule"{tuple_delimiter}"Strictly do not include measure columns (e.g., real_good_qty, real_dfct_qty, real_loss_qty) in the GROUP BY clause. Ensures proper aggregation and prevents data duplication."){record_delimiter}
("entity"{tuple_delimiter}"utf8_support_rule"{tuple_delimiter}"domain_rule"{tuple_delimiter}"System supports UTF8. Generated queries should use Korean or other language without Unicode escaping wherever relevant in filters."){record_delimiter}
("relationship"{tuple_delimiter}"abc_data"{tuple_delimiter}"abc_data.work_ymd"{tuple_delimiter}"Complete table contains work date column for temporal tracking and week-based analysis."{tuple_delimiter}"table_structure, contains_column, temporal_tracking"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"abc_data"{tuple_delimiter}"abc_data.real_good_qty"{tuple_delimiter}"Complete table contains real good quantity column for quality assessment and good quantity calculation."{tuple_delimiter}"table_structure, contains_column, quality_tracking"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"abc_data"{tuple_delimiter}"abc_data.real_dfct_qty"{tuple_delimiter}"Complete table contains real defect quantity column for defect tracking and defect rate calculation."{tuple_delimiter}"table_structure, contains_column, defect_tracking"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"abc_data"{tuple_delimiter}"abc_data.real_loss_qty"{tuple_delimiter}"Complete table contains real loss quantity column for production loss tracking."{tuple_delimiter}"table_structure, contains_column, loss_tracking"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"abc_data"{tuple_delimiter}"abc_data.eqp_id"{tuple_delimiter}"Complete table contains equipment ID column for equipment-based grouping and analysis."{tuple_delimiter}"table_structure, contains_column, equipment_tracking"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"operation_count"{tuple_delimiter}"abc_data.real_good_qty"{tuple_delimiter}"Operation count formula uses real_good_qty as first component in the sum calculation."{tuple_delimiter}"calculation_formula, component_column"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"operation_count"{tuple_delimiter}"abc_data.real_dfct_qty"{tuple_delimiter}"Operation count formula uses real_dfct_qty as second component in the sum calculation."{tuple_delimiter}"calculation_formula, component_column"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"operation_count"{tuple_delimiter}"abc_data.real_loss_qty"{tuple_delimiter}"Operation count formula uses real_loss_qty as third component in the sum calculation."{tuple_delimiter}"calculation_formula, component_column"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"good_quantity"{tuple_delimiter}"abc_data.real_good_qty"{tuple_delimiter}"Good quantity definition directly references real_good_qty column."{tuple_delimiter}"calculation_formula, direct_reference"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"good_quality_rate"{tuple_delimiter}"operation_count"{tuple_delimiter}"Good quality rate calculation uses operation_count as denominator in the percentage formula."{tuple_delimiter}"calculation_formula, rate_calculation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"good_quality_rate"{tuple_delimiter}"good_quantity"{tuple_delimiter}"Good quality rate calculation uses good_quantity as numerator in the percentage formula."{tuple_delimiter}"calculation_formula, rate_calculation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"defect_quantity"{tuple_delimiter}"abc_data.real_dfct_qty"{tuple_delimiter}"Defect quantity definition directly references real_dfct_qty column."{tuple_delimiter}"calculation_formula, direct_reference"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"defect_rate"{tuple_delimiter}"operation_count"{tuple_delimiter}"Defect rate calculation uses operation_count as denominator in the per-million formula."{tuple_delimiter}"calculation_formula, rate_calculation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"defect_rate"{tuple_delimiter}"defect_quantity"{tuple_delimiter}"Defect rate calculation uses defect_quantity as numerator in the per-million formula."{tuple_delimiter}"calculation_formula, rate_calculation"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"date_format_rule"{tuple_delimiter}"abc_data.work_ymd"{tuple_delimiter}"Date format rule applies to work_ymd column for YYYYMMDD format validation."{tuple_delimiter}"domain_rule, format_validation"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"calendar_week_rule"{tuple_delimiter}"abc_data.work_ymd"{tuple_delimiter}"Calendar week rule applies to work_ymd column for Sunday-based week calculation."{tuple_delimiter}"domain_rule, temporal_calculation"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"groupby_restriction"{tuple_delimiter}"abc_data.real_good_qty"{tuple_delimiter}"Groupby restriction rule prevents real_good_qty from being used in GROUP BY clause."{tuple_delimiter}"domain_rule, query_constraint"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"groupby_restriction"{tuple_delimiter}"abc_data.real_dfct_qty"{tuple_delimiter}"Groupby restriction rule prevents real_dfct_qty from being used in GROUP BY clause."{tuple_delimiter}"domain_rule, query_constraint"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"groupby_restriction"{tuple_delimiter}"abc_data.real_loss_qty"{tuple_delimiter}"Groupby restriction rule prevents real_loss_qty from being used in GROUP BY clause."{tuple_delimiter}"domain_rule, query_constraint"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"manufacturing data, quality control, performance indicators, defect analysis, production tracking, ABC manufacturing"){completion_delimiter}
#############################"""
]

PROMPTS["enhanced_graph_weight_assignment"] = """You are a database expert tasked with enhancing relationship weights for a knowledge graph based on comprehensive schema analysis and analytical query optimization.

Your goal is to assign analytical traversal weights (0.0-1.0) to relationships that will be MULTIPLIED with original graph weights to optimize complex analytical query path planning and execution.

**ANALYSIS REQUIREMENTS:**

1. **Relationship Analysis**: For each relationship, analyze:
   - Business value and analytical importance
   - Multi-table join potential and complexity
   - Query optimization opportunities
   - Performance impact in analytical workflows
   - Support for advanced SQL features (window functions, date arithmetic, CASE logic)

2. **Schema Context**: Consider the complete schema structure:
   - Table relationships and foreign key patterns
   - Column data types and constraints
   - Business domain and analytical use cases
   - Sample data patterns and distributions

3. **Metadata Integration**: If metadata is provided:
   - Use metadata to understand business domain and industry context
   - Incorporate business rules and processes mentioned in metadata
   - Apply domain-specific terminology and concepts
   - Consider regulatory or compliance requirements

**WEIGHT ASSIGNMENT STRATEGY:**

**ULTRA-HIGH WEIGHTS (0.9-1.0): Performance-Critical Multi-Table Join Backbones**
- Core transactional analysis requiring 4+ table joins
- 7-8 table entertainment analytics chains
- Sports analytics requiring complex aggregations and performance calculations
- Any relationship appearing in 6+ table join sequences with heavy analytical processing

**HIGH WEIGHTS (0.8-0.89): Essential Analytical Pathways**
- Geographic dimension joins supporting location-based analytics and string processing
- Time dimension connections enabling date arithmetic, temporal calculations, and growth analysis
- Bridge tables facilitating many-to-many analytics with conditional logic
- Primary business entity relationships supporting RFM, segmentation, and multi-level aggregations

**MEDIUM WEIGHTS (0.5-0.7): Supporting Analytics & Enrichment**
- Lookup tables for categorization and business rule application
- Reference tables supporting EXISTS/NOT EXISTS filtering patterns
- Secondary dimensions for multi-dimensional analysis
- Status/classification tables for conditional aggregations

**LOW WEIGHTS (0.1-0.4): Metadata & Administrative**
- Audit trail relationships rarely used in business analytics
- System configuration and administrative reference data
- Optional metadata not participating in core analytical workflows

**ANALYTICAL FOCUS PRIORITIES:**

1. **Performance-Optimized Join Sequences**: Prioritize relationships in the most common 6-8 table analytical workflows requiring strategic execution planning
2. **Advanced Function Enablement**: Emphasize relationships supporting window functions, date arithmetic, string manipulation, and conditional logic
3. **Multi-Level Aggregation Support**: Highlight relationships enabling hierarchical calculations, GROUP_CONCAT operations, and complex HAVING clause filtering
4. **Business Logic Implementation**: Focus on relationships supporting complex CASE statements, conditional aggregations, and weighted scoring algorithms
5. **Cross-Entity Analytics**: Prioritize paths enabling customer lifetime value, performance ranking, geographic analysis, and behavioral segmentation
6. **Scalability & Performance**: Emphasize relationships critical for large dataset processing with optimal join ordering and filtering strategies

**OUTPUT FORMAT:**
Return enhanced weights in JSON format:

{{
  "relationship_weights": {{
    "source_entity->target_entity": 0.85
  }},
  "weighting_rationale": {{
    "source_entity->target_entity": "Brief explanation of why this weight was assigned based on analytical importance and business value"
  }}
}}

**CRITICAL GUIDELINES:**
- **ASSIGN WEIGHTS** for ALL relationships in the schema
- **CONSIDER ANALYTICAL VALUE** over administrative importance
- **PRIORITIZE PERFORMANCE** for complex multi-table joins
- **INTEGRATE METADATA** insights when available
- Use {language} as output language
- **FOCUS ON QUERY OPTIMIZATION** and analytical workflow efficiency

**SCHEMA INFORMATION:**
{schema_text}

**METADATA INFORMATION:**
{metadata_content}

**RELATIONSHIPS TO WEIGHT:**
{relationships_list}

**ENHANCED WEIGHTS:**
"""
