# CHESS for spider2: Usage Guide

Please refer to the original CHESS github for full usage guide and technical details:
https://github.com/ShayanTalaei/CHESS

Following are the instructions to run CHESS pipeline on the spider2 snowflake and sqlite datasets.

## 1. Installation

Make sure you have Python 3.10+ and install the required dependencies using the provided requirements file:

```
pip install -r requirements.txt
```

## 2. Basic Usage

### 2.1. Modify the .env file in the root directory:

* To run CHESS with spider2 sqlite, make sure to provide the DB_ROOT_PATH, which leads to the root directory where all .sqlite database files are stored. 
* This repository is set up to be used with Openai GPT-4o models. Please provide the OPENAI_API_KEY_PATH, which leads to a file with openai api key. 
* Other path specifications are optional but recommended to speed up inference especially for snowflake data. 

### 2.2. Modify the config file:

Modify the config file to choose the right tools.  
All the settings we used in our experiments are as they appear in ./configs/CHESS_IR_SS_CG_GPT.yaml.
For snowflake, the information_retriever tool is switched off.
For sqlite, switch on (uncomment) the information_retriever specification. 

Please, pay attention to the template names in each tool, as they need to be modified for each data type:
- For snowflake, use generate_candidate_one_snow_o3_full_revised and revise_one_snow_o3
- For sqlite, use generate_candidate_one and revise_one

### 2.3. Run the preprocessing script:

Run this script to create the minhash, LSH, and vector databases for sqlite data. 
We did not use this preprocessing script with snowflake data.  

```bash
sh run/run_preprocess.sh
```

### 2.3. Run the main script

```bash
sh run/run_main_ir_cg.sh
```

In the main script, choose data parameters (data_mode, data_path, data_type) and other (doc_path, snowflake_credentials and config) according to your setup. 

**Acknowledgments:** This repository is built from the original CHESS: https://github.com/ShayanTalaei/CHESS