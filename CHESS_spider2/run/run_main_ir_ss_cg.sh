#!/bin/bash
data_mode='dev' # Options: 'dev', 'train' 

data_path='' # path to spider2-snow.jsonl

doc_path='' # path to spider2-snow/resource/documents or spider2-lite/resource/documents

snowflake_credentials='' # path to snowflake_credential.json

data_type='spider2-sqlite' # or 'spider2-snow'
config="./configs/CHESS_IR_SS_CG_GPT.yaml"

num_workers=1 # Number of workers to use for parallel processing, set to 1 for no parallel processing

python3 -u ./src/main.py \
        --data_mode ${data_mode} \
        --data_path ${data_path} \
        --doc_path ${doc_path} \
        --data_type ${data_type} \
        --config "$config" \
        --num_workers ${num_workers} \
        --pick_final_sql true \
        --snowflake_credentials ${snowflake_credentials}


