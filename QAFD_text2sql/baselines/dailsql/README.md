# Installation 
Download the resource folder and place it in the root folder of baselines. You can download it from https://drive.google.com/file/d/1Y1zSbZ9cBUVJFI_W3t5EZZTzcUi1Ma4i/view

The following installation guidance is derived from [the original repository of Dail-SQL](https://github.com/BeachWang/DAIL-SQL).

Set up the Python environment:
```
conda create -n DAIL-SQL python=3.8
conda activate DAIL-SQL
cd spider2-lite/baselines/dailsql
pip install -r requirements.txt
python nltk_downloader.py
```

Download the model for spacy:
```
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl
```

# Running

Export your OpenAI API key:
```
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

Replace the VPN launch approach below with your own method, to gain access to OpenAI and Google BigQuery:
```
export https_proxy=http://127.0.0.1:15777 http_proxy=http://127.0.0.1:15777
```

Finally, simply run :laughing::
```
bash run.sh
```
this script automatically conducts all procedures: 1) data preprocess, 2) executing Dail-SQL, 3) evaluation. You can find the predicted SQL in `spider2-lite/baselines/dailsql/postprocessed_data`.

To switch from lite/snow change DEV to either spider2-lite or spider2-snow.

