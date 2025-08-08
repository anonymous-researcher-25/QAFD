# Installation 
Download the resource folder and place it in the root folder of baselines. You can download it from https://drive.google.com/file/d/1Y1zSbZ9cBUVJFI_W3t5EZZTzcUi1Ma4i/view

Set up the Python environment:
```
conda create -n DIN-SQL python=3.9
conda activate DIN-SQL
cd spider2-lite/baselines/dinsql
pip install -r requirements.txt
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

To switch from lite/snow change DEV to either spider2-lite or spider2-snow.
