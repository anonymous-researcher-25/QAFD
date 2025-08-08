
# Installation

The following installation guidance is derived from [the original repository of CodeS](https://github.com/RUCKBReasoning/codes).

Download the resource folder and place it in the root folder of baselines. You can download it from https://drive.google.com/file/d/1Y1zSbZ9cBUVJFI_W3t5EZZTzcUi1Ma4i/view
#### Step1: Install Java
```
apt-get update
apt-get install -y openjdk-11-jdk
```
If you already have a Java environment installed, you can skip this step.

#### Step2: Create Python Environments
```
conda create -n CodeS python=3.9 -y
conda activate CodeS
pip install -r requirements.txt
git clone https://github.com/lihaoyang-ruc/SimCSE.git
cd SimCSE
python setup.py install
cd ..
```

#### Step3: Download Checkpoints
Download the schema item classifier checkpoints [sic_ckpts.zip](https://drive.google.com/file/d/1V3F4ihTSPbV18g3lrg94VMH-kbWR_-lY/view?usp=sharing) and unzip it:
```
unzip sic_ckpts.zip
```

Download the SFT model `seeklhy/codes-7b-merged`  from:
```
https://huggingface.co/seeklhy/codes-7b-merged/tree/main
```

Since we release Spider2-SQL as a pure test-set (without its corresponding train-set), few-shot in-context learning is infeasible. By default we use `seeklhy/codes-7b-merged` to enhance the model's ability to generalize to the unseen Spider2-SQL domain. **For quick deployment, you may opt to use the smaller `seeklhy/codes-1b` model instead.**


# Running
Replace the VPN launch approach below with your own method, to gain access to Google BigQuery:
```
export https_proxy=http://127.0.0.1:15777 http_proxy=http://127.0.0.1:15777
```
Then, simply run :laughing::
```
bash run.sh
```
this script automatically conducts all procedures: 1) data preprocess, 2) executing CodeS, 3) evaluation. You can find the output SQL in `spider2-lite/baselines/codes/postprocessed_data`.

To switch from lite/snow change DEV to either spider2-lite or spider2-snow.
