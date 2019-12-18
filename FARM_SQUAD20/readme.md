# [FARM](https://github.com/deepset-ai/FARM) for QA on [SQUAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)

#### on hpc-cluster gateway
very first run: on frontend/gateway (in order to download pretrained models) in `DataSilo`-constructor set `automatic_loading=False` and run python-script on frontend just right after the line where the language model is loaded

    cd natural-language-processing
    sbatch FARM_SQUAD20/hpc_gpu_job_farm_squad20.sh
    
#### on local system
1. mount hpc-filesystem with sshfs
2. cd to `natural-language-processing` where FARM should have already created `mlruns` folder
3. run `mlflow ui` + goto localhost:5000 (see training progress)