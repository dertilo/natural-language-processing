# [FARM](https://github.com/deepset-ai/FARM) for QA on [SQUAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)

#### on hpc-cluster gateway
very first run: on frontend/gateway (in order to download pretrained models) in `DataSilo`-constructor set `automatic_loading=False` and run python-script on frontend just right after the line where the language model is loaded

    cd natural-language-processing
    sbatch FARM_SQUAD20/hpc_gpu_job_farm_squad20.sh
    
#### on local system
1. mount hpc-filesystem with sshfs
2. cd to `natural-language-processing` where FARM should have already created `mlruns` folder
3. run `mlflow ui` + goto localhost:5000 (see training progress)

#### results
* ran for ~9 hours on HPC-cluster
    
    
         Parameters
        Name	Value
        ave_seq_len	172.15870608705072
        batch_size	16
        clipped	0.11980474859011421
        dev_split	0
        device	cuda
        epochs	2
        fp16	False
        learning_rate	1e-05
        lm_name	xlnet-large-cased
        lm_output_types	per_token
        lm_type	XLNet
        max_seq_len	256
        n_gpu	2
        n_samples_dev	13888
        n_samples_test	0
        n_samples_train	147707
        num_train_optimization_steps	18464
        prediction_heads	QuestionAnsweringHead
        processor	SquadProcessor
        tokenizer	XLNetTokenizer
        warmup_proportion	0.2
        
        Metrics
        Name	Value
        Dev_EM_question_answering	0.831
        Dev_f1_question_answering	0.868
        Dev_loss_question_answering	0.624
        Train_loss_total	1.045