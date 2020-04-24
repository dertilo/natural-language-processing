
# BART 
* [fairseq-BART](https://github.com/pytorch/fairseq/tree/master/examples/bart)

## huggingface 
(hugginface implementation see: transformers/examples/summarization/bart)

`rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --delete --exclude=.git --max-size=1m /home/tilo/code/NLP/NLU/Transformers/transformers tilo-himmelsbach@gateway.hpc.tu-berlin.de:/home/users/t/tilo-himmelsbach/`

1. download data `wget https://s3.amazonaws.com/datasets.huggingface.co/summarization/cnn_dm.tgz && tar -xzvf cnn_dm.tgz`
    
### inference

1. `export PYTHONPATH=/beegfs/home/users/t/tilo-himmelsbach/transformers:/beegfs/home/users/t/tilo-himmelsbach/transformers/examples`
2. on front-end to download model: `OMP_NUM_THREADS=8 python evaluate_cnn.py ~/data/cnn_dm/test.source cnn_predicted_summaries_finetuned.txt ~/data/BART_summarization/bart_sum/checkpointepoch\=0.ckpt --bs 16`
* from fine-tuned checkpoint: `python evaluate_cnn.py ~/data/cnn_dm/test.source cnn_predicted_summaries_finetuned.txt ~/data/BART_summarization/bart_sum/checkpointepoch\=0.ckpt --bs 16`
* from pretrained model :`python evaluate_cnn.py ~/data/cnn_dm/test.source cnn_predicted_summaries.txt bart-large-cnn`

### evaluation
[rouge-calculation](https://github.com/pltrdy/rouge)