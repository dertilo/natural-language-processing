# [huggingface transformers glue finetuning](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py)


## setup
    rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress --exclude=.git --max-size=1m /home/tilo/code/NLP/NLU/Transformers/transformers gunther@guntherhamachi:/home/gunther/tilo_data/
on gunther I had to manually `pip uninstall transformers` to force python to use the source code 

### apex
`git clone https://github.com/NVIDIA/apex`
`cd apex && pip install -v --no-cache-dir .`
### glue
    git clone https://gist.github.com/60c2bdb54d156a41194446737ce03e2e.git glue_downloading
    python glue_downloading/download_glue_data.py --data_dir /docker-share/data/glue --tasks all
    python glue_downloading/download_glue_data.py --data_dir data/glue --tasks all

### finetune on glue-task

    export GLUE_DIR=../data/glue
    export TASK_NAME=MNLI

    python examples/run_glue.py \
      --model_type bert \
      --model_name_or_path bert-base-uncased \
      --task_name $TASK_NAME \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir $GLUE_DIR/$TASK_NAME \
      --max_seq_length 128 \
      --per_gpu_train_batch_size 8 \
      --learning_rate 2e-5 \
      --num_train_epochs 3.0 \
      --output_dir checkpoints/$TASK_NAME/ \
      --save_steps 1000 \
      --overwrite_output_dir \
      --fp16 \
      --max_steps 2 \
      --seed 1

      
#### MRPC-dataset

    11/29/2019 16:02:46 - INFO - __main__ -   ***** Eval results  *****
    11/29/2019 16:02:46 - INFO - __main__ -     acc = 0.8627450980392157
    11/29/2019 16:02:46 - INFO - __main__ -     acc_and_f1 = 0.8825920612147299
    11/29/2019 16:02:46 - INFO - __main__ -     f1 = 0.902439024390244
      
#### [MNLI](https://huggingface.co/transformers/examples.html#mnli)
* one epoch


    12/04/2019 12:04:45 - INFO - __main__ -   ***** Eval results  *****
    12/04/2019 12:04:45 - INFO - __main__ -     acc = 0.8351502801833928
    12/04/2019 12:04:45 - INFO - __main__ -   Loading features from cached file ../data/glue/MNLI/cached_dev_bert-base-uncased_128_mnli-mm
    12/04/2019 12:04:46 - INFO - __main__ -   ***** Running evaluation  *****
    12/04/2019 12:04:46 - INFO - __main__ -     Num examples = 9832
    12/04/2019 12:04:46 - INFO - __main__ -     Batch size = 16
    
    12/04/2019 12:05:15 - INFO - __main__ -   ***** Eval results  *****
    12/04/2019 12:05:15 - INFO - __main__ -     acc = 0.8425549227013832
