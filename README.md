# transformer finetuning
## setup
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
  --max_steps 30 \
  --seed 1


      
#### MRPC-dataset

    11/29/2019 16:02:46 - INFO - __main__ -   ***** Eval results  *****
    11/29/2019 16:02:46 - INFO - __main__ -     acc = 0.8627450980392157
    11/29/2019 16:02:46 - INFO - __main__ -     acc_and_f1 = 0.8825920612147299
    11/29/2019 16:02:46 - INFO - __main__ -     f1 = 0.902439024390244
      
  
