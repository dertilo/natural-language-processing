

# GermEval 2014 
#### 0. 
    pip install seqeval
#### 1. 
    `bash preprocess_germeval_2014.sh`
#### 2.     
    export MAX_LENGTH=128
    export BERT_MODEL=bert-base-multilingual-cased
    export OUTPUT_DIR=germeval-model
    export NUM_EPOCHS=3
    export SAVE_STEPS=750
    export SEED=1
#### 3.  
    cp /docker-share/data/germEval_2014/labels.txt ./
#### 4. 
    python3 run_ner.py --data_dir /docker-share/data/germEval_2014 \
        --model_type bert \
        --labels ./labels.txt \
        --model_name_or_path $BERT_MODEL \
        --output_dir $OUTPUT_DIR \
        --max_seq_length  $MAX_LENGTH \
        --num_train_epochs $NUM_EPOCHS \
        --per_gpu_train_batch_size 8 \
        --save_steps $SAVE_STEPS \
        --seed $SEED \
        --do_train \
        --do_eval \
        --do_predict
        
#### results    
    3 epochs took about 1 hour; 1500 batches (2 in parallel) per epoch
        
    12/10/2019 14:40:49 - INFO - __main__ -   ***** Running evaluation  *****
    12/10/2019 14:40:49 - INFO - __main__ -     Num examples = 2200
    12/10/2019 14:40:49 - INFO - __main__ -     Batch size = 16
    Evaluating: 100%|█ 138/138 [00:54<00:00,  2.53it/s]
    12/10/2019 14:41:44 - INFO - __main__ -   ***** Eval results  *****
    12/10/2019 14:41:44 - INFO - __main__ -     f1 = 0.8623281393217233
    12/10/2019 14:41:44 - INFO - __main__ -     loss = 0.07727124436549249
    12/10/2019 14:41:44 - INFO - __main__ -     precision = 0.8457389428263214
    12/10/2019 14:41:44 - INFO - __main__ -     recall = 0.8795811518324608
    
    
    12/10/2019 14:41:55 - INFO - __main__ -   ***** Running evaluation  *****
    12/10/2019 14:41:55 - INFO - __main__ -     Num examples = 5100
    12/10/2019 14:41:55 - INFO - __main__ -     Batch size = 16
    Evaluating: 100%|█| 319/319 [02:06<00:00,  2.52it/s]
    12/10/2019 14:44:03 - INFO - __main__ -   ***** Eval results  *****
    12/10/2019 14:44:03 - INFO - __main__ -     f1 = 0.8600209525344509
    12/10/2019 14:44:03 - INFO - __main__ -     loss = 0.07628801183886008
    12/10/2019 14:44:03 - INFO - __main__ -     precision = 0.8563633445674852
    12/10/2019 14:44:03 - INFO - __main__ -     recall = 0.8637099384914212

