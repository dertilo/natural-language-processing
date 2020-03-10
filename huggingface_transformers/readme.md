### language model on wikitext
* [wikitext-long-term-dependency-language-modeling-dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)

    wget --trust-server-names https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
    unzip wikitext-2-raw-v1.zip
    export TRAIN_FILE=$HOME/data/wikitext-2-raw/wiki.train.raw
    export TEST_FILE=$HOME/data/wikitext-2-raw/wiki.test.raw
    
finetune
    
    python run_language_modeling.py     --output_dir=output     --model_type=gpt2     --model_name_or_path=gpt2     --do_train     --train_data_file=$TRAIN_FILE     --do_eval     --eval_data_file=$TEST_FILE --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2

from scratch

    python run_language_modeling.py     --output_dir=output     --model_type=gpt2     --do_train     --train_data_file=$TRAIN_FILE     --do_eval     --eval_data_file=$TEST_FILE --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2
    
    
needs `drop_last=True` in DataLoader

    ***** Eval results  *****
    03/10/2020 18:15:37 - INFO - __main__ -     perplexity = tensor(22.3560)
