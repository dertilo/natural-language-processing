# BERT Text Classifier
* based on https://github.com/minimalist-nlp/lightning-text-classification

used libraries:
- [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [Transformers](https://huggingface.co/transformers/index.html)
- [PyTorch-NLP](https://pytorchnlp.readthedocs.io/en/latest/index.html)


### Train:
```bash
python training.py     --gpus 2     --batch_size 8     --loader_workers 4
```
after 1hour on two 16GB-GPUs reached ~94% accuracy on `imdb_reviews_test.csv`

using these params
```bash
    cat meta_tags.csv
    
    key,value
    seed,3
    save_top_k,1
    monitor,val_acc
    metric_mode,max
    patience,3
    min_epochs,1
    max_epochs,10
    batch_size,8
    accumulate_grad_batches,2
    gpus,2
    val_percent_check,1.0
    encoder_model,bert-base-uncased
    encoder_learning_rate,1e-05
    learning_rate,3e-05
    nr_frozen_epochs,1
    label_set,"pos,neg"
    train_csv,data/imdb_reviews_train.csv
    dev_csv,data/imdb_reviews_test.csv
    loader_workers,4
    hpc_exp_number,None
```
