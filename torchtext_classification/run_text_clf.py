import os
import logging
import argparse

import torch

from torchtext.datasets import text_classification

from text_clf_models import EmbeddingBagClfModel, ConvNNClassifier
from torch.utils.data.dataset import random_split

from train_util import train_and_valid, evaluate, TrainParams

if __name__ == "__main__":

    num_ngrams = 2
    num_epochs = 5
    embed_dim = 32
    split_ratio = 0.95

    data = ".data"
    if not os.path.exists(data):
        os.mkdir(data)

    dataset_Name = "AG_NEWS"
    assert dataset_Name in text_classification.DATASETS
    train_data_file = os.path.join(data, 'train_dataset.pt')
    test_data_file = os.path.join(data, 'test_dataset.pt')

    if not os.path.isfile(train_data_file):
        print('creating datasets')
        train_dataset, test_dataset = text_classification.DATASETS[dataset_Name](
            root=data, ngrams=num_ngrams
        )
        torch.save(train_dataset, train_data_file)
        torch.save(test_dataset, test_data_file)
    else:
        print('loading preprocessed datasets')
        train_dataset = torch.load(train_data_file)
        test_dataset = torch.load(test_data_file)

    vocab_size = len(train_dataset.get_vocab())
    num_class = len(train_dataset.get_labels())

    model = EmbeddingBagClfModel(vocab_size, embed_dim, num_class)
    # model = ConvNNClassifier(vocab_size, embed_dim, num_class,16)

    criterion = torch.nn.CrossEntropyLoss()

    train_len = int(len(train_dataset) * split_ratio)
    sub_train_, sub_valid_ = random_split(
        train_dataset, [train_len, len(train_dataset) - train_len]
    )
    params = TrainParams(4.0, 0.8)
    train_and_valid(model, criterion, sub_train_, sub_valid_, params)
    print("Test - Accuracy: {}".format(evaluate(model, test_dataset)))


'''
# EmbeddingBagClfModel
120000lines [00:09, 13095.38lines/s]
120000lines [00:16, 7197.67lines/s]
7600lines [00:00, 8408.97lines/s]
Progress:  20% lr: 4.000 loss: 0.236
Valid - Accuracy: 0.9003333333333333
Progress:  40% lr: 2.560 loss: 0.160
Valid - Accuracy: 0.9098333333333334
Progress:  60% lr: 2.048 loss: 0.090
Valid - Accuracy: 0.9081666666666667
Progress:  80% lr: 1.638 loss: 0.265
Valid - Accuracy: 0.9113333333333333
Progress: 100% lr: 1.311 loss: 0.017
Valid - Accuracy: 0.913
Test - Accuracy: 0.8743421052631579
... another run ... 
Valid - Accuracy: 0.9135
Test - Accuracy: 0.9019736842105263

'''