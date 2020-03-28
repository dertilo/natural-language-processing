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
    train_dataset, test_dataset = text_classification.DATASETS[dataset_Name](
        root=data, ngrams=num_ngrams
    )
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
