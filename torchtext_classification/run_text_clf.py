import os
import logging
import argparse

import torch

from torchtext.datasets import text_classification

from text_clf_models import EmbeddingBagClfModel
from torch.utils.data.dataset import random_split

from train_util import train_and_valid, evaluate, TrainParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a text classification model on text classification datasets."
    )
    parser.add_argument(
        "--dataset", default="AG_NEWS", choices=text_classification.DATASETS
    )
    parser.add_argument("--ngrams", type=int, default=2, help="ngrams (default=2)")
    args = parser.parse_args()
    num_ngrams = 2

    num_epochs = 5
    embed_dim = 32
    data = ".data"
    split_ratio = 0.95

    if not os.path.exists(data):
        os.mkdir(data)

    train_dataset, test_dataset = text_classification.DATASETS[args.dataset](
        root=data, ngrams=num_ngrams
    )
    model = EmbeddingBagClfModel(
        len(train_dataset.get_vocab()), embed_dim, len(train_dataset.get_labels())
    )

    criterion = torch.nn.CrossEntropyLoss()

    train_len = int(len(train_dataset) * split_ratio)
    sub_train_, sub_valid_ = random_split(
        train_dataset, [train_len, len(train_dataset) - train_len]
    )
    params = TrainParams(4.0, 0.8)
    train_and_valid(model, criterion, sub_train_, sub_valid_, params)
    print("Test - Accuracy: {}".format(evaluate(model, test_dataset)))

    if args.save_model_path:
        print("Saving model to {}".format(args.save_model_path))
        torch.save(model.to("cpu"), args.save_model_path)

    if args.dictionary is not None:
        print("Save vocab to {}".format(args.dictionary))
        torch.save(train_dataset.get_vocab(), args.dictionary)
