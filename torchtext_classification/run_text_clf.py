import os
import logging
import argparse

import torch

from torchtext.datasets import text_classification

from text_clf_models import TextSentiment
from torch.utils.data.dataset import random_split

from train_util import train_and_valid, evaluate

r"""
This file shows the training process of the text classification model.
"""

r"""
torch.utils.data.DataLoader is recommended for PyTorch users to load data.
We use DataLoader here to load datasets and send it to the train_and_valid()
and text() functions.

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a text classification model on text classification datasets."
    )
    parser.add_argument(
        "--dataset", default="AG_NEWS", choices=text_classification.DATASETS
    )
    parser.add_argument(
        "--num-epochs", type=int, default=5, help="num epochs (default=5)"
    )
    parser.add_argument(
        "--embed-dim", type=int, default=32, help="embed dim. (default=32)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="batch size (default=16)"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.95,
        help="train/valid split ratio (default=0.95)",
    )
    parser.add_argument(
        "--lr", type=float, default=4.0, help="learning rate (default=4.0)"
    )
    parser.add_argument(
        "--lr-gamma", type=float, default=0.8, help="gamma value for lr (default=0.8)"
    )
    parser.add_argument("--ngrams", type=int, default=2, help="ngrams (default=2)")
    parser.add_argument(
        "--num-workers", type=int, default=1, help="num of workers (default=1)"
    )
    parser.add_argument("--device", default="cpu", help="device (default=cpu)")
    parser.add_argument(
        "--data", default=".data", help="data directory (default=.data)"
    )
    parser.add_argument(
        "--use-sp-tokenizer",
        type=bool,
        default=False,
        help="use sentencepiece tokenizer (default=False)",
    )
    parser.add_argument(
        "--sp-vocab-size",
        type=int,
        default=20000,
        help="vocab size in sentencepiece model (default=20000)",
    )
    parser.add_argument("--dictionary", help="path to save vocab")
    parser.add_argument("--save-model-path", help="path for saving model")
    parser.add_argument(
        "--logging-level", default="WARNING", help="logging level (default=WARNING)"
    )
    args = parser.parse_args()

    num_epochs = args.num_epochs
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    lr = args.lr
    device = args.device
    data = args.data
    split_ratio = args.split_ratio
    # two args for sentencepiece tokenizer
    use_sp_tokenizer = args.use_sp_tokenizer
    sp_vocab_size = args.sp_vocab_size

    logging.basicConfig(level=getattr(logging, args.logging_level))

    if not os.path.exists(data):
        print("Creating directory {}".format(data))
        os.mkdir(data)

    if use_sp_tokenizer:
        import spm_dataset

        train_dataset, test_dataset = spm_dataset.setup_datasets(
            args.dataset, root=".data", vocab_size=sp_vocab_size
        )
        model = TextSentiment(
            sp_vocab_size, embed_dim, len(train_dataset.get_labels())
        ).to(device)

    else:
        train_dataset, test_dataset = text_classification.DATASETS[args.dataset](
            root=data, ngrams=args.ngrams
        )
        model = TextSentiment(
            len(train_dataset.get_vocab()), embed_dim, len(train_dataset.get_labels())
        ).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    # split train_dataset into train and valid
    train_len = int(len(train_dataset) * split_ratio)
    sub_train_, sub_valid_ = random_split(
        train_dataset, [train_len, len(train_dataset) - train_len]
    )
    train_and_valid(lr, sub_train_, sub_valid_)
    print("Test - Accuracy: {}".format(evaluate(test_dataset)))

    if args.save_model_path:
        print("Saving model to {}".format(args.save_model_path))
        torch.save(model.to("cpu"), args.save_model_path)

    if args.dictionary is not None:
        print("Save vocab to {}".format(args.dictionary))
        torch.save(train_dataset.get_vocab(), args.dictionary)
