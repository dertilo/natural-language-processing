# -*- coding: utf-8 -*-
import pandas as pd

from test_tube import HyperOptArgumentParser
from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import collate_tensors
from typing import Dict, List


def collate_lists(text: list, label: list) -> List[Dict]:
    """ Converts each line into a dictionary. """
    collated_dataset = []
    for i in range(len(text)):
        collated_dataset.append({"text": str(text[i]), "label": str(label[i])})
    return collated_dataset


def sentiment_analysis_dataset(
    hparams: HyperOptArgumentParser, train=True, val=True, test=True
):
    """
    Loads the Dataset from the csv files passed to the parser.
    :param hparams: HyperOptArgumentParser obj containg the path to the data files.
    :param train: flag to return the train set.
    :param val: flag to return the validation set.
    :param test: flag to return the test set.

    Returns:
        - Training Dataset, Development Dataset, Testing Dataset
    """

    def load_dataset(path):
        df = pd.read_csv(path)
        text = list(df.text)
        label = list(df.sentiment)
        assert len(text) == len(label)
        return Dataset(collate_lists(text, label))

    func_out = []
    if train:
        func_out.append(load_dataset(hparams.train_csv))
    if val:
        func_out.append(load_dataset(hparams.dev_csv))
    if test:
        func_out.append(load_dataset(hparams.test_csv))

    return tuple(func_out)


def prepare_sample(sample: list,tokenizer, label_encoder,prepare_target: bool = True) -> (dict, dict):
    """
    Function that prepares a sample to input the model.
    :param sample: list of dictionaries.

    Returns:
        - dictionary with the expected model inputs.
        - dictionary with the expected target labels.
    """
    sample = collate_tensors(sample)
    tokens, lengths = tokenizer.batch_encode(sample["text"])

    inputs = {"tokens": tokens, "lengths": lengths}

    if not prepare_target:
        return inputs, {}

    # Prepare target:
    try:
        targets = {"labels": label_encoder.batch_encode(sample["label"])}
        return inputs, targets
    except RuntimeError:
        raise Exception("Label encoder found an unknown label.")