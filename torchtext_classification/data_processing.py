import abc
import os
import re
from typing import List

import torch
from torchtext.data import Dataset, Example, Field
from torchtext.data.utils import ngrams_iterator
from torchtext.datasets import text_classification
from tqdm import tqdm
from util import data_io


def build_seqconcat_batch(raw_batch: List):  # TODO(tilo): for EmbeddingBag only!
    label = torch.tensor([entry[0] for entry in raw_batch])
    text = [entry[1] for entry in raw_batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return (text, offsets), label


def get_datasets(dataset_Name, data, num_ngrams):
    assert dataset_Name in text_classification.DATASETS
    train_data_file = os.path.join(data, "train_dataset.pt")
    test_data_file = os.path.join(data, "test_dataset.pt")
    if not os.path.isfile(train_data_file):
        print("creating datasets")
        train_dataset, test_dataset = text_classification.DATASETS[dataset_Name](
            root=data, ngrams=num_ngrams
        )
        torch.save(train_dataset, train_data_file)
        torch.save(test_dataset, test_data_file)
    else:
        print("loading preprocessed datasets")
        train_dataset = torch.load(train_data_file)
        test_dataset = torch.load(test_data_file)
    train_dataset.collate_fn = build_seqconcat_batch
    test_dataset.collate_fn = build_seqconcat_batch
    return train_dataset, test_dataset


class TextClfDataset(Dataset):
    def get_vocab(self):
        return self.fields["text"].vocab

    def get_labels(self):
        return set(self.fields["label"].vocab.freqs.keys())

    @abc.abstractmethod
    def collate_fn(self, raw_batch):
        raise NotImplementedError


class NgramsDataset(TextClfDataset):
    def __init__(self, examples, fields, filter_pred=None, ngrams=1):
        super().__init__(examples, fields, filter_pred)
        self.ngrams = ngrams

    def __getitem__(self, i):
        raw_datum: Example = super().__getitem__(i)
        tokens = raw_datum.text
        ngrams = list(ngrams_iterator(tokens, self.ngrams))
        text = self.fields["text"].numericalize([ngrams]).squeeze()
        label = int(self.fields["label"].numericalize([raw_datum.label]))
        return label, text

    def collate_fn(self, raw_batch: List):
        return build_seqconcat_batch(raw_batch)


def parse_csv_to_examples_build_fields(
    train_csv=".data/ag_news_csv/train.csv", test_csv=".data/ag_news_csv/test.csv",
):
    def regex_tokenizer(
        text, pattern=r"(?u)\b\w\w+\b"
    ) -> List[str]:  # pattern stolen from scikit-learn
        return [m.group() for m in re.finditer(pattern, text)]

    def parse_line(line):
        splits = line.split(",")
        text = " ".join(splits[1:])
        label = splits[0]
        return text, label

    text_field = Field(
        include_lengths=False, batch_first=True, tokenize=regex_tokenizer
    )
    label_field = Field(batch_first=True, sequential=False, unk_token=None)
    fields = [("text", text_field), ("label", label_field)]

    g = (parse_line(l) for l in tqdm(data_io.read_lines(train_csv)))
    train_examples = [Example.fromlist([text, label], fields) for text, label in g]

    text_field.build_vocab([example.text for example in train_examples])
    label_field.build_vocab([example.label for example in train_examples])

    g = (parse_line(l) for l in tqdm(data_io.read_lines(test_csv)))
    test_examples = [Example.fromlist([text, label], fields) for text, label in g]
    return train_examples, test_examples, fields


class PaddedTextClfDataset(TextClfDataset):
    def collate_fn(self, raw_batch: List):
        batch = self.fields["text"].process([e.text for e in raw_batch])
        batch_targets = self.fields["label"].process([e.label for e in raw_batch])
        return batch, batch_targets
