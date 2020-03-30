"""
 MNIST example with training and validation monitoring using TensorboardX and Tensorboard.

 Requirements:
    Optionally TensorboardX (https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`
    Tensorboard: `pip install tensorflow` (or just install tensorboard without the rest of tensorflow)

 Usage:

    Start tensorboard:
    ```bash
    tensorboard --logdir=/tmp/tensorboard_logs/
    ```

    Run the example:
    ```bash
    python mnist_with_tensorboard_logger.py --log_dir=/tmp/tensorboard_logs
    ```
"""
import os
import re
import sys
import logging
from typing import NamedTuple, List

import torch
from ignite.contrib.handlers import ProgressBar
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torchtext.data import Field, Example, Dataset
from torchtext.datasets import text_classification, TextClassificationDataset

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers.tensorboard_logger import *
from torch.utils.data.dataset import random_split
from tqdm import tqdm
from util import data_io

from text_clf_models import EmbeddingBagClfModel

LOG_INTERVAL = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainParams(NamedTuple):
    lr: float
    lr_gamma: float
    batch_size: int = 16
    eval_batch_size: int = 32
    num_workers: int = 0
    num_epochs: int = 5
    momentum: float = 0.5
    log_dir: str = "tensorboard_logs"


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

    return train_dataset, test_dataset


class TextClfDataset(Dataset):
    def get_vocab(self):
        return self.fields["text"].vocab

    def get_labels(self):
        return set(self.fields["label"].vocab.freqs.keys())

    def __getitem__(self, i):
        raw_datum: Example = super().__getitem__(i)
        text = self.fields["text"].numericalize([raw_datum.text]).squeeze()
        label = int(self.fields["label"].numericalize([raw_datum.label]))
        return label, text


def build_text_clf_datasets(
    train_csv=".data/ag_news_csv/train.csv", test_csv=".data/ag_news_csv/test.csv"
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
    label_field = Field(batch_first=True, sequential=False,unk_token=None)
    fields = [("text", text_field), ("label", label_field)]

    text_label_data = (parse_line(l) for l in tqdm(data_io.read_lines(train_csv)))
    examples = [
        Example.fromlist([text, label], fields) for text, label in text_label_data
    ]

    text_field.build_vocab([example.text for example in examples])
    label_field.build_vocab([example.label for example in examples])

    train_dataset = TextClfDataset(examples, fields)

    text_label_data = (parse_line(l) for l in tqdm(data_io.read_lines(test_csv)))
    examples = [
        Example.fromlist([text, label], fields) for text, label in text_label_data
    ]
    test_dataset = TextClfDataset(examples, fields)
    return train_dataset, test_dataset


def generate_seqconcat_batch(raw_batch: List):  # TODO(tilo): for EmbeddingBag only!
    label = torch.tensor([entry[0] for entry in raw_batch])
    text = [entry[1] for entry in raw_batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return (text, offsets), label


def get_data_loaders(train_dataset, params: TrainParams):

    train_len = int(len(train_dataset) * 0.95)
    sub_train_, sub_valid_ = random_split(
        train_dataset, [train_len, len(train_dataset) - train_len]
    )

    train_loader = DataLoader(
        sub_train_,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=generate_seqconcat_batch,
        num_workers=params.num_workers,
    )

    val_loader = DataLoader(
        sub_valid_,
        batch_size=params.eval_batch_size,
        collate_fn=generate_seqconcat_batch,
    )

    return train_loader, val_loader


def gpuinfo_metrics(trainer):
    if sys.version_info > (3,):
        from ignite.contrib.metrics.gpu_info import GpuInfo

        try:
            GpuInfo().attach(trainer)
        except RuntimeError:
            print(
                "INFO: By default, in this example it is possible to log GPU information (used memory, utilization). "
                "As there is no pynvml python package installed, GPU information won't be logged. Otherwise, please "
                "install it : `pip install pynvml`"
            )


def run(params: TrainParams):
    # train_dataset, test_dataset = get_datasets("AG_NEWS", ".data", 2)
    train_dataset, test_dataset = build_text_clf_datasets()
    vocab_size = len(train_dataset.get_vocab())
    num_class = len(train_dataset.get_labels())

    train_loader, val_loader = get_data_loaders(train_dataset, params)
    model = EmbeddingBagClfModel(vocab_size, 32, num_class)

    optimizer = Adam(model.parameters(), lr=params.lr)
    criterion = nn.CrossEntropyLoss()
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    gpuinfo_metrics(trainer)

    metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    validation_evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        train_evaluator.run(train_loader)
        validation_evaluator.run(val_loader)

    RunningAverage(output_transform=lambda x: x, alpha=0.99).attach(trainer, "loss")
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=["loss"])

    RunningAverage(src=metrics["accuracy"], alpha=0.99).attach(
        validation_evaluator, "running-acc"
    )
    pbar_eval = ProgressBar(persist=True)
    pbar_eval.attach(validation_evaluator, metric_names=["running-acc"])

    tb_logger = TensorboardLogger(log_dir=params.log_dir)

    tb_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="training",
            output_transform=lambda loss: {"batchloss": loss},
            metric_names="all",
        ),
        event_name=Events.ITERATION_COMPLETED(every=100),
    )

    tb_logger.attach(
        train_evaluator,
        log_handler=OutputHandler(
            tag="training", metric_names=["loss", "accuracy"], another_engine=trainer
        ),
        event_name=Events.EPOCH_COMPLETED,
    )

    tb_logger.attach(
        validation_evaluator,
        log_handler=OutputHandler(
            tag="validation", metric_names=["loss", "accuracy"], another_engine=trainer
        ),
        event_name=Events.EPOCH_COMPLETED,
    )

    tb_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer),
        event_name=Events.ITERATION_COMPLETED(every=100),
    )

    tb_logger.attach(
        trainer,
        log_handler=WeightsScalarHandler(model),
        event_name=Events.ITERATION_COMPLETED(every=100),
    )

    tb_logger.attach(
        trainer,
        log_handler=WeightsHistHandler(model),
        event_name=Events.EPOCH_COMPLETED(every=100),
    )

    tb_logger.attach(
        trainer,
        log_handler=GradsScalarHandler(model),
        event_name=Events.ITERATION_COMPLETED(every=100),
    )

    tb_logger.attach(
        trainer,
        log_handler=GradsHistHandler(model),
        event_name=Events.EPOCH_COMPLETED(every=100),
    )

    # kick everything off
    trainer.run(train_loader, max_epochs=params.num_epochs)
    tb_logger.close()


if __name__ == "__main__":
    logger = logging.getLogger("ignite.engine.engine.Engine")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    run(TrainParams(0.01, 0.8, num_epochs=1))
