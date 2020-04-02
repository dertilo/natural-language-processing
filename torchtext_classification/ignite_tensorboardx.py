import os
import re
import shutil
import sys
import logging
from typing import NamedTuple, List

import torch
from ignite.contrib.handlers import ProgressBar
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch import nn

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers.tensorboard_logger import *
from torch.utils.data.dataset import random_split

from data_processing import NgramsDataset, parse_csv_to_examples_build_fields, \
    PaddedTextClfDataset
from text_clf_models import EmbeddingBagClfModel, ConvNNClassifier

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


def get_data_loaders(collate_fn, train_dataset, params: TrainParams):

    train_len = int(len(train_dataset) * 0.95)
    sub_train_, sub_valid_ = random_split(
        train_dataset, [train_len, len(train_dataset) - train_len]
    )

    train_loader = DataLoader(
        sub_train_,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=params.num_workers,
    )

    val_loader = DataLoader(
        sub_valid_, batch_size=params.eval_batch_size, collate_fn=collate_fn,
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


def setup_tensorboard(
    model,
    optimizer,
    params: TrainParams,
    train_evaluator,
    trainer,
    validation_evaluator,
):
    log_dir = params.log_dir
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    tb_logger = TensorboardLogger(log_dir=log_dir)
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
        event_name=Events.EPOCH_COMPLETED(every=1),
    )
    tb_logger.attach(
        trainer,
        log_handler=GradsScalarHandler(model),
        event_name=Events.ITERATION_COMPLETED(every=100),
    )
    tb_logger.attach(
        trainer,
        log_handler=GradsHistHandler(model),
        event_name=Events.EPOCH_COMPLETED(every=1),
    )
    return tb_logger


def run(params: TrainParams):
    # train_dataset, test_dataset = get_datasets("AG_NEWS", ".data", 2)
    train_examples, test_examples, fields = parse_csv_to_examples_build_fields()
    # ngrams = 1
    # train_dataset = NgramsDataset(train_examples, fields, ngrams=ngrams)
    # test_dataset = NgramsDataset(test_examples, fields, ngrams=ngrams)
    train_dataset = PaddedTextClfDataset(train_examples, fields)
    test_dataset = PaddedTextClfDataset(test_examples, fields)

    vocab_size = len(train_dataset.get_vocab())
    num_class = len(train_dataset.get_labels())

    train_loader, val_loader = get_data_loaders(
        train_dataset.collate_fn, train_dataset, params
    )
    # model = EmbeddingBagClfModel(vocab_size, 32, num_class)
    model = ConvNNClassifier(vocab_size, 32, num_class,32)
    print(model)

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

    tb_logger = setup_tensorboard(
        model, optimizer, params, train_evaluator, trainer, validation_evaluator
    )

    trainer.run(train_loader, max_epochs=params.num_epochs)
    tb_logger.close()


if __name__ == "__main__":
    logger = logging.getLogger("ignite.engine.engine.Engine")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    run(TrainParams(0.01, 0.8, num_epochs=2,log_dir='tensorboard_logs/convNN'))
