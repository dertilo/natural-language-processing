import sys
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


class TrainParams(NamedTuple):
    lr: float
    lr_gamma: float
    batch_size: int = 16
    num_workers: int = 2
    num_epochs: int = 5


def train_and_valid(model, criterion, sub_train_, sub_valid_, params: TrainParams):
    model.train()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=params.lr_gamma)
    train_data = DataLoader(
        sub_train_,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=generate_batch,
        num_workers=params.num_workers,
    )
    num_lines = params.num_epochs * len(train_data)

    for epoch in range(params.num_epochs):

        # Train the model
        for i, (text, offsets, cls) in enumerate(train_data):
            optimizer.zero_grad()
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss.backward()
            optimizer.step()
            processed_lines = i + len(train_data) * epoch
            progress = processed_lines / float(num_lines)
            if processed_lines % 128 == 0:
                sys.stderr.write(
                    "\rProgress: {:3.0f}% lr: {:3.3f} loss: {:3.3f}".format(
                        progress * 100, scheduler.get_lr()[0], loss
                    )
                )
        # Adjust the learning rate
        scheduler.step()

        # Test the model on valid set
        print("")
        print("Valid - Accuracy: {}".format(evaluate(model, sub_valid_)))


def evaluate(model, data_, batch_size=16):
    data = DataLoader(data_, batch_size=batch_size, collate_fn=generate_batch)
    total_accuracy = []
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            accuracy = (output.argmax(1) == cls).float().mean().item()
            total_accuracy.append(accuracy)

    if total_accuracy == []:
        return 0.0

    return sum(total_accuracy) / len(total_accuracy)
