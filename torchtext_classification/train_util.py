import sys

import torch
from torch.utils.data import DataLoader

from train import model, args, batch_size, num_epochs, device, criterion


def generate_batch(batch):
    r"""
    Since the text entries have different lengths, a custom function
    generate_batch() is used to generate data batches and offsets,
    which are compatible with EmbeddingBag. The function is passed
    to 'collate_fn' in torch.utils.data.DataLoader. The input to
    'collate_fn' is a list of tensors with the size of batch_size,
    and the 'collate_fn' function packs them into a mini-batch.
    Pay attention here and make sure that 'collate_fn' is declared
    as a top level def. This ensures that the function is available
    in each worker.

    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        offsets: the offsets is a tensor of delimiters to represent the beginning
            index of the individual sequence in the text tensor.
        cls: a tensor saving the labels of individual text entries.
    """
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def train_and_valid(lr_, sub_train_, sub_valid_):
    r"""
    We use a SGD optimizer to train the model here and the learning rate
    decreases linearly with the progress of the training process.

    Arguments:
        lr_: learning rate
        sub_train_: the data used to train the model
        sub_valid_: the data used for validation
    """

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.lr_gamma)
    train_data = DataLoader(
        sub_train_,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=generate_batch,
        num_workers=args.num_workers,
    )
    num_lines = num_epochs * len(train_data)

    for epoch in range(num_epochs):

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
        print("Valid - Accuracy: {}".format(evaluate(sub_valid_)))


def evaluate(data_):
    r"""
    Arguments:
        data_: the data used to train the model
    """
    data = DataLoader(data_, batch_size=batch_size, collate_fn=generate_batch)
    total_accuracy = []
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            accuracy = (output.argmax(1) == cls).float().mean().item()
            total_accuracy.append(accuracy)

    # In case that nothing in the dataset
    if total_accuracy == []:
        return 0.0

    return sum(total_accuracy) / len(total_accuracy)
