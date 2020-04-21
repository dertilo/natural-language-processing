import pandas as pd
from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import collate_tensors


def load_dataset(path):
    df = pd.read_csv(path)
    text = list(df.text)
    label = list(df.sentiment)
    assert len(text) == len(label)
    return Dataset([{"text": str(t), "label": str(l)} for t, l in zip(text, label)])


def prepare_sample(
    sample: list, tokenizer, label_encoder, prepare_target: bool = True
) -> (dict, dict):
    sample = collate_tensors(sample)
    tokens, lengths = tokenizer.batch_encode(sample["text"])

    inputs = {"tokens": tokens, "lengths": lengths}

    if not prepare_target:
        targets = {}
    else:
        targets = {"labels": label_encoder.batch_encode(sample["label"])}
    return inputs, targets
