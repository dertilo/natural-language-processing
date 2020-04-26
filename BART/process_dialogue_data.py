import os
import random

from tqdm import tqdm
from transformers import BartTokenizer
from typing import List
from util import data_io


def generate_coqa_seq2seq(file_name, SEP_TOKEN, hist_len=3):

    data = data_io.read_json(os.environ["HOME"] + "/data/QA/coqa/" + file_name)["data"]

    def get_history(l: List, k, hist_len):
        return [d["input_text"] for d in l[max(0, k - hist_len) : (k + 1)]]

    for datum in data:
        dialogue_len = len(datum["questions"])
        for k in range(dialogue_len):
            q_hist = get_history(datum["questions"], k, hist_len)
            a_hist = get_history(datum["answers"], k, hist_len)
            dialogue, target = build_input_target(
                datum["story"], q_hist, a_hist, SEP_TOKEN
            )
            yield dialogue, target


def build_input_target(background, q_hist, a_hist, SEP_TOKEN):
    def process(s):
        return s.replace("\n", "")

    turns = [process(x) for turn in zip(q_hist, a_hist) for x in turn]
    target = process(turns.pop(-1))
    dialogue = SEP_TOKEN.join([process(background)] + turns)
    return dialogue, target


if __name__ == "__main__":
    tokenizer = BartTokenizer.from_pretrained("bart-large")
    # BOS = tokenizer.special_tokens_map['bos_token']
    SEP = tokenizer.special_tokens_map["sep_token"]
    datagenerators = {
        "train": [generate_coqa_seq2seq("coqa-train-v1.0.json", SEP)],
        "val": [generate_coqa_seq2seq("coqa-dev-v1.0.json", SEP)],
    }
    data_path = os.environ["HOME"] + "/data/seq2seq_dialogue"
    os.makedirs(data_path, exist_ok=True)

    for ds, gs in datagenerators.items():
        with open(data_path + "/" + ds + ".source", mode="w") as s, open(
            data_path + "/" + ds + ".target", mode="w"
        ) as t:
            for x, y in tqdm(((x, y) for g in gs for x, y in g)):
                s.write(x + "\n")
                t.write(y + "\n")
