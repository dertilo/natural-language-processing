import os

from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer as Tokenizer
from transformers import GPT2Tokenizer

if __name__ == '__main__':
    # # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # HOME = os.environ['HOME']
    # data_file = HOME + '/data/wikitext-2-raw/wiki.train.raw'
    data_file ='/tmp/wikitext-2-raw/wiki.train.raw'
    tokenizer.train(files=[data_file], vocab_size=20_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    tokenizer_name = "Tokenizer"
    os.makedirs(tokenizer_name)
    tokenizer.save(tokenizer_name)

    with open(data_file, encoding="utf-8") as f:
        text = f.read()

    tok = GPT2Tokenizer.from_pretrained('Tokenizer')
    x = tok.convert_tokens_to_ids(tok.tokenize(text[:100]))
    y = tok.build_inputs_with_special_tokens(x)
    print(x)
    print(y)