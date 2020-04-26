import torch
from summarization.bart.finetune import SummarizationTrainer
from tqdm import tqdm
from transformers import BartTokenizer

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_interaction(
    model_name: str, device: str = DEFAULT_DEVICE
):

    assert model_name.endswith('.ckpt')
    model = SummarizationTrainer.load_from_checkpoint(model_name).model.to(device)

    tokenizer = BartTokenizer.from_pretrained("bart-large-cnn")

    max_length = 140
    min_length = 55

    while True:
        batch = [input(': ')]
        dct = tokenizer.batch_encode_plus(batch, max_length=1024, return_tensors="pt", pad_to_max_length=True)
        encoded = model.generate(
            input_ids=dct["input_ids"].to(device),
            attention_mask=dct["attention_mask"].to(device),
            num_beams=4,
            length_penalty=2.0,
            max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
            min_length=min_length + 1,  # +1 from original because we start at step=1
            no_repeat_ngram_size=3,
            early_stopping=True,
            decoder_start_token_id=model.config.eos_token_id,
        )[0]
        answer = tokenizer.decode(encoded, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
        print(answer)
        print()


if __name__ == '__main__':
    model_file = '/home/tilo/data/bart_coqa_seq2seq/checkpointepoch=0.ckpt'
    run_interaction(model_file)