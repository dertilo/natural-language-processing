import os
import random

from transformers import BartTokenizer
from util import data_io

def generate_input_output(data,    hist_len = 3
):
    def process(s):
        return s.replace('\n','')

    for d in data:
        for k in range(0,len(d['questions'])):
            turns = [process(x['input_text']) for turn in zip(d['questions'][max(0,k-hist_len):(k+1)], d['answers'][max(0,k-hist_len):(k+1)])
                     for x in turn]
            target = process(turns.pop(-1))
            dialogue = SEP.join([process(d['story'])]+turns)
            yield dialogue,target

if __name__ == '__main__':
    tokenizer = BartTokenizer.from_pretrained("bart-large")
    # BOS = tokenizer.special_tokens_map['bos_token']
    SEP = tokenizer.special_tokens_map['sep_token']
    for file_name,ds in [('coqa-train-v1.0.json','train'),('coqa-dev-v1.0.json','val')]:
        data = data_io.read_json(os.environ['HOME'] + '/data/QA/coqa/' + file_name)['data']
        dialogues = list(generate_input_output(data))
        random.shuffle(dialogues)
        folder = 'coqa_seq2eq'
        os.makedirs(folder,exist_ok=True)
        data_io.write_lines(folder + '/' + ds + '.source', (x for x, _ in dialogues))
        data_io.write_lines(folder + '/' + ds + '.target',(x for _,x in dialogues))
