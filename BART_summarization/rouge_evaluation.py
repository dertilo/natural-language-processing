import os
from pprint import pprint
import json
from rouge import Rouge

from util import data_io
#
# with open('./tests/data.json') as f:
#   data = json.load(f)
#
# hyps, refs = map(list, zip(*[[d['hyp'], d['ref']] for d in data]))
rouge = Rouge()
# scores = rouge.get_scores(hyps, refs)
hyps = list(data_io.read_lines(os.environ['HOME']+'/hpc/transformers/examples/summarization/bart/cnn_predicted_summaries.txt',limit=1000))
refs = list(data_io.read_lines(os.environ['HOME']+'/hpc/data/cnn_dm/test.target',limit=1000))
scores = rouge.get_scores(hyps, refs, avg=True)
pprint(scores)

'''
{'rouge-1': {'f': 0.2923597368088391,
             'p': 0.2430148556164662,
             'r': 0.38675644246961155},
 'rouge-2': {'f': 0.11613720277501925,
             'p': 0.09577817141610762,
             'r': 0.15578198219928469},
 'rouge-l': {'f': 0.28175826380865326,
             'p': 0.23867975605955946,
             'r': 0.35882717263780856}}
'''