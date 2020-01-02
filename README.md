# NLP-stuff that I'm working on
* [NLP-progress](https://github.com/sebastianruder/NLP-progress)

## transformer fine-tuning
* [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
* [huggingface](https://github.com/huggingface/transformers)
* [FARM](https://github.com/deepset-ai/FARM)

### germEval-2014 NER 
* [huggingface](https://github.com/huggingface/transformers/blob/7296f1010b6faaf3b1fb409bc5a9ebadcea51973/examples/run_ner.py#L28) vs. [FARM](https://github.com/deepset-ai/FARM/blob/7b60e4ea12c82185196bd6de9c33baeefe8bd75b/farm/metrics.py#L5) use [span-level-micro-averaged-f1](https://github.com/chakki-works/seqeval/blob/445d99798f6969f606dbf5978d936d5a8b33bbd1/seqeval/metrics/sequence_labeling.py#L116)
* [Deepset](https://deepset.ai/german-bert) reports 0.84
* could reproduce with FARM: Test_seq_f1_ner = 0.84 vs. huggingface: Test_seq_f1_ner =  0.86
* what are reasons for performance-gap?
    * differenced caused by random seed? dropout?
    
### GLUE
* [MNLI-kaggle](https://inclass.kaggle.com/c/multinli-matched-open-evaluation/data); [MNLI](https://www.nyu.edu/projects/bowman/multinli/)
### [SQUAD 2.0](https://rajpurkar.github.io/SQuAD-explorer)

* [huggingface](https://huggingface.co/transformers/examples.html#squad)

## Elasticsearch
* [bertsearch](https://github.com/Hironsan/bertsearch)

## discourse parsing
* [discopy](https://github.com/rknaebel/discopy)
* [shallow-discourse-parser](https://github.com/atreyasha/shallow-discourse-parser)

# TODO: 
* [fairseq](https://github.com/pytorch/fairseq)
* [mednli](https://github.com/jgc128/mednli)
* [Pytorch-OCR-Fully-Convolutional](https://github.com/lysecret2/Pytorch-OCR-Fully-Convolutional)
* [SmartDataAnalytics/OpenResearch](https://github.com/SmartDataAnalytics/OpenResearch)
* [QUD-comp](https://github.com/QUD-comp/QUD-comp)
* [openlaws.com](https://openlaws.com/home)
* [germeval-2019](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html)
* [QALD-9](http://2018.nliwod.org/challenge)
* [recon](https://github.com/DFKI-NLP/recon)

# NLP-tasks+datasets

## closed domain QA
* [haystack](https://github.com/deepset-ai/haystack) vs. [cdQA](https://github.com/cdqa-suite/cdQA#Evaluating-models) -> who got inspired by who?

### [Natural Questions](https://ai.google.com/research/NaturalQuestions/dataset) on [github](https://github.com/google-research-datasets/natural-questions)
* Natural Questions (NQ) contains real user questions issued to Google search; answers found from Wikipedia by annotators
* paper: `Natural Questions: a Benchmark for Question Answering Research` by `Tom Kwiatkowski`

### [SQUAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
* Extractive reading comprehension
* SQuAD 2.0 combines existing SQuAD data with over 50,000 unanswerable questions 
* system must learn to determine when no answer is supported by the paragraph and abstain from answering.
* we report average exact match and F1 scores; For negative examples, abstaining receives a score of 1, and any other response gets 0, for both exact match and F1

* [hugginface-people](https://github.com/huggingface/transformers/issues/947) did not yet reproduce SQUAD2.0 results?