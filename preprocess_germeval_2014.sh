
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-train.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > train.txt.tmp
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-dev.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > dev.txt.tmp
curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-test.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > test.txt.tmp

wget "https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py"

export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased
mkdir germEval_2014

python3 preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > germEval_2014/train.txt
python3 preprocess.py dev.txt.tmp $BERT_MODEL $MAX_LENGTH > germEval_2014/dev.txt
python3 preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH > germEval_2014/test.txt

cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > germEval_2014/labels.txt

rm test.txt.tmp train.txt.tmp dev.txt.tmp