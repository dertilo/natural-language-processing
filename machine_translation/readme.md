# fairseq
[see](https://github.com/pytorch/fairseq/tree/master/examples/translation)
[also see](https://modelzoo.co/model/fairseq-py)

    
    fairseq-train \
    data-bin/wmt17_en_de \
    --arch fconv_wmt_en_de \
    -s en -t de \
    --tensorboard-logdir tensorboard_logdir/wmt_en_de \
    --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler fixed --force-anneal 50 \
    --save-dir checkpoints/fconv_wmt_en_de
    
    fairseq-generate data-bin/wmt17_en_de \
    --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt \
    --beam 5 --remove-bpe
    
    2020-03-09 23:29:55 | INFO | fairseq_cli.generate | Generate test with beam=5: BLEU4 = 18.06, 50.4/23.4/12.6/7.1 (BP=1.000, ratio=1.031, syslen=66478, reflen=64506)

    
### portuguese

    TEXT=examples/translation/pt_en
    fairseq-preprocess \
    --source-lang pt --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid  \
    --destdir data-bin/pt_en --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

pt-en training

    fairseq-train \
    data-bin/pt_en \
    --arch fconv_wmt_en_de \
    -s pt -t en \
    --tensorboard-logdir tensorboard_logdir/pt_en \
    --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler fixed --force-anneal 50 \
    --save-dir checkpoints/pt_en

en-pt training

    fairseq-train \
    data-bin/pt_en \
    --arch fconv_wmt_en_de \
    -s en -t pt \
    --tensorboard-logdir tensorboard_logdir/en_pt \
    --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler fixed --force-anneal 50 \
    --save-dir checkpoints/en_pt
    
### deployment
    rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress /home/tilo/data/models/MT gunther@gunther:/home/gunther/tilo_data/

used standard ml-container
pip install fairseq==0.9.0