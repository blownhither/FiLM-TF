#!/usr/bin/env bash
set -xe

date;
DATE_STR=$(date +"%Y%m%d%H%M");

MACHINE="martin@alchemist.smn.cs.brown.edu";
REMOTE_DIR="/home/martin/projects/myFilm";
REMOTE_PY="/home/martin/anaconda3/envs/tf2-conda/bin/python"

TRAIN_QUESTIONS="/home/martin/data/CLEVR_v1.0/questions/CLEVR_train_questions.json"
TRAIN_TOKENIZER="$REMOTE_DIR/tmp/CLEVR_train_tokenizer.pickle"
TRAIN_IMAGE_DIR="/home/martin/data/CLEVR_v1.0/images/train"
VAL_QUESTIONS="/home/martin/data/CLEVR_v1.0/questions/CLEVR_val_questions.json"
VAL_IMAGE_DIR="/home/martin/data/CLEVR_v1.0/images/val"

scp *.py .comet.config "$MACHINE:$REMOTE_DIR"
#scp *.py .comet.config "$MACHINE:$REMOTE_DIR"
#ssh "$MACHINE" "cd $REMOTE_DIR; $REMOTE_PY tokenizer.py --json_path=$TRAIN_QUESTIONS --save_to=$TRAIN_TOKENIZER 2>&1 | tee tok.log"
#ssh "$MACHINE" "cd $REMOTE_DIR; $REMOTE_PY film.py --train_questions=$TRAIN_QUESTIONS --tokenizer_path=$TRAIN_TOKENIZER --image_dir=$TRAIN_IMAGE_DIR 2>&1 | tee film.log"
ssh "$MACHINE" "screen -S myFilm -dm bash -c 'cd $REMOTE_DIR; $REMOTE_PY film.py --train_questions=$TRAIN_QUESTIONS --val_questions=$VAL_QUESTIONS --tokenizer_path=$TRAIN_TOKENIZER --train_image_dir=$TRAIN_IMAGE_DIR --val_image_dir=$VAL_IMAGE_DIR 2>&1 | tee film.log ' "




