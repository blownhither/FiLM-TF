#!/usr/bin/env bash
set -xe

date;
DATE_STR=$1;
MAIN_SCRIPT=$2;
MAIN_SCRIPT_HASH=$3;
actual_hash=$(md5sum "$MAIN_SCRIPT" | cut -f1 -d" ");
if [[ "$actual_hash" -ne "$MAIN_SCRIPT_HASH" ]]; then
    echo "Hash mismtach. Expect $MAIN_SCRIPT_HASH but got $actual_hash";
    exit 1;
else
    echo "Hash check matches $MAIN_SCRIPT_HASH"
fi

REMOTE_DIR="/home/zma17/projects/myFilm";
REMOTE_ENV="/home/zma17/py37-tf2"
TRAIN_QUESTIONS="/nbu/people/zma17/CLEVR_v1.0/questions/CLEVR_train_questions.json"
TRAIN_TOKENIZER="$REMOTE_DIR/tmp/CLEVR_train_tokenizer.pickle"
TRAIN_IMAGE_DIR="/nbu/people/zma17/CLEVR_v1.0/images/train"
VAL_QUESTIONS="/nbu/people/zma17/CLEVR_v1.0/questions/CLEVR_val_questions.json"
VAL_IMAGE_DIR="/nbu/people/zma17/CLEVR_v1.0/images/val"


source "$REMOTE_ENV"/bin/activate;
export LD_LIBRARY_PATH=/local/projects/cuda10/cuda-10.1/lib64:/local/projects/cuda10/cuda-10.1/extras/CUPTI;

python film.py --train_questions="$TRAIN_QUESTIONS" \
    --val_questions="$VAL_QUESTIONS" \
    --tokenizer_path="$TRAIN_TOKENIZER" \
    --train_image_dir="$TRAIN_IMAGE_DIR" \
    --val_image_dir="$VAL_IMAGE_DIR" 2>&1 | tee rawT.log "film-$DATE_STR.log";
