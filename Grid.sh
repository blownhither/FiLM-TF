#!/usr/bin/env bash
set -xe

date;
DATE_STR=$(date +"%Y%m%d%H%M%S");
MACHINE="zma17@pk-ssh.cs.brown.edu";
REMOTE_DIR="/home/zma17/projects/myFilm";

MAIN_SCRIPT="film.py"
MAIN_SCRIPT_HASH=$(md5 -q "$MAIN_SCRIPT");

scp *.py *.sh .comet.config "$MACHINE:$REMOTE_DIR"
ssh "$MACHINE" "cd $REMOTE_DIR; qsub -l gpus=1 -cwd -m abes job.sh $DATE_STR $MAIN_SCRIPT $MAIN_SCRIPT_HASH"
#ssh "$MACHINE" "cd $REMOTE_DIR; qsub -now y -l gpus=1 -cwd -m abes job.sh $DATE_STR $MAIN_SCRIPT $MAIN_SCRIPT_HASH"
ssh "$MACHINE" "cd $REMOTE_DIR; tail -f film-$DATE_STR.log"


