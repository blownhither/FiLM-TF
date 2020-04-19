import json
import random
import tensorflow as tf


def read_paired_dataset(question_file, image_dir, read_question_family=False):
    questions = json.load(open(question_file, 'r'))['questions']
    if read_question_family:
        questions = [(questions['image_filename'], questions['question']) for x in questions]
    else:
        questions = [(questions['image_filename'], questions['question'],
                      questions['question_family_index']) for x in questions]
    