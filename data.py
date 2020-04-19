import json
import os
import tensorflow as tf


def read_paired_dataset(question_file, image_dir, tokenizer, read_question_family=False):
    def load_image(*args):
        nonlocal read_question_family
        fn, q = args[:2]
        image = tf.io.decode_png(tf.io.read_file(fn))
        if read_question_family:
            return image, q, args[2]
        else:
            return image, q

    pairs = json.load(open(question_file, 'r'))['questions']
    file_names = [os.path.join(image_dir, x['image_filename']) for x in pairs]
    question_text = [tokenizer.encode_sentence(x['question'].strip('?')) for x in pairs]
    question_text = tf.keras.preprocessing.sequence.pad_sequences(question_text)
    if read_question_family:
        question_family_index = [x['question_family_index'] for x in pairs]
        dataset = tf.data.Dataset.from_tensor_slices((file_names, question_text,
                                                      question_family_index))
        del question_family_index
    else:
        dataset = tf.data.Dataset.from_tensor_slices((file_names, question_text))
    del pairs, file_names, question_text
    dataset = dataset.shuffle(1024).map(load_image)



