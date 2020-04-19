import os
import json
import tensorflow as tf

from tokenizer import Tokenizer


def read_paired_dataset(question_file, image_dir, tokenizer, read_question_family=False,
                        epoch=80, batch_size=64):
    def load_image(*args):
        nonlocal read_question_family
        fn, q = args[:2]
        image = tf.io.decode_png(tf.io.read_file(fn))
        if read_question_family:
            return q, image, args[2]
        else:
            return q, image

    pairs = json.load(open(question_file, 'r'))['questions']
    file_names = [os.path.join(image_dir, x['image_filename']) for x in pairs]
    question_text = [tokenizer.encode_sentence(x['question'].strip('?')) for x in pairs]
    question_text = tf.keras.preprocessing.sequence.pad_sequences(question_text, padding='post',
                                                                  value=tokenizer.pad_id)
    if read_question_family:
        question_family_index = [x['question_family_index'] for x in pairs]
        dataset = tf.data.Dataset.from_tensor_slices((file_names, question_text,
                                                      question_family_index))
        del question_family_index
    else:
        dataset = tf.data.Dataset.from_tensor_slices((file_names, question_text))
    del pairs, file_names, question_text
    dataset = dataset.repeat(epoch).shuffle(1024).map(load_image,
                                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


def _test_read_paired_dataset():
    from matplotlib import pyplot as plt
    tokenizer = Tokenizer().load('tmp/tiny-tokenizer.pickle')
    dataset = read_paired_dataset(
        question_file='data/CLEVR_v1.0/questions/CLEVR_tiny_questions.json',
        image_dir='data/CLEVR_v1.0/images/test', tokenizer=tokenizer, batch_size=2)
    for x in dataset:
        text, image = x
        print(tokenizer.decode_sentence(text.numpy()[0]))
        plt.imshow(image.numpy()[0])
        plt.show()
        return


if __name__ == '__main__':
    _test_read_paired_dataset()
