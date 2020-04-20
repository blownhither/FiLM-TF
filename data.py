import os
import json
import tensorflow as tf

from tokenizer import Tokenizer, LabelEncoder


def read_paired_dataset(question_file, image_dir, tokenizer: Tokenizer, label_encoder: LabelEncoder,
                        read_question_family=False, read_label=False, epoch=80, batch_size=64,
                        image_size=224):
    def load_image(d: dict):
        nonlocal read_question_family, image_size
        image = tf.io.decode_png(tf.io.read_file(d['image_file']))
        image = tf.image.resize(image, [image_size, image_size])
        d.pop('image_file')
        d['image'] = image
        return d

    pairs = json.load(open(question_file, 'r'))['questions']
    file_names = [os.path.join(image_dir, x['image_filename']) for x in pairs]
    question_text = [tokenizer.encode_sentence(x['question'].strip('?')) for x in pairs]
    question_text = tf.keras.preprocessing.sequence.pad_sequences(question_text, padding='post',
                                                                  value=tokenizer.pad_id)
    raw_data = {
        'question': question_text,
        'image_file': file_names,
    }
    if read_question_family:
        raw_data['question_family_index'] = [x['question_family_index'] for x in pairs]
    if read_label:
        raw_data['label'] = [label_encoder.encode(x['answer']) for x in pairs]

    dataset = tf.data.Dataset.from_tensor_slices(raw_data)
    del pairs, file_names, question_text, raw_data
    dataset = dataset.repeat(epoch).shuffle(1024).map(load_image,
                                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


def _test_read_paired_dataset():
    from matplotlib import pyplot as plt
    tokenizer = Tokenizer().load('tmp/tiny-tokenizer.pickle')
    label_encoder = LabelEncoder(LabelEncoder.TRAIN_LABELS)
    dataset = read_paired_dataset(
        question_file='data/CLEVR_v1.0/questions/CLEVR_tiny_val_questions.json',
        image_dir='data/CLEVR_v1.0/images/val/', tokenizer=tokenizer, label_encoder=label_encoder,
        batch_size=2, read_question_family=True, read_label=True)
    for x in dataset:
        question, image, answer, family_index = x['question'], x['image'], x['label'], \
                                                x['question_family_index']
        print(tokenizer.decode_sentence(question.numpy()[0]))
        print(answer, family_index)
        plt.imshow(image.numpy()[0])
        plt.show()
        return


if __name__ == '__main__':
    _test_read_paired_dataset()
