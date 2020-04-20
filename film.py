import numpy as np
import tensorflow as tf
from absl import app, flags

from model import FilmModel
from tokenizer import Tokenizer, LabelEncoder
from data import read_paired_dataset


class TrainableFilmModel:
    def __init__(self, vocab_size, predict_size, lr=3e-4):
        self.model = FilmModel(vocab_size, predict_size)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(lr)

    def train_step(self, batch):
        with tf.GradientTape() as t:
            logits = self.model((batch['question'], batch['image']))
            loss = self.loss_fn(batch['label'], logits)
        grads = t.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss.numpy()

    def train_epoch(self, dataset):
        loss_records = []
        for batch in dataset:
            loss = self.train_step(batch)
            loss_records.append(loss)
        return np.mean(loss_records)


def main():
    FLAGS = flags.FLAGS
    flags.DEFINE_string()


    tokenizer = Tokenizer().load('tmp/tiny-tokenizer.pickle')
    label_encoder = LabelEncoder(LabelEncoder.TRAIN_LABELS)
    dataset = read_paired_dataset(
        question_file='data/CLEVR_v1.0/questions/CLEVR_tiny_val_questions.json',
        image_dir='data/CLEVR_v1.0/images/val/', tokenizer=tokenizer, label_encoder=label_encoder,
        batch_size=2, read_question_family=True, read_label=True)

    model = TrainableFilmModel(tokenizer.get_vocab_size(), len(label_encoder.TRAIN_LABELS))
    for batch in dataset:
        loss = model.train_step(batch)
        print(loss)


if __name__ == '__main__':
    main()

