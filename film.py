import numpy as np
from comet_ml import Experiment
import tensorflow as tf
from absl import app, flags

from model import FilmModel
from tokenizer import Tokenizer, LabelEncoder
from data import read_paired_dataset


hyper_parameters = {
    'lr': 3e-4,
    'batch_size': 64,
}


class TrainableFilmModel:
    def __init__(self, vocab_size, predict_size):
        self.model = FilmModel(vocab_size, predict_size)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(hyper_parameters['lr'])

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


def main(argv):
    FLAGS = flags.FLAGS
    experiment = Experiment()
    experiment.log_asset(__file__)
    experiment.log_parameters(hyper_parameters)

    tokenizer = Tokenizer().load(FLAGS.tokenizer_path)
    label_encoder = LabelEncoder(LabelEncoder.TRAIN_LABELS)
    train_dataset = read_paired_dataset(
        question_file=FLAGS.train_questions,
        image_dir=FLAGS.image_dir, tokenizer=tokenizer, label_encoder=label_encoder,
        batch_size=hyper_parameters['batch_size'], read_question_family=True, read_label=True)

    model = TrainableFilmModel(tokenizer.get_vocab_size(), len(label_encoder.TRAIN_LABELS))
    for batch in train_dataset:
        loss = model.train_step(batch)
        print(loss, flush=True)


if __name__ == '__main__':
    flags.DEFINE_string('tokenizer_path', 'tmp/tiny-tokenizer.pickle', 'path to load tokenizer')
    flags.DEFINE_string('train_questions',
                        'data/CLEVR_v1.0/questions/CLEVR_tiny_val_questions.json',
                        'path to read question json')
    flags.DEFINE_string('image_dir', 'data/CLEVR_v1.0/images/val/', 'path to read question json')
    app.run(main)

