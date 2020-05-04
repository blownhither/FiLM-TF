import os
import datetime
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
    'epoch': 80,
    'question_family_supervision': True,
    'question_family_weight': 1e-2
}
DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')


class TrainableFilmModel:
    def __init__(self, vocab_size, predict_size, pad_id):
        self.model = FilmModel(vocab_size, predict_size, pad_id=pad_id)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                     name='answer_loss')
        self.question_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                              name='question_family_loss')
        self.accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()
        self.optimizer = tf.keras.optimizers.Adam(hyper_parameters['lr'])
        self.global_step = 0

    def train_step(self, batch):
        with tf.GradientTape() as t:
            logits, question_family_logits = self.model((batch['question'], batch['image']))
            loss = self.loss_fn(batch['label'], logits)
            if hyper_parameters['question_family_supervision']:
                loss += self.question_loss_fn(batch['question_family_index'],
                                              question_family_logits) * hyper_parameters[
                            'question_family_weight']
        grads = t.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.global_step += 1
        if self.global_step % 100 == 0:
            print('step', self.global_step, 'loss', loss.numpy())
        return loss.numpy()

    def train_epoch(self, dataset, comet_experiment=None, epoch=None):
        loss_records = []
        for batch in dataset:
            loss = self.train_step(batch)
            loss_records.append(loss)
            if comet_experiment is not None:
                comet_experiment.log_metric('train_loss', loss, epoch=epoch, step=self.global_step)
        mean_loss = np.mean(loss_records)
        comet_experiment.log_metric('epoch_train_loss', mean_loss, epoch=epoch,
                                    step=self.global_step)
        print(epoch, 'train', 'loss=', mean_loss, flush=True)
        return mean_loss

    def eval_step(self, batch):
        logits = self.model((batch['question'], batch['image']))
        loss = self.loss_fn(batch['label'], logits)
        prob = tf.nn.softmax(logits, axis=1)
        choice = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
        correct_count = tf.reduce_sum(tf.cast(batch['label'] == choice, dtype=tf.float32))
        return loss.numpy(), correct_count.numpy()

    def eval_all(self, dataset, comet_experiment=None, epoch=None):
        loss_records = []
        total_count = 0
        total_correct_count = 0
        for batch in dataset:
            batch_size = len(batch['label'])
            loss, correct_count = self.eval_step(batch)
            total_correct_count += correct_count
            total_count += batch_size
            loss_records.append(loss)
            if comet_experiment is not None:
                comet_experiment.log_metric('eval_loss', loss, step=self.global_step, epoch=epoch)
                comet_experiment.log_metric('eval_acc', correct_count / batch_size,
                                            step=self.global_step, epoch=epoch)
        mean_loss = np.mean(loss_records)
        print(epoch, 'eval', 'loss=', mean_loss, 'acc=', total_correct_count / total_count,
              flush=True)
        if comet_experiment is not None:
            comet_experiment.log_metric('epoch_eval_acc', total_correct_count / total_count,
                                        step=self.global_step, epoch=epoch)
            comet_experiment.log_metric('epoch_eval_loss', mean_loss, step=self.global_step,
                                        epoch=epoch)
        return mean_loss


def main(argv):
    FLAGS = flags.FLAGS
    experiment = Experiment()
    experiment.log_asset(__file__)
    experiment.log_parameters(hyper_parameters)
    experiment.log_parameter('datetime', DATETIME)
    model_dir = os.path.join(os.path.dirname(__file__), f'tmp/model-{DATETIME}/')
    os.mkdir(model_dir)
    experiment.log_parameter('model_dir', model_dir)

    tokenizer = Tokenizer().load(FLAGS.tokenizer_path)
    label_encoder = LabelEncoder(LabelEncoder.TRAIN_LABELS)
    train_dataset = read_paired_dataset(question_file=FLAGS.train_questions,
                                        image_dir=FLAGS.train_image_dir, tokenizer=tokenizer,
                                        label_encoder=label_encoder, epoch=1,
                                        batch_size=hyper_parameters['batch_size'],
                                        read_question_family=True, read_label=True)
    val_dataset = read_paired_dataset(question_file=FLAGS.val_questions,
                                      image_dir=FLAGS.val_image_dir, tokenizer=tokenizer,
                                      label_encoder=label_encoder, epoch=1,
                                      batch_size=hyper_parameters['batch_size'],
                                      read_question_family=True, read_label=True)

    model = TrainableFilmModel(tokenizer.get_vocab_size(), len(label_encoder.TRAIN_LABELS),
                               pad_id=tokenizer.pad_id)
    for epoch in range(hyper_parameters['epoch']):
        model.train_epoch(train_dataset, comet_experiment=experiment, epoch=epoch)
        if epoch == 0:
            param_count = model.model.count_params()
            print('param_size', param_count)
            experiment.log_parameter('param_size', param_count)
        model.eval_all(val_dataset, comet_experiment=experiment, epoch=epoch)

        # save weights
        if epoch % 5 == 0:
            model.model.save_weights(
                os.path.join(model_dir, f'model-e{epoch}s{model.global_step}.savedmodel'))


if __name__ == '__main__':
    flags.DEFINE_string('tokenizer_path', 'tmp/tiny-tokenizer.pickle', 'path to load tokenizer')
    flags.DEFINE_string('train_questions',
                        'data/CLEVR_v1.0/questions/CLEVR_tiny_val_questions.json',
                        'path to read train question json')
    flags.DEFINE_string('val_questions', 'data/CLEVR_v1.0/questions/CLEVR_tiny_val_questions.json',
                        'path to read val question json')
    flags.DEFINE_string('train_image_dir', 'data/CLEVR_v1.0/images/val/',
                        'path to read train images')
    flags.DEFINE_string('val_image_dir', 'data/CLEVR_v1.0/images/val/', 'path to read val images')
    app.run(main)
