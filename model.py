import numpy as np
import tensorflow as tf


class FilmModel(tf.keras.Model):
    def __init__(self, vocab_size, predict_size, embedding_dim=200, gru_size=4096, n_res_blocks=4,
                 cnn_channels=128, classifier_channels=512, classifier_hidden=1024, pad_id=1):
        super(FilmModel, self).__init__()
        self.pad_id = pad_id

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(gru_size, return_sequences=False)
        self.conv_blocks = [(
            tf.keras.layers.Conv2D(cnn_channels, 4, strides=(1, 1),
                                   activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(trainable=True),
        ) for _ in range(4)]
        self.res_blocks = [(
            tf.keras.layers.Conv2D(cnn_channels, 1, strides=(2, 2),
                                   activation='relu', padding='same'),
            tf.keras.layers.Conv2D(cnn_channels, 3, strides=(1, 1),
                                   activation=None, padding='same'),
            tf.keras.layers.BatchNormalization(trainable=False),
            tf.keras.layers.Dense(cnn_channels),
            tf.keras.layers.Dense(cnn_channels),
            tf.keras.layers.ReLU()
        ) for _ in range(n_res_blocks)]
        self.classifier_conv = tf.keras.layers.Conv2D(classifier_channels, 1, padding='same')
        self.max_pool = tf.keras.layers.GlobalMaxPool2D()
        self.hidden = tf.keras.layers.Dense(classifier_hidden, activation='relu')
        self.classify = tf.keras.layers.Dense(predict_size)

        self.coordinate_feature_cache = {}

    def call(self, inputs, training=None, mask=None):
        language_input, image_input = inputs
        emb = self.embedding(language_input)
        gru_out = self.gru(emb, mask=language_input != self.pad_id)
        tensor = image_input
        for b in self.conv_blocks:
            conv, bn = b
            tensor = bn(conv(tensor))
        for b in self.res_blocks:
            tensor = self.append_coordinate_feature_map(tensor)
            tensor = self.call_resblock(b, tensor, gru_out)
        tensor = self.append_coordinate_feature_map(tensor)
        tensor = self.max_pool(self.classifier_conv(tensor))
        logits = self.classify(self.hidden(tensor))
        return logits

    def append_coordinate_feature_map(self, tensor):
        size = tensor.get_shape().as_list()[1]
        if size not in self.coordinate_feature_cache:
            coord_feature = self.get_coordinate_feature(size)  # [size, size, 2]
            coord_feature = tf.expand_dims(coord_feature, 0)
            self.coordinate_feature_cache[size] = coord_feature
        else:
            coord_feature = self.coordinate_feature_cache[size]  # [1, size, size, 2]
        coord_feature_4d = tf.tile(coord_feature, [tensor.get_shape()[0], 1, 1, 1])
        tensor = tf.concat([tensor, coord_feature_4d], axis=3)
        return tensor

    @staticmethod
    def call_resblock(layers, tensor, side_input):
        # side input: [B, -]
        conv1, conv3, batch_norm, dense1, dense2, relu = layers
        residual = conv1(tensor)
        before_film = batch_norm(conv3(residual))
        scale = dense1(side_input)
        bias = dense2(side_input)
        scale_4d = tf.expand_dims(tf.expand_dims(scale, 1), 1)
        bias_4d = tf.expand_dims(tf.expand_dims(bias, 1), 1)
        after_film = before_film * scale_4d + bias_4d
        out = relu(after_film) + residual
        return out

    @staticmethod
    def get_coordinate_feature(size):
        # [size, size, 2]
        a = np.linspace(-1, 1, size)
        x, y = np.meshgrid(a, a)
        return tf.constant(np.dstack((x, y)), dtype=tf.float32)


def _test_model():
    model = FilmModel(vocab_size=5, predict_size=4)
    language = tf.keras.preprocessing.sequence.pad_sequences([
        [1, 2],
        [0, 3, 3]
    ])
    images = np.random.rand(2, 64, 64, 3)
    logits = model((language, images))
    print(logits.shape)
    print(logits)


if __name__ == '__main__':
    _test_model()

