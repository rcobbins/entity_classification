import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM

tfpl = tfp.layers

class BayesianNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, output_dim, num_layers=3, units=64):
        super(BayesianNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.units = units
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.lstm = Bidirectional(LSTM(units, return_sequences=True))
        self.bayesian_layers = [tfpl.DenseFlipout(units, activation='relu') for _ in range(num_layers)]
        self.output_layer = tfpl.DenseFlipout(output_dim, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        for bayesian_layer in self.bayesian_layers:
            x = bayesian_layer(x)
        return self.output_layer(x)

    def update_output_layer(self, new_output_dim):
        self.output_dim = new_output_dim
        self.output_layer = tfpl.DenseFlipout(new_output_dim, activation='softmax')

