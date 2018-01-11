import numpy as np
import os
from keras.models import Model
from keras.layers import Dense, Embedding, Activation, Permute
from keras.layers import Input, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from .custom_recurrents import AttentionDecoder


def simpleNMT(in_padding,
              in_vocab_size,
              embedding_mul,
              encoder_units,
              decoder_units,
              out_padding,
              out_vocab_size,
              trainable=True,
              embedding_learnable=False,
              return_probabilities=False):
    """
    Builds a Neural Machine Translator that has alignment attention
    :param [in|out]_padding:    the unified lengths of the batch input or output sequences
    :param [in|out]_vocab_size: the sizes of the input or output vocabulary
    :param embedding_mul:       the multiplyer of one char to the length of its vector
    :param embedding_learnable: decides if the one hot embedding should be refinable.
    :return:                    keras.models.Model that can be compiled and fit'ed

    *** REFERENCES ***
    Lee, Jason, Kyunghyun Cho, and Thomas Hofmann.
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    """
    input_ = Input(shape = (in_padding,), dtype = 'float32')

    input_embed = Embedding(input_dim    = in_vocab_size,
                            output_dim   = embedding_mul,
                            input_length = in_padding,
                            trainable    = embedding_learnable,
                            weights      = [np.eye(in_vocab_size)],
                            name         = 'OneHot')(input_)

    rnn_encoded = Bidirectional(LSTM(encoder_units, return_sequences=True),
                                name        = 'bidirectional_1',
                                merge_mode  = 'concat',
                                trainable   = trainable)(input_embed)

    y_hat = AttentionDecoder(decoder_units, out_padding, out_vocab_size,
                             name                   = 'attention_decoder_1',
                             return_probabilities   = return_probabilities,
                             trainable              = trainable)(rnn_encoded)

    return Model(inputs = input_, outputs = y_hat)


if __name__ == '__main__':
    model = simpleNMT()
    model.summary()
