# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras.backend as K
from keras.layers import Embedding, Dense
from utils import show_layer_info

class EmbedText(Embedding):
    def __init__(self, config, output_dim, **kwargs):
        self.config = config
        self.output_dim = output_dim
        #super(EmbedText, self).__init__(**kwargs)

    def build(self, input_shape):
        self.vocabW = self.add_weight(name='vocabW',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(EmbedText, self).build(input_shape)

    def call(self, text):
        # Create a trainable weights variable for the embedding layer.
        embed = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                          trainable=self.config['train_embed'])  # embed should be a matrix with word_ids x vectors
        # show_layer_info('Embedding', embed)
        # embed_w = Dense(self.config['vocab_size'], use_bias=False)(embed)  # weight all terms in the vocabulary
        text_embed = embed(text)
        embed_w = K.dot(text_embed, self.vocabW)
        # show_layer_info('Dense', embed_w)
        return K.sum(embed_w)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def embed_text(input, config, input_shape):
        """Functional interface to the `EmbedText` layer.
        # Arguments
            input: text input
        # Returns
            A tensor, the weighted text vector.
        """
        return EmbedText(input, config, input_shape)
