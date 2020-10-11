"""
################################################################################
## Module Name: models.py
## Created by: Patrick La Rosa
## Created on: 30/09/2020
##
##
###############################################################################
"""

# import libraries
import tensorflow as tf

class EncoderModel(tf.keras.Model):
    """
    input: extracted image features from pretrained model
    output shape: (batch_size, 64, units)
    """

    def __init__(self, units):
        self.units = units
        super(EncoderModel, self).__init__()
        self.fc = tf.keras.layers.Dense(self.units, activation=tf.nn.relu)

    def call(self, img_features):
        
        x = self.fc(img_features)

        return x

    def get_config(self):
        return {'units', self.units}

class AttentionModel(tf.keras.Model):
    """
    Bahdanau Attention model

    input: output of EncoderModel with shape (batch_size, 64, embedding_dim)
           hidden state of LSTM
    output shape: (batch_size, units)
    """
    def __init__(self, units):
        super(AttentionModel, self).__init__()
        # fully connected layer for the output of encoder model
        self.fc1 = tf.keras.layers.Dense(units)
        # fully connected layer for the hidden state of LSTM
        self.fc2 = tf.keras.layers.Dense(units)
        # fully connected layer to compute for the score
        self.fc3 = tf.keras.layers.Dense(1)

    def call(self, encoded_img, hidden_state):
        # shape after fc1 (batch_size, 64, units)
        features = self.fc1(encoded_img)
        # shape after fc2 (batch_size, units)
        hidden_state = self.fc2(hidden_state)
        
        # expand to be able to add features and hidden_state
        hidden_state = tf.expand_dims(hidden_state, axis=1)
        attention_vector = tf.keras.activations.tanh(features + hidden_state)

        # shape after fc3 (batch_size, 1)
        attention_score = self.fc3(attention_vector)
        attention_score = tf.nn.softmax(attention_score, axis=1)

        # context vector shape (batch_size, units)
        context_vector = attention_score * encoded_img
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

class DecoderModel(tf.keras.Model):
    """
    one step of LSTM cell
    input: output of image encoder, input word, 
                 memory state and carry state of LSTM
    output shape: (batch_size, vocab_size) to represent each word in vocab
    """
    def __init__(self, units, embedding_dim, vocab_size):
        super(DecoderModel, self).__init__()
        self.units = units
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.attention = AttentionModel(self.units)
        self.lstm = tf.keras.layers.LSTM(self.units, return_state=True)
        self.fc = tf.keras.layers.Dense(self.vocab_size)

    def call(self, encoded_img, word_input, memory_state, carry_state):
        # x shape after embedding (batch_size, 1, embedding_dim) 
        x = self.embedding(word_input)
        # contex_vector shape (batch_size, units)
        context_vector = self.attention(encoded_img, memory_state)
        
        # x shape after concat (batch_size, 1, units + embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # perform one step of lstm cell, output shape (batch_size, units)
        (output, 
        memory_state, 
        carry_state) = self.lstm(x, initial_state=[memory_state, carry_state])
        
        # output shape after fc (batch_size, vocab_size)
        output = self.fc(output)
        # output = tf.nn.softmax(output)

        return output, memory_state, carry_state

    def get_config(self):
        return {'units':self.units, 
                'embedding_dim':self.embedding_dim,
                'vocab_size':self.vocab_size}
