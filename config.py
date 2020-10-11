"""
################################################################################
## Module Name: config.py
## Created by: Patrick La Rosa
## Created on: 30/09/2020
##
##
################################################################################
"""

import tensorflow as tf

# CONSTANTS
seed = 0
# save path of images
save_path = 'images/'

# image size based on the pretrained model 
IMG_SIZE = {'EfficientNetB2': (260,260)}
# output shape of image after forward pass to pretrained model
img_shape = {'EfficientNetB2' : [64, 1408]}
model_name = 'EfficientNetB2'

# number of words to be included in the dictionary
n_words = 10000
max_len = 80

# hyperparameters
# Dataset batch and buffer size
BATCH_SIZE = 128
BUFFER_SIZE = 1024
NUM_EPOCHS = 5
lr = 0.001

# num parallel calls
AUTOTUNE = tf.data.experimental.AUTOTUNE

# word and image embeddings size
embedding_dim = 256
# hidden state units of LSTM
units = 512


