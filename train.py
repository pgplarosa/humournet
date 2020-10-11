"""
################################################################################
## Module Name: train.py
## Created by: Patrick La Rosa
## Created on: 30/09/2020
##
##
###############################################################################
"""

# import libraries
import os
from argparse import ArgumentParser
import pickle
from tqdm import tqdm
import tensorflow as tf

import config
import models
from dataset import MemesDataset
import generate_meme

@tf.function
def train_step(batch_set, tokenizer, vocab_size):
  """ train step for 1 batch"""
  img_tensor, cap_seqs = batch_set
  
  loss = 0
  batch_size = cap_seqs.shape[0]
  # initialize memory and carry state to zero
  memory_state = tf.zeros((batch_size, config.units))
  carry_state = tf.zeros((batch_size, config.units))
  
  # initialize word_input seqs, shape (batch_size, 1)
  # word to input to embedding layer 
  word_input = [tokenizer.word_index['[cls]']] * batch_size
  word_input = tf.expand_dims(word_input, axis=1)
  
  with tf.GradientTape() as tape:
    # get encoded image from encoder
    encoded_img = encoder(img_tensor)
    
    for index in range(cap_seqs.shape[1]):
      preds, memory_state, carry_state = decoder(encoded_img, word_input,  
                                                memory_state, carry_state)
      
      # use the previous input as next word input
      word_input = tf.expand_dims(cap_seqs[:, index], axis=1)
      # convert labels to one hot as expected by loss function
      true_label = tf.one_hot(word_input, depth=vocab_size)
      true_label = tf.squeeze(true_label)
      # compute loss
      loss += loss_f(true_label, preds, vocab_size)
      
  trainable_variables = (encoder.trainable_variables 
                        + decoder.trainable_variables)
  
  # back propagate and update parameters
  gradients = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(gradients, trainable_variables))
  
  return loss

def loss_f(real, pred, vocab_size):
  """ mask [pad] token for computing of loss"""

  mask = tf.argmax(real, axis=-1) != 0
  mask = tf.cast(mask, dtype=tf.float32)
  loss = loss_object(real, pred)
  loss *= mask
  
  return tf.reduce_mean(loss)


if __name__ == '__main__':
  parser  = ArgumentParser()
  parser.add_argument('--fname', default='memes_dataset_small.csv', 
    help='filename of dataset to be used')
  parser.add_argument('--save_path', default=config.save_path, 
    help='save path of images')
  parser.add_argument('--model_name', default=config.model_name, 
    help='name of pretrained model for image features extraction')
  parser.add_argument('--units', default=config.units, 
    help='hidden units')
  parser.add_argument('--emb_dim', default=config.embedding_dim, 
    help='embedding dimensions', type=int)
  parser.add_argument('--n_words', default=config.n_words, 
    help='vocabulary size', type=int)
  parser.add_argument('--lr', default=config.lr, help='learning rate', 
    type=float)
  parser.add_argument('--num_epochs', default=config.NUM_EPOCHS, 
    help='number of epochs', type=int)
  parser.add_argument('--load_models', default=1, 
    help='load pretrained models', type=int)
  parser.add_argument('--sample_img', default='imagination.jpg', 
    help='sample image to eval during training', type=str)
  parser.add_argument('--enc_weights', default='model_weights/encoder.h5', 
    help='encoder model weights location')
  parser.add_argument('--dec_weights', default='model_weights/decoder.h5', 
    help='decoder model weights location')

  args = parser.parse_args()

  # load train dataset
  dataset = MemesDataset(args.fname, args.save_path, 
                        args.n_words, args.model_name)

  train_dataset, train_len = dataset.get_dataset(train=True)
  
  # create models
  if args.load_models:
    weights_fname = (args.enc_weights, args.dec_weights)
    models = generate_meme.load_models(weights_fname, 
                                      args.model_name, args.units, 
                                      args.emb_dim, args.n_words + 1)
    encoder, decoder = models
  else:
    encoder = models.EncoderModel(args.units)
    decoder = models.DecoderModel(args.units, args.emb_dim, args.n_words + 1)
    models = encoder, decoder
  
  encoder_config = encoder.get_config()
  decoder_config = decoder.get_config()

  # create directory to save model weights
  if not os.path.isdir('model_config'):
    os.mkdir('model_config')


  # save encoder/decoder config
  with open('model_config/encoder_config.pickle', 'wb') as handle:
      pickle.dump(encoder_config, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open('model_config/decoder_config.pickle', 'wb') as handle:
      pickle.dump(decoder_config, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # create optimizer and loss object
  optimizer = tf.keras.optimizers.Adam(args.lr)
  loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, 
                                                        reduction='none')

  # load tokenizer object
  with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

  # create directory to save model weights
  if not os.path.isdir('model_weights'):
    os.mkdir('model_weights')

  train_losses = []
  for epoch in range(args.num_epochs):
    train_loss = 0
    print(f'EPOCH: {epoch + 1} / {args.num_epochs} ..')
    for batch_train in tqdm(train_dataset, position=0, leave=True):
      train_loss += train_step(batch_train, tokenizer, args.n_words + 1)
    
    print(f'Train loss = {train_loss/train_len}')
    train_losses.append(train_loss/train_len)

    result = generate_meme.eval_img(args.sample_img, models, tokenizer)
    print(result)

    if epoch and epoch % 5 == 0:
      # create checkpoints every 5 epochs
      encoder.save_weights(f'model_weights/encoder_{epoch}.h5')
      decoder.save_weights(f'model_weights/decoder_{epoch}.h5')

  encoder.save_weights('model_weights/encoder.h5')
  decoder.save_weights('model_weights/decoder.h5')
