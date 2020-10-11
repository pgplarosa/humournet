"""
################################################################################
## Module Name: dataset.py
## Created by: Patrick La Rosa
## Created on: 30/09/2020
##
##
################################################################################
"""

# import libraries
import re
import os

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pickle

import config
# TODO: save_path, n_words, model_name
class MemesDataset(object):
   """
   preprocess images and captions, and creates memes tensorflow dataset object
   """
   def __init__(self, fname, save_path, n_words, model_name):
      # read memesdataset 
      self.df = pd.read_csv(fname)
      # get full path
      self.df['full_path'] = [os.path.join(save_path, fname) 
                              for fname in self.df['filename']]
      # drop any null images/captions
      self.df.dropna(inplace=True)
      
      # extract image features using pretrained model
      print('extracting image features..')
      self.img_extract = self.extract_image_features(self.df['full_path'].unique(),
                                                      model_name=model_name)
      
      # peform necessary preprocessing for captions
      print('preprocessing and tokenizing captions..')
      self.df['caption'] = self.preprocess_captions(self.df['caption'])

      # tokenize captions 
      (self.tokenizer, 
      self.padded_seqs, 
      self.max_len) = self.tokenize(list(self.df['caption'].values), n_words)

      # split to training and test sets
      print('splitting train and eval set..')
      self.df['padded_seqs'] = list(self.padded_seqs)
      (self.train_set, 
      self.eval_set) = train_test_split(self.df.loc[:,['full_path', 'padded_seqs']], 
                                       test_size=0.1, 
                                       random_state=config.seed, 
                                       stratify=self.df['full_path'])

   @staticmethod
   def preprocess_image(img_path, model_name='EfficientNetB2', img_size=(260,260)):
       """resize and normalize image based on the chosen pretrained model"""
       
       img = tf.io.read_file(img_path)
       img = tf.image.decode_jpeg(img, channels=3)
       img = tf.image.resize(img, img_size) 
       
       # normalize image based on model
       if model_name == 'EfficientNetB2':
         img = tf.keras.applications.efficientnet.preprocess_input(img)
       else:
         print(f'{model_name} preprocess is not yet implemented') 
       
       return img

   @staticmethod
   def load_image_features_model(model_name='EfficientNetB2'):
       """
       load model to be used for extraction of image features 
       model_name='EfficientNetB2'
         input: shape=(None, None, None, 3)
         output: shape=(None, None, None, 1408)
       """
       if model_name == 'EfficientNetB2':
         image_model = tf.keras.applications.EfficientNetB2(include_top=False,
                                                             weights='imagenet')
       else:
         print(f'{model_name} model is not yet implemented')

       model_input = image_model.input
       model_output = image_model.output
       
       return tf.keras.Model(model_input, model_output)

   def extract_image_features(self, fnames, model_name='EfficientNetB2'):
       """extract and cache image features using pretrained model"""

       img_extract = {}
       image_features_model = MemesDataset.load_image_features_model()
       
       for img_path in tqdm(fnames, position=0, leave=True):
         img_tensor = MemesDataset.preprocess_image(img_path, model_name, 
                                             config.IMG_SIZE[model_name])
         # shape after (1, 8, 8, 1408) after getting image features
         img_features = image_features_model(img_tensor)
         # flatten tensor to be passed to fully connected encoder
         img_features = tf.reshape(img_features, shape=(img_features.shape[0], 
                                                      -1, img_features.shape[-1]))
         
         img_extract[img_path] = img_features
       
       return img_extract

   def preprocess_captions(self, captions):
      """
      convert characters to lower case, trim spaces, etc 
      input: series object
      """

      # add start and end token for each sentence
      captions = '[cls] ' + captions + ' [sep]'

      # convert to lower case and strip unnecessary spaces
      captions = captions.str.lower().str.strip()

      # correct spelling for some words/phrase manually for training
      captions = captions.str.replace('y u no', 'why you not')
      captions = captions.str.replace('y u', 'why you')
      captions = captions.str.replace(' u ', ' you ')

      return captions

   def tokenize(self, captions, n_words=10000):
      """
      tokenize and return tokenizer, padded word sequences, 
      and maximum length of sentence.
      """

      # vocab size :22153
      # get most n_words frequent words
      tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=n_words,
                                                      oov_token='[unk]',
                                                      filters='!"#$%&()*+.,-/:;=?@^_`{|}~')
      tokenizer.fit_on_texts(captions)
      caps_seqs = tokenizer.texts_to_sequences(captions)
      
      # get maximum sentence length for padding and attention model
      max_len = max([len(caps_seq) for caps_seq in caps_seqs])

      tokenizer.word_index['[pad]'] = 0
      tokenizer.index_word[0] = '[pad]'

      padded_seqs = tf.keras.preprocessing.sequence.pad_sequences(caps_seqs, 
                                                                  padding='post',
                                                                  maxlen=max_len)

      # save tokenizer object
      with open('tokenizer.pickle', 'wb') as handle:
          pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
      assert padded_seqs.shape[0] == len(captions)

      return tokenizer, padded_seqs, max_len

   def get_img_features(self, full_path, padded_seqs):
      """returns image tensor"""

      img_tensor = self.img_extract[full_path.decode('utf-8')]
      img_tensor = tf.squeeze(img_tensor)
      return img_tensor, padded_seqs

   def create_dataset(self, dataset):
      """create training tensorflow dataset object"""

      dataset = tf.data.Dataset.from_tensor_slices((dataset['full_path'].values, 
                                             list(dataset['padded_seqs'].values)))

      dataset = dataset.map(lambda full_path, padded_seqs: 
                                          tf.numpy_function(self.get_img_features, 
                                                         [full_path, padded_seqs], 
                                                         [tf.float32, tf.int32]),
                                          num_parallel_calls=config.AUTOTUNE)

      dataset = dataset.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)
      dataset = dataset.prefetch(buffer_size=config.AUTOTUNE)

      return dataset

   def get_dataset(self, train=True):
      """returns tensorflow dataset object"""

      if train:
         dataset = self.create_dataset(self.train_set)
         dataset_len =  len(self.train_set)
      else:
         dataset = self.create_dataset(self.eval_set)
         dataset_len = len(self.eval_set)

      return dataset, dataset_len
