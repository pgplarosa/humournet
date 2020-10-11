"""
################################################################################
## Module Name: generate_meme.py
## Created by: Patrick La Rosa
## Created on: 30/09/2020
##
##
###############################################################################
"""

# import libraries
import os
import tensorflow as tf
import pickle
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from argparse import ArgumentParser

from dataset import MemesDataset
import config
import models
import write_caption

def eval_img(img_path, models, tokenizer):
    """evaluate image and return caption"""
    
    # storage of results
    result = []
    encoder, decoder = models

    # preprocess image
    img = MemesDataset.preprocess_image(img_path)
    img_features_model = MemesDataset.load_image_features_model()
    img_tensor = img_features_model(img)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    # flatten tensor to be passed to fully connected encoder
    img_tensor = tf.reshape(img_tensor, shape=(img_tensor.shape[0],
        -1, img_tensor.shape[-1]))

    # initialize memory and carry state
    memory_state = tf.zeros((1, config.units))
    carry_state = tf.zeros((1, config.units))

    # initialize word input
    word_input = [tokenizer.word_index['[cls]']] 
    word_input = tf.expand_dims(word_input, axis=0)
    
    encoded_img = encoder(img_tensor)

    for index in range(config.max_len):
        output, memory_state, carry_state = decoder(encoded_img, word_input,  
                                                    memory_state, carry_state)

        # get random output based on probability given by output
        output = tf.math.argmax(output, axis=-1)[0].numpy()

        if tokenizer.word_index['[sep]'] == output:
            return ' '.join(result) 

        word_input = tf.expand_dims([output], axis=0)

        if (tokenizer.word_index['[cls]'] == output or 
              tokenizer.word_index['[pad]'] == output):
            continue

        result.append(tokenizer.index_word[output])

    return ' '.join(result) 

def load_models(weights_fname, model_name, units, embedding_dim, vocab_size):
    """load models weights"""
    
    # instantiate model
    encoder = models.EncoderModel(units)
    decoder = models.DecoderModel(units, embedding_dim, vocab_size)

    enc_weights, dec_weights = weights_fname

    # call model to load weights
    encoder(tf.zeros([1] + config.img_shape[model_name]))
    decoder(tf.zeros([1] + [config.img_shape[model_name][0]] + [units]), 
        tf.zeros([1, 1]) , tf.zeros([1] + [units]), tf.zeros([1] + [units]))

    # load trained weights
    encoder.load_weights(enc_weights)
    decoder.load_weights(dec_weights)

    return encoder, decoder

if __name__ == '__main__':
    parser  = ArgumentParser()
    parser.add_argument('--img_path', default='imagination.jpg', 
    help='image file location to be used to generate caption')
    parser.add_argument('--model_name', default=config.model_name, 
    help='name of pretrained model for image features extraction')
    parser.add_argument('--save', default=1, 
    help='save an image with caption')
    parser.add_argument('--enc_weights', default='model_weights/encoder.h5', 
    help='encoder model weights location')
    parser.add_argument('--dec_weights', default='model_weights/decoder.h5', 
    help='decoder model weights location')

    args = parser.parse_args()

    # load model configs
    with open('model_config/encoder_config.pickle', 'rb') as handle:
        encoder_config = pickle.load(handle)

    with open('model_config/decoder_config.pickle', 'rb') as handle:
        decoder_config = pickle.load(handle)

    weights_fname = (args.enc_weights, args.dec_weights)
    models = load_models(weights_fname,
                        args.model_name, 
                        decoder_config['units'], 
                        decoder_config['embedding_dim'], 
                        decoder_config['vocab_size'])

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    result = eval_img(args.img_path, models, tokenizer)
    print(result)

    if args.save:
        fname = os.path.basename(args.img_path)
        fname_output = 'caption_' + fname

        if not os.path.isdir('output'):
            os.mkdir('output')

        full_path = os.path.join('output', fname_output)

        img = Image.open(args.img_path)
        draw = ImageDraw.Draw(img)
        write_caption.drawText(img, draw, result.upper(), "top")

        img.save(full_path)
