from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from numpy import array
from pickle import load,dump
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from numpy import argmax

from tqdm import tqdm

# Step 1: Read the text file
with open('trainuc.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Step 2 & 3: Tokenize lines and assign indices
vocab = {}
index = 0
for line in lines:
    words = line.strip().split()  # Tokenize by space assuming each line contains a single word
    for word in words:
        if word not in vocab:
            vocab[word] = index
            index += 1

# Step 4: Create inverse vocabulary
inv_vocab = {v: k for k, v in vocab.items()}

# Optionally, save the vocab dictionary to a numpy file


# Now you have the vocab and inv_vocab dictionaries



print("+"*50)
print("vocabulary loaded")
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split(' ')[0]
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    #print(all_features)
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

def create_sequences(tokenizer, max_length, desc_list, photo):
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	for desc in desc_list:
		# encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)
 
#data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
			yield [[in_img, in_seq], out_word]



# load training dataset (6K)
filename = 'trainuc.txt'
train = load_set(filename)

# descriptions
train_descriptions = load_clean_descriptions('trainuc.txt', train)
train_features = load_photo_features('features_uc_resnet152_updated.pkl', train)
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


class Attention(Layer):
    max_length = 100
    def __init__(self, step_dim=max_length,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
      assert len(input_shape) == 3

      self.W = self.add_weight(shape=(input_shape[-1],),
                             initializer=self.init,
                             name='{}_W'.format(self.name),
                             regularizer=self.W_regularizer,
                             constraint=self.W_constraint)
      self.features_dim = input_shape[-1]

      if self.bias:
          self.b = self.add_weight(shape=(input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.b_regularizer,
                                 constraint=self.b_constraint)
      else:
          self.b = None

      self.built = True



    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim





embedding_size = 128
max_length = 24
inputs1 = Input(shape=(2048,))
    #fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(inputs1)
    
inputs2 = Input(shape=(21,))
merged = concatenate([fe2,inputs2])
    
fe3 = RepeatVector(max_length)(merged)
    
    # sequence model
inputs3 = Input(shape=(max_length,))
    #se1 = load_embedding(tokenizer, vocab_size, max_length)(inputs3)
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs3)
    #print(se1.shape)
se2 = LSTM(256,return_sequences=True)(se1)
    #print(se2.shape)
    #x = Attention(max_length)(se2) 
    #print(x.shape)
se3 = TimeDistributed(Dense(256,activation='relu'))(se2)
    #print(se3.shape)
    
    # decoder model
decoder1 = concatenate([fe3, se3])
decoder2 = Bidirectional(LSTM(150,return_sequences=True))(decoder1)
decoder3 = Attention(max_length)(decoder2)
outputs = Dense(vocab_size, activation='softmax')(decoder3)
    # tie it together [image, seq] [word]
model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile model
model.compile(loss='categorical_crossentropy', optimizer='adam')


model.load_weights('model_4.h5')

print("="*150)
print("MODEL LOADED")

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def index():
    return render_template('index.html')

# Load the pre-trained ResNet50 model
from tensorflow.keras.applications.resnet50 import preprocess_input

resnet_model = ResNet50(weights='imagenet', include_top=False)

# Function to preprocess the image
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Function to extract image features using ResNet50
def extract_features(img):
    img = preprocess_image(img)
    features = resnet_model.predict(img)
    # Reshape the features to (1, 2048)
    features = features.flatten()[:2048]
    features = np.reshape(features, (1, -1))
    return features


# Update the 'after' route to extract image features before prediction
@app.route('/after', methods=['GET', 'POST'])
def after():
    global model, tokenizer, max_length

    photo = request.files['file1']
    photo.save('static/file.jpg')

    print("="*50)
    print("IMAGE SAVED")

    photo = cv2.imread('static/file.jpg')
    photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    
    # Extract image features
    photo_features = extract_features(photo)
    
    print("Predicting Caption")

    in_text = 'startseq'
    # integer encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
    # predict next word
    yhat = model.predict([photo_features, np.zeros((1, 21)), sequence], verbose=0)
    # convert probability to integer
    yhat = argmax(yhat)
    # map integer to word
    word = word_for_id(yhat, tokenizer)
    
    while word != 'endseq':
        in_text += ' ' + word
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo_features, np.zeros((1, 21)), sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)

    return render_template('after.html', data=in_text)


    











if __name__ == "__main__":
    app.run(debug=True)


