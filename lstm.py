import numpy as np
import utils
from keras.preprocessing.text import Tokenizer
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

RANDOM_SEED = 420

VOCAB_SIZE = 5000
EMBEDDING_DIM = 64
MAX_TEXT_LEN = 250
OOV_TOKEN = "<OOV>"
NUM_GENRES = 20

INIT_LR = .005
NUM_EPOCHS = 10

# TODO replace this later one
data = utils.load_data()
data = utils.format_data(data)

genres, summaries = data[:, 1], data[:, 0]

# Pre-process Data
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(summaries)
word_index = tokenizer.word_index
print(word_index[:10])

# Tokenize to sequences and pad to have same length
text_sequences = tokenizer.texts_to_sequences(summaries)
x = pad_sequences(text_sequences, maxlen=MAX_TEXT_LEN)

# Multi-Label Binarize the Gentres
binarizer = MultiLabelBinarizer()
binarizer.fit(genres)
y = binarizer.classes_

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=RANDOM_SEED)

# Actual LSTM Architecture

"""
Architecture here

Layers
    - Embedding
    - LSTM
    - Fully Connected? 
    - TBD
"""

def create_baseline():
    """
    Dummy baseline Model
    :return: model, simple fully connected 1 hidden layer
    """

    model = Sequential()
    model.add(Dense(64, input_dim=MAX_TEXT_LEN, activation='relu'))
    model.add(Dropout(rate=.2))
    model.add(Dense(NUM_GENRES, activation='sigmoid'))

    optimizer = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


def create_lstm():
    # TODO
    pass


