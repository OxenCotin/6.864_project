import numpy as np
import utils
import keras
from keras.preprocessing.text import Tokenizer
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPool1D, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

RANDOM_SEED = 420

VOCAB_SIZE = 5000
EMBEDDING_DIM = 64
MAX_TEXT_LEN = 250
OOV_TOKEN = "<OOV>"
NUM_GENRES = 19

INIT_LR = .075
NUM_EPOCHS = 10

# TODO replace this later one
data = utils.get_data()

genres, summaries = data["genres"], data["summary"]

# Pre-process Data
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(summaries)
word_index = tokenizer.word_index

#
# import pdb
# pdb.set_trace()

# Tokenize to sequences and pad to have same length
text_sequences = tokenizer.texts_to_sequences(summaries)
x = pad_sequences(text_sequences, maxlen=MAX_TEXT_LEN)

# Multi-Label Binarize the Gentres
multilabel_binarizer = MultiLabelBinarizer()
y = multilabel_binarizer.fit_transform(genres)

import pdb
pdb.set_trace()

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
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_TEXT_LEN))
    model.add(Dropout(rate=.2))
    model.add(GlobalMaxPool1D())
    model.add(Dense(NUM_GENRES, activation='sigmoid'))

    optimizer = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['categorical_accuracy'])

    return model

def create_lstm():
    # TODO
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_TEXT_LEN))
    model.add(SpatialDropout1D(.2))
    model.add(LSTM(100, dropout=.2, recurrent_dropout=.2))
    # model.add(GlobalMaxPool1D())
    model.add(Dense(NUM_GENRES, activation='sigmoid'))

    optimizer = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['categorical_accuracy'])

    return model

# print("Creating Model")
#
# baseline_model = create_baseline()
# print("Model created")
#
# callbacks = [
#     EarlyStopping(patience=4),
#     ModelCheckpoint(filepath='baseline-nn.h5', save_best_only=True)
# ]
#
# history = baseline_model.fit(x_train, y_train,
#                              epochs=NUM_EPOCHS,
#                              batch_size=1,
#                              validation_split=.1,
#                              callbacks=callbacks
#
# )

# lstm = create_lstm()
# print("Model created")
#
# callbacks = [
#     EarlyStopping(patience=4),
#     ModelCheckpoint(filepath='lstm-nn.h5', save_best_only=True)
# ]
#
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
#
# history = lstm.fit(x_train, y_train,
#                              epochs=NUM_EPOCHS,
#                              batch_size=1,
#                              validation_split=.1,
#                              callbacks=callbacks
#
# )

simple_model = keras.models.load_model('lstm-nn.h5')
metrics = simple_model.evaluate(x_test, y_test)
print("{}: {}".format(simple_model.metrics_names[0], metrics[0]))
print("{}: {}".format(simple_model.metrics_names[1], metrics[1]))