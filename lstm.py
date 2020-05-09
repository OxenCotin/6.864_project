import numpy as np
import utils
import keras
from keras.preprocessing.text import Tokenizer
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, MaxPool1D, GlobalMaxPool1D, SpatialDropout1D, CuDNNLSTM, CuDNNGRU, SimpleRNN, Conv1D, RNN, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.activations import sigmoid
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

RANDOM_SEED = 420

VOCAB_SIZE = 5000
EMBEDDING_DIM = 64
MAX_TEXT_LEN = 500
OOV_TOKEN = "<OOV>"
NUM_GENRES = 8

INIT_LR = .2
NUM_EPOCHS = 30

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

# import pdb
# pdb.set_trace()

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

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def create_baseline():
    """
    Dummy baseline Model
    :return: model, simple fully connected 1 hidden layer
    """

    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_TEXT_LEN))
    model.add(SpatialDropout1D(rate=.2))
    model.add(GlobalMaxPool1D())
    model.add(Dense(NUM_GENRES, activation='sigmoid'))

    optimizer = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['categorical_accuracy'])

    return model

def create_lstm():
    # TODO
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_TEXT_LEN))
    model.add(SpatialDropout1D(rate=.2))
    # model.add(MaxPool1D())
    model.add(LSTM(100, dropout=.2, recurrent_dropout=.2))
    model.add(Dense(NUM_GENRES, activation='sigmoid'))

    optimizer = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['categorical_accuracy'])

    return model

def create_lstm_f1():
    # TODO
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_TEXT_LEN))
    model.add(SpatialDropout1D(rate=.2))
    # model.add(MaxPool1D())
    model.add(LSTM(100, dropout=.2, recurrent_dropout=.2))
    model.add(Dense(NUM_GENRES, activation='sigmoid'))

    optimizer = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)

    model.compile(optimizer=optimizer, loss=f1_loss, metrics=['categorical_accuracy', f1])

    return model

def create_cnn():
    # TODO
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_TEXT_LEN))
    model.add(SpatialDropout1D(.2))
    model.add(Conv1D(filters=100, kernel_size=5))
    model.add(SpatialDropout1D(.15))
    model.add(GlobalMaxPool1D())
    model.add(Dense(NUM_GENRES, activation='sigmoid'))

    optimizer = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['categorical_accuracy', f1])

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

# lstm = create_lstm_f1()
# print("Model created")
#
# callbacks = [
#     EarlyStopping(patience=4),
#     ModelCheckpoint(filepath='cuddnnlstm-f1-loss-nn.h5', save_best_only=True)
# ]
#
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
#
# history = lstm.fit(x_train, y_train,
#                              epochs=NUM_EPOCHS,
#                              batch_size=32,
#                              validation_split=.1,
#                              callbacks=callbacks
#
# )

simple_model = keras.models.load_model('lstm-f1-loss-nn.h5', custom_objects={'loss': f1_loss})
# metrics = simple_model.evaluate(x_test, y_test)

def f(n):
    return 1 if n > .5 else 0

import pdb
pdb.set_trace()
y_pred = simple_model.predict(x_test, verbose=1)
f_v = np.vectorize(f)
y_pred = f_v(y_pred)

print(classification_report(y_test, y_pred))

# print("{}: {}".format(simple_model.metrics_names[0], metrics[0]))
# print("{}: {}".format(simple_model.metrics_names[1], metrics[1]))
