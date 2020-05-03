import numpy as np
import torch
from nltk.corpus import stopwords
import re

STOPWORDS = set(stopwords.words('english'))
REPLACE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS = re.compile('[^0-9a-z #+_]')

def tf_idf(summary):
    """
    Calculate tf_idf score for a given summary

    :param summary: string of book summary to analyze
    :return: score
    """
    # TODO: implement this
    pass


def clean_summary(text):
    """
    Remove StopWords, punctuation, and send to lower
    :param text:
    :return:
    """
    text = text.lower()
    text = REPLACE.sub(' ', text)
    text = BAD_SYMBOLS.sub('', text)

    # Filter out stopwords
    text = ''.join([word for word in text.split() if word not in STOPWORDS])
    return text


def tokenize():

