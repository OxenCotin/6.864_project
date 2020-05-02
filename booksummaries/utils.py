import numpy as np
import pandas as pd
import sklearn.model_selection
import re
import torch
import math 
import sys

BOOK_SUMMARY_FILE = "clean_summaries.csv"


def load_data():
    names = ["article_id", "freebase_id", "book_title", "author", "pub_date", "genres",
    "summary"]

    book_summary_data = pd.read_csv(BOOK_SUMMARY_FILE)
    print("loaded")
    return book_summary_data


def format_data(df):
    """

    Remove all other columns other than genres and summary. Get genres from string
    of dictionary to list of genres

    :param df: Dataframe to format
    :return: df trimmed and formatted
    """

    # Cols taken currently, only genres and summaries


    df["genres"] = df["genres"].apply(format_genres)
    df = df[["genres", "summary"]]

    # Scramble
    return df

def format_genres(s):
    """

    :param s: string rep of genres from original source
    :return: list of strings: list of strings of genres
    """

    regex = '(?<=: \").+?(?=\")'
    genres = re.findall(regex, s)

    return genres

def filter_genres(df, genres):
    """

    :param df: Dataframe to filter
    :param genres: list of strings, genres to filter
    :return: df with filtered genres
    """
    # TODO
    pass

def get_train_and_test_data():
    data = load_data()
    data = format_data(data)

    train, test = sklearn.model_selection.train_test_split(data, test_size=.2)

    # TODO: Filter genres and One Hot,

    x_train, y_train, x_test, y_test = train[:, 1], train[:, 0], test[:, 1], test[:0]
    
    return x_train, y_train, x_test, y_test


data = load_data()
data = format_data(data)
counts = data['genres'].explode().value_counts()

# data_n = data.to_numpy()
# unique_elts, count_elts = np.unique(data_n[:, 0], return_counts=True)


