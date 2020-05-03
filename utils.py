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

    
    for g in genres:
        df = df[~df['genres'].apply(lambda x: g in x)] 

    return df


def replace_genre(df, col, key, val):
    """
    :param df: Dataframe to filter
    :param col: column to filter by
    :param key: string to replace
    :param val: string to use instead

    """

    m = [key in v for v in df[col]]
    df.loc[m, col] = val

    return df


def get_train_and_test_data():
    data = load_data()
    data = format_data(data)

    train, test = sklearn.model_selection.train_test_split(data, test_size=.2)

    # TODO: Filter genres and One Hot,

    x_train, y_train, x_test, y_test = train[:, 1], train[:, 0], test[:, 1], test[:0]

    return x_train, y_train, x_test, y_test


data = load_data()
data = format_data(data)
counts = data['genres'].explode().value_counts().to_dict()

remove_genres = {key:val for key, val in counts.items() if val < 87}
data = filter_genres(data, remove_genres.keys())


replacement_dict = {'Science fiction': ['Speculative fiction', 'Hard science fiction'],
                    'Mystery': ['Detective fiction', 'Crime fiction'],
                    'Historical novel': ['Historical fiction', 'History']}


data = [[replace_genre(data, 'genres', v, k) for v in replacement_dict[k]] for k in replacement_dict]

print(data)

# data_n = data.to_numpy()
# unique_elts, count_elts = np.unique(data_n[:, 0], return_counts=True)


