import numpy as np
import pandas as pd
import sklearn.model_selection
import re
import torch
import math 
import sys
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

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

    df['genres'] = df['genres'].apply(lambda x: [val for val in x if val not in genres]) 

    return df


def replace_genres(df, col, keys, val):
    """
    :param df: Dataframe to filter
    :param col: column to filter by
    :param key: string to replace
    :param val: string to use instead

    """

    df['genres'] = df['genres'].apply(lambda x: [val for val in x if val not in genres]) 

    return df

def replace_genres_d(df, col, dict):
    """
    :param df: Dataframe to filter
    :param col: column to filter by
    :param key: string to replace
    :param val: string to use instead

    """

    df['genres'] = df['genres'].apply(lambda x: [dict.get(val, val) for val in x]) 

    return df


def get_data():
    data = load_data()
    data = format_data(data)

    train, test = sklearn.model_selection.train_test_split(data, test_size=.2)


    counts = data['genres'].explode().value_counts().to_dict()

    remove_genres = {key:val for key, val in counts.items() if val < 124}
    data = filter_genres(data, remove_genres.keys()) # remove genres with too few titles


    replacement_dict = {'Science fiction': ['Speculative fiction','Science Fiction'],
                'Fantasy': ['Alternate history','Dystopia','Adventure novel','Fantasy'],
                'Mystery': ['Crime Fiction','Detective fiction','Mystery'],
                'Suspense': ['Horror','Thriller','Suspense', 'Spy fiction'],
                'Historical fiction': ['Historical fiction','Historical novel'],
                'Non-fiction': ['Autobiography','Non-fiction'],
                'Romance novel': ['Romance novel'],
                'Children\'s literature': ['Children\'s literature'],
                'Young adult literature': ['Young adult literature'],
                'Comedy': ['Comedy']
                }

    replacement_dict = {v: k for k, values in replacement_dict.items() for v in values}

    data = replace_genres_d(data, 'genres', replacement_dict)

    data = filter_genres(data, ['Fiction', 'Novel'])

    data = data[data['genres'].astype(bool)]


    # mlb = MultiLabelBinarizer()
    # data = data.join(pd.DataFrame(mlb.fit_transform(data.pop('genres')),
    #                           columns=mlb.classes_,
    #                           index=data.index))

    data.reset_index(drop=True, inplace=True)
    # print(mlb.classes_)
    # data['genres'] = data[data.columns[1:]].values.tolist()

    return data

def get_train_and_test_data():
    data = load_data()
    data = format_data(data)

    train, test = sklearn.model_selection.train_test_split(data, test_size=.2)

    counts = data['genres'].explode().value_counts().to_dict()

    remove_genres = {key:val for key, val in counts.items() if val < 600}
    data = filter_genres(data, remove_genres.keys()) # remove genres with too few titles

    replacement_dict = {'Science fiction': ['Speculative fiction','Science Fiction'],
                    'Fantasy': ['Alternative history','Dystopia','Adventure novel','Fantasy'],
                    'Mystery': ['Crime Fiction','Detective fiction','Mystery'],
                    'Suspense': ['Horror','Thriller','Suspense'],
                    'Historical fiction': ['Historical fiction','Historical novel'],
                    'Non-fiction': ['Autobiography','Non-fiction'],
                    'Romance novel': ['Romance novel'],
                    'Children\'s literature': ['Children\'s literature'],
                    'Young adult literature': ['Young adult literature']
                    }


    for k in replacement_dict:
        for v in replacement_dict[k]:
            data = replace_genre(data, 'genres', v, k)

    data = replace_genre(data, 'genres', 'Fiction', 'NaN')
    data = replace_genre(data, 'genres', 'Novel', 'NaN')

    data = data[data.genres != 'NaN']

    data['genres'] = data[data.columns[1:]].values.tolist()

    x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(data["summary"], data["genres"], test_size=.2)
    return x_train, y_train, x_test, y_test

# data_n = data.to_numpy()
# unique_elts, count_elts = np.unique(data_n[:, 0], return_counts=True)

'''
data = load_data()
data = format_data(data)
counts = data.genres.explode().value_counts().tolist()
sum_counts = sum(counts)
leftover = sum_counts - sum(counts[0:19])
counts = counts[0:19]
counts.append(leftover)
labels = data.genres.explode().value_counts().keys().tolist()[0:19]
labels.append('Other')
'''


data = get_data()
counts = data.genres.explode().value_counts().tolist()
labels = data.genres.explode().value_counts().keys().tolist()

fig1, ax1 = plt.subplots()
ax1.pie(counts, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=120, labeldistance = 1.05)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.show()
