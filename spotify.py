import ast
import pandas
import numpy as np
import re

from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder
from typing import List


def parse_file(file_name: str) -> DataFrame:
    data: DataFrame = pandas.read_csv(file_name, header=0)

    # Can uncomment this line if we want to only use a subset of the rows
    data = data.iloc[0:10]

    artists_df = DataFrame(columns=['artists'])
    for i, line in enumerate(data['artists']):
        # convert the string to a list
        # line = re.sub('\'', '', line)
        # artists_df = artists_df.append({'artists': line}, ignore_index=True)
        converted = ast.literal_eval(line)
        artists_df = artists_df.append({'artists': converted}, ignore_index=True)

        # artists = list()
        # for artist in converted:
        #     artists.append(artist)
        #     if artist not in artists:
        #         artists.append(artist)
        # artists = ''.join(artists)
        # print(artists)
        # artists_df = artists_df.append({'artists': artists}, ignore_index=True)
    print(artists_df)
    # data = data['artists'].str.join('|').str.get_dummies()
    mlb = MultiLabelBinarizer()
    test = DataFrame(mlb.fit_transform(artists_df['artists']), columns=mlb.classes_, index=artists_df.index)
    # test = artists_df['artists'].str.join('').str.get_dummies()

    print(test)
    # OneHotEncode the 'artists' field
    # enc = OneHotEncoder()
    # artists: List[str] = list()

    # # artists = artists.values.reshape(-1, 1)
    # enc.fit(artists_df)
    # print(enc.categories_)
    # data.drop('artists', axis=1, inplace=True)

    return data


def predict_popularity(data: DataFrame) -> None:
    # Drop the columns that aren't relevant
    # data.drop(['id', 'release_date', 'artists'], axis=1, inplace=True)
    data.drop(['id', 'release_date'], axis=1, inplace=True)

    # Save the identifiers/labels to add back in after
    names = data['name']
    labels = data['popularity']
    data.drop(['name', 'popularity'], axis=1, inplace=True)

    # Scale the data
    scaler = MinMaxScaler()
    scaled = DataFrame(scaler.fit_transform(data.iloc[:, data.columns != 'name']), columns=data.columns)

    # Add the identifiers/labels back to the dataframe
    scaled['name'] = names
    scaled['popularity'] = labels
    print(scaled)

    # The 'name' field should not be used for the model, but I'm keeping it
    #   for printing purposes

    # todo: implement model
    return


def main():
    data = parse_file('data.csv')
    # predict_popularity(data)
    return


main()
