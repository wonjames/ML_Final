import ast
import pandas

from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer


def parse_file(file_name: str) -> DataFrame:
    data: DataFrame = pandas.read_csv(file_name, header=0)

    print('Encoding the data...')
    # Can modify this line depending on how many rows we want to use
    data = data.iloc[0:10000]

    artists = DataFrame(columns=['artists'])
    for i, line in enumerate(data['artists']):
        # convert the string to a list
        converted = ast.literal_eval(line)
        artists = artists.append({'artists': converted}, ignore_index=True)

    mlb = MultiLabelBinarizer()
    artists = DataFrame(mlb.fit_transform(artists['artists']), columns=mlb.classes_, index=artists.index)

    # drop the artists column; no longer needed
    data.drop('artists', axis=1, inplace=True)
    # Add the encoded artists back to the original dataframe
    for col in artists:
        data[col] = artists[col]

    print('Done.')
    return data


def predict_popularity(data: DataFrame) -> None:
    # Drop the columns that aren't relevant
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
    print(scaled.head())

    # The 'name' field should not be used for the model, but I'm keeping it
    #   for printing purposes

    # todo: implement model
    return

'''
def main():
    data = parse_file('data.csv')
    predict_popularity(data)
    return


main()


'''


