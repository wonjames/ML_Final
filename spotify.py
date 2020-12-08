import ast
import pandas

from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder


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


def parse_file2(file_name: str) -> DataFrame:
    data: DataFrame = pandas.read_csv(file_name, header=0)

    print('Encoding the data...')
    data=data.dropna()
    data =data.loc[data['year'].isin(['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020'])]
    data = data.tail(4200)

    #labelencoder the artists
    data['artists']=LabelEncoder().fit_transform(data['artists'].tolist())

    # artists = DataFrame(columns=['artists'])
    # for i, line in enumerate(data['artists']):

    print('Done.')
    return data


def predict_popularity(data: DataFrame) -> None:
    # Drop the columns that aren't relevant
    data.drop(['id', 'release_date'], axis=1, inplace=True)
    print(data)

    # Save the identifiers/labels to add back in after
    # names = data.loc[:,'name']
    # labels = data.loc[:,'popularity']
    # data.drop(['name', 'popularity'], axis=1, inplace=True)

    # Scale the data
    scaler = MinMaxScaler()
    scaled = DataFrame(scaler.fit_transform(data[data.columns.difference(['name','popularity'])]))
    print(scaled)
    mv_abalone = scaled.isnull().sum().sort_values(ascending=False)
    pmv_abalone = (mv_abalone / len(scaled)) * 100
    missing_abalone = pandas.concat([mv_abalone, pmv_abalone], axis=1, keys=['Missing value', '% Missing'])
    print(missing_abalone)
    print("\n")
    # print(names)
    # print(labels,"\n")
    # # Add the identifiers/labels back to the dataframe
    # scaled['name'] = names
    # scaled['popularity'] = labels
    print(scaled)
    # checking missing value
    mv_abalone = scaled.isnull().sum().sort_values(ascending=False)
    pmv_abalone = (mv_abalone / len(scaled)) * 100
    missing_abalone = pandas.concat([mv_abalone, pmv_abalone], axis=1, keys=['Missing value', '% Missing'])
    print(missing_abalone)
    # The 'name' field should not be used for the model, but I'm keeping it
    #   for printing purposes

    # todo: implement model
    return

