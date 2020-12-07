import random

from sklearn.metrics import mean_squared_error

import spotify as sp
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def predict_year(data: DataFrame) -> None:
    # Drop the columns that aren't relevant
    data.drop(['id', 'release_date'], axis=1, inplace=True)

    # Save the identifiers/labels to add back in after
    names = data['name']
    labels = data['year']
    data.drop(['name', 'year'], axis=1, inplace=True)

    # Scale the data
    scaler = MinMaxScaler()
    scaled = DataFrame(scaler.fit_transform(data.iloc[:, data.columns != 'name']), columns=data.columns)


    # The 'name' field should not be used for the model, but I'm keeping it
    #   for printing purposes

    # todo: implement model
    x = scaled
    y = labels

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    '''
    # train the model, but I'm commenting out it. You can just load the model with the following code.
    clf = LinearRegression(random.seed(1))
    clf.fit(x_train, y_train)

    # save the trained model as a file, so that you can load the model and do not need to train the model again
    joblib.dump(clf, 'regression_predict_year.model')
    '''



    clf = joblib.load('regression_predict_year.model')

    #  for printing purposes
    predictions = clf.predict(x_test)
    print("mse: ", mean_squared_error(y_test, predictions))

    # todo: implement evaluation



    return


def main():
    data = sp.parse_file('data.csv')
    predict_year(data)

main()