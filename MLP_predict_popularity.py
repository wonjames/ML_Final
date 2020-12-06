import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix
import spotify as sp
import ast
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer



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
    # scaled.to_csv("todomodel.csv")
    print(scaled)

    # The 'name' field should not be used for the model, but I'm keeping it
    #   for printing purposes


    #implement model
    # data split
    # extract needed column
    cols = [i for i in scaled.columns if i not in ['name', 'popularity']]
    feature_space = scaled[cols]
    label = scaled['popularity']
    # #checkpoint
    # print(feature_space.head())
    # print(label.head())
    x_train, x_test, y_train, y_test = train_test_split(feature_space, label, random_state=3, test_size=0.3)


    # mlp classifier
    print("start to fit data:\n")
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10, 10), max_iter=1000, solver='adam')
    # change the parameter if needed
    mlp.fit(x_train, y_train)
    prediction_mlp = mlp.predict(x_test)
    print("Done!")

    # save model to model.pkl
    joblib.dump(mlp, 'predict_popularity.pkl')


    # todo : model evaluation

    return


def main():
    data = sp.parse_file('data.csv')
    predict_popularity(data)

    return



main()




