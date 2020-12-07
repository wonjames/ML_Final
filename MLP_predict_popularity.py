import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPRegressor
import spotify as sp
import ast
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


def predict_popularity(data: DataFrame) -> None:
    # Drop the columns that aren't relevant
    data.drop(['id', 'release_date'], axis=1, inplace=True)
    print(data)
    labels = data.loc[:,'popularity']
    # Scale the data
    scaler = MinMaxScaler()
    scaled = DataFrame(scaler.fit_transform(data[data.columns.difference(['name','popularity'])]))

    print("\n")

    print(scaled)

    # The 'name' field should not be used for the model, but I'm keeping it
    #   for printing purposes


    #implement model
    # data split
    cols = [i for i in scaled.columns if i not in ['name', 'popularity']]
    feature_space = scaled[cols]
    label = labels

    x_train, x_test, y_train, y_test = train_test_split(feature_space, label, random_state=3, test_size=0.3)

    #lr clssifier
    print("start to fit data:\n")
    mlp = MLPRegressor( max_iter=5000, solver='adam')
    # change the parameter if needed
    mlp.fit(x_train, y_train)
    prediction_mlp = mlp.predict(x_test)
    print("Done!")
    print("mse: ", mean_squared_error(y_test, prediction_mlp))
    print(mlp.score(x_test,y_test))

    # save model to model.pkl
    joblib.dump(mlp, 'predict_popularity.pkl')


    # todo : model evaluation

    return


def main():
    data = sp.parse_file2('data.csv')
    predict_popularity(data)

    return

main()




