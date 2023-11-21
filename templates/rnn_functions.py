import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def get_month(date):
    d = datetime.strptime('{}'.format(date), '%m/%d/%Y')
    return d.month


def get_weekday(date):
    d = datetime.strptime('{}'.format(date), '%m/%d/%Y')
    return d.weekday()


def prepare_data(data_frame, features):
    if 'Month' in features:
        data_frame['Month'] = pd.DataFrame(data=data_frame['Date']).applymap(get_month)
    if 'WeekDay' in features:
        data_frame['WeekDay'] = pd.DataFrame(data=data_frame['Date']).applymap(get_weekday)

    data_frame = data_frame[features].values

    #    pp=make_column_transformer(
    #        (['Open'], 'passthrough' ),
    #        (['Month'], OneHotEncoder(categories=[range(1,13)])),
    #        (['WeekDay'], OneHotEncoder(categories=[range(5)]))
    #        )
    #
    #    data_frame=pp.fit_transform(data_frame).toarray()

    return data_frame


def load_dataset(input_file, n_test_examples, features):
    input_set = pd.read_csv(input_file)
    input_set = input_set.dropna()

    training_set = input_set[:len(input_set) - n_test_examples]
    training_set = training_set[features].values

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    test_set = input_set[len(input_set) - n_test_examples:]
    test_set = test_set[features].values

    test_set_scaled = sc.transform(test_set)

    return training_set, training_set_scaled, sc, test_set, test_set_scaled


def build_series(inputs, n_obs):
    A = []
    for v in range(inputs.shape[1]):
        X = []
        for i in range(n_obs, len(inputs)):
            X.append(inputs[i - n_obs:i, v])
        X = np.array(X)
        A.append(X)
    A = np.swapaxes(np.swapaxes(np.array(A), 0, 1), 1, 2)
    return A


def build_regressor(layers, input_shape, dropout=0.2, optimizer='adam', loss_function='mean_squared_error'):
    # Initialising the RNN
    regressor = Sequential()

    n_layers = len(layers)

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=layers[0], return_sequences=True, input_shape=input_shape))
    regressor.add(Dropout(dropout))

    if n_layers > 1:
        for i in range(1, n_layers - 1):
            # Adding another LSTM layer and some Dropout regularisation
            regressor.add(LSTM(units=layers[i], return_sequences=True))
            regressor.add(Dropout(dropout))

    # Adding the last LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=layers[-1]))
    regressor.add(Dropout(dropout))

    # Adding the output layer
    regressor.add(Dense(units=1))

    # Compiling the RNN
    regressor.compile(optimizer=optimizer, loss=loss_function)

    return regressor


def get_subset(training_set_scaled, real_stock_price_scaled, n_obs):
    dataset_total = np.concatenate((training_set_scaled, real_stock_price_scaled), axis=0)
    inputs = dataset_total[len(training_set_scaled) - n_obs:]
    return inputs


def apply_inverse_transform(predicted_stock_price, n_features, sc):
    # creation d'une matrice intermédiaire avec 4 colonnes de zéros
    trainPredict_dataset_like = np.zeros(shape=(len(predicted_stock_price), n_features))
    # rajout en première colonne du vecteur de résultats prédits
    trainPredict_dataset_like[:, 0] = predicted_stock_price[:, 0]
    # inversion et retour vers les valeurs non scalées
    predicted_stock_price = sc.inverse_transform(trainPredict_dataset_like)[:, 0]
    return predicted_stock_price


def load_regressor(file_name='regressor'):
    from keras.models import model_from_json
    json_file = open('{}.json'.format(file_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    regressor = model_from_json(loaded_model_json)
    # load weights into new model
    regressor.load_weights("{}.h5".format(file_name))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    print("Loaded model from disk")
    return regressor


def save_regressor(regressor, file_name='regressor'):
    model_json = regressor.to_json()
    with open("{}.json".format(file_name), "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        regressor.save_weights("{}.h5".format(file_name))
    print("Saved model to disk")


def plot_stock_prices(real_stock_price, predicted_stock_price):
    plt.plot(real_stock_price[:, 0], color='red', label='Real Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()