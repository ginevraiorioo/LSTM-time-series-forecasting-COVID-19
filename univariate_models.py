import data_handler
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


def vanilla_LSTM(df, dataset_name):
    # choosing a number of time steps
    n_steps = 3

    # defining the input sequence
    raw_seq = df['NumCases']

    # splitting the sequence into samples
    X, y = data_handler.split_sequence(raw_seq, n_steps)

    # reshaping from [samples, timesteps] into [samples, timesteps, features]
    # one feature since we are working with a univariate sequence
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # splitting samples into training, validation and testing sets
    X_train, y_train, X_val, y_val, X_test, y_test = data_handler.split_samples(X, y)

    # finding the dates for which the testing set is defined, to be used on the x axis of the graph
    test_dates = df.index[len(X_train) + len(X_val):len(X_train) + len(X_val) + len(X_test)]

    # ---------------- VANILLA LSTM MODEL ----------------
    model = Sequential()
    model.add(LSTM(units=32, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(units=1))
    model.summary()

    # compiling, fitting and predicting function
    test_predictions = data_handler.build_model(model, X_train, y_train, X_val, y_val, X_test)

    # plotting results
    data_handler.plot_results(y_test, test_predictions, test_dates, 'Vanilla LSTM on ' + dataset_name)


def bidirectional_LSTM(df, dataset_name):
    # choosing a number of time steps
    n_steps = 3

    # defining the input sequence
    raw_seq = df['NumCases']

    # splitting the sequence into samples with the previously defined function
    X, y = data_handler.split_sequence(raw_seq, n_steps)

    # reshaping from [samples, timesteps] into [samples, timesteps, features]
    # one feature since we are working with a univariate sequence
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # splitting samples into training, validation and testing sets
    X_train, y_train, X_val, y_val, X_test, y_test = data_handler.split_samples(X, y)

    # finding the dates for which the testing set is defined, to be used on the x axis of the graph
    test_dates = df.index[len(X_train)+len(X_val):len(X_train)+len(X_val)+len(X_test)]

    # ---------------- BIDIRECTIONAL LSTM MODEL ----------------
    model = Sequential()
    model.add(Bidirectional(LSTM(units=32, activation='relu'), input_shape=(n_steps, n_features)))
    model.add(Dense(units=1))
    model.summary()

    # compiling, fitting and predicting function
    test_predictions = data_handler.build_model(model, X_train, y_train, X_val, y_val, X_test)

    # plotting results
    data_handler.plot_results(y_test, test_predictions, test_dates, 'Bidirectional LSTM on '+ dataset_name)


def CNN_LSTM(df, dataset_name):
    # choosing a number of time steps
    n_steps = 4

    # defining the input sequence
    raw_seq = df['NumCases']

    # splitting the sequence into samples with the previously defined function
    X, y = data_handler.split_sequence(raw_seq, n_steps)

    # reshaping from [samples, timesteps] into [samples, timesteps, features]
    # one feature since we are working with a univariate sequence
    n_features = 1
    n_seq = 2
    n_steps = 2
    X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

    # splitting samples into training, validation and testing sets
    X_train, y_train, X_val, y_val, X_test, y_test = data_handler.split_samples(X, y)

    # finding the dates for which the testing set is defined, to be used on the x axis of the graph
    test_dates = df.index[len(X_train)+len(X_val):len(X_train)+len(X_val)+len(X_test)]

    # ---------------- CNN LSTM MODEL ----------------
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                              input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(units=32, activation='relu'))
    model.add(Dense(units=1))
    model.summary()

    # compiling, fitting and predicting function
    test_predictions = data_handler.build_model(model, X_train, y_train, X_val, y_val, X_test)

    # plotting results
    data_handler.plot_results(y_test, test_predictions, test_dates, 'CNN LSTM on ' + dataset_name)


def stacked_LSTM(df, dataset_name):
    # choosing a number of time steps
    n_steps = 3

    # defining the input sequence
    raw_seq = df['NumCases']

    # splitting the sequence into samples with the previously defined function
    X, y = data_handler.split_sequence(raw_seq, n_steps)

    # reshaping from [samples, timesteps] into [samples, timesteps, features]
    # one feature since we are working with a univariate sequence
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # print(X.shape, y.shape)

    # splitting samples into training, validation and testing sets
    X_train, y_train, X_val, y_val, X_test, y_test = data_handler.split_samples(X, y)

    # finding the dates for which the testing set is defined, to be used on the x axis of the graph
    test_dates = df.index[len(X_train)+len(X_val):len(X_train)+len(X_val)+len(X_test)]

    # ---------------- STACKED LSTM MODEL ----------------
    model = Sequential()
    model.add(LSTM(units=32, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(units=32, activation='relu'))
    model.add(Dense(units=1))
    model.summary()

    # compiling, fitting and predicting function
    test_predictions = data_handler.build_model(model, X_train, y_train, X_val, y_val, X_test)

    # plotting results
    data_handler.plot_results(y_test, test_predictions, test_dates, 'Stacked LSTM on ' + dataset_name)


def conv_LSTM(df, dataset_name):
    # choosing a number of time steps
    n_steps = 4

    # defining the input sequence
    raw_seq = df['NumCases']

    # splitting the sequence into samples with the previously defined function
    X, y = data_handler.split_sequence(raw_seq, n_steps)

    # reshaping from [samples, timesteps] into [samples, timesteps, features]
    # one feature since we are working with a univariate sequence
    n_features = 1
    n_seq = 2
    n_steps = 2
    X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
    # print(X.shape, y.shape)

    # splitting samples into training, validation and testing sets
    X_train, y_train, X_val, y_val, X_test, y_test = data_handler.split_samples(X, y)

    # finding the dates for which the testing set is defined, to be used on the x axis of the graph
    test_dates = df.index[len(X_train)+len(X_val):len(X_train)+len(X_val)+len(X_test)]

    # ---------------- CONVLSTM MODEL ----------------
    model = Sequential()
    model.add(
        ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
    model.add(Flatten())
    model.add(Dense(units=1))
    model.summary()

    # compiling, fitting and predicting function
    test_predictions = data_handler.build_model(model, X_train, y_train, X_val, y_val, X_test)

    # plotting results
    data_handler.plot_results(y_test, test_predictions, test_dates, 'Vanilla LSTM on ' + dataset_name)
