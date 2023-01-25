import pandas as pd
import matplotlib.pyplot as plt
import univariate_models
from numpy import array
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


# function that splits a sequence into input and target datasets, based on the number of chosen time steps
# if a sequence s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] is split with n_steps = 3
# the first input dataset would be X = [0, 1, 2] and target dataset y = [3]
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    i = 1

    while i < (len(sequence) - n_steps):
        # finding the end of this pattern
        end_pat = i + n_steps

        # gathering input and output parts of the pattern
        X.append(sequence[i:end_pat])
        y.append(sequence[end_pat])

        i = i + 1

    return array(X), array(y)


# function that splits a sequence into train, validation and test sets
def split_samples(sequence1, sequence2):
    train_length, val_length = define_lengths(sequence1)
    X_train, y_train = sequence1[:train_length], sequence2[:train_length]
    X_val, y_val = sequence1[train_length:train_length+val_length], sequence2[train_length:train_length+val_length]
    X_test, y_test = sequence1[train_length+val_length:], sequence2[train_length+val_length:]

    return X_train, y_train, X_val, y_val, X_test, y_test


# function that defines the length in percentage of the train and validation sets
def define_lengths(seq):
    train_length = int(len(seq)*0.8)
    val_length = int(len(seq)*0.1)

    return train_length, val_length


def build_model(model, X_train, y_train, X_val, y_val, X_test):
    # compiling model
    model.compile(optimizer='adam', loss='mse')
    # model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

    # fitting model
    cp1 = ModelCheckpoint('model1/', save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp1])

    # modeling the testing dataset and printing the results
    model1 = load_model('model1/')
    test_predictions = model1.predict(X_test).flatten()

    return test_predictions


def plot_results(y_test, test_predictions, test_dates, title):

    # plotting the testing dataset in comparison with the actual dataset
    test_results = pd.DataFrame(data={'Test Predictions': test_predictions, 'Actuals': y_test})
    plt.plot(test_dates, test_results['Test Predictions'], label='Test Predictions')
    plt.plot(test_dates, test_results['Actuals'], label='Actual Values')
    plt.xlabel('Dates')
    plt.ylabel('Cases')
    plt.legend()
    plt.title(title)
    plt.show(block=True)


# function that, depending on the number chosen by the user, reads the right csv file
def switch_file(num, currDir):
    if num == "1":
        return pd.read_csv(currDir + '/csv_files/cases.csv'), 'COVID Cases'
    elif num == "2":
        return pd.read_csv(currDir + '/csv_files/discharged.csv'), 'Discharged Patients'
    elif num == "3":
        return pd.read_csv(currDir + '/csv_files/allCOVbeds.csv'), 'All Hospital COVID Beds'
    elif num == "4":
        return pd.read_csv(currDir + '/csv_files/accumulatedCases.csv'), 'Accumulated COVID Cases'
    elif num == "5":
        return pd.read_csv(currDir + '/csv_files/accumulatedDis.csv'), 'Accumulated Discharged Patients'
    elif num == "6":
        return pd.read_csv(currDir + '/csv_files/inCOVpatients.csv'), 'Incoming COVID Patients'
    elif num == "7":
        return pd.read_csv(currDir + '/csv_files/inTOTpatients.csv'), 'Total Incoming Patients'
    elif num == "8":
        return pd.read_csv(currDir + '/csv_files/resPatients.csv'), 'Respirator Patients'
    elif num == "9":
        return pd.read_csv(currDir + '/csv_files/deaths.csv'), 'COVID Deaths'
    elif num == "10":
        return pd.read_csv(currDir + '/csv_files/COVnoRes.csv'), 'COVID Patients Without Respirator'
    elif num == "11":
        return pd.read_csv(currDir + '/csv_files/COVwithRes.csv'), 'COVID Patients With Respirator'
    elif num == "12":
        return pd.read_csv(currDir + '/csv_files/allBeds.csv'), 'All Hospital Beds'
    elif num == "13":
        return pd.read_csv(currDir + '/csv_files/noResPatients.csv'), 'Patients Without Respirator'
    elif num == "14":
        return pd.read_csv(currDir + '/csv_files/accumulatedDeaths.csv'), 'Accumulated COVID Deaths'


# function that, depending on the number chosen by the user, calls the right function
def switch_model(num, df, dataset_name):
    if num == "1":
        return univariate_models.vanilla_LSTM(df, dataset_name)
    elif num == "2":
        return univariate_models.stacked_LSTM(df, dataset_name)
    elif num == "3":
        return univariate_models.bidirectional_LSTM(df, dataset_name)
    elif num == "4":
        return univariate_models.CNN_LSTM(df, dataset_name)
    elif num == "5":
        return univariate_models.conv_LSTM(df, dataset_name)
