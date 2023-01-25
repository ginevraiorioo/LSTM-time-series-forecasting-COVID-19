import pandas as pd
import os
import pathlib
import absl.logging
import data_handler
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    # getting directory's path and changing its format to string
    currDir = str(pathlib.Path(__file__).parent.resolve())

    # asking the user to choose the dataset to study
    print("\n\nWhich csv file would you like to open?\n")
    i = 1
    for file in os.listdir(currDir + '/csv_files'):
        if file == '.DS_Store':
            continue
        print(str(i) + ' - ' + file)
        i = i+1

    # asking for an input
    print("\nInsert number: ")
    file_num = input()

    # checking for wrong inputs
    if not file_num.isdigit():
        file_num = 0

    while not (1 <= int(file_num) <= 14):
        print("Invalid number! Try again: ")
        file_num = input()

        if not file_num.isdigit():
            file_num = 0

    # loading file and its title
    df, dataset_name = data_handler.switch_file(str(file_num), currDir)

    # indexing the data with the Date column, after transforming it to datetime
    df.index = pd.to_datetime(df['Date'], format='%d/%b/%Y')

    # asking the user to choose one of the LSTM univariate models
    print("\nWhich LSTM univariate model would you like to use?\n")
    print("1 - Vanilla LSTM\n2 - Stacked LSTM\n3 - Bidirectional LSTM\n4 - CNN LSTM\n5 - ConvLSTM\n")
    print("Insert number: ")
    model_num = input()

    # checking for wrong inputs
    if not file_num.isdigit():
        file_num = 0

    while not (1 <= int(file_num) <= 5):
        print("Invalid number! Try again: ")
        file_num = input()

        if not file_num.isdigit():
            file_num = 0

    # executing the chosen model
    data_handler.switch_model(str(model_num), df, dataset_name)


main()
