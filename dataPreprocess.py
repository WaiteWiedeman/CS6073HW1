# import packages
import tensorflow as tf
from tensorflow import keras
import csv
import sklearn
import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
import warnings


warnings.filterwarnings('ignore')  # ignore warnings with warnings package


# defining pre_process function to preprocess data
# parameter csv_filename is a string
# output is training, test, and evaluation features and target data in a list


def pre_process(csv_filename):
    df = pd.read_csv(csv_filename, encoding='latin-1')  # reads csv into df variable
    # print statements for debugging
    #print(df.head)
    #print(df.describe())
    #print(df.info())
    #print(df.isna().sum())
    # binnedInc feature is a set of two values separated by a comma
    # following code replaces the two binnedInc values with their average
    df['binnedInc'] = df['binnedInc'].str.replace('(', '')  # remove parentheses from string
    df['binnedInc'] = df['binnedInc'].str.replace('[', '')  # remove brackets from string
    df['binnedInc'] = df['binnedInc'].str.replace(']', '')
    x = df['binnedInc'].str.split(',', expand=True).astype(float)  # convert string to float
    y = (x[0] + x[1]) / 2  # mean of two values from binnedInc
    df['binnedInc'] = y  # replace binnedInc column with mean
    #print(df.head())  # print statement for debug
    # following three lines replaces missing values with median values for the respective features
    df['PctEmployed16_Over'] = df['PctEmployed16_Over'].fillna(df['PctEmployed16_Over'].median())
    df['PctSomeCol18_24'] = df['PctSomeCol18_24'].fillna(df['PctSomeCol18_24'].median())
    df['PctPrivateCoverageAlone'] = df['PctPrivateCoverageAlone'].fillna(df['PctPrivateCoverageAlone'].median())

    #print(df.isna().sum())  # print statement for debug

    x1 = df.drop(['TARGET_deathRate', 'Geography'], axis=1)  # drops target values and geography from features
    sc = StandardScaler().fit(x1)  # scales features
    x1 = sc.transform(x1)
    normalizer = Normalizer(norm='l2').fit(x1)  # normalizes features
    x1 = normalizer.transform(x1)

    y1 = df['TARGET_deathRate']  # target array
    # train, validate, test percentages
    train_ratio = 0.80
    validation_ratio = 0.10
    test_ratio = 0.10
    # split data into train, test, and validate
    X_train, X_test, Y_train, Y_test = train_test_split(x1, y1, test_size=1 - train_ratio, random_state=1)  #
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,test_size=test_ratio / (test_ratio + validation_ratio),random_state=1)

    #print(X_train, Y_train, X_test, Y_test, X_val, Y_val)  # print statement for debug

    return [X_train, Y_train, X_test, Y_test, X_val, Y_val]  # returns list of train, test, and validate features and targets

