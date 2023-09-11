# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import tensorflow as tf
from tensorflow import keras
import csv
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def pre_process():
    df = pd.read_csv("cancer_reg.csv", encoding='latin-1')

    print(df.head)
    print(df.describe())
    print(df.info())
    df['binnedInc'] = df['binnedInc'].str.replace('(', '')
    df['binnedInc'] = df['binnedInc'].str.replace('[', '')
    df['binnedInc'] = df['binnedInc'].str.replace(']', '')
    x = df['binnedInc'].str.split(',', expand=True).astype(float)
    y = (x[0] + x[1]) / 2
    df['binnedInc'] = y
    print(df.head())
    df['PctEmployed16_Over'] = df['PctEmployed16_Over'].fillna(df['PctEmployed16_Over'].median())
    df['PctSomeCol18_24'] = df['PctSomeCol18_24'].fillna(df['PctSomeCol18_24'].median())
    df['PctPrivateCoverageAlone'] = df['PctPrivateCoverageAlone'].fillna(df['PctPrivateCoverageAlone'].median())

    print(df.isna().sum())

    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    x1 = df.drop(['TARGET_deathRate', 'Geography'], axis=1) #'PctSomeCol18_24','PctPrivateCoverageAlone',
    sc = StandardScaler()
    x1 = sc.fit_transform(x1)
    y1 = df['TARGET_deathRate']

    X_train, X_test, Y_train, Y_test = train_test_split(x1, y1, test_size=1 - train_ratio , random_state=1) #
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=test_ratio / (test_ratio + validation_ratio), random_state=1)
    print(X_train, Y_train, X_test, Y_test, X_val, Y_val)
    clf = LinearRegression()
    clf.fit(X_train, Y_train)
    print(clf.predict(X_test))
    print(clf.score(X_test,Y_test))


def load_data():

    print('yo')



def main():
    pre_process()
    load_data()


if __name__ == '__main__':
    main()
