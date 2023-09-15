import tensorflow as tf
from tensorflow import keras
import csv
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import warnings


warnings.filterwarnings('ignore')


def training(models, X_train, Y_train):
        clf = models[0]
        ann_oneL_16 = models[1]
        ann_twoL_32_8 = models[2]
        ann_threeL_32_16_8 = models[3]
        ann_fourL_32_16_8_4 = models[4]

        # normalize = tf.keras.layers.Normalization()
        # normalize.adapt(df)
        # normalize(x1)

        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        print(clf.predict(X_test))
        print(clf.score(X_test, Y_test))
        mse = mean_squared_error(Y_test, Y_pred)
        print(mse)



def save_model():
        print('yo')
        print('yo')