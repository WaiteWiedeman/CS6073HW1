# import packages
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
from tensorflow.keras.optimizers.legacy import SGD
import warnings


warnings.filterwarnings('ignore')  # ignore warnings with warnings package

# defining get_model function to build neural network and linear regression models
# takes float input parameter learn_rate
# returns list of models


def get_model(learn_rate):
    clf = LinearRegression()  # assign sklearn linear regression model to variable

    # Neural networks
    # Stochastic Gradient Descent optimizer with learning rate
    opt = SGD(learning_rate=learn_rate)  # tf.keras.optimizers.legacy.
    # ANN-oneL-16 model
    ann_oneL_16 = tf.keras.models.Sequential()
    ann_oneL_16.add(tf.keras.Input(shape=(32,)))  # input size 32
    ann_oneL_16.add(tf.keras.layers.Dense(16))  # add layer of size 16
    # compile model for testing
    ann_oneL_16.compile(optimizer=opt, loss='MeanSquaredError')
    # ANN-twoL-32-8
    ann_twoL_32_8 = tf.keras.models.Sequential()
    ann_twoL_32_8.add(tf.keras.Input(shape=(32,)))
    ann_twoL_32_8.add(tf.keras.layers.Dense(32))  # add layer of size 32
    ann_twoL_32_8.add(tf.keras.layers.Dense(8))  # add layer of size 8

    ann_twoL_32_8.compile(optimizer=opt, loss='MeanSquaredError')
    # ANN-threeL-32-16-8
    ann_threeL_32_16_8 = tf.keras.models.Sequential()
    ann_threeL_32_16_8.add(tf.keras.Input(shape=(32,)))
    ann_threeL_32_16_8.add(tf.keras.layers.Dense(32))  # add layer of size 32
    ann_threeL_32_16_8.add(tf.keras.layers.Dense(16))  # add layer of size 16
    ann_threeL_32_16_8.add(tf.keras.layers.Dense(8))  # add layer of size 8

    ann_threeL_32_16_8.compile(optimizer=opt, loss='MeanSquaredError')
    # ANN-fourL-32-16-8-4
    ann_fourL_32_16_8_4 = tf.keras.models.Sequential()
    ann_fourL_32_16_8_4.add(tf.keras.Input(shape=(32,)))
    ann_fourL_32_16_8_4.add(tf.keras.layers.Dense(32))  # add layer of size 32
    ann_fourL_32_16_8_4.add(tf.keras.layers.Dense(16))  # add layer of size 16
    ann_fourL_32_16_8_4.add(tf.keras.layers.Dense(8))  # add layer of size 8
    ann_fourL_32_16_8_4.add(tf.keras.layers.Dense(4))  # add layer of size 4

    ann_fourL_32_16_8_4.compile(optimizer=opt, loss='MeanSquaredError')

    # return list of models
    return [clf, ann_oneL_16, ann_twoL_32_8, ann_threeL_32_16_8, ann_fourL_32_16_8_4]