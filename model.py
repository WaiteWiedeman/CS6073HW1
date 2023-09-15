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


def get_model(learn_rate):
    print('yo')
    clf = LinearRegression()

    # Neural networks
    # optimizer with learning rate
    opt = tf.keras.optimizers.SGD(learning_rate=learn_rate)
    # ANN-oneL-16
    ann_oneL_16 = tf.keras.models.Sequential()
    ann_oneL_16.add(tf.keras.Input(shape=(32,)))
    ann_oneL_16.add(tf.keras.layers.Dense(16))

    ann_oneL_16.compile(optimizer=opt, loss='MeanSquaredError', metrics=['accuracy', 'mse'])
    # ANN-twoL-32-8
    ann_twoL_32_8 = tf.keras.models.Sequential()
    ann_twoL_32_8.add(tf.keras.Input(shape=(32,)))
    ann_twoL_32_8.add(tf.keras.layers.Dense(32))
    ann_twoL_32_8.add(tf.keras.layers.Dense(8))

    ann_twoL_32_8.compile(optimizer=opt, loss='MeanSquaredError', metrics=['accuracy', 'mse'])
    # ANN-threeL-32-16-8
    ann_threeL_32_16_8 = tf.keras.models.Sequential()
    ann_threeL_32_16_8.add(tf.keras.Input(shape=(32,)))
    ann_threeL_32_16_8.add(tf.keras.layers.Dense(32))
    ann_threeL_32_16_8.add(tf.keras.layers.Dense(16))
    ann_threeL_32_16_8.add(tf.keras.layers.Dense(8))

    ann_threeL_32_16_8.compile(optimizer=opt, loss='MeanSquaredError', metrics=['accuracy', 'mse'])
    # ANN-fourL-32-16-8-4
    ann_fourL_32_16_8_4 = tf.keras.models.Sequential()
    ann_fourL_32_16_8_4.add(tf.keras.Input(shape=(32,)))
    ann_fourL_32_16_8_4.add(tf.keras.layers.Dense(32))
    ann_fourL_32_16_8_4.add(tf.keras.layers.Dense(16))
    ann_fourL_32_16_8_4.add(tf.keras.layers.Dense(8))
    ann_fourL_32_16_8_4.add(tf.keras.layers.Dense(4))

    ann_fourL_32_16_8_4.compile(optimizer=opt, loss='MeanSquaredError', metrics=['accuracy', 'mse'])

    # return
    return [clf, ann_oneL_16, ann_twoL_32_8, ann_threeL_32_16_8, ann_fourL_32_16_8_4]