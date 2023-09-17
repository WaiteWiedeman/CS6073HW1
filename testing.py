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
import matplotlib.pyplot as plt
import pickle
import warnings


warnings.filterwarnings('ignore')  # ignore warnings with warnings package

# defining function testing to test model performance
# takes list of models and testing features and targets as input
# returns evaluation of test


def testing(models, X_test, Y_test):
    # unpack models from list
    clf = models[0]
    ann_oneL_16 = models[1]
    ann_twoL_32_8 = models[2]
    ann_threeL_32_16_8 = models[3]
    ann_fourL_32_16_8_4 = models[4]
    # test regression model
    # predict target with features
    Y_pred = clf.predict(X_test)
    # print statements for debugging
    # print(clf.predict(X_test))
    # print(clf.score(X_test, Y_test))
    # get mean squared error of regression model
    mse = mean_squared_error(Y_test, Y_pred)
    # print regression mean squared error
    print('linear regression mean squared error: ', mse)
    # evaluate one layer model
    oneLeval = ann_oneL_16.evaluate(X_test, Y_test)
    # evaluate two layer model
    twoLeval = ann_twoL_32_8.evaluate(X_test, Y_test)
    # evaluate three layer model
    threeLeval = ann_threeL_32_16_8.evaluate(X_test, Y_test)
    # evaluate four layer model
    fourLeval = ann_fourL_32_16_8_4.evaluate(X_test, Y_test)
    # return list of model evaluation loss values
    return [oneLeval, twoLeval, threeLeval, fourLeval]

# defining get_plot function to plot evaluation and training loss data
# function takes lists of model histories and model evaluation as input


def get_plot(hist_models):
    # unpack model histories
    oneL_hist = hist_models[0]
    twoL_hist = hist_models[1]
    threeL_hist = hist_models[2]
    fourL_hist = hist_models[3]

    # one layer model plot of loss and validation loss vs epochs
    plt.plot(oneL_hist.history['loss'])
    plt.plot(oneL_hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # two layer model plot of loss and validation loss vs epochs
    plt.plot(twoL_hist.history['loss'])
    plt.plot(twoL_hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # three layer model plot of loss and validation loss vs epochs
    plt.plot(threeL_hist.history['loss'])
    plt.plot(threeL_hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # four layer model plot of loss and validation loss vs epochs
    plt.plot(fourL_hist.history['loss'])
    plt.plot(fourL_hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
