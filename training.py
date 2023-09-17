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
import pickle
import warnings


warnings.filterwarnings('ignore')  # ignore warnings with warnings package

# defining function training to train models
# takes list of models, training features and targets, validation
# features and targets, and desired number of epochs as input parameters
# returns model training histories in a list


def training(models, X_train, Y_train, X_val, Y_val, epochs):
        # unpack models from list
        clf = models[0]
        ann_oneL_16 = models[1]
        ann_twoL_32_8 = models[2]
        ann_threeL_32_16_8 = models[3]
        ann_fourL_32_16_8_4 = models[4]

        clf.fit(X_train, Y_train)  # fit linear regression
        # one layer model history
        oneL_hist = ann_oneL_16.fit(X_train, Y_train, batch_size=5, epochs=epochs, validation_data=(X_val, Y_val))
        # two layer model history
        twoL_hist = ann_twoL_32_8.fit(X_train, Y_train, batch_size=20, epochs=epochs, validation_data=(X_val, Y_val))
        # three layer model history
        threeL_hist = ann_threeL_32_16_8.fit(X_train, Y_train, batch_size=5, epochs=epochs, validation_data=(X_val, Y_val))
        # four layer model history
        fourL_hist = ann_fourL_32_16_8_4.fit(X_train, Y_train, batch_size=5, epochs=epochs, validation_data=(X_val, Y_val))
        # return list of model training histories
        return [oneL_hist, twoL_hist, threeL_hist, fourL_hist]

# defining function save_model that saves all trained models
# takes list of models as input parameter


def save_model(trained_models):
        # unpack models from list
        clf = trained_models[0]
        ann_oneL_16 = trained_models[1]
        ann_twoL_32_8 = trained_models[2]
        ann_threeL_32_16_8 = trained_models[3]
        ann_fourL_32_16_8_4 = trained_models[4]
        # save regression model
        reg_filename = 'reg_model.sav'
        pickle.dump(clf, open(reg_filename, 'wb'))
        # save neural network models
        ann_oneL_16.save('ann_oneL_16.keras')
        ann_twoL_32_8.save('ann_twoL_32_8.keras')
        ann_threeL_32_16_8.save('ann_threeL_32_16_8.keras')
        ann_fourL_32_16_8_4.save('ann_fourL_32_16_8_4.keras')


