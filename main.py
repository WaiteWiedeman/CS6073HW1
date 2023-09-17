# import functions from files
from dataPreprocess import pre_process
from model import get_model
from training import training
from training import save_model
from testing import testing
from testing import get_plot
import warnings
from sys import argv


warnings.filterwarnings('ignore')  # ignore warnings with warnings package

def main():
    argument = argv[0]  # run main in terminal
    csv_filename = 'cancer_reg.csv'  # declare filename of csv
    data = pre_process(csv_filename)  # store pre_process results in variable 'data'
    # unpack training, test, and validation features and targets from list
    X_train = data[0]
    Y_train = data[1]
    X_test = data[2]
    Y_test = data[3]
    X_val = data[4]
    Y_val = data[5]

    learn_rate = 0.00001  # declare learning rate variable for get_model function
    models = get_model(learn_rate)  # run get_model and store results in 'models'
    epochs = 20  # declare number of epochs
    hist_models = training(models, X_train, Y_train, X_val, Y_val, epochs)
    save_model(models)
    evals = testing(models, X_test, Y_test)
    # unpack testing data
    oneLeval = evals[0]
    twoLeval = evals[1]
    threeLeval = evals[2]
    fourLeval = evals[3]
    #unpack models
    ann_oneL_16 = models[1]
    ann_twoL_32_8 = models[2]
    ann_threeL_32_16_8 = models[3]
    ann_fourL_32_16_8_4 = models[4]
    # print model test data
    print('one layer test loss: ', oneLeval)
    print('two layer test loss: ', twoLeval)
    print('three layer test loss: ', threeLeval)
    print('four layer test loss: ', fourLeval)
    # save model weights
    ann_fourL_32_16_8_4.save_weights('bestfourlayermodelweights')
    get_plot(hist_models)


if __name__ == '__main__':
    main()  # run main
