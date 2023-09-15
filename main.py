from dataPreprocess import pre_process
from model import get_model
import warnings
from sys import argv
warnings.filterwarnings('ignore')


def main():
    argument = argv[0]
    filename = 'cancer_reg.csv'
    data = pre_process(filename)
    X_train = data[0]
    Y_train = data[1]
    X_test = data[2]
    Y_test = data[3]
    X_val = data[4]
    Y_val = data[5]
    models = get_model()


    print('yo')

if __name__ == '__main__':
    main()
