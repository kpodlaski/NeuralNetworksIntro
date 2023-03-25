import pandas as pd
import numpy as np
import os

from sklearn.preprocessing._encoders import OneHotEncoder
from sklearn.preprocessing._label import LabelEncoder

def create_acc_from_raw_uci_har(type):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    path_in = "../../datasets/UCI_HAR_Dataset/{0}/Inertial_Signals/total_acc_{1}_{0}.txt"
    path_out = "../../datasets/UCI_HAR_Dataset/{0}/total_acc_{0}.txt"
    print(os.getcwd())
    acc_X = np.loadtxt(path_in.format(type,"X"))
    acc_Y = np.loadtxt(path_in.format(type, "Y"))
    acc_Z = np.loadtxt(path_in.format(type, "Z"))
    # print(acc_X.shape, acc_Y.shape, acc_Z.shape)
    # print(acc_X[0,0],acc_Y[0,0],acc_Z[0,0])
    acc = np.sqrt(acc_X**2+acc_Y**2+acc_Z**2)
    np.savetxt(path_out.format(type),acc)

def read_acc_data_one_hot():
    train_X, train_Y, test_X, test_Y = read_acc_data()
    train_Y = OneHotEncoder().fit_transform(train_Y).toarray()
    test_Y = OneHotEncoder().fit_transform(test_Y).toarray()
    return train_X, train_Y, test_X, test_Y

def read_acc_data():
    actual_path = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    path_X = "../../datasets/UCI_HAR_Dataset/{0}/total_acc_{0}.txt"
    path_Y = "../../datasets/UCI_HAR_Dataset/{0}/y_{0}.txt"
    train_X = pd.read_csv(path_X.format("train"), header=None, delim_whitespace=True).values
    train_Y = pd.read_csv(path_Y.format("train"), header=None, delim_whitespace=True).values
    test_X = pd.read_csv(path_X.format("test"), header=None, delim_whitespace=True).values
    test_Y = pd.read_csv(path_Y.format("test"), header=None, delim_whitespace=True).values
    os.chdir(actual_path)
    return train_X, train_Y, test_X, test_Y

#create_acc_from_raw_uci_har("train")
#create_acc_from_raw_uci_har("test")
#train_X, train_Y, test_X, test_Y = read_acc_data()
