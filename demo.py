import numpy as np
from sklearn.model_selection import train_test_split
import mne
#读取mat文件信息
import scipy.io as sio
file_path = 'F:\SEED\SEED_EEG\Preprocessed_EEG\/1_20131027.mat'
data = sio.loadmat(file_path) # 读取.mat文件





#读取mat文件信息
if __name__ == '__main__':
    ##file_path = 'F:\SEED\SEED_EEG\Preprocessed_EEG\/1_20131027.mat'
    ##data = sio.loadmat(file_path) # 读取.mat文件
    #查找key
    ##dd = data.keys()
    ##keys = list(data.keys())[3:] # 第一个key是__header__，第二个key是__version__，第三个key是__globals__
    ##print(dd)
 print(data.info)
