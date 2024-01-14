import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import feature_ext
from read_file import read_all_files
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

if __name__ == '__main__':
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    # 读取数据，假设 read_all_files 函数正确读取 EEG 数据和标签
    raws, labels = read_all_files('F:\\SEED\\SEED_EEG\\Preprocessed_EEG\\', 1)

    # 特征提取，假设 temporal_feature 函数能从原始数据中提取出特征
    feature_matrix = feature_ext.temporal_feature(raws)
    feature_matrix1 = feature_ext.power_spectrum(raws)
    print(type(feature_matrix))
    print(type(feature_matrix1))

    # 将特征矩阵转换为 numpy 数组
    feature_matrix = np.array(feature_matrix)
    feature_matrix1 = np.array(feature_matrix1)
    #归一化
    feature_matrix = preprocessing.scale(feature_matrix)
    #setting an array element with a sequence.

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.2, random_state=42) # 80% 作为训练集，20% 作为测试集, random_state 是随机数种子, 保证每次运行程序时划分的训练集和测试集都相同
    X1_train, X1_test, y1_train, y1_test = train_test_split(feature_matrix1, labels, test_size=0.2, random_state=42) # 80% 作为训练集，20% 作为测试集, random_state 是随机数种子, 保证每次运行程序时划分的训练集和测试集都相同
    # 使用 SVM 模型
    clf = svm.SVC()
    clf.fit(X_train, y_train)  # 训练模型
    print('temporal_feature')
    print(clf.score(X_test, y_test))  # 输出测试集上的准确度



    # 使用 LogisticRegression 模型
    clf = LogisticRegression()
    clf.fit(X_train, y_train)  # 训练模型
    print("temporal_feature")
    print(clf.score(X_test, y_test))  # 输出测试集上的准确度

    #使用SVM模型
    clf = svm.SVC()
    clf.fit(X1_train, y1_train)  # 训练模型
    print("power_spectrum")
    print(clf.score(X1_test, y1_test))  # 输出测试集上的准确度

    # 使用 LogisticRegression 模型
    clf = LogisticRegression()
    clf.fit(X1_train, y1_train)  # 训练模型

    print("power_spectrum")
    print(clf.score(X1_test, y1_test))  # 输出测试集上的准确度



