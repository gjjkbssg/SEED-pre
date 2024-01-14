import mne
from mne.time_frequency import psd_array_multitaper
from mne_connectivity import spectral_connectivity_epochs
import numpy as np
# 引入前文中构建的文件读取函数
from read_file import read_all_files


def frequency_spectrum(raws):
    # 提取EEG数据在五个频段的能量特征
    # delta(0.5-4Hz) theta(4-8Hz) alpha(8-13Hz) beta(13-30Hz) gamma(30-100Hz)
    # 特定频带
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}
    # 特征矩阵
    feature_matrix = [] # 每个元素是一个特征向量
    # 遍历每个raw
    for raw in raws:
        # 生成频谱特征向量
        feature_vector = []
        # 遍历每个频段
        for band in FREQ_BANDS:
            # 提取每个频段的数据，不打印信息
            raw_band = raw.copy().filter(l_freq=FREQ_BANDS[band][0], h_freq=FREQ_BANDS[band][1], verbose=False)
            # 计算能量
            power = np.sum(raw_band.get_data() ** 2, axis=1) / raw_band.n_times # axis=1表示按行求和
            # 添加到特征向量
            feature_vector.extend(power)
        # 添加到特征矩阵
        feature_matrix.append(feature_vector)
    # 返回特征矩阵
    print("频谱特征矩阵的shape为：{}".format(np.array(feature_matrix).shape))
    print("频谱特征矩阵内容为：{}".format(np.array(feature_matrix)))
    return np.array(feature_matrix, dtype=float)


import numpy as np
import mne


def power_spectrum(raws):
    print("功率谱密度特征提取开始...")
    # 计算每个频段的功率谱密度
    # 特定频带
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}
    # 特征矩阵
    feature_matrix = []
    # 遍历每个raw
    for raw in raws: # raws是一个列表，每个元素是一个raw对象
        # 生成频谱特征向量
        feature_vector = []
        # 遍历每个频段
        for band in FREQ_BANDS:
            # 计算功率谱密度
            power = raw.compute_psd(picks='all', method='welch', fmin=FREQ_BANDS[band][0],
                                           fmax=FREQ_BANDS[band][1], verbose=False)
            # print(power.shape)
            # 添加到特征向量，在第二个维度方向扩展
            for i in range(power.shape[0]):
                feature_vector.extend(power[i])
        # 添加到特征矩阵
        # print(len(feature_vector))
        feature_matrix.append(feature_vector)
    # 将特征矩阵转换为numpy数组
    feature_matrix = np.array(feature_matrix, dtype=object)
    # 返回特征矩阵
    print("功率谱密度特征矩阵的shape为：{}".format(feature_matrix.shape))
    # print("功率谱密度特征矩阵内容为：{}".format(np.array(feature_matrix)))
    print("功率谱密度特征提取结束")
    return feature_matrix


def spectral_connectivity(raws):
    print("相干性特征提取开始...")
    # 计算EEG数据的相干性特征
    # 特征矩阵
    feature_matrix = []
    # 遍历每个raw
    for raw in raws:
        # 生成events, 2s为一个event，重叠0.2s
        events = mne.make_fixed_length_events(raw, duration=2.0, overlap=0.2)
        # print('event.shape: ', events.shape)
        # 生成epochs
        epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, verbose=False)
        # print('epochs.shape: ', epochs.get_data().shape)
        # 计算相干性矩阵
        con = spectral_connectivity_epochs(epochs, method='coh', mode='multitaper', verbose=False)
        # print('con info: ', con)
        feature_matrix.append(con.get_data())
    # 将特征矩阵转换为numpy数组
    feature_matrix = np.array(feature_matrix, dtype=float)
    # 返回特征矩阵
    print("相干性特征矩阵的shape为：{}".format(feature_matrix.shape))
    print("相干性特征矩阵内容为：{}".format(np.array(feature_matrix)))
    print("相干性特征提取结束")
    return feature_matrix




# 时域特征提取

def temporal_feature(raws):
    # 提取EEG数据的时域特征
    print("时域特征提取开始...")
    # 特征矩阵
    feature_matrix = []
    # 遍历每个raw
    for raw in raws:
        # 生成时域特征向量
        feature_vector = []
        # 计算EEG数据的时域特征
        # 1.均值
        mean = raw.get_data().mean(axis=1)
        # 2.方差
        var = raw.get_data().var(axis=1)
        # 3.标准差
        std = raw.get_data().std(axis=1)
        # 4.最大值
        max = raw.get_data().max(axis=1)
        # 5.最小值
        min = raw.get_data().min(axis=1)
        # 6.峰峰值
        p_p = max - min
        # 7.斜率
        slope = np.diff(raw.get_data(), axis=1) # 沿着第二个维度计算差分
        # 8.峭度
        kurtosis = np.mean((raw.get_data() - mean[:, None]) ** 4, axis=1) / var ** 2 - 3
        # 9.偏度
        skewness = np.mean((raw.get_data() - mean[:, None]) ** 3, axis=1) / var ** (3 / 2)
        # 10.能量
        power = np.sum(raw.get_data() ** 2, axis=1) / raw.n_times
        # 11.方根幅值
        rms = np.sqrt(np.mean(raw.get_data() ** 2, axis=1))
        # 12.脑电活动时域特征向量
        feature_vector.extend(mean) # extend()函数用于在列表末尾一次性追加另一个序列中的多个值
        feature_vector.extend(var)
        feature_vector.extend(std)
        feature_vector.extend(max)
        feature_vector.extend(min)
        feature_vector.extend(p_p)
        feature_vector.extend(slope)
        feature_vector.extend(kurtosis)
        feature_vector.extend(skewness)
        feature_vector.extend(power)
        feature_vector.extend(rms)
        # 添加到特征矩阵
        feature_matrix.append(feature_vector)
    # 将特征矩阵转换为numpy数组
    feature_matrix = np.array(feature_matrix, dtype=object) # dtype=object 保证每个元素的类型都是list
    # 返回特征矩阵
    print("时域特征矩阵的shape为：{}".format(feature_matrix.shape))
    print("时域特征矩阵内容为：{}".format(np.array(feature_matrix)))
    return feature_matrix

if __name__ == '__main__':
    raws, label = read_all_files("F:\SEED\SEED_EEG\Preprocessed_EEG/", 1)
    #temporal_feature(raws)
    #frequency_spectrum(raws)
    #spectral_connectivity(raws)
    data = power_spectrum(raws)
    print(type(data))
