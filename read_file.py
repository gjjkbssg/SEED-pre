import mne
import scipy.io as sio
import os

# 超参数
# 通道名顺序
ch_names = ['FP1', 'FPz', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
            'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
            'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
            'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
            'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
            'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
            'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'] # 64个
# 采样频率
sfreq = 200
# 每个.mat文件中的数据label
basic_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1] # 15个


def read_one_file(file_path):
    """
    input:单个.mat文件路径
    output:raw格式数据
    """
    data = sio.loadmat(file_path) # 读取.mat文件
    # 获取keys并转化为list，获取数据所在key
    keys = list(data.keys())[3:] # 第一个key是__header__，第二个key是__version__，第三个key是__globals__,第四个key才是数据
    # print(keys)
    # 获取数据
    raw_list = [] # 用于存储每个.mat文件中的15段数据
    for i in range(len(keys)):
        # 获取数据
        stamp = data[keys[i]] # stamp是一个numpy数组
        # print(stamp.shape)
        # 创建info
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        # 创建raw，取第5秒开始的数据
        raw = mne.io.RawArray(stamp, info).crop(tmin=5) # crop()函数用于截取数据，raw是一个raw格式的数据
        # 添加到raw_list
        raw_list.append(raw) #  为什么要添加到raw_list？ 不能直接返回raw吗？
    return raw_list # raw_list中的每个元素是一个raw格式的数据，共15个，raw是list格式吗？ 是的raw.info 怎么取出来？ raw.info['sfreq']


def read_all_files(path, max_files_num=1): # max_files_num=1是为了测试方便
    # 读取文件夹下所有.mat文件
    print("read_all_files start...")
    # 遍历Preprocessed_EEG文件夹下所有.mat文件
    data_list = []
    # 读取文件数量（每个文件中有15段数据）
    files_num = 0
    for root, dirs, files in os.walk(path): # root是当前目录路径，dirs是当前路径下所有子目录，files是当前路径下所有非目录子文件
        for file in files: # 遍历所有文件
            if os.path.splitext(file)[1] == '.mat': # 判断文件后缀是否为.mat
                file_path = os.path.join(root, file) # 拼接文件路径
                raw_list = read_one_file(file_path) # 读取单个文件
                # 将raw_list中的每一个元素添加到data_list
                data_list.extend(raw_list) # data_list中的每个元素是一个raw格式的数据
                files_num += 1 # 读取文件数量+1
                if files_num == max_files_num:
                    break

    # 生成所有数据的label（每个文件中有15段数据，每段数据的label相同）
    label_list = []
    for i in range(int(files_num)):
        label_list.extend(basic_label)
    # 将label_list添加到data_list
    print("共读取了{}个文件".format(files_num))
    print("共有{}段数据".format(len(data_list)))
    print("read ended...")
    return data_list, label_list

if __name__ == '__main__':
    path = 'F:\SEED\SEED_EEG\Preprocessed_EEG\/1_20131027.mat'
    data = read_one_file(path )
    print(type(data))


