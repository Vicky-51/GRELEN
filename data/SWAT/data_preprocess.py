import pandas as pd
import numpy as np
import copy
import sys


def downsampling(mat, interval):
    num_row, num_col = mat.shape
    res = num_row%interval
    if res != 0:
        add_num = interval - res
        add_mat = np.zeros((add_num, num_col))
        mat = np.concatenate((mat, add_mat))
    num_row, num_col = mat.shape
    mat_tmp = np.zeros((interval, int(num_row/interval), num_col))
    for i in range(interval):
        mat_tmp[i, ...] = mat[i::interval, :]
    return np.mean(mat_tmp, 0)

def max_min_norm(mat, max_=None, min_=None):
    if max_ is None:
        max_ = np.max(mat, 0)
    if min_ is None:
        min_ = np.min(mat, 0)
    nrow, ncol = mat.shape
    for i in range(ncol):
        if max_[i] == min_[i]:

            mat[:, i] = mat[:, i] - min_[i]
        else:
            mat[:, i] = (mat[:, i] - min_[i]) / (max_[i] - min_[i])
    return mat, max_, min_

def swat_generate(xx, split, length, filename=None, max_=None, min_=None):
    mat_, max_, min_ = max_min_norm(xx, max_, min_)
    nrow, ncol = xx.shape

    xx1 = xx[:int(nrow * split), :]
    xx2 = xx[int(nrow * split):, :]
    train_x = np.zeros((xx1.shape[0] - length + 1, length, ncol))
    for i in range(train_x.shape[0]):
        train_x[i, ...] = xx1[i:i + length, :]
    valid_x = np.zeros((xx2.shape[0] - length + 1, length, ncol))
    for i in range(valid_x.shape[0]):
        valid_x[i, ...] = xx2[i:i + length, :]
    all_data = {
        'train': {
            'x': np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2),
            'target': np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2),
        },
        'val': {
            'x': np.expand_dims(np.transpose(valid_x, (0, 2, 1)), 2),
            'target': np.expand_dims(np.transpose(valid_x, (0, 2, 1)), 2),
        },
        'stats': {
            '_mean': np.zeros((1, 1, 3, 1)),
            '_std': np.zeros((1, 1, 3, 1)),
        }
    }
    if filename == None:
        return max_, min_, all_data
    np.savez_compressed(filename,
                        train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                        val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                        mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                        )

    return max_, min_, all_data


def swat_generate_test(xx, length, filename=None, max_=None, min_=None):

    mat_, max_, min_ = max_min_norm(xx, max_, min_)
    nrow, ncol = xx.shape

    train_x = np.zeros((xx.shape[0] - length + 1, length, ncol))
    for i in range(train_x.shape[0]):
        train_x[i, ...] = xx[i:i + length, :]
    all_data = {
        'test': {
            'x': np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2),  # (7456, 89, 1, 24)
            'target': np.expand_dims(np.transpose(train_x, (0, 2, 1)), 2),
        },
        'stats': {
            '_mean': np.zeros((1, 1, 3, 1)),
            '_std': np.zeros((1, 1, 3, 1)),
        }
    }
    if filename == None:
        return max_, min_, all_data
    np.savez_compressed(filename,
                        test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                        mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                        )

    return max_, min_, all_data


if __name__ == '__main__':
    sys.path.append("../..")
    from config_files.SWAT_config import Config
    config = Config()
    swat_normal = pd.read_csv('SWaT_Dataset_Normal_v0.csv')
    swat_abnormal = pd.read_csv('SWaT_Dataset_Attack_v0.csv')
    swat_normal_np = np.array(swat_normal.iloc[:, 1: -1])
    swat_abnormal_np = np.array(swat_abnormal.iloc[:, 1: -1])

    train_x = downsampling(swat_normal_np, config.downsampling_fre)[3000:, :]
    split = 0.8
    length = config.target_len
    max_, min_, all_data = swat_generate(copy.copy(train_x), split, length, 'train_swat.npz')
    test_x = downsampling(swat_abnormal_np, config.downsampling_fre)
    # max_, min_, all_data_test = swat_generate_test(copy.copy(test_x), length, 'test_swat.npz')
    max_, min_, all_data_test = swat_generate_test(copy.copy(test_x), length, 'test_swat.npz', copy.copy(max_), copy.copy(min_))

    test_gr = all_data_test['test']['x']
    test_old = np.load("/home/zwq/Test/test_60_complete (1).npz")['train_x']

