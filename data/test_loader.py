import numpy as np
import _pickle
import math
import _pickle
from matplotlib import pyplot as plt
import random
from data.REFIT_dataset.PreDefine import *


class DataGenerator(object):
    def __init__(self, pkl, batch_size, app, ex=True, threshold=False):
        '''pkl_list: .pkl files contaiing the data set'''
        self.batch_size = batch_size
        self.app = app
        dict = _pickle.load(open(pkl, 'rb'))
        samples = dict['tr']
        valid = dict['va']
        test = dict['te']
        thres = {}
        for key in samples:
            if key == 'dish washer':
                key = 'dishwasher'
            if threshold:
                thres[key] = params_appliance[key]['on_power_threshold']
            else:
                thres[key] = 0
        else:
            MAX = {}
            for key in samples:
                if key == 'dish washer':
                    key = 'dishwasher'
                MAX[key] = [612, 612]
        self.MAX = MAX
        if ex:
            data = {}
            data_va = {}
            data_te = {}
            i = 0
            j = 0
            m = 0
            veri_num = len(samples)
            for key, value in samples.items():
                if key == 'dish washer':
                    key = 'dishwasher'
                if key in app:
                    class_label = np.ones(batch_size)
                    class_label = i*class_label
                    i += 1
                    for k in range(len(value)):
                        value[k][0] = value[k][0] / MAX[key][0]
                        value[k][1] = value[k][1] / MAX[key][1]
                        value[k][2] = value[k][2] / MAX[key][1]
                    for lis in value:
                        lis.append(class_label)
                    data[key] = value
                else:
                    continue
            for key, value in valid.items():
                if key == 'dish washer':
                    key = 'dishwasher'
                if key in app:
                    for k in range(len(value)):
                        value[k][0] = value[k][0] / MAX[key][0]
                        value[k][1] = value[k][1] / MAX[key][1]
                        value[k][2] = value[k][2] / MAX[key][1]
                    class_label = np.ones(batch_size)
                    class_label = j * class_label
                    j += 1
                    for lis in value:
                        lis.append(class_label)
                    data_va[key] = value
                else:
                    continue
            for key, value in test.items():
                if key == 'dish washer':
                    key = 'dishwasher'
                if key in app:
                    for k in range(len(value)):
                        value[k][0] = value[k][0] / MAX[key][0]
                        value[k][1] = value[k][1] / MAX[key][1]
                        value[k][2] = value[k][2] / MAX[key][1]
                    class_label = np.ones(batch_size)
                    class_label = m * class_label
                    m += 1
                    for lis in value:
                        lis.append(class_label)
                    data_te[key] = value
                else:
                    continue
        self.data = data
        self.va = data_va
        self.te = data_te
        del data, data_te, data_va
        batch_dic = {}


def draw_losses(lis):
    tr_loss = []
    va_loss = []
    te_loss = []
    for i in lis[0]:
        tr_loss.append(i.cpu().detach())
    for i in lis[1]:
        va_loss.append(i.cpu().detach())
    for i in lis[2]:
        te_loss.append(i.cpu().detach())
    plt.plot(tr_loss)
    plt.plot(va_loss)
    plt.plot(te_loss)
    plt.title('Losses in tr, va and te:')
    plt.legend(['train', 'validation', 'test'])
    plt.show()
    pass


def get_f1_score(y_hat, y_true, THRESHOLD):
    # https://blog.csdn.net/zjn295771349/article/details/84961596#/
    epsilon = 1e-7
    y_hat = y_hat > THRESHOLD
    y_hat = np.int8(y_hat)
    y_true = y_true > THRESHOLD
    y_true = np.int8(y_true)
    tp = np.sum(y_hat * y_true, axis=0)
    fp = np.sum(y_hat * (1 - y_true), axis=0)
    fn = np.sum((1 - y_hat) * y_true, axis=0)
    p = tp / (tp + fp + epsilon)  # The significance of epsilon is to prevent the denominator from being 0
    r = tp / (tp + fn + epsilon)

    f1 = 2 * p * r / (p + r + epsilon)
    f1 = np.where(np.isnan(f1), np.zeros_like(f1), f1)

    return np.mean(f1)

