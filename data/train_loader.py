"""
single app version
    1.in order to save Ram, i split the big data into pieces so i don't need to read  the whole dataset into Ram, i can only
read one-batch-data per step. and this file write to read the one-batch-data
    2. different from the old version, this dataset of this project contains 5mixs data (raw,clean,aux,class_label)

----------------------

Class DataGenerator:
    Read in the .pkl datasets generated in datagenerator.py
    and present the batch data for the model
    !!fixed size dataloader(batch must be the batch_size)
    dataloader.py can have any batch whose size are smaller than  batch_size
"""

import numpy as np
import _pickle
import torch
import os
from data.REFIT_dataset.PreDefine import *


def save_checkpoint(name, model, path, optimizer, no_improve_epochs, epoch, batch_size, loss_list, ce_list, best_para=[]):
    """ save model best: the best model"""
    os.makedirs(path, exist_ok=True)
    torch.save({
        'no_improve_epochs':no_improve_epochs,
        'epoch': epoch,
        'batch_size': batch_size,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'trandom_state': torch.get_rng_state(),
        'nrandom_state': np.random.get_state(),
        'mae_loss_list': loss_list,
        'ce_loss_list': ce_list,
        'best_para': best_para
    }, os.path.join(path, name+'.pt'))
    return


class DataGenerator(object):
    def __init__(self, path, batch_size, app, ex=True, norm=True, threshold=False):
        '''pkl_list: .pkl files contaiing the data set'''
        self.batch_size = batch_size
        self.app = app
        self.path = path
        f_list_tr = os.listdir(path+'/tr')
        self.tr_num = len(f_list_tr)
        f_list_va = os.listdir(path + '/va')
        self.va_num = len(f_list_va)
        f_list_te = os.listdir(path + '/te')
        self.te_num = len(f_list_te)

        self.ind_tr = [ind for ind in range(self.tr_num)]
        self.ind_va = [ind for ind in range(self.va_num)]
        self.ind_te = [ind for ind in range(self.te_num)]
        np.random.shuffle(self.ind_tr)
        self.count = 0
        self.count_va = 0
        self.count_te = 0

    def gen_tr(self):
        f = open(self.path + '/tr/' + str(self.ind_tr[self.count]) + '.pickle', 'rb')
        out = _pickle.load(f)
        f.close()
        if self.count >= (self.tr_num-1):
            self.count = 0
            np.random.shuffle(self.ind_tr)
        else:
            self.count += 1
        yield out

    def gen_val(self):
        f = open(self.path + '/va/' + str(self.ind_va[self.count_va]) + '.pickle', 'rb')
        out = _pickle.load(f)
        f.close()
        if self.count_va >= (self.va_num - 1):
            self.count_va = 0
        else:
            self.count_va += 1
        yield out

    def gen_te(self):
        f = open(self.path + '/te/' + str(self.ind_te[self.count_te]) + '.pickle', 'rb')
        out = _pickle.load(f)
        f.close()
        if self.count_te >= (self.te_num - 1):
            self.count_te = 0
        else:
            self.count_te += 1
        yield out
