import _pickle
from PreDefine import *
import numpy as np
import shutil
import os

path0 = '.'
tar_ap = [ap_name]  # target app

if data_name == 'REDD':
    batch_si = 32  # batch size
    all_ap = ['dishwasher', 'microwave', 'fridge']  # all app
    f_name = 'redd_'+str(batch_si)+tar_ap[0][0:3]+'_sam6_sen1'
elif data_name == 'UKDale':
    batch_si = 64  # batch size
    all_ap = ['washingmachine', 'dishwasher',
              'microwave', 'fridge', 'kettle']  # all app
    f_name = 'ukdale_'+str(batch_si)+tar_ap[0][0:3]+'_sam6_sen0'
elif data_name == 'REFIT':
    batch_si = 64  # batch size
    all_ap = ['washingmachine', 'dishwasher',
              'microwave', 'fridge', 'kettle']  # all app
    f_name = 'refit_'+str(batch_si)+tar_ap[0][0:3]+'_sam8_sen0'

# check
if os.path.exists(f_name):
    # delete
    shutil.rmtree(f_name)
    print(f"Folder '{f_name}' has been removed!")
else:
    print(f"Folder '{f_name}' does not exist!")


def save(file_name, path, batch_size, all_app, tar_app):
    """By reading the dictionary file of filename, put it in different folders according to tr, te, va,
    and package the dictionary files in each folder according to batch, all_app, tar_app are a list composed of
    the names of all appliances and target appliances respectively"""
    dataset = _pickle.load(open(file_name + '.pickle', 'rb'))  # read
    os.mkdir(path + '/' + file_name)
    os.mkdir(path + '/' + file_name + '/' + 'tr')
    os.mkdir(path + '/' + file_name + '/' + 'va')
    os.mkdir(path + '/' + file_name + '/' + 'te')
    count_tr,count_va,count_te = 0, 0, 0
    thres, prob = {}, {}
    for key in all_app:
        thres[key] = params_appliance[key]['on_power_threshold']
    MAX = {}
    for key in all_app:
        MAX[key] = [612, 612]
    i = 0
    veri_num = len(dataset['tr'])

    for key in all_app:
        class_label = np.ones(batch_size)
        class_label = i * class_label
        if key in tar_app:
            for k in range(len(dataset['tr'][key])):
                dataset['tr'][key][k][0] = dataset['tr'][key][k][0] / MAX[key][0]
                dataset['tr'][key][k][1] = dataset['tr'][key][k][1] / MAX[key][1]
                dataset['tr'][key][k][2] = dataset['tr'][key][k][2] / MAX[key][1]
                dataset['tr'][key][k].append(class_label)
                _pickle.dump(dataset['tr'][key][k], open(path+'/'+file_name+'/'+'tr/'+str(count_tr)+'.pickle', 'wb'))
                count_tr += 1
            for k in range(len(dataset['va'][key])):
                dataset['va'][key][k][0] = dataset['va'][key][k][0] / MAX[key][0]
                dataset['va'][key][k][1] = dataset['va'][key][k][1] / MAX[key][1]
                dataset['va'][key][k][2] = dataset['va'][key][k][2] / MAX[key][1]
                dataset['va'][key][k].append(class_label)
                _pickle.dump(dataset['va'][key][k], open(path+'/'+file_name+'/'+'va/'+str(count_va)+'.pickle', 'wb'))
                count_va += 1
            for k in range(len(dataset['te'][key])):
                dataset['te'][key][k][0] = dataset['te'][key][k][0] / MAX[key][0]
                dataset['te'][key][k][1] = dataset['te'][key][k][1] / MAX[key][1]
                dataset['te'][key][k][2] = dataset['te'][key][k][2] / MAX[key][1]
                dataset['te'][key][k].append(class_label)
                _pickle.dump(dataset['te'][key][k], open(path+'/'+file_name+'/'+'te/'+str(count_te)+'.pickle', 'wb'))
                count_te += 1
        else:  # te doesn't need to calculate regression CEloss, so it doesn't need to be labeled
            for k in range(len(dataset['tr'][key])):
                dataset['tr'][key][k][0] = dataset['tr'][key][k][0] / MAX[key][0]
                dataset['tr'][key][k].append(class_label)
                _pickle.dump(dataset['tr'][key][k], open(path+'/'+file_name+'/'+'tr/'+str(count_tr)+'.pickle', 'wb'))
                count_tr += 1
            for k in range(len(dataset['va'][key])):
                dataset['va'][key][k][0] = dataset['va'][key][k][0] / MAX[key][0]
                dataset['va'][key][k].append(class_label)
                _pickle.dump(dataset['va'][key][k], open(path+'/'+file_name+'/'+'va/'+str(count_va)+'.pickle', 'wb'))
                count_va += 1
        i += 1


if __name__ == '__main__':
    save(file_name=f_name, path=path0, batch_size=batch_si,
         all_app=all_ap, tar_app=tar_ap)
    print('All save over')
