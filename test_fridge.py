"""test program, all electrical appliances corresponding to REFIT have a python file of train and test"""
import torch
from data.test_loader import DataGenerator, draw_losses, get_f1_score
from nnet.tans import TAN_Plus
from lossfunc import Metrics
from matplotlib import pyplot as plt
import numpy as np
from data.REFIT_dataset.PreDefine import *
data_set_name = 'REFIT'  # 'REDD', 'UKDale'
app_name = 'fridge'  # todo: 'washingmachine', 'dishwasher', 'microwave', 'fridge', 'kettle'
if data_set_name == 'REDD':
    batch_si = 32
else:
    batch_si = 64
folder_name = './data/REFIT_dataset/'+'refit_64fri_sam8_sen0'+'.pickle'  # todo:
stri = was_appliance[app_name]['stride']

checkpoint = torch.load('check_'+app_name+'/spexlast154.pt')
model = TAN_Plus(L1=3, N=8, B=3, O=8, P=8, Q=3, num_spks=5, spk_embed_dim=8, Head=4, causal=False).cuda()
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']

dataset = DataGenerator(folder_name, batch_size=batch_si, app=[app_name], threshold=False)
val = dataset.va
te = dataset.te
tr = dataset.data
MAX = dataset.MAX


def plot(set, num_batch, begin, stride):
    model.eval()
    with torch.no_grad():
        for key, value in set.items():
            est, clean, mix, auxplt = [], [], [], []
            batch_len = value[0][0].shape[1]
            batch_size = len(value[0][0])
            seq_len = num_batch * len(value[0][0]) * stride + batch_len - stride
            est_sum = np.zeros(seq_len)
            clean_sum = np.zeros(seq_len)
            mix_sum = np.zeros(seq_len)
            count = np.zeros(seq_len)
            for i in range(num_batch):
                mix_wavs = torch.tensor(value[i + begin][0], dtype=torch.float32).cuda()
                aux = torch.tensor(value[i+begin][2], dtype=torch.float32).cuda()
                aux_len = torch.tensor(value[i+begin][3]).cuda()
                rec_sources_wavs = model(mix_wavs, aux, aux_len)
                for j in range(value[0][0].shape[0]):
                    est_sum[(i * batch_size + j) * stride:(i * batch_size + j) * stride + batch_len] += np.array(
                        rec_sources_wavs[0][j, :].cpu() * MAX[key][1])
                    clean_sum[(i * batch_size + j) * stride:(i * batch_size + j) * stride + batch_len] += np.array(
                        value[i][1][j, :] * MAX[key][1])
                    mix_sum[(i * batch_size + j) * stride:(i * batch_size + j) * stride + batch_len] += np.array(
                        value[i][0][j, :] * MAX[key][0])
                    count[(i * batch_size + j) * stride:(i * batch_size + j) * stride + batch_len] += 1
            clean = clean_sum / count
            est = est_sum / count
            est[est < 0] = 0
            mix = mix_sum / count
            plt.figure()
            plt.subplot(len(set), 1, 1)
            p1, = plt.plot(mix, color = 'k')
            p2, = plt.plot(clean)
            p3, = plt.plot(est)
            legend = plt.legend([p1, p2, p3], ['Mains', 'Ground Truth', 'TAN-NILM'], loc='upper left')
            plt.title(key)
            legend.get_frame().set_alpha(0)
            plt.xlim([911000,919000])
            plt.ylim([-10,3000])
            plt.xticks([])
            plt.yticks([])
    plt.show()


def test_whole(set, stride):
    model.eval()
    m = Metrics()
    with torch.no_grad():
        for key, value in set.items():
            est, clean, mix, auxplt = [], [], [], []
            batch_len = value[0][0].shape[1]
            batch_size = value[0][0].shape[0]
            seq_len = len(value)*len(value[0][0])*stride+batch_len-stride
            est_sum = np.zeros(seq_len)
            clean_sum = np.zeros(seq_len)
            mix_sum = np.zeros(seq_len)
            count = np.zeros(seq_len)
            for i in range(len(value)):
                m1wavs = torch.tensor(value[i][0], dtype=torch.float32).cuda()
                aux = torch.tensor(value[i][2], dtype=torch.float32).cuda()
                aux_len = torch.tensor(value[i][3]).cuda()
                rec_sources_wavs = model(m1wavs, aux, aux_len)
                for j in range(value[0][0].shape[0]):
                    est_sum[(i * batch_size + j) * stride:(i * batch_size + j) * stride + batch_len] += np.array(rec_sources_wavs[0][j, :].cpu() * MAX[key][1])
                    clean_sum[(i * batch_size + j) * stride:(i * batch_size + j) * stride + batch_len] += np.array(value[i][1][j,:] * MAX[key][1])
                    count[(i * batch_size + j) * stride:(i * batch_size + j) * stride + batch_len] += 1
            clean = clean_sum/count
            est = est_sum/count
            est[est<0] = 0
            metrics = m.compute_metrics(est, clean)
            print(key + ', mae: ', metrics['regression']['mean_absolute_error'])
            print(key + ', sae: ', metrics['regression']['relative_error_in_total_energy'])
            print(key + ', sae*: ', metrics['regression']['signal_aggregate_error'])


if __name__ == '__main__':
    print(checkpoint['epoch'])
    plot(te, 200, 0, stri)
    test_whole(te, stri)